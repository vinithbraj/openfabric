import sys
import types
import unittest
from unittest.mock import patch


pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.common import run_local_reducer_loop


class LocalReducerLoopTests(unittest.TestCase):
    def test_reducer_loop_repairs_after_failure(self):
        commands = []

        class _Completed:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def fake_run(command, **_kwargs):
            commands.append(command)
            if command == "bad command":
                return _Completed(1, stderr="syntax error")
            return _Completed(0, stdout="final reduced output\n")

        def generate_command(previous_command, previous_error):
            if not previous_error:
                return "bad command"
            self.assertEqual(previous_command, "bad command")
            self.assertIn("syntax error", previous_error)
            return "good command"

        with patch("agent_library.common.subprocess.run", side_effect=fake_run):
            result = run_local_reducer_loop(
                {"rows": [{"value": 1}]},
                generate_command,
                validate_output=lambda output: bool(output.strip()),
            )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.output, "final reduced output")
        self.assertEqual(result.command, "good command")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(commands, ["bad command", "good command"])


if __name__ == "__main__":
    unittest.main()
