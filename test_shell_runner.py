import sys
import types
import unittest
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
sys.modules.setdefault("fastapi", fastapi_stub)

from agent_library.common import task_plan_context
from agent_library.agents.shell_runner import (
    _build_preprocess_prompt,
    _execute_command_with_retries,
    _shell_repair_max_attempts,
    handle_event,
)


class ShellRunnerDataFlowTests(unittest.TestCase):
    def test_task_plan_context_exposes_structured_context(self):
        context = task_plan_context(
            {
                "task": "check whether any listed patient is named Test",
                "original_task": "list patients, then check whether any listed patient is named Test",
                "target_agent": "shell_runner",
                "previous_step_result": {"rows": [{"PatientName": "Alice"}]},
            }
        )
        self.assertIn("Structured workflow context JSON:", context.execution_task)
        self.assertEqual(context.structured_context["previous_step_result"]["rows"][0]["PatientName"], "Alice")

    def test_preprocess_prompt_mentions_structured_input(self):
        prompt = _build_preprocess_prompt("check this list", {"rows": [{"PatientName": "Test"}]})
        self.assertIn("Structured workflow input JSON", prompt)
        self.assertIn("stdin", prompt)

    def test_explicit_shell_command_receives_instruction_input_on_stdin(self):
        captured = {}

        class _Completed:
            stdout = "true\n"
            stderr = ""
            returncode = 0

        def fake_run(_args, input=None, **_kwargs):
            captured["input"] = input
            return _Completed()

        payload = {
            "task": "check prior rows",
            "target_agent": "shell_runner",
            "instruction": {
                "operation": "run_command",
                "command": "python3 -c 'print(1)'",
                "input": [{"PatientName": "Test"}],
            },
        }
        with patch("agent_library.agents.shell_runner.subprocess.run", side_effect=fake_run):
            response = handle_event(types.SimpleNamespace(event="task.plan", payload=payload))

        self.assertEqual(captured["input"], '[{"PatientName": "Test"}]')
        emitted = response["emits"][0]
        self.assertEqual(emitted["event"], "shell.result")
        self.assertEqual(emitted["payload"]["stdout"], "true")

    def test_derived_shell_command_receives_structured_context_on_stdin(self):
        captured = {}

        class _Completed:
            stdout = "false\n"
            stderr = ""
            returncode = 0

        def fake_run(_args, input=None, **_kwargs):
            captured["input"] = input
            return _Completed()

        payload = {
            "task": "check whether anyone in this list is named Test",
            "original_task": "list patients, then check whether anyone in this list is named Test",
            "target_agent": "shell_runner",
            "previous_step_result": {
                "rows": [{"PatientName": "Alice"}],
                "step_id": "step1",
            },
        }
        with patch("agent_library.agents.shell_runner._derive_command_from_task", return_value="python3 -c 'print(0)'"), patch(
            "agent_library.agents.shell_runner.subprocess.run", side_effect=fake_run
        ):
            response = handle_event(types.SimpleNamespace(event="task.plan", payload=payload))

        self.assertIn('"current_step": "check whether anyone in this list is named Test"', captured["input"])
        self.assertIn('"previous_step_result"', captured["input"])
        emitted = response["emits"][0]
        self.assertEqual(emitted["event"], "shell.result")
        self.assertEqual(emitted["payload"]["stdout"], "false")


class ShellRunnerRepairTests(unittest.TestCase):
    def test_shell_repair_max_attempts_defaults_to_ten(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_shell_repair_max_attempts(), 10)

    def test_shell_command_retries_until_success(self):
        calls = []

        def fake_execute(command, stdin_data=None):
            calls.append((command, stdin_data))
            if len(calls) == 1:
                return {"command": command, "stdout": "Mounted\n/dev\n/boot/efi", "stderr": "ValueError: invalid literal for int()", "returncode": 1}
            return {"command": command, "stdout": "42.75", "stderr": "", "returncode": 0}

        def fake_repair(task, failing_command, error_text, stdout_text="", previous_repair_command="", previous_repair_error=""):
            self.assertIn("free space", task.lower())
            self.assertIn("invalid literal", error_text.lower())
            self.assertIn("/boot/efi", stdout_text)
            return "python -c \"print(42.75)\""

        with patch("agent_library.agents.shell_runner._execute_command_once", side_effect=fake_execute), patch(
            "agent_library.agents.shell_runner._repair_command_from_failure", side_effect=fake_repair
        ), patch.dict("os.environ", {"SHELL_AGENT_MAX_REPAIR_ATTEMPTS": "3"}, clear=False):
            result, stats = _execute_command_with_retries(
                "python -c \"print(round(int('Mounted /dev') / (1024**3), 2))\"",
                "How much free space do I have on this machine and compute the size in GB?",
            )

        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "42.75")
        self.assertEqual(stats["shell_repair_attempts"], 1.0)
        self.assertEqual(len(calls), 2)

    def test_shell_command_stops_after_max_retries(self):
        def fake_execute(command, stdin_data=None):
            return {"command": command, "stdout": "", "stderr": "still bad", "returncode": 1}

        def fake_repair(task=None, failing_command=None, error_text=None, stdout_text="", previous_repair_command="", previous_repair_error=""):
            return "printf bad"

        with patch("agent_library.agents.shell_runner._execute_command_once", side_effect=fake_execute), patch(
            "agent_library.agents.shell_runner._repair_command_from_failure", side_effect=fake_repair
        ), patch.dict("os.environ", {"SHELL_AGENT_MAX_REPAIR_ATTEMPTS": "2"}, clear=False):
            result, stats = _execute_command_with_retries("printf bad", "bad task")

        self.assertEqual(result["returncode"], 1)
        self.assertEqual(stats["shell_repair_attempts"], 2.0)


if __name__ == "__main__":
    unittest.main()
