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

    def get(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


class _HTTPExceptionStub(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.HTTPException = _HTTPExceptionStub
existing_fastapi = sys.modules.get("fastapi")
if existing_fastapi is not None:
    existing_fastapi.FastAPI = _FastAPIStub
    existing_fastapi.HTTPException = _HTTPExceptionStub
else:
    sys.modules["fastapi"] = fastapi_stub

pydantic_stub = types.ModuleType("pydantic")


class _BaseModelStub:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModelStub
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.agents.slurm_runner import (
    _deterministic_reduce_slurm_result,
    _deterministic_slurm_command,
    _instruction_to_command,
    _parse_command_input,
)
from agent_library.agents.slurm_runner import _gateway_base_url, _result_payload
from dep_agent_library.slurm_gateway_agent.app import _allowed_commands, _resolve_command


class SlurmRunnerTests(unittest.TestCase):
    def test_parse_command_input_from_string(self):
        command, args = _parse_command_input("squeue -u vinith")
        self.assertEqual(command, "squeue")
        self.assertEqual(args, ["-u", "vinith"])

    def test_instruction_cluster_status(self):
        command, args = _instruction_to_command({"operation": "cluster_status"}, "show nodes")
        self.assertEqual(command, "sinfo")
        self.assertEqual(args, ["-Nel"])

    def test_result_payload_marks_control_kind(self):
        payload = _result_payload(
            "scancel",
            ["12345"],
            {"returncode": 0, "stdout": "", "stderr": "", "duration_ms": 12.5},
            {},
        )
        self.assertEqual(payload["result"]["kind"], "control")
        self.assertEqual(payload["detail"], "Slurm control command executed.")

    def test_gateway_base_url_uses_host_and_port(self):
        with patch.dict("os.environ", {"SLURM_GATEWAY_HOST": "10.0.0.8", "SLURM_GATEWAY_PORT": "9001"}, clear=False):
            self.assertEqual(_gateway_base_url(), "http://10.0.0.8:9001")

    def test_is_slurm_task_new_keywords(self):
        from agent_library.agents.slurm_runner import _is_slurm_task
        self.assertTrue(_is_slurm_task("check gpu availability"))
        self.assertTrue(_is_slurm_task("scheduler status"))
        self.assertTrue(_is_slurm_task("list compute nodes"))
        self.assertFalse(_is_slurm_task("list files"))

    def test_instruction_to_command_scancel(self):
        # We need to test how scancel is handled via execute_command
        instruction = {
            "operation": "execute_command",
            "command": "scancel 12345"
        }
        command, args = _instruction_to_command(instruction, "cancel job")
        self.assertEqual(command, "scancel")
        self.assertEqual(args, ["12345"])

    def test_deterministic_slurm_command_for_node_inventory_summary(self):
        command = _deterministic_slurm_command("how many nodes are currently in my slurm cluster and what is their state ?")
        self.assertEqual(command["command"], "sinfo")
        self.assertEqual(command["args"], ["-N", "-h", "-o", "%N|%T"])

    def test_deterministic_reduce_slurm_result_for_node_inventory_summary(self):
        stdout = "\n".join(
            [
                "node-a|idle",
                "node-b|mixed*",
                "node-c|idle",
            ]
        )
        reduced, reducer_command = _deterministic_reduce_slurm_result(
            "how many nodes are currently in my slurm cluster and what is their state ?",
            "sinfo -N -h -o %N|%T",
            stdout,
        )
        self.assertIn("Total nodes: 3", reduced)
        self.assertIn("State idle: 2", reduced)
        self.assertIn("State mixed: 1", reduced)
        self.assertIn("python3 -c", reducer_command)


class SlurmGatewayTests(unittest.TestCase):
    def test_allowed_commands_default_contains_sinfo(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertIn("sinfo", _allowed_commands())

    def test_resolve_command_rejects_unallowed_command(self):
        with patch.dict("os.environ", {"SLURM_GATEWAY_ALLOWED_COMMANDS": "sinfo,squeue"}, clear=False):
            with self.assertRaisesRegex(Exception, "not allowed"):
                _resolve_command("bash")


if __name__ == "__main__":
    unittest.main()
