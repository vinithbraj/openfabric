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
    handle_event,
    _build_deterministic_slurm_plan,
    _deterministic_reduce_slurm_result,
    _deterministic_slurm_command,
    _heuristic_slurm_selection,
    _instruction_to_command,
    _normalize_slurm_selection,
    _parse_command_input,
)
from agent_library.agents.slurm_runner import _gateway_base_url, _result_payload
from agent_library.common import EventRequest
from dep_agent_library.slurm_gateway_agent.app import _allowed_commands, _resolve_command


class SlurmRunnerTests(unittest.TestCase):
    def test_parse_command_input_from_string(self):
        command, args = _parse_command_input("squeue -u vinith")
        self.assertEqual(command, "squeue")
        self.assertEqual(args, ["-u", "vinith"])

    def test_parse_command_input_merges_explicit_args_with_string_command(self):
        command, args = _parse_command_input("sshare", ["-u", "vinith"])
        self.assertEqual(command, "sshare")
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

    def test_gateway_base_url_defaults_to_dedicated_slurm_gateway_port(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_gateway_base_url(), "http://127.0.0.1:8312")

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

    def test_heuristic_slurm_selection_for_pending_job_count(self):
        selection = _heuristic_slurm_selection(
            "how many pending jobs are there for user vinith",
            {"partitions": ["hpc"], "node_states": {}, "allowed_commands": ["squeue"]},
        )
        self.assertEqual(selection["primitive_id"], "slurm.jobs.queue_count")
        self.assertEqual(selection["parameters"]["user"], "vinith")
        self.assertEqual(selection["parameters"]["job_states"], ["PENDING"])

    def test_normalize_slurm_selection_rejects_unknown_primitive(self):
        normalized = _normalize_slurm_selection(
            {
                "primitive_id": "slurm.unknown",
                "selection_reason": "bad",
                "parameters": {"partition_name": "hpc"},
                "fallback_command": "squeue",
                "fallback_args": ["-h"],
                "fallback_reason": "fallback",
            },
            {"partitions": ["hpc"]},
        )
        self.assertEqual(normalized["primitive_id"], "fallback_only")
        self.assertEqual(normalized["fallback_command"], "squeue")

    def test_build_deterministic_slurm_plan_for_queue_count(self):
        plan = _build_deterministic_slurm_plan(
            {
                "primitive_id": "slurm.jobs.queue_count",
                "parameters": {"user": "vinith", "partition_name": "hpc", "job_states": ["PENDING"]},
            },
            "how many pending jobs are there for user vinith",
            {"partitions": ["hpc"]},
        )
        self.assertEqual(plan["command"], "squeue")
        self.assertIn("-u", plan["args"])
        self.assertIn("vinith", plan["args"])
        self.assertIn("-p", plan["args"])
        self.assertIn("hpc", plan["args"])
        self.assertIn("-t", plan["args"])

    def test_handle_event_prefers_deterministic_slurm_primitive(self):
        req = EventRequest(
            event="task.plan",
            payload={
                "task": "how many pending jobs are there for user vinith",
                "target_agent": "slurm_runner_cluster",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "how many pending jobs are there for user vinith",
                },
            },
        )

        with patch("agent_library.agents.slurm_runner._slurm_gateway_ready", return_value=True), patch(
            "agent_library.agents.slurm_runner._get_slurm_context",
            return_value={"partitions": ["hpc"], "node_states": {}, "allowed_commands": ["squeue"]},
        ), patch(
            "agent_library.agents.slurm_runner._heuristic_slurm_selection",
            return_value={
                "primitive_id": "slurm.jobs.queue_count",
                "selection_reason": "heuristic",
                "parameters": {"user": "vinith", "job_states": ["PENDING"]},
                "fallback_command": "",
                "fallback_args": [],
                "fallback_reason": "",
            },
        ), patch(
            "agent_library.agents.slurm_runner._build_deterministic_slurm_plan",
            return_value={"primitive_id": "slurm.jobs.queue_count", "command": "squeue", "args": ["-h", "-o", "%i|%u|%T|%P|%j"], "reason": "deterministic"},
        ), patch(
            "agent_library.agents.slurm_runner._gateway_execute",
            return_value={"returncode": 0, "stdout": "1|vinith|PENDING|hpc|job-a\n2|vinith|PENDING|hpc|job-b\n", "stderr": "", "duration_ms": 11},
        ), patch("agent_library.agents.slurm_runner._llm_slurm_command") as llm_fallback:
            response = handle_event(req)

        self.assertEqual(response["emits"][0]["event"], "slurm.result")
        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["execution_strategy"], "deterministic")
        self.assertEqual(payload["deterministic_primitive"], "slurm.jobs.queue_count")
        self.assertEqual(payload["reduction_request"]["kind"], "slurm.line_count")
        self.assertIsNone(payload["reduced_result"])
        llm_fallback.assert_not_called()

    def test_handle_event_marks_job_id_list_requests_for_reduction(self):
        req = EventRequest(
            event="task.plan",
            payload={
                "task": "list the pending job IDs for user vinith in the Slurm cluster",
                "target_agent": "slurm_runner_cluster",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "list the pending job IDs for user vinith in the Slurm cluster",
                },
            },
        )

        with patch("agent_library.agents.slurm_runner._slurm_gateway_ready", return_value=True), patch(
            "agent_library.agents.slurm_runner._get_slurm_context",
            return_value={"partitions": ["hpc", "gpu"], "node_states": {}, "allowed_commands": ["squeue"]},
        ), patch(
            "agent_library.agents.slurm_runner._heuristic_slurm_selection",
            return_value={
                "primitive_id": "slurm.jobs.queue_list",
                "selection_reason": "heuristic",
                "parameters": {"user": "vinith", "job_states": ["PENDING"]},
                "fallback_command": "",
                "fallback_args": [],
                "fallback_reason": "",
            },
        ), patch(
            "agent_library.agents.slurm_runner._build_deterministic_slurm_plan",
            return_value={"primitive_id": "slurm.jobs.queue_list", "command": "squeue", "args": ["-h", "-o", "%i|%u|%T|%P|%j"], "reason": "deterministic"},
        ), patch(
            "agent_library.agents.slurm_runner._gateway_execute",
            return_value={"returncode": 0, "stdout": "101|vinith|PENDING|hpc|align\n104|vinith|PENDING|gpu|sim\n", "stderr": "", "duration_ms": 11},
        ):
            response = handle_event(req)

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["execution_strategy"], "deterministic")
        self.assertEqual(payload["deterministic_primitive"], "slurm.jobs.queue_list")
        self.assertEqual(payload["reduction_request"]["kind"], "slurm.job_id_list")

    def test_handle_event_uses_selector_fallback_after_deterministic_failure(self):
        req = EventRequest(
            event="task.plan",
            payload={
                "task": "how many pending jobs are there",
                "target_agent": "slurm_runner_cluster",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "how many pending jobs are there",
                },
            },
        )

        gateway_results = [
            {"returncode": 1, "stdout": "", "stderr": "deterministic failed", "duration_ms": 5},
            {"returncode": 0, "stdout": "1|vinith|PENDING|hpc|job-a\n2|vinith|PENDING|hpc|job-b\n", "stderr": "", "duration_ms": 7},
        ]

        with patch("agent_library.agents.slurm_runner._slurm_gateway_ready", return_value=True), patch(
            "agent_library.agents.slurm_runner._get_slurm_context",
            return_value={"partitions": ["hpc"], "node_states": {}, "allowed_commands": ["squeue"]},
        ), patch(
            "agent_library.agents.slurm_runner._heuristic_slurm_selection",
            return_value={
                "primitive_id": "slurm.jobs.queue_count",
                "selection_reason": "heuristic",
                "parameters": {"job_states": ["PENDING"]},
                "fallback_command": "squeue",
                "fallback_args": ["-h", "-t", "PENDING"],
                "fallback_reason": "selector fallback",
            },
        ), patch(
            "agent_library.agents.slurm_runner._build_deterministic_slurm_plan",
            return_value={"primitive_id": "slurm.jobs.queue_count", "command": "squeue", "args": ["-h", "-o", "%i|%u|%T|%P|%j"], "reason": "deterministic"},
        ), patch(
            "agent_library.agents.slurm_runner._gateway_execute",
            side_effect=gateway_results,
        ), patch("agent_library.agents.slurm_runner._llm_slurm_command") as llm_fallback:
            response = handle_event(req)

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["execution_strategy"], "selector_fallback_command")
        self.assertEqual(payload["reduction_request"]["kind"], "slurm.line_count")
        llm_fallback.assert_not_called()


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
