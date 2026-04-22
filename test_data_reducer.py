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
fastapi_stub.HTTPException = Exception


class _RequestStub:
    pass


fastapi_stub.Request = _RequestStub
sys.modules.setdefault("fastapi", fastapi_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.agents.data_reducer import handle_event


class DataReducerTests(unittest.TestCase):
    def test_data_reducer_executes_structured_reduction_request(self):
        response = handle_event(
            types.SimpleNamespace(
                event="data.reduce",
                payload={
                    "task": "summarize current jobs",
                    "original_task": "summarize current jobs",
                    "step_id": "step1",
                    "target_agent": "slurm_runner",
                    "source_event": "slurm.result",
                    "reduction_request": {
                        "kind": "slurm.line_count",
                        "task": "summarize current jobs",
                        "source_command": "squeue -h",
                        "label": "Matching jobs",
                    },
                    "input_data": "101|vinith|PENDING\n102|vinith|RUNNING\n",
                },
            )
        )

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["reduced_result"], "Matching jobs: 2")
        self.assertEqual(payload["strategy"], "deterministic_local_reduction_command")
        self.assertIn("python3 -c", payload["local_reduction_command"])
        self.assertEqual(payload["node"]["agent"], "data_reducer")
        self.assertEqual(payload["node"]["role"], "reducer")
        self.assertEqual(payload["node"]["request_event"], "data.reduce")
        self.assertEqual(payload["node"]["emitted_event"], "data.reduced")
        self.assertEqual(payload["node"]["step_id"], "step1")
        self.assertEqual(payload["node"]["scope"], "step")

    def test_data_reducer_omits_empty_local_reduction_command_for_summary_results(self):
        with patch(
            "agent_library.agents.data_reducer.execute_reduction_request",
            return_value=types.SimpleNamespace(
                reduced_result="There are 8 tables in the dicom schema.",
                strategy="summary",
                local_reduction_command="",
                attempts=1,
                error="",
            ),
        ):
            response = handle_event(
                types.SimpleNamespace(
                    event="data.reduce",
                    payload={
                        "task": "List the tables in the dicom schema and tell me how many there are.",
                        "original_task": "List the tables in the dicom schema and tell me how many there are.",
                        "step_id": "step1",
                        "target_agent": "sql_runner_dicom_mock",
                        "source_event": "sql.result",
                        "reduction_request": {
                            "kind": "sql.summary",
                            "task": "List the tables in the dicom schema and tell me how many there are.",
                            "source_sql": "select table_name from information_schema.tables where table_schema = 'dicom'",
                            "columns": ["table_name"],
                            "row_count": 8,
                            "sample_rows": [{"table_name": "patients"}],
                        },
                        "input_data": {"rows": [{"table_name": "patients"}]},
                    },
                )
            )

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["strategy"], "summary")
        self.assertEqual(payload["reduced_result"], "There are 8 tables in the dicom schema.")
        self.assertNotIn("local_reduction_command", payload)

    def test_data_reducer_runs_local_command_against_input_data(self):
        class _Completed:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        with patch("agent_library.agents.data_reducer.subprocess.run", return_value=_Completed(0, stdout="Reduced lines: 2\n")) as mocked_run:
            response = handle_event(
                types.SimpleNamespace(
                    event="data.reduce",
                    payload={
                        "task": "count lines",
                        "original_task": "count lines in this output",
                        "step_id": "step1",
                        "target_agent": "shell_runner",
                        "source_event": "shell.result",
                        "local_reduction_command": "python3 -c 'print(2)'",
                        "input_data": "alpha\nbeta\n",
                    },
                )
            )

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["reduced_result"], "Reduced lines: 2")
        self.assertEqual(payload["strategy"], "local_reduction_command")
        self.assertEqual(payload["local_reduction_command"], "python3 -c 'print(2)'")
        mocked_run.assert_called_once()

    def test_data_reducer_falls_back_to_existing_reduced_result_on_failure(self):
        class _Completed:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        with patch("agent_library.agents.data_reducer.subprocess.run", return_value=_Completed(1, stderr="bad reducer")):
            response = handle_event(
                types.SimpleNamespace(
                    event="data.reduce",
                    payload={
                        "task": "summarize rows",
                        "original_task": "summarize rows",
                        "step_id": "step1",
                        "target_agent": "sql_runner",
                        "source_event": "sql.result",
                        "local_reduction_command": "python3 -c 'raise SystemExit(1)'",
                        "existing_reduced_result": "Patients over 45: 10",
                        "input_data": {"rows": [{"age": 46}]},
                    },
                )
            )

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["reduced_result"], "Patients over 45: 10")
        self.assertEqual(payload["strategy"], "existing_reduced_result_fallback")
        self.assertIn("bad reducer", payload["error"])


if __name__ == "__main__":
    unittest.main()
