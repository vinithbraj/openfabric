import sys
import types
import unittest


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
sys.modules.setdefault("fastapi", fastapi_stub)

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.agents.synthesizer import _build_source_payload, _fallback_answer


class SynthesizerReducedResultTests(unittest.TestCase):
    def test_sql_source_payload_preserves_reduced_result(self):
        req = types.SimpleNamespace(
            event="sql.result",
            payload={
                "detail": "SQL query executed.",
                "reduced_result": "- TableA\n- TableB",
                "local_reduction_command": "python3 -c '...'",
                "sql": "select * from tables",
                "result": {
                    "columns": ["table"],
                    "rows": [{"table": "TableA"}, {"table": "TableB"}],
                    "row_count": 2,
                },
            },
        )
        source = _build_source_payload(req)
        self.assertEqual(source["reduced_result"], "- TableA\n- TableB")
        self.assertEqual(source["local_reduction_command"], "python3 -c '...'")

    def test_workflow_source_payload_uses_reduced_step_outcome(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "give me all tables",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "list all tables",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "reduced_result": "- TableA\n- TableB",
                            "local_reduction_command": "python3 -c '...'",
                            "result": {
                                "columns": ["table"],
                                "rows": [{"table": "TableA"}, {"table": "TableB"}],
                                "row_count": 2,
                            },
                        },
                    }
                ],
            },
        )
        source = _build_source_payload(req)
        self.assertEqual(len(source["step_outcomes"]), 1)
        self.assertEqual(source["step_outcomes"][0]["outcome"], "- TableA\n- TableB")
        self.assertEqual(source["steps"], [])

    def test_sql_fallback_prefers_reduced_result(self):
        req = types.SimpleNamespace(
            event="sql.result",
            payload={
                "reduced_result": "- TableA\n- TableB",
                "result": {"columns": ["table"], "rows": [{"table": "TableA"}], "row_count": 1},
            },
        )
        self.assertEqual(_fallback_answer(req), "- TableA\n- TableB")

    def test_workflow_source_payload_uses_shell_stdout_for_saved_path(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "save tables and give me the final path",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "save the rows to tables.txt",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "stdout": "/tmp/tables/tables.txt",
                            "stderr": "",
                            "returncode": 0,
                        },
                    }
                ],
            },
        )
        source = _build_source_payload(req)
        self.assertEqual(source["step_outcomes"][0]["outcome"], "/tmp/tables/tables.txt")

    def test_workflow_source_payload_prefers_compact_step_evidence_summary(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "count matching rows",
                "task_shape": "count",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "count matching rows",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {"detail": "SQL query executed."},
                        "evidence": {
                            "summary_text": "There are 42 matching rows.",
                            "payload": {"reduced_result": "There are 42 matching rows."},
                        },
                    }
                ],
            },
        )
        source = _build_source_payload(req)
        self.assertEqual(source["task_shape"], "count")
        self.assertEqual(source["step_outcomes"][0]["outcome"], "There are 42 matching rows.")


if __name__ == "__main__":
    unittest.main()
