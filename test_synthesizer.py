import os
import sys
import types
import unittest
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

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

requests_stub = types.ModuleType("requests")
requests_stub.post = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.agents.synthesizer import (
    _build_source_payload,
    _fallback_answer,
    _format_agent_result,
    _format_sql_result_answer,
    _format_workflow_answer,
    _synthesize,
)


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

    def test_workflow_fallback_uses_shell_stdout_excerpt_when_stdout_not_present(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "show current git branch",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "show current git branch",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "command": "git branch --show-current",
                            "stdout_excerpt": "Version_3_small_model",
                            "stderr_excerpt": "",
                            "returncode": 0,
                        },
                    }
                ],
            },
        )
        answer = _fallback_answer(req)
        self.assertIn("Version_3_small_model", answer)

    def test_format_agent_result_prefers_stdout_over_detail_when_output_exists(self):
        answer = _format_agent_result(
            {
                "command": 'find . -type f -name "*.sh"',
                "returncode": 0,
                "stdout_excerpt": "./a.sh\n./b.sh",
                "stderr_excerpt": "find: './private': Permission denied",
                "detail": "find returned partial results and only reported non-fatal permission warnings.",
            },
            "shell.result",
        )
        self.assertIn("./a.sh", answer)
        self.assertNotEqual(
            answer.strip(),
            "find returned partial results and only reported non-fatal permission warnings.",
        )

    def test_workflow_with_shell_output_bypasses_llm_synthesis(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "list all running docker containers",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "list all running docker containers",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "command": "docker ps",
                            "stdout_excerpt": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db",
                            "stderr_excerpt": "",
                            "returncode": 0,
                        },
                    }
                ],
            },
        )
        with patch("agent_library.agents.synthesizer._llm_synthesize", side_effect=AssertionError("LLM should not be called")):
            answer = _synthesize(req)
        self.assertIn("docker ps", answer)
        self.assertIn("postgres_db", answer)

    def test_workflow_markdown_table_uses_deterministic_shell_table_rendering(self):
        answer = _format_workflow_answer(
            {
                "task": "All Docker containers on the machine:",
                "status": "completed",
                "presentation": {
                    "format": "markdown_table",
                    "include_context": True,
                    "include_internal_steps": False,
                },
                "steps": [
                    {
                        "id": "step1",
                        "task": "list all docker containers",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "command": "docker ps -a",
                            "stdout": (
                                "CONTAINER ID   IMAGE          COMMAND         CREATED      STATUS      PORTS      NAMES\n"
                                "abc123def456   nginx:latest   nginx -g ...    2 hours ago  Up 2 hours  80->80/tcp web_server\n"
                                "def456ghi789   redis:alpine   redis-server    3 hours ago  Up 3 hours  6379/tcp   cache_server"
                            ),
                            "stderr": "",
                            "returncode": 0,
                        },
                    }
                ],
            }
        )
        self.assertIn("| CONTAINER ID | IMAGE | COMMAND |", answer)
        self.assertIn("| abc123def456 | nginx:latest | nginx -g ... |", answer)
        self.assertIn("| def456ghi789 | redis:alpine | redis-server |", answer)
        self.assertNotIn("Workflow complete", answer)

    def test_schema_summary_workflow_formats_schema_listing_instead_of_workflow_json(self):
        answer = _format_workflow_answer(
            {
                "task": "list all schemas in dicom_mock",
                "status": "completed",
                "task_shape": "schema_summary",
                "presentation": {
                    "format": "markdown_table",
                    "include_context": True,
                    "include_internal_steps": False,
                },
                "steps": [
                    {
                        "id": "step1",
                        "task": "list all schemas in dicom_mock",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "detail": "Database schemas listed.",
                            "result": {
                                "columns": ["schema"],
                                "rows": [{"schema": "dicom"}],
                                "row_count": 1,
                                "limit": 1,
                            },
                        },
                    }
                ],
            }
        )
        self.assertIn("| schema |", answer)
        self.assertIn("| dicom |", answer)
        self.assertNotIn("Workflow completed", answer)

    def test_workflow_internal_steps_include_raw_stage_outputs(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "list all docker containers in this machine",
                "status": "completed",
                "presentation": {
                    "task": "Show all Docker containers and include raw outputs for each stage.",
                    "format": "markdown_table",
                    "audience": "openwebui",
                    "include_context": True,
                    "include_internal_steps": True,
                },
                "steps": [
                    {
                        "id": "step1",
                        "task": "list all docker containers",
                        "status": "completed",
                        "event": "shell.result",
                        "stdout": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db\n456def   open-webui   open-webui",
                        "stderr": "",
                        "payload": {
                            "command": "docker ps -a",
                            "stdout_excerpt": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db",
                            "stderr_excerpt": "",
                            "returncode": 0,
                        },
                    }
                ],
            },
        )
        with patch("agent_library.agents.synthesizer._llm_synthesize", side_effect=AssertionError("LLM should not be called")):
            answer = _synthesize(req)
        self.assertIn("**Stage Outputs**", answer)
        self.assertIn("docker ps -a", answer)
        self.assertIn("postgres_db", answer)
        self.assertIn("open-webui", answer)

    def test_multi_sql_workflow_combines_count_and_detail_table(self):
        answer = _format_workflow_answer(
            {
                "task": "Count matching users and provide their details",
                "status": "completed",
                "presentation": {
                    "format": "markdown_table",
                    "include_context": True,
                    "include_internal_steps": False,
                },
                "steps": [
                    {
                        "id": "step1",
                        "task": "Count the qualifying rows for the request",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "sql": "SELECT COUNT(*) AS user_count FROM qualifying_users",
                            "result": {
                                "columns": ["user_count"],
                                "rows": [{"user_count": 2}],
                                "row_count": 1,
                            },
                        },
                    },
                    {
                        "id": "step2",
                        "task": "List the qualifying rows with requested details",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "sql": "SELECT mrn, name FROM qualifying_users",
                            "result": {
                                "columns": ["mrn", "name"],
                                "rows": [
                                    {"mrn": "A001", "name": "Alice"},
                                    {"mrn": "B002", "name": "Bob"},
                                ],
                                "row_count": 2,
                                "limit": 50,
                            },
                        },
                    },
                ],
            }
        )
        self.assertIn("Count the qualifying rows for the request: `2`", answer)
        self.assertIn("| mrn | name |", answer)
        self.assertIn("| A001 | Alice |", answer)

    def test_sql_export_workflow_includes_saved_artifact_path(self):
        export_path = "/tmp/openfabric/artifacts/exports/patients.json"
        answer = _format_workflow_answer(
            {
                "task": "get matching patients, save them to a file, and provide the location",
                "status": "completed",
                "task_shape": "save_artifact",
                "presentation": {
                    "format": "markdown_table",
                    "include_context": True,
                    "include_internal_steps": False,
                },
                "steps": [
                    {
                        "id": "step1",
                        "task": "List the qualifying rows with requested details for export",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "sql": "SELECT patient_id, patient_name FROM patients",
                            "result": {
                                "columns": ["patient_id", "patient_name"],
                                "rows": [
                                    {"patient_id": "P001", "patient_name": "Alice"},
                                    {"patient_id": "P002", "patient_name": "Bob"},
                                ],
                                "row_count": 2,
                            },
                        },
                    },
                    {
                        "id": "step2",
                        "task": "Save the exported rows to artifacts/exports/patients.json and print the absolute file path",
                        "status": "completed",
                        "event": "shell.result",
                        "result": export_path,
                        "payload": {
                            "command": "python3 -c '...'",
                            "stdout": export_path,
                            "stderr": "",
                            "returncode": 0,
                        },
                    },
                ],
            }
        )
        self.assertIn("| patient_id | patient_name |", answer)
        self.assertIn(
            f"Saved results to: [{export_path}]({export_path})",
            answer,
        )

    def test_markdown_table_links_file_path_cells(self):
        result = _format_sql_result_answer(
            {
                "columns": ["file_path"],
                "rows": [{"file_path": "artifacts/exports/patients.json"}],
                "row_count": 1,
                "limit": 1,
            }
        )
        expected_target = os.path.abspath("artifacts/exports/patients.json")
        self.assertIn(
            f"| [artifacts/exports/patients.json]({expected_target}) |",
            result,
        )

    def test_file_content_preview_links_path(self):
        answer = _fallback_answer(
            types.SimpleNamespace(
                event="file.content",
                payload={
                    "path": "artifacts/exports/patients.json",
                    "content": '{"ok": true}',
                },
            )
        )
        expected_target = os.path.abspath("artifacts/exports/patients.json")
        self.assertIn(
            f"File [artifacts/exports/patients.json]({expected_target}) content preview:",
            answer,
        )


if __name__ == "__main__":
    unittest.main()
