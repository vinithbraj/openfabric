import sys
import types
import unittest


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

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

from agent_library.agents.synthesizer import _fallback_answer


class SynthesizerWorkflowPriorityTests(unittest.TestCase):
    def test_workflow_schema_summary_prefers_reduced_sql_summary(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count.",
                "task_shape": "schema_summary",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "list the tables in the dicom schema alphabetically",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "reduced_result": (
                                "The data shows 8 tables contained within the 'dicom' schema, listed alphabetically. "
                                "The tables are: dicom_tags, instances, patients, rtdose, rtplan, rtstruct, series, and studies."
                            ),
                            "result": {
                                "columns": ["schema", "table", "type"],
                                "rows": [
                                    {"schema": "dicom", "table": "dicom_tags", "type": "table"},
                                    {"schema": "dicom", "table": "instances", "type": "table"},
                                    {"schema": "dicom", "table": "patients", "type": "table"},
                                    {"schema": "dicom", "table": "rtdose", "type": "table"},
                                    {"schema": "dicom", "table": "rtplan", "type": "table"},
                                ],
                                "row_count": 8,
                                "returned_row_count": 5,
                                "total_matching_rows": 8,
                                "truncated": True,
                            },
                        },
                    },
                    {
                        "id": "step2",
                        "task": "count the number of tables from the previous result",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "stdout": "8\n",
                            "returncode": 0,
                        },
                    },
                ],
            },
        )

        answer = _fallback_answer(req)
        self.assertIn("8 tables", answer)
        self.assertIn("rtstruct", answer)
        self.assertIn("studies", answer)
        self.assertNotIn("| schema | table | type |", answer)

    def test_sql_result_prefers_reduced_result_over_raw_rows(self):
        req = types.SimpleNamespace(
            event="sql.result",
            payload={
                "reduced_result": "34",
                "result": {
                    "columns": ["patient_id", "count"],
                    "rows": [{"patient_id": "PAT_20", "count": 6}],
                    "row_count": 85,
                },
            },
        )

        self.assertEqual(_fallback_answer(req), "34")

    def test_mixed_workflow_count_answer_aggregates_all_completed_steps_and_difference(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": (
                    "In the dicom_mock database count patients, and in the repository root count Python files, "
                    "then report both counts and the difference."
                ),
                "task_shape": "count",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "Count patients in the dicom_mock database",
                        "target_agent": "sql_runner_dicom_mock",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "reduced_result": "10000",
                            "result": {
                                "columns": ["count"],
                                "rows": [{"count": 10000}],
                                "row_count": 1,
                            },
                        },
                    },
                    {
                        "id": "step2",
                        "task": "Count Python files in the repository root",
                        "target_agent": "shell_runner",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "stdout": "21\n",
                            "returncode": 0,
                        },
                    },
                    {
                        "id": "step3",
                        "task": "compute the absolute difference between the previous counts",
                        "target_agent": "shell_runner",
                        "status": "pending",
                    },
                ],
            },
        )

        answer = _fallback_answer(req)
        self.assertIn("10000", answer)
        self.assertIn("21", answer)
        self.assertIn("9979", answer)
        self.assertIn("**Difference:** `9979`", answer)
        self.assertNotIn("**Count:** `9979`", answer)

    def test_lookup_workflow_aggregates_mixed_agent_counts(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": (
                    "Count pending Slurm jobs for vinith, count patients in the dicom_mock database, "
                    "and count Python files in the repository root. Report all three counts in one answer."
                ),
                "task_shape": "lookup",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "Count pending Slurm jobs for vinith",
                        "target_agent": "slurm_runner_cluster",
                        "status": "completed",
                        "event": "slurm.result",
                        "payload": {
                            "reduced_result": "Matching jobs: 2",
                        },
                    },
                    {
                        "id": "step2",
                        "task": "Count patients in the dicom_mock database",
                        "target_agent": "sql_runner_dicom_mock",
                        "status": "completed",
                        "event": "sql.result",
                        "payload": {
                            "reduced_result": "10000",
                            "result": {
                                "columns": ["count"],
                                "rows": [{"count": 10000}],
                                "row_count": 1,
                            },
                        },
                    },
                    {
                        "id": "step3",
                        "task": "Count Python files in the repository root",
                        "target_agent": "shell_runner",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "stdout": "21\n",
                            "returncode": 0,
                        },
                    },
                ],
            },
        )

        answer = _fallback_answer(req)
        self.assertIn("2", answer)
        self.assertIn("10000", answer)
        self.assertIn("21", answer)

    def test_shell_inventory_workflow_preserves_list_details_and_count(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "In the repository root, list the first five Python files alphabetically and tell me the total count.",
                "task_shape": "lookup",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "list the first five Python files alphabetically in the repository root",
                        "target_agent": "shell_runner",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "stdout": (
                                "cli.py\n"
                                "openwebui_gateway.py\n"
                                "test_common_reducer.py\n"
                                "test_data_reducer.py\n"
                                "test_engine_validation.py\n"
                            ),
                            "returncode": 0,
                        },
                    },
                    {
                        "id": "step2",
                        "task": "count Python files in the repository root",
                        "target_agent": "shell_runner",
                        "status": "completed",
                        "event": "shell.result",
                        "payload": {
                            "stdout": "21\n",
                            "returncode": 0,
                        },
                    },
                ],
            },
        )

        answer = _fallback_answer(req)
        self.assertIn("21", answer)
        self.assertIn("cli.py", answer)
        self.assertIn("test_engine_validation.py", answer)

    def test_slurm_lookup_workflow_reports_count_and_job_ids(self):
        req = types.SimpleNamespace(
            event="workflow.result",
            payload={
                "task": "In the Slurm cluster, how many pending jobs does vinith have, and list the job IDs.",
                "task_shape": "lookup",
                "status": "completed",
                "steps": [
                    {
                        "id": "step1",
                        "task": "Count pending Slurm jobs for vinith",
                        "target_agent": "slurm_runner_cluster",
                        "status": "completed",
                        "event": "slurm.result",
                        "payload": {
                            "reduced_result": "Matching jobs: 2",
                        },
                    },
                    {
                        "id": "step2",
                        "task": "list the pending job IDs for user vinith in the Slurm cluster",
                        "target_agent": "slurm_runner_cluster",
                        "status": "completed",
                        "event": "slurm.result",
                        "payload": {
                            "reduced_result": "101\n104",
                        },
                    },
                ],
            },
        )

        answer = _fallback_answer(req)
        self.assertIn("2", answer)
        self.assertIn("101", answer)
        self.assertIn("104", answer)


if __name__ == "__main__":
    unittest.main()
