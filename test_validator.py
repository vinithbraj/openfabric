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

from agent_library.agents.validator import handle_event


class ValidatorTests(unittest.TestCase):
    def test_validation_heuristic_accepts_completed_attempt_with_output(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "list the files",
                    "task_shape": "list",
                    "workflow_status": "completed",
                    "result": ["a.txt", "b.txt"],
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["retry_recommended"])
        self.assertEqual(payload["verdict"], "valid")
        self.assertTrue(payload["trace"])

    def test_validation_heuristic_rejects_failed_attempt(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "list the files",
                    "task_shape": "list",
                    "workflow_status": "failed",
                    "error": "command crashed",
                    "steps": [{"id": "step1", "status": "failed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["retry_recommended"])
        self.assertEqual(payload["verdict"], "invalid")
        self.assertIn("command crashed", payload["reason"])

    def test_validation_respects_zero_llm_budget_on_uncertain_result(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "summarize the dataset",
                    "task_shape": "summarize_dataset",
                    "workflow_status": "completed",
                    "validation_llm_budget_remaining": 0,
                    "result": {},
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["verdict"], "uncertain")
        self.assertFalse(payload["retry_recommended"])

    def test_validation_accepts_multiline_shell_output_for_list_tasks(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "list all running docker containers",
                    "task_shape": "list",
                    "workflow_status": "completed",
                    "result": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db",
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["verdict"], "valid")
        self.assertFalse(payload["retry_recommended"])

    def test_validation_accepts_singleton_path_for_list_task(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "list database files in this repo",
                    "task_shape": "list",
                    "workflow_status": "completed",
                    "result": "./services/postgres/pgadmin_data/pgadmin4.db",
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["verdict"], "valid")

    def test_validation_accepts_installation_path_for_boolean_check(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "show whether docker is installed",
                    "task_shape": "boolean_check",
                    "workflow_status": "completed",
                    "result": "/bin/docker",
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["verdict"], "valid")

    def test_workflow_validation_accepts_combined_count_and_state_output(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "how many nodes are currently in my slurm cluster and what is their state ?",
                    "task_shape": "lookup",
                    "workflow_status": "completed",
                    "result": "Total nodes: 3\nState idle: 2\nState mixed: 1",
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["verdict"], "valid")
        self.assertFalse(payload["retry_recommended"])

    def test_workflow_validation_accepts_structured_count_output(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "task": "how many rows are in dicom.patients",
                    "task_shape": "lookup",
                    "workflow_status": "completed",
                    "result": {"result": {"rows": [{"count": 100}], "columns": ["count"], "row_count": 1}},
                    "steps": [{"id": "step1", "status": "completed"}],
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["verdict"], "valid")

    def test_step_validation_rejects_combined_count_and_state_step_without_count(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "validation_scope": "step",
                    "task": "how many nodes are currently in my slurm cluster and what is their state ?",
                    "original_task": "how many nodes are currently in my slurm cluster and what is their state ?",
                    "step_id": "step1",
                    "step_task": "how many nodes are currently in my slurm cluster and what is their state ?",
                    "task_shape": "count",
                    "workflow_status": "completed",
                    "step_event": "slurm.result",
                    "result": {"reduced_result": 'The state of the nodes is either "idle" or "mixed".'},
                    "step_value": 'The state of the nodes is either "idle" or "mixed".',
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["verdict"], "invalid")
        self.assertTrue(payload["retry_recommended"])
        self.assertIn("count", payload["missing_requirements"])

    def test_step_validation_accepts_structured_single_row_count(self):
        response = handle_event(
            types.SimpleNamespace(
                event="validation.request",
                payload={
                    "validation_scope": "step",
                    "task": "how many rows are in dicom.patients",
                    "original_task": "how many rows are in dicom.patients",
                    "step_id": "step1",
                    "step_task": "how many rows are in dicom.patients",
                    "task_shape": "count",
                    "workflow_status": "completed",
                    "step_event": "sql.result",
                    "result": {"result": {"rows": [{"count": 100}], "columns": ["count"], "row_count": 1}},
                    "step_value": {"result": {"rows": [{"count": 100}], "columns": ["count"], "row_count": 1}},
                },
            )
        )
        payload = response["emits"][0]["payload"]
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["verdict"], "valid")


if __name__ == "__main__":
    unittest.main()
