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


if __name__ == "__main__":
    unittest.main()
