import copy
import sys
import types
import unittest

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)
jsonschema_stub = types.ModuleType("jsonschema")
jsonschema_stub.validate = lambda instance, schema: None
sys.modules.setdefault("jsonschema", jsonschema_stub)

from runtime.engine import Engine
from runtime.registry import ADAPTER_REGISTRY


class _ExecAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "task.plan":
            return []
        task = payload.get("task")
        if task == "primary attempt":
            return [("task.result", {"detail": "Primary execution finished.", "result": "partial answer"})]
        if task == "fallback attempt":
            return [("task.result", {"detail": "Fallback execution finished.", "result": "final answer"})]
        return [("task.result", {"detail": "Unknown task", "status": "failed", "error": "unsupported"})]


class _ValidatorAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "validation.request":
            return []
        if payload.get("option_id") == "option1":
            return [
                (
                    "validation.result",
                    {
                        "valid": False,
                        "reason": "Primary attempt only returned a partial answer.",
                        "retry_recommended": True,
                        "missing_requirements": ["final answer completeness"],
                        "trace": ["Checked the option output against the original task.", "Detected incomplete fulfillment."],
                    },
                )
            ]
        return [
            (
                "validation.result",
                {
                    "valid": True,
                    "reason": "Fallback attempt satisfies the task.",
                    "retry_recommended": False,
                    "missing_requirements": [],
                    "trace": ["Checked the option output against the original task.", "Detected complete fulfillment."],
                },
            )
        ]


class _RecorderAdapter:
    events = []

    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        self.__class__.events.append((event_name, copy.deepcopy(payload)))
        return []


TEST_SPEC = {
    "contracts": {
        "TaskPlan": {
            "type": "object",
            "required": ["task"],
            "properties": {
                "task": {"type": "string"},
                "steps": {"type": "array"},
                "plan_options": {"type": "array"},
            },
        },
        "TaskResult": {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {"type": "string"},
                "status": {"type": "string"},
                "error": {"type": "string"},
                "result": {},
            },
        },
        "ValidationRequest": {
            "type": "object",
            "required": ["task", "workflow_status"],
            "properties": {
                "task": {"type": "string"},
                "workflow_status": {"type": "string"},
                "option_id": {"type": "string"},
                "steps": {"type": "array"},
                "result": {},
                "error": {"type": "string"},
            },
        },
        "ValidationResult": {
            "type": "object",
            "required": ["valid", "reason", "retry_recommended"],
            "properties": {
                "valid": {"type": "boolean"},
                "reason": {"type": "string"},
                "retry_recommended": {"type": "boolean"},
                "missing_requirements": {"type": "array"},
                "trace": {"type": "array"},
            },
        },
        "ValidationProgress": {
            "type": "object",
            "required": ["stage", "task", "message"],
            "properties": {
                "stage": {"type": "string"},
                "task": {"type": "string"},
                "message": {"type": "string"},
            },
        },
        "WorkflowResult": {
            "type": "object",
            "required": ["task", "status", "steps"],
            "properties": {
                "task": {"type": "string"},
                "status": {"type": "string"},
                "steps": {"type": "array"},
                "result": {},
            },
        },
    },
    "events": {
        "task.plan": {"contract": "TaskPlan"},
        "task.result": {"contract": "TaskResult"},
        "validation.request": {"contract": "ValidationRequest"},
        "validation.result": {"contract": "ValidationResult"},
        "validation.progress": {"contract": "ValidationProgress"},
        "workflow.result": {"contract": "WorkflowResult"},
    },
    "agents": {
        "executor": {
            "runtime": {"adapter": "test_exec"},
            "subscribes_to": ["task.plan"],
            "emits": ["task.result"],
        },
        "validator": {
            "runtime": {"adapter": "test_validator"},
            "subscribes_to": ["validation.request"],
            "emits": ["validation.result"],
        },
        "recorder": {
            "runtime": {"adapter": "test_recorder"},
            "subscribes_to": ["validation.progress", "workflow.result"],
            "emits": [],
        },
    },
}


class EngineValidationTests(unittest.TestCase):
    def setUp(self):
        self.original_registry = dict(ADAPTER_REGISTRY)
        ADAPTER_REGISTRY["test_exec"] = _ExecAdapter
        ADAPTER_REGISTRY["test_validator"] = _ValidatorAdapter
        ADAPTER_REGISTRY["test_recorder"] = _RecorderAdapter
        _RecorderAdapter.events = []

    def tearDown(self):
        ADAPTER_REGISTRY.clear()
        ADAPTER_REGISTRY.update(self.original_registry)

    def test_engine_retries_next_plan_option_when_validator_rejects_first_attempt(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "produce the final answer",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "primary attempt",
                        "instruction": {"operation": "run_command", "command": "primary"},
                    }
                ],
                "plan_options": [
                    {
                        "id": "option1",
                        "label": "Primary plan",
                        "reason": "Try the direct path first.",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "executor",
                                "task": "primary attempt",
                                "instruction": {"operation": "run_command", "command": "primary"},
                            }
                        ],
                    },
                    {
                        "id": "option2",
                        "label": "Fallback plan",
                        "reason": "Use the safer fallback path.",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "executor",
                                "task": "fallback attempt",
                                "instruction": {"operation": "run_command", "command": "fallback"},
                            }
                        ],
                    },
                ],
            },
        )
        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["selected_option"]["id"], "option2")
        self.assertEqual(len(workflow["attempts"]), 2)
        self.assertTrue(workflow["validation"]["valid"])

        validation_progress = [payload for event_name, payload in _RecorderAdapter.events if event_name == "validation.progress"]
        stages = [item["stage"] for item in validation_progress]
        self.assertIn("retrying", stages)
        self.assertIn("passed", stages)


if __name__ == "__main__":
    unittest.main()
