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
        if task == "uncertain attempt":
            return [("task.result", {"detail": "Execution finished with ambiguous result.", "result": ""})]
        return [("task.result", {"detail": "Unknown task", "status": "failed", "error": "unsupported"})]


class _ValidatorAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "validation.request":
            return []
        if payload.get("task") == "uncertain goal":
            return [
                (
                    "validation.result",
                    {
                        "valid": False,
                        "verdict": "uncertain",
                        "reason": "The result is too ambiguous to accept confidently.",
                        "retry_recommended": False,
                        "missing_requirements": ["narrower target"],
                        "trace": ["Validation could not determine whether the result satisfies the request."],
                    },
                )
            ]
        if payload.get("option_id") == "option1":
            return [
                (
                    "validation.result",
                    {
                        "valid": False,
                        "verdict": "invalid",
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
                    "verdict": "valid",
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


class _FallbackPlannerAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "planner.replan.request":
            return []
        return [
            (
                "planner.replan.result",
                {
                    "replace_step_id": payload.get("step_id", "__workflow__"),
                    "reason": "Use the fallback workflow.",
                    "steps": [
                        {
                            "id": "step1",
                            "target_agent": "executor",
                            "task": "fallback attempt",
                            "instruction": {"operation": "run_command", "command": "fallback"},
                        }
                    ],
                },
            )
        ]


TEST_SPEC = {
    "contracts": {
        "TaskPlan": {
            "type": "object",
            "required": ["task"],
            "properties": {
                "task": {"type": "string"},
                "task_shape": {"type": "string"},
                "retry_budget": {"type": "object"},
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
                "task_shape": {"type": "string"},
                "validation_llm_budget_remaining": {"type": "number"},
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
                "verdict": {"type": "string"},
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
        "PlannerReplanRequest": {
            "type": "object",
            "required": ["task", "step_id", "reason"],
            "properties": {
                "task": {"type": "string"},
                "task_shape": {"type": "string"},
                "step_id": {"type": "string"},
                "reason": {"type": "string"},
                "available_context": {"type": "object"},
                "partial_result": {},
                "presentation": {"type": "object"},
            },
        },
        "PlannerReplanResult": {
            "type": "object",
            "required": ["replace_step_id", "steps"],
            "properties": {
                "replace_step_id": {"type": "string"},
                "reason": {"type": "string"},
                "steps": {"type": "array"},
            },
        },
        "WorkflowResult": {
            "type": "object",
            "required": ["task", "status", "steps"],
            "properties": {
                "task": {"type": "string"},
                "task_shape": {"type": "string"},
                "status": {"type": "string"},
                "steps": {"type": "array"},
                "result": {},
            },
        },
        "ClarificationRequired": {
            "type": "object",
            "required": ["task", "detail"],
            "properties": {
                "task": {"type": "string"},
                "task_shape": {"type": "string"},
                "detail": {"type": "string"},
                "question": {"type": "string"},
                "available_context": {"type": "object"},
                "missing_information": {"type": "array"},
            },
        },
    },
    "events": {
        "task.plan": {"contract": "TaskPlan"},
        "task.result": {"contract": "TaskResult"},
        "validation.request": {"contract": "ValidationRequest"},
        "validation.result": {"contract": "ValidationResult"},
        "validation.progress": {"contract": "ValidationProgress"},
        "planner.replan.request": {"contract": "PlannerReplanRequest"},
        "planner.replan.result": {"contract": "PlannerReplanResult"},
        "workflow.result": {"contract": "WorkflowResult"},
        "clarification.required": {"contract": "ClarificationRequired"},
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
        "fallback_planner": {
            "runtime": {"adapter": "test_fallback_planner"},
            "subscribes_to": ["planner.replan.request"],
            "emits": ["planner.replan.result"],
        },
        "recorder": {
            "runtime": {"adapter": "test_recorder"},
            "subscribes_to": ["validation.progress", "validation.request", "workflow.result", "planner.replan.request", "clarification.required"],
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
        ADAPTER_REGISTRY["test_fallback_planner"] = _FallbackPlannerAdapter
        _RecorderAdapter.events = []

    def tearDown(self):
        ADAPTER_REGISTRY.clear()
        ADAPTER_REGISTRY.update(self.original_registry)

    def test_engine_derives_fallback_only_after_validator_rejects_primary_attempt(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "produce the final answer",
                "task_shape": "lookup",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "primary attempt",
                        "instruction": {"operation": "run_command", "command": "primary"},
                    }
                ]
            },
        )
        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["task_shape"], "lookup")
        self.assertEqual(workflow["selected_option"]["id"], "option2")
        self.assertEqual(len(workflow["attempts"]), 2)
        self.assertTrue(workflow["validation"]["valid"])

        validation_progress = [payload for event_name, payload in _RecorderAdapter.events if event_name == "validation.progress"]
        stages = [item["stage"] for item in validation_progress]
        self.assertIn("retrying", stages)
        self.assertIn("passed", stages)

        validation_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "validation.request"]
        self.assertEqual(validation_requests[0]["task_shape"], "lookup")
        self.assertEqual(
            validation_requests[0]["steps"],
            [{"result": "partial answer", "status": "completed"}],
        )
        replan_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "planner.replan.request"]
        self.assertEqual(len(replan_requests), 1)
        self.assertEqual(replan_requests[0]["step_id"], "__workflow__")

    def test_engine_escalates_to_clarification_after_uncertainty_threshold(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "uncertain goal",
                "task_shape": "lookup",
                "retry_budget": {
                    "max_attempts": 2,
                    "max_replans": 1,
                    "max_uncertain_attempts": 0,
                    "max_validation_llm_calls": 0,
                },
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "uncertain attempt",
                        "instruction": {"operation": "run_command", "command": "ambiguous"},
                    }
                ],
            },
        )
        clarification_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "clarification.required"]
        self.assertEqual(len(clarification_events), 1)
        clarification = clarification_events[0]
        self.assertEqual(clarification["task"], "uncertain goal")
        self.assertIn("narrower target", clarification.get("question", ""))
        replan_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "planner.replan.request" and payload.get("task") == "uncertain goal"]
        self.assertEqual(len(replan_requests), 0)


if __name__ == "__main__":
    unittest.main()
