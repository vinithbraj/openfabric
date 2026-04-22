import unittest

from runtime.semantic_validator import validate_semantics


class SemanticValidatorTests(unittest.TestCase):
    def test_accepts_explicit_trigger_event_and_emits_contract(self):
        spec = {
            "contracts": {
                "agent_execution_request": {"type": "object"},
                "agent_execution_result": {"type": "object"},
                "UserQuestion": {"type": "object"},
                "TaskPlan": {"type": "object"},
                "TaskResult": {"type": "object"},
            },
            "events": {
                "user.ask": {"contract": "UserQuestion"},
                "task.plan": {"contract": "TaskPlan"},
                "task.result": {"contract": "TaskResult"},
            },
            "agents": {
                "planner": {
                    "runtime": {
                        "adapter": "http",
                        "endpoint": "http://127.0.0.1:8123/handle",
                    },
                    "apis": [
                        {
                            "name": "plan_task",
                            "trigger_event": "user.ask",
                            "emits": ["task.plan"],
                            "request_contract": "agent_execution_request",
                            "result_contract": "agent_execution_result",
                            "request_envelope_fields": ["node", "task", "instruction"],
                            "result_envelope_fields": ["node", "status", "detail"],
                        }
                    ],
                    "subscribes_to": ["user.ask"],
                    "emits": ["task.plan"],
                    "request_contract": "agent_execution_request",
                    "result_contract": "agent_execution_result",
                }
            },
        }

        validate_semantics(spec)

    def test_rejects_unknown_api_emitted_event(self):
        spec = {
            "contracts": {
                "UserQuestion": {"type": "object"},
                "TaskPlan": {"type": "object"},
            },
            "events": {
                "user.ask": {"contract": "UserQuestion"},
                "task.plan": {"contract": "TaskPlan"},
            },
            "agents": {
                "planner": {
                    "runtime": {
                        "adapter": "http",
                        "endpoint": "http://127.0.0.1:8123/handle",
                    },
                    "apis": [
                        {
                            "name": "plan_task",
                            "trigger_event": "user.ask",
                            "emits": ["missing.event"],
                        }
                    ],
                    "subscribes_to": ["user.ask"],
                    "emits": ["task.plan"],
                }
            },
        }

        with self.assertRaisesRegex(ValueError, "emits undefined event 'missing.event'"):
            validate_semantics(spec)


if __name__ == "__main__":
    unittest.main()
