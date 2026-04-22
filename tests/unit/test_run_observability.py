import os
import sys
import tempfile
import types
import unittest

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)
jsonschema_stub = types.ModuleType("jsonschema")
jsonschema_stub.validate = lambda instance, schema: None
sys.modules.setdefault("jsonschema", jsonschema_stub)

from runtime.engine import Engine
from runtime.registry import ADAPTER_REGISTRY
from runtime.run_store import RunStore
from tests.unit.test_engine_validation import (
    TEST_SPEC,
    _ExecAdapter,
    _FallbackPlannerAdapter,
    _NoRecoveryPlannerAdapter,
    _RecorderAdapter,
    _ReducerAdapter,
    _ValidatorAdapter,
)


class RunObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.original_registry = dict(ADAPTER_REGISTRY)
        self.original_run_store_dir = os.environ.get("OPENFABRIC_RUN_STORE_DIR")
        self.run_store_dir = tempfile.TemporaryDirectory()
        os.environ["OPENFABRIC_RUN_STORE_DIR"] = self.run_store_dir.name
        ADAPTER_REGISTRY["test_exec"] = _ExecAdapter
        ADAPTER_REGISTRY["test_validator"] = _ValidatorAdapter
        ADAPTER_REGISTRY["test_reducer"] = _ReducerAdapter
        ADAPTER_REGISTRY["test_recorder"] = _RecorderAdapter
        ADAPTER_REGISTRY["test_fallback_planner"] = _FallbackPlannerAdapter
        ADAPTER_REGISTRY["test_no_recovery_planner"] = _NoRecoveryPlannerAdapter
        _RecorderAdapter.events = []

    def tearDown(self):
        ADAPTER_REGISTRY.clear()
        ADAPTER_REGISTRY.update(self.original_registry)
        if self.original_run_store_dir is None:
            os.environ.pop("OPENFABRIC_RUN_STORE_DIR", None)
        else:
            os.environ["OPENFABRIC_RUN_STORE_DIR"] = self.original_run_store_dir
        self.run_store_dir.cleanup()

    def test_completed_run_persists_observability_metrics(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "produce the final answer",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option1",
                        "label": "Primary plan",
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
        run_id = workflow_events[0]["run_id"]

        store = RunStore()
        observability = store.load_observability(run_id)
        self.assertIsInstance(observability, dict)
        self.assertEqual(observability["run_id"], run_id)
        self.assertEqual(observability["counts"]["attempt_count"], 2)
        self.assertEqual(observability["counts"]["step_count"], 2)
        self.assertEqual(observability["validation_counts"]["workflow"]["invalid"], 1)
        self.assertEqual(observability["validation_counts"]["workflow"]["valid"], 1)
        self.assertEqual(observability["routing_action_counts"]["try_next_option"], 1)
        self.assertEqual(observability["routing_action_counts"]["accept_attempt"], 1)
        self.assertIn("executor", observability["audit"]["agents"])
        self.assertGreaterEqual(len(observability["slowest_steps"]), 1)

        inspection = engine.inspect_run(run_id)
        self.assertEqual(inspection["observability"]["run_id"], run_id)
        self.assertEqual(inspection["summary"]["step_count"], 2)
        self.assertEqual(inspection["summary"]["agent_count"], 1)

        direct_observability = engine.inspect_run_observability(run_id)
        self.assertEqual(direct_observability["run_id"], run_id)

    def test_run_listing_supports_audit_filters(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "healthy executor run",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option2",
                        "label": "Healthy workflow",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "executor",
                                "task": "fallback attempt",
                                "instruction": {"operation": "run_command", "command": "fallback"},
                            }
                        ],
                    }
                ],
            },
        )
        engine.emit(
            "task.plan",
            {
                "task": "broken executor run",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option1",
                        "label": "Broken workflow",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "executor",
                                "task": "unsupported attempt",
                                "instruction": {"operation": "run_command", "command": "broken"},
                            }
                        ],
                    }
                ],
            },
        )

        store = RunStore()
        errored_runs = store.list_runs(limit=10, has_errors=True)
        self.assertEqual(len(errored_runs), 1)
        self.assertEqual(errored_runs[0]["task"], "broken executor run")
        self.assertTrue(errored_runs[0]["has_errors"])
        self.assertGreater(errored_runs[0]["error_count"], 0)

        executor_runs = store.list_runs(limit=10, agent="executor")
        self.assertEqual(len(executor_runs), 2)

        filtered_runs = store.list_runs(limit=10, task_contains="healthy")
        self.assertEqual(len(filtered_runs), 1)
        self.assertEqual(filtered_runs[0]["task"], "healthy executor run")


if __name__ == "__main__":
    unittest.main()
