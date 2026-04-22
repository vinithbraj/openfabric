import copy
import json
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
    _CrashOnceExecAdapter,
    _ExecAdapter,
    _FallbackPlannerAdapter,
    _NoRecoveryPlannerAdapter,
    _RecorderAdapter,
    _ReducerAdapter,
    _ValidatorAdapter,
)


class RunInspectorTests(unittest.TestCase):
    def setUp(self):
        self.original_registry = dict(ADAPTER_REGISTRY)
        self.original_run_store_dir = os.environ.get("OPENFABRIC_RUN_STORE_DIR")
        self.run_store_dir = tempfile.TemporaryDirectory()
        os.environ["OPENFABRIC_RUN_STORE_DIR"] = self.run_store_dir.name
        ADAPTER_REGISTRY["test_exec"] = _ExecAdapter
        ADAPTER_REGISTRY["test_crash_exec"] = _CrashOnceExecAdapter
        ADAPTER_REGISTRY["test_validator"] = _ValidatorAdapter
        ADAPTER_REGISTRY["test_reducer"] = _ReducerAdapter
        ADAPTER_REGISTRY["test_recorder"] = _RecorderAdapter
        ADAPTER_REGISTRY["test_fallback_planner"] = _FallbackPlannerAdapter
        ADAPTER_REGISTRY["test_no_recovery_planner"] = _NoRecoveryPlannerAdapter
        _RecorderAdapter.events = []
        _CrashOnceExecAdapter.crash_counts = {}

    def tearDown(self):
        ADAPTER_REGISTRY.clear()
        ADAPTER_REGISTRY.update(self.original_registry)
        if self.original_run_store_dir is None:
            os.environ.pop("OPENFABRIC_RUN_STORE_DIR", None)
        else:
            os.environ["OPENFABRIC_RUN_STORE_DIR"] = self.original_run_store_dir
        self.run_store_dir.cleanup()

    def test_run_store_persists_summary_graph_and_mermaid_for_completed_run(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "replay the final answer",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option2",
                        "label": "Replayable workflow",
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

        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        run_id = workflow_events[0]["run_id"]

        store = RunStore()
        summary = store.load_summary(run_id)
        graph = store.load_graph(run_id)
        graph_mermaid = store.load_graph_mermaid(run_id)
        observability = store.load_observability(run_id)

        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["run_id"], run_id)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["graph_available"])
        self.assertEqual(summary["terminal_event"], "workflow.result")

        self.assertIsInstance(graph, dict)
        self.assertEqual(graph["kind"], "workflow_execution")
        self.assertEqual(graph["run_id"], run_id)

        self.assertIsInstance(graph_mermaid, str)
        self.assertIn("flowchart TD", graph_mermaid)
        self.assertIn("Replayable workflow", graph_mermaid)
        self.assertIsInstance(observability, dict)
        self.assertEqual(observability["run_id"], run_id)
        self.assertEqual(observability["counts"]["attempt_count"], 1)

        inspection = engine.inspect_run(run_id)
        self.assertEqual(inspection["summary"]["run_id"], run_id)
        self.assertEqual(inspection["summary"]["graph_node_count"], graph["statistics"]["node_count"])
        self.assertEqual(inspection["graph"]["run_id"], run_id)
        self.assertEqual(inspection["observability"]["run_id"], run_id)

        mermaid = engine.render_run_graph(run_id, format="mermaid")
        self.assertIn("flowchart TD", mermaid)
        graph_payload = engine.render_run_graph(run_id, format="json")
        self.assertEqual(graph_payload["run_id"], run_id)

    def test_list_runs_filters_and_orders_summaries(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "first completed run",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option2",
                        "label": "Replayable workflow",
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
                "task": "second completed run",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option2",
                        "label": "Replayable workflow",
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

        runs = engine.list_runs(limit=10)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["task"], "second completed run")
        self.assertEqual(runs[1]["task"], "first completed run")

        completed_runs = engine.list_runs(limit=10, status="completed")
        self.assertEqual(len(completed_runs), 2)
        self.assertTrue(all(item["status"] == "completed" for item in completed_runs))

    def test_non_terminal_run_inspection_reconstructs_graph_and_summary(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"]["executor"]["runtime"]["adapter"] = "test_crash_exec"
        engine = Engine(spec)
        engine.setup()
        run_id = "inspect_interrupted_run"

        with self.assertRaisesRegex(RuntimeError, "simulated executor crash"):
            engine.emit(
                "task.plan",
                {
                    "task": "resume after crash workflow",
                    "run_id": run_id,
                    "task_shape": "lookup",
                    "steps": [],
                    "plan_options": [
                        {
                            "id": "option2",
                            "label": "Crashy workflow",
                            "steps": [
                                {
                                    "id": "step1",
                                    "target_agent": "executor",
                                    "task": "resume after crash step",
                                    "instruction": {"operation": "run_command", "command": "resume"},
                                }
                            ],
                        }
                    ],
                },
            )

        store = RunStore()
        inspection = store.inspect(run_id, include_timeline=True)
        self.assertIsInstance(inspection, dict)
        self.assertEqual(inspection["summary"]["status"], "running")
        self.assertTrue(inspection["summary"]["resumable"])
        self.assertEqual(inspection["summary"]["active_step_id"], "step1")
        self.assertEqual(inspection["graph"]["run_id"], run_id)
        self.assertEqual(inspection["graph"]["kind"], "workflow_execution")
        self.assertGreaterEqual(inspection["summary"]["timeline_entries"], 2)
        self.assertIn("flowchart TD", inspection["graph_mermaid"])

        summary_path = store.summary_path(run_id)
        graph_path = store.graph_path(run_id)
        mermaid_path = store.graph_mermaid_path(run_id)
        observability_path = store.observability_path(run_id)
        self.assertTrue(summary_path.exists())
        self.assertTrue(graph_path.exists())
        self.assertTrue(mermaid_path.exists())
        self.assertTrue(observability_path.exists())

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["status"], "running")
        self.assertTrue(summary["resumable"])


if __name__ == "__main__":
    unittest.main()
