import json
import os
import sys
import tempfile
import types
import unittest

fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.routes = []

    def get(self, *_args, **_kwargs):
        def decorator(func):
            self.routes.append({"method": "GET", "args": _args, "kwargs": _kwargs, "func": func})
            return func

        return decorator

    def post(self, *_args, **_kwargs):
        def decorator(func):
            self.routes.append({"method": "POST", "args": _args, "kwargs": _kwargs, "func": func})
            return func

        return decorator


class _HTTPExceptionStub(Exception):
    pass


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.HTTPException = _HTTPExceptionStub
sys.modules.setdefault("fastapi", fastapi_stub)

responses_stub = types.ModuleType("fastapi.responses")


class _ResponseStub:
    def __init__(self, content=None, *args, **kwargs):
        self.content = content
        self.args = args
        self.kwargs = kwargs


responses_stub.HTMLResponse = _ResponseStub
responses_stub.JSONResponse = _ResponseStub
responses_stub.PlainTextResponse = _ResponseStub
sys.modules.setdefault("fastapi.responses", responses_stub)

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)
jsonschema_stub = types.ModuleType("jsonschema")
jsonschema_stub.validate = lambda instance, schema: None
sys.modules.setdefault("jsonschema", jsonschema_stub)

from runtime.engine import Engine
from runtime.registry import ADAPTER_REGISTRY
from runtime.run_store import RunStore
from runtime.run_visualizer import (
    build_graph_index,
    build_graph_view_model,
    build_run_visualization_payload,
    create_run_visualizer_app,
    list_run_visualizations,
    load_run_graph_payload,
    load_run_observability_payload,
    load_run_visualization,
    render_run_visualizer_html,
)
from test_engine_validation import (
    TEST_SPEC,
    _FallbackPlannerAdapter,
    _NoRecoveryPlannerAdapter,
    _RecorderAdapter,
    _ReducerAdapter,
    _ValidatorAdapter,
    _ExecAdapter,
)


class RunVisualizerTests(unittest.TestCase):
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

    def _completed_run_id(self) -> str:
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "visualize the completed workflow",
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
        return workflow_events[0]["run_id"]

    def test_build_graph_view_model_positions_nodes_by_depth(self):
        run_id = self._completed_run_id()
        store = RunStore()
        inspection = store.inspect(run_id, include_timeline=True)
        self.assertIsInstance(inspection, dict)

        payload = build_run_visualization_payload(inspection)
        graph_view = payload["graph_view"]
        self.assertGreaterEqual(graph_view["width"], 960)
        self.assertGreaterEqual(graph_view["height"], 320)
        self.assertEqual(len(graph_view["nodes"]), payload["graph"]["statistics"]["node_count"])

        node_by_id = {item["node_id"]: item for item in graph_view["nodes"]}
        root = node_by_id[payload["graph"]["root_node_id"]]
        attempt = node_by_id[f"{payload['graph']['root_node_id']}:attempt:1"]
        step = node_by_id[f"{payload['graph']['root_node_id']}:attempt:1:step:step1"]
        self.assertLess(root["x"], attempt["x"])
        self.assertLess(attempt["x"], step["x"])

        graph_index = build_graph_index(payload["graph"])
        self.assertEqual(graph_index["root_node_id"], payload["graph"]["root_node_id"])
        self.assertIn(payload["graph"]["root_node_id"], graph_index["nodes"])
        self.assertIn(attempt["node_id"], [item["node_id"] for item in graph_index["outgoing"][payload["graph"]["root_node_id"]]])

    def test_run_visualization_helpers_load_store_payloads(self):
        run_id = self._completed_run_id()
        store = RunStore()

        listing = list_run_visualizations(store, limit=10)
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["runs"][0]["run_id"], run_id)

        visualization = load_run_visualization(store, run_id)
        self.assertEqual(visualization["run_id"], run_id)
        self.assertIn("graph_view", visualization)
        self.assertIn("graph_index", visualization)
        self.assertIn("timeline", visualization)
        self.assertIn("observability", visualization)
        self.assertIn("nodes", visualization["graph_index"])
        self.assertIn("incoming", visualization["graph_index"])
        self.assertIn("outgoing", visualization["graph_index"])

        graph_json = load_run_graph_payload(store, run_id, format="json")
        self.assertEqual(graph_json["run_id"], run_id)
        graph_view = load_run_graph_payload(store, run_id, format="view")
        self.assertIn("nodes", graph_view)
        graph_mermaid = load_run_graph_payload(store, run_id, format="mermaid")
        self.assertIn("flowchart TD", graph_mermaid)
        observability = load_run_observability_payload(store, run_id)
        self.assertEqual(observability["run_id"], run_id)
        self.assertIn("timings", observability)

    def test_render_run_visualizer_html_contains_dashboard_shell(self):
        html = render_run_visualizer_html(base_dir="/tmp/openfabric_runs")
        self.assertIn("Run Atlas", html)
        self.assertIn("/api/runs", html)
        self.assertIn("graph-shell", html)
        self.assertIn("run-search", html)
        self.assertIn("node-search", html)
        self.assertIn("graph-legend", html)
        self.assertIn("signal-shell", html)
        self.assertIn("agent-metrics-shell", html)
        self.assertIn("failure-shell", html)
        self.assertIn("auto-refresh", html)
        self.assertIn("/tmp/openfabric_runs", html)

    def test_graph_endpoint_disables_response_model_generation(self):
        app = create_run_visualizer_app(run_store=RunStore())
        graph_routes = [
            route
            for route in getattr(app, "routes", [])
            if route.get("method") == "GET" and route.get("args") and route["args"][0] == "/api/runs/{run_id}/graph"
        ]
        self.assertEqual(len(graph_routes), 1)
        self.assertIsNone(graph_routes[0]["kwargs"].get("response_model"))


if __name__ == "__main__":
    unittest.main()
