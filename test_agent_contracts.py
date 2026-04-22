import os
import unittest
from importlib import import_module

from agent_library.contracts import (
    AGENT_CONTRACT_VERSION,
    SHARED_REQUEST_CONTRACT_NAME,
    SHARED_RESULT_CONTRACT_NAME,
    build_agent_api,
    build_agent_descriptor,
    normalize_agent_metadata,
)
from agent_library.template import emit_many, emit_sequence, failure_result, needs_decomposition, noop
from runtime.engine import Engine
from runtime.graph import build_agent_graph_node


class AgentContractTests(unittest.TestCase):
    def test_build_agent_descriptor_normalizes_api_specs_and_legacy_methods(self):
        descriptor = build_agent_descriptor(
            name="example_agent",
            role="executor",
            description="Example executor.",
            capability_domains=["example"],
            action_verbs=["run"],
            side_effect_policy="read_only",
            safety_enforced_by_agent=True,
            apis=[
                build_agent_api(
                    name="do_example_work",
                    trigger_event="task.plan",
                    emits=["task.result"],
                    summary="Run example work.",
                    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    deterministic=True,
                    side_effect_level="read_only",
                )
            ],
        )

        self.assertEqual(descriptor["contract_version"], AGENT_CONTRACT_VERSION)
        self.assertEqual(descriptor["apis"][0]["name"], "do_example_work")
        self.assertEqual(descriptor["apis"][0]["trigger_event"], "task.plan")
        self.assertEqual(descriptor["apis"][0]["emits"], ["task.result"])
        self.assertEqual(descriptor["methods"][0]["name"], "do_example_work")
        self.assertEqual(descriptor["methods"][0]["event"], "task.plan")
        self.assertEqual(descriptor["methods"][0]["trigger_event"], "task.plan")
        self.assertEqual(descriptor["request_contract"], SHARED_REQUEST_CONTRACT_NAME)
        self.assertEqual(descriptor["result_contract"], SHARED_RESULT_CONTRACT_NAME)
        self.assertIn("request_schema", descriptor)
        self.assertIn("result_schema", descriptor)

    def test_engine_load_agent_metadata_normalizes_real_agent_metadata(self):
        engine = Engine({"contracts": {}, "events": {}, "agents": {}})
        metadata = engine._load_agent_metadata(
            "planner",
            {
                "adapter": "http",
                "endpoint": "http://127.0.0.1:9999/handle",
                "autostart": {"app": "agent_library.agents.planner:app"},
            },
        )

        self.assertEqual(metadata["contract_version"], AGENT_CONTRACT_VERSION)
        self.assertTrue(metadata["apis"])
        self.assertTrue(metadata["methods"])
        self.assertIn("request_schema", metadata)
        self.assertIn("result_schema", metadata)

    def test_build_agent_graph_node_projects_contract_and_rich_api_specs(self):
        metadata = normalize_agent_metadata(
            "filesystem",
            {
                "description": "Reads files.",
                "capability_domains": ["filesystem"],
                "action_verbs": ["read"],
                "side_effect_policy": "read_only",
                "safety_enforced_by_agent": True,
                "methods": [
                    {
                        "name": "read_workspace_file",
                        "event": "task.plan",
                        "when": "Reads a file path from a task plan.",
                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                        "output_schema": {"type": "object", "properties": {"content": {"type": "string"}}},
                    }
                ],
            },
        )
        node = build_agent_graph_node(
            "filesystem",
            {
                "runtime": {"adapter": "http", "endpoint": "http://127.0.0.1:8123/handle"},
                "subscribes_to": ["task.plan"],
                "emits": ["file.content"],
            },
            metadata,
        )

        self.assertEqual(node["contract"]["version"], AGENT_CONTRACT_VERSION)
        self.assertIn("request_schema", node["contract"])
        self.assertIn("result_schema", node["contract"])
        self.assertEqual(node["contract"]["request_contract"], SHARED_REQUEST_CONTRACT_NAME)
        self.assertEqual(node["contract"]["result_contract"], SHARED_RESULT_CONTRACT_NAME)
        self.assertIn("input_schema", node["capabilities"]["apis"][0])
        self.assertIn("output_schema", node["capabilities"]["apis"][0])
        self.assertEqual(node["capabilities"]["apis"][0]["trigger_event"], "task.plan")

    def test_all_agent_modules_expose_normalizable_contract_metadata(self):
        agent_dir = os.path.join(os.path.dirname(__file__), "agent_library", "agents")
        module_names = []
        for filename in sorted(os.listdir(agent_dir)):
            if not filename.endswith(".py") or filename == "__init__.py":
                continue
            module_names.append(f"agent_library.agents.{filename[:-3]}")

        for module_name in module_names:
            module = import_module(module_name)
            raw = getattr(module, "AGENT_DESCRIPTOR", None)
            if raw is None:
                raw = getattr(module, "AGENT_METADATA", None)
            self.assertIsNotNone(raw, module_name)
            normalized = normalize_agent_metadata(module_name.rsplit(".", 1)[-1], raw)
            self.assertEqual(normalized["contract_version"], AGENT_CONTRACT_VERSION, module_name)
            self.assertTrue(normalized["apis"], module_name)
            self.assertTrue(normalized["methods"], module_name)
            self.assertIn("request_schema", normalized, module_name)
            self.assertIn("result_schema", normalized, module_name)
            for api in normalized["apis"]:
                self.assertTrue(api["trigger_event"], module_name)
                self.assertIn("request_contract", api, module_name)
                self.assertIn("result_contract", api, module_name)
                self.assertIn("request_envelope_fields", api, module_name)
                self.assertIn("result_envelope_fields", api, module_name)

    def test_router_agents_use_explicit_trigger_and_emit_contracts(self):
        planner = import_module("agent_library.agents.planner")
        normalized_planner = normalize_agent_metadata("planner", planner.AGENT_DESCRIPTOR)
        planner_apis = {item["name"]: item for item in normalized_planner["apis"]}
        self.assertEqual(planner_apis["plan_research"]["trigger_event"], "user.ask")
        self.assertEqual(planner_apis["plan_research"]["emits"], ["research.query"])
        self.assertEqual(planner_apis["plan_task"]["trigger_event"], "user.ask")
        self.assertEqual(planner_apis["plan_task"]["emits"], ["task.plan"])

        llm_planner = import_module("agent_library.agents.llm_operations_planner")
        normalized_llm = normalize_agent_metadata("ops_planner", llm_planner.AGENT_DESCRIPTOR)
        llm_apis = {item["name"]: item for item in normalized_llm["apis"]}
        self.assertEqual(llm_apis["assess_processable_request"]["trigger_event"], "user.ask")
        self.assertEqual(llm_apis["assess_processable_request"]["emits"], ["plan.progress", "task.plan"])
        self.assertEqual(llm_apis["decompose_failed_step"]["trigger_event"], "planner.replan.request")
        self.assertEqual(llm_apis["decompose_failed_step"]["emits"], ["planner.replan.result"])

    def test_template_helpers_emit_runtime_compatible_responses(self):
        self.assertEqual(noop(), {"emits": []})
        combined = emit_many(
            ("task.result", {"detail": "ok"}),
            ("answer.final", {"answer": "done"}),
        )
        self.assertEqual(len(combined["emits"]), 2)
        rebuilt = emit_sequence([{"event": "task.result", "payload": {"detail": "ok"}}])
        self.assertEqual(rebuilt["emits"][0]["event"], "task.result")

        failure = failure_result("something failed", error="boom")
        self.assertEqual(failure["emits"][0]["event"], "task.result")
        self.assertEqual(failure["emits"][0]["payload"]["status"], "failed")

        replan = needs_decomposition("needs more detail", suggested_capabilities=["shell_runner"])
        self.assertEqual(replan["emits"][0]["payload"]["status"], "needs_decomposition")
        self.assertEqual(
            replan["emits"][0]["payload"]["replan_hint"]["suggested_capabilities"],
            ["shell_runner"],
        )


if __name__ == "__main__":
    unittest.main()
