import unittest

from runtime.graph import build_agent_graph_node, build_capability_graph, build_workflow_graph


class RuntimeGraphTests(unittest.TestCase):
    def test_build_agent_graph_node_projects_common_agent_shape(self):
        node = build_agent_graph_node(
            "sql_runner_mydb",
            {
                "runtime": {"adapter": "http", "endpoint": "http://127.0.0.1:8010/handle"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
            },
            {
                "description": "Query a configured SQL database.",
                "capability_domains": ["sql", "database"],
                "action_verbs": ["query", "describe"],
                "execution_model": "deterministic_first_with_llm_fallback",
                "methods": [
                    {
                        "name": "query_from_request",
                        "event": "task.plan",
                        "when": "Answers a natural-language SQL request.",
                        "examples": ["how many rows are in patients"],
                    }
                ],
            },
        )

        self.assertEqual(node["node_id"], "agent:sql_runner_mydb")
        self.assertEqual(node["role"], "executor")
        self.assertEqual(node["interfaces"]["receives"], ["task.plan"])
        self.assertEqual(node["interfaces"]["emits"], ["task.result"])
        self.assertEqual(node["capabilities"]["domains"], ["sql", "database"])
        self.assertEqual(node["capabilities"]["apis"][0]["name"], "query_from_request")

    def test_build_capability_graph_connects_agent_and_event_nodes(self):
        planner_node = build_agent_graph_node(
            "planner",
            {
                "runtime": {"adapter": "http", "endpoint": "http://127.0.0.1:8001/handle"},
                "subscribes_to": ["user.ask"],
                "emits": ["task.plan"],
            },
            {
                "description": "Plan and route tasks.",
                "capability_domains": ["planning", "routing"],
                "action_verbs": ["plan", "route"],
            },
        )
        executor_node = build_agent_graph_node(
            "shell_runner",
            {
                "runtime": {"adapter": "http", "endpoint": "http://127.0.0.1:8002/handle"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
            },
            {
                "description": "Execute shell commands.",
                "capability_domains": ["general_shell"],
                "action_verbs": ["run"],
            },
        )
        graph = build_capability_graph(
            [
                {
                    "name": "planner",
                    "graph_node": planner_node,
                    "subscribes_to": ["user.ask"],
                    "emits": ["task.plan"],
                },
                {
                    "name": "shell_runner",
                    "graph_node": executor_node,
                    "subscribes_to": ["task.plan"],
                    "emits": ["task.result"],
                },
            ]
        )

        node_ids = {item["node_id"] for item in graph["nodes"]}
        self.assertIn("agent:planner", node_ids)
        self.assertIn("agent:shell_runner", node_ids)
        self.assertIn("event:user.ask", node_ids)
        self.assertIn("event:task.plan", node_ids)
        self.assertIn("event:task.result", node_ids)

        edge_relations = {(item["source"], item["target"], item["relation"]) for item in graph["edges"]}
        self.assertIn(("agent:planner", "event:task.plan", "emits"), edge_relations)
        self.assertIn(("event:task.plan", "agent:shell_runner", "consumes"), edge_relations)

    def test_build_agent_graph_node_infers_reducer_role(self):
        node = build_agent_graph_node(
            "data_reducer",
            {
                "runtime": {"adapter": "http", "endpoint": "http://127.0.0.1:8313/handle"},
                "subscribes_to": ["data.reduce"],
                "emits": ["data.reduced"],
            },
            {
                "description": "Reduce raw step outputs.",
                "capability_domains": ["data_reduction"],
                "action_verbs": ["reduce"],
            },
        )

        self.assertEqual(node["role"], "reducer")

    def test_build_workflow_graph_models_attempts_steps_reducers_and_validators(self):
        graph = build_workflow_graph(
            task="show cluster node inventory",
            task_shape="lookup",
            status="completed",
            selected_option={"id": "option1", "label": "Primary plan"},
            result="Total nodes: 3\nState idle: 2\nState mixed: 1",
            presentation={"format": "markdown"},
            attempts=[
                {
                    "option": {"id": "option1", "label": "Primary plan"},
                    "status": "completed",
                    "result": "Total nodes: 3\nState idle: 2\nState mixed: 1",
                    "validation": {
                        "valid": True,
                        "verdict": "valid",
                        "reason": "Workflow satisfied the request.",
                    },
                    "routing": {
                        "scope": "workflow",
                        "stage": "validation",
                        "action": "accept_attempt",
                        "reason": "Workflow satisfied the request.",
                    },
                    "steps": [
                        {
                            "id": "step1",
                            "task": "query cluster node inventory",
                            "target_agent": "slurm_runner",
                            "status": "completed",
                            "instruction": {"operation": "query_from_request", "question": "show cluster node inventory"},
                            "event": "slurm.result",
                            "duration_ms": 12.5,
                            "payload": {
                                "command": "sinfo -Nel",
                                "stdout": "node001 idle",
                                "reduction_request": {
                                    "kind": "slurm.node_inventory_summary",
                                    "task": "show cluster node inventory",
                                    "source_command": "sinfo -Nel",
                                },
                                "reduced_result": "Total nodes: 3\nState idle: 2\nState mixed: 1",
                            },
                            "result": "Total nodes: 3\nState idle: 2\nState mixed: 1",
                            "validation": {
                                "valid": True,
                                "verdict": "valid",
                                "reason": "The step output satisfies the step intent.",
                            },
                            "routing": {
                                "scope": "step",
                                "stage": "validation",
                                "action": "accept_step",
                                "reason": "The step output satisfies the step intent.",
                            },
                        }
                    ],
                }
            ],
        )

        node_kinds = {item["node_id"]: item["kind"] for item in graph["nodes"]}
        workflow_node_id = graph["root_node_id"]
        attempt_node_id = f"{workflow_node_id}:attempt:1"
        step_node_id = f"{attempt_node_id}:step:step1"
        attempt_validator_node_id = f"{attempt_node_id}:validator"
        attempt_router_node_id = f"{attempt_validator_node_id}:router"
        step_reducer_node_id = f"{step_node_id}:reducer"
        step_validator_node_id = f"{step_reducer_node_id}:validator"
        step_router_node_id = f"{step_validator_node_id}:router"
        self.assertEqual(graph["selected_option_id"], "option1")
        self.assertEqual(node_kinds[step_node_id], "step")
        self.assertEqual(node_kinds[step_reducer_node_id], "reducer")
        self.assertEqual(node_kinds[step_validator_node_id], "validator")
        self.assertEqual(node_kinds[step_router_node_id], "router")
        self.assertEqual(node_kinds[attempt_validator_node_id], "validator")
        self.assertEqual(node_kinds[attempt_router_node_id], "router")
        self.assertEqual(graph["statistics"]["reducer_count"], 1)
        self.assertEqual(graph["statistics"]["validator_count"], 2)
        self.assertEqual(graph["statistics"]["router_count"], 2)

        edge_relations = {(item["source"], item["target"], item["relation"]) for item in graph["edges"]}
        self.assertIn((workflow_node_id, attempt_node_id, "attempt"), edge_relations)
        self.assertIn((attempt_node_id, step_node_id, "contains"), edge_relations)
        self.assertIn((step_node_id, step_reducer_node_id, "reduced_by"), edge_relations)
        self.assertIn((step_reducer_node_id, step_validator_node_id, "validated_by"), edge_relations)
        self.assertIn((step_validator_node_id, step_router_node_id, "routed_by"), edge_relations)
        self.assertIn((attempt_validator_node_id, attempt_router_node_id, "routed_by"), edge_relations)

    def test_build_workflow_graph_models_replans_and_clarification_nodes(self):
        graph = build_workflow_graph(
            task="recover workflow after invalid attempt",
            task_shape="lookup",
            status="needs_clarification",
            selected_option=None,
            result=None,
            attempts=[
                {
                    "option": {"id": "option1", "label": "Primary plan"},
                    "status": "failed",
                    "result": "partial answer",
                    "validation": {
                        "valid": False,
                        "verdict": "invalid",
                        "reason": "Primary attempt only returned a partial answer.",
                    },
                    "routing": {
                        "scope": "workflow",
                        "stage": "validation",
                        "action": "replan_workflow",
                        "reason": "Primary attempt only returned a partial answer.",
                    },
                    "replan": {
                        "scope": "workflow",
                        "status": "derived",
                        "replace_step_id": "__workflow__",
                        "derived_option_id": "option2",
                        "reason": "Recovered fallback",
                        "steps": [{"id": "step1", "task": "fallback attempt"}],
                    },
                    "steps": [
                        {
                            "id": "step1",
                            "task": "primary attempt",
                            "target_agent": "executor",
                            "status": "completed",
                            "event": "task.result",
                            "payload": {"detail": "Primary execution finished.", "result": "partial answer"},
                            "result": "partial answer",
                            "routing": {
                                "scope": "step",
                                "stage": "validation",
                                "action": "replan_step",
                                "reason": "Step diverged from the request.",
                            },
                            "replan": {
                                "scope": "step",
                                "status": "received",
                                "step_id": "step1",
                                "replace_step_id": "step1",
                                "reason": "Use a repaired step.",
                                "steps": [{"id": "step1_1", "task": "fallback attempt"}],
                            },
                            "steps": [
                                {
                                    "id": "step1_1",
                                    "task": "fallback attempt",
                                    "target_agent": "executor",
                                    "status": "failed",
                                    "event": "task.result",
                                    "payload": {"detail": "Need a clearer target."},
                                    "result": None,
                                    "routing": {
                                        "scope": "step",
                                        "stage": "execution",
                                        "action": "clarify",
                                        "reason": "Need a clearer target.",
                                    },
                                    "clarification": {
                                        "detail": "Need a clearer target.",
                                        "question": "Which fallback target should I use?",
                                        "missing_information": ["target system"],
                                    },
                                }
                            ],
                        }
                    ],
                },
                {
                    "option": {
                        "id": "option2",
                        "label": "Recovered fallback",
                        "derived_from_attempt": 1,
                    },
                    "status": "needs_clarification",
                    "result": None,
                    "validation": {
                        "valid": False,
                        "verdict": "uncertain",
                        "reason": "The fallback path still needs clarification.",
                    },
                    "routing": {
                        "scope": "workflow",
                        "stage": "validation",
                        "action": "clarify",
                        "reason": "The fallback path still needs clarification.",
                    },
                    "clarification": {
                        "detail": "The fallback path still needs clarification.",
                        "question": "Which fallback target should I use?",
                        "missing_information": ["target system"],
                    },
                    "steps": [],
                },
            ],
        )

        node_kinds = {item["node_id"]: item["kind"] for item in graph["nodes"]}
        workflow_node_id = graph["root_node_id"]
        attempt1_node_id = f"{workflow_node_id}:attempt:1"
        attempt2_node_id = f"{workflow_node_id}:attempt:2"
        step1_node_id = f"{attempt1_node_id}:step:step1"
        self.assertEqual(node_kinds[f"{attempt1_node_id}:validator:router"], "router")
        self.assertEqual(node_kinds[f"{attempt1_node_id}:validator:router:replan"], "replan")
        self.assertEqual(node_kinds[f"{attempt2_node_id}:validator:router:clarification"], "clarification")
        self.assertEqual(node_kinds[f"{step1_node_id}:router"], "router")
        self.assertEqual(node_kinds[f"{step1_node_id}:router:replan"], "replan")
        self.assertEqual(node_kinds[f"{attempt1_node_id}:step:step1_1:router:clarification"], "clarification")
        self.assertEqual(graph["statistics"]["replan_count"], 2)
        self.assertEqual(graph["statistics"]["clarification_count"], 2)

        edge_relations = {(item["source"], item["target"], item["relation"]) for item in graph["edges"]}
        self.assertIn((f"{attempt1_node_id}:validator:router:replan", attempt2_node_id, "activates"), edge_relations)
        self.assertIn((f"{step1_node_id}:router:replan", f"{attempt1_node_id}:step:step1_1", "expands_to"), edge_relations)


if __name__ == "__main__":
    unittest.main()
