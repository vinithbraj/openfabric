import copy
import sys
import types
import unittest
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.HTTPException = Exception


class _RequestStub:
    pass


fastapi_stub.Request = _RequestStub
sys.modules.setdefault("fastapi", fastapi_stub)

requests_stub = types.ModuleType("requests")
requests_stub.post = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

import agent_library.agents.llm_operations_planner as planner_module
from agent_library.agents.llm_operations_planner import (
    _capability_summary,
    _format_discovered_agents,
    _llm_plan_with_retries,
    _normalize_followup_shell_instruction,
    _parse_decision,
    handle_event,
)


CAPABILITIES = {
    "agents": [
        {
            "name": "slurm_runner_cluster",
            "subscribes_to": ["task.plan"],
            "description": "Slurm cluster agent",
            "execution_model": "llm_selects_local_execution",
            "deterministic_catalog_version": "v4-initial",
            "deterministic_catalog_size": 16,
            "deterministic_catalog_families": ["cluster", "queue", "history"],
            "cluster_name": "default",
            "cluster_aliases": ["slurm", "cluster"],
            "planning_hints": {
                "keywords": ["slurm", "cluster", "job"],
                "preferred_task_shapes": ["lookup", "count"],
                "instruction_operations": ["query_from_request"],
                "routing_priority": 10,
            },
        },
        {
            "name": "sql_runner_mydb",
            "subscribes_to": ["task.plan"],
            "description": "SQL database agent",
            "database_name": "mydb",
            "database_aliases": ["mydb"],
            "execution_model": "llm_selects_local_execution",
            "deterministic_catalog_version": "v4-initial",
            "deterministic_catalog_size": 11,
            "deterministic_catalog_families": ["schema", "table", "aggregate"],
            "planning_hints": {
                "keywords": ["database", "sql", "patients"],
                "preferred_task_shapes": ["lookup", "count", "list"],
                "instruction_operations": ["query_from_request", "inspect_schema"],
            },
        },
        {
            "name": "sql_runner_dicom_mock",
            "subscribes_to": ["task.plan"],
            "description": "SQL database agent",
            "database_name": "dicom_mock",
            "database_aliases": ["dicom_mock", "dicom mock"],
            "planning_hints": {
                "keywords": ["dicom", "schema", "patients"],
                "preferred_task_shapes": ["lookup", "count", "list"],
                "instruction_operations": ["query_from_request", "inspect_schema"],
            },
        },
        {
            "name": "shell_runner",
            "subscribes_to": ["task.plan"],
            "description": "Shell agent",
            "planning_hints": {
                "keywords": ["shell", "file", "docker", "git"],
                "preferred_task_shapes": ["lookup", "count", "command_execution"],
                "instruction_operations": ["run_command"],
                "structured_followup": True,
            },
        },
    ]
}


class PlannerActivePathTests(unittest.TestCase):
    def setUp(self):
        self.original_capabilities = copy.deepcopy(planner_module.CAPABILITIES)
        planner_module.CAPABILITIES.clear()
        planner_module.CAPABILITIES.update(
            {
                "agents": copy.deepcopy(CAPABILITIES["agents"]),
                "available_events": {"task.plan", "plan.progress", "task.result", "planner.replan.result"},
                "execution_policy": {},
            }
        )

    def tearDown(self):
        planner_module.CAPABILITIES.clear()
        planner_module.CAPABILITIES.update(self.original_capabilities)

    def test_llm_plan_with_retries_requests_second_candidate_after_semantic_failure(self):
        question = (
            "In the mydb database count the tables in the dicom schema, and in the repository root using the shell "
            "count directories. Report both counts and the difference."
        )
        invalid_decision = {
            "processable": True,
            "reason": "first attempt",
            "task_shape": "count",
            "steps": [
                {
                    "id": "step1",
                    "target_agent": "sql_runner_mydb",
                    "task": "Count the tables in the dicom schema of the mydb database",
                    "instruction": {
                        "operation": "inspect_schema",
                        "focus": "Count the tables in the dicom schema of the mydb database",
                    },
                }
            ],
            "presentation": {"format": "plain"},
        }
        valid_decision = {
            "processable": True,
            "reason": "second attempt",
            "task_shape": "compare",
            "steps": [
                {
                    "id": "step1",
                    "target_agent": "sql_runner_mydb",
                    "task": "count the tables in the dicom schema of the mydb database",
                    "instruction": {
                        "operation": "query_from_request",
                        "question": "count the tables in the dicom schema of the mydb database",
                    },
                },
                {
                    "id": "step2",
                    "target_agent": "shell_runner",
                    "task": "count directories in the repository root",
                    "instruction": {
                        "operation": "run_command",
                        "command": "find . -mindepth 1 -maxdepth 1 -type d | wc -l",
                        "capture": {"mode": "stdout_stripped"},
                    },
                },
            ],
            "presentation": {"format": "plain"},
        }

        with patch.object(planner_module, "_llm_decide", return_value=invalid_decision), patch.object(
            planner_module,
            "_llm_retry_decide",
            return_value=valid_decision,
        ) as retry_mock, patch.object(
            planner_module,
            "_llm_validate_plan_semantics",
            side_effect=[
                {
                    "valid": False,
                    "reason": "collapsed workflow",
                    "issues": ["collapsed"],
                    "goal_coverage": "partial",
                    "decomposition": "collapsed",
                    "user_action_alignment": "weak",
                    "rewarded_paths": [],
                    "disallowed_paths": ["step1"],
                },
                {
                    "valid": True,
                    "reason": "good workflow",
                    "issues": [],
                    "goal_coverage": "complete",
                    "decomposition": "good",
                    "user_action_alignment": "strong",
                    "rewarded_paths": ["step1", "step2"],
                    "disallowed_paths": [],
                },
            ],
        ), patch.object(planner_module, "_debug_log"):
            result = _llm_plan_with_retries(question, planner_module.CAPABILITIES)

        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][0]["target_agent"], "sql_runner_mydb")
        self.assertEqual(result["steps"][1]["target_agent"], "shell_runner")
        retry_mock.assert_called_once()

    def test_handle_event_emits_progress_with_llm_task_shape(self):
        with patch.object(
            planner_module,
            "_llm_plan_with_retries",
            return_value={
                "decision": {
                    "processable": True,
                    "reason": "good routing",
                    "presentation": {"format": "plain"},
                    "task_shape": "boolean_check",
                },
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "shell_runner",
                        "task": "check whether the conda environment named vinith exists",
                        "instruction": {
                            "operation": "run_command",
                            "command": "conda env list --json",
                            "capture": {"mode": "json"},
                        },
                    }
                ],
            },
        ):
            response = handle_event(
                types.SimpleNamespace(
                    event="user.ask",
                    payload={"question": "Is there a conda environment named vinith?"},
                )
            )

        progress = next(item["payload"] for item in response["emits"] if item["event"] == "plan.progress")
        task_plan = next(item["payload"] for item in response["emits"] if item["event"] == "task.plan")
        self.assertEqual(progress["task_shape"], "boolean_check")
        self.assertEqual(task_plan["task_shape"], "boolean_check")
        self.assertEqual(task_plan["steps"][0]["target_agent"], "shell_runner")

    def test_handle_event_replan_progress_uses_existing_task_shape(self):
        with patch.object(
            planner_module,
            "_llm_replan_with_retries",
            return_value={
                "decision": {"replace_step_id": "step2"},
                "steps": [
                    {
                        "id": "step2_1",
                        "target_agent": "shell_runner",
                        "task": "list all conda environments",
                        "instruction": {"operation": "run_command", "command": "conda env list"},
                    },
                    {
                        "id": "step2_2",
                        "target_agent": "shell_runner",
                        "task": "check whether the conda environment named vinith exists",
                        "instruction": {"operation": "run_command", "command": "conda env list --json"},
                        "depends_on": ["step2_1"],
                    },
                ],
            },
        ):
            response = handle_event(
                types.SimpleNamespace(
                    event="planner.replan.request",
                    payload={
                        "task": "list all conda environments and check whether vinith exists",
                        "step_id": "step2",
                        "task_shape": "lookup",
                        "presentation": {"format": "markdown"},
                    },
                )
            )

        progress = next(item["payload"] for item in response["emits"] if item["event"] == "plan.progress")
        result = next(item["payload"] for item in response["emits"] if item["event"] == "planner.replan.result")
        self.assertEqual(progress["task_shape"], "lookup")
        self.assertEqual(result["replace_step_id"], "step2")
        self.assertEqual(len(result["steps"]), 2)

    def test_handle_event_returns_validation_failure_when_llm_exhausts_retries(self):
        with patch.object(
            planner_module,
            "_llm_plan_with_retries",
            return_value={
                "decision": {
                    "processable": True,
                    "reason": "bad condensed plan",
                    "task_shape": "lookup",
                    "presentation": {"format": "markdown"},
                },
                "steps": [],
                "attempt": 3,
                "validation_reason": "Planner kept collapsing the request into one broad step.",
            },
        ):
            response = handle_event(
                types.SimpleNamespace(
                    event="user.ask",
                    payload={"question": "list all conda environemtns and check if there is an environment named vinith ?"},
                )
            )

        payload = next(item["payload"] for item in response["emits"] if item["event"] == "task.result")
        self.assertEqual(payload["detail"], "Planner kept collapsing the request into one broad step.")

    def test_handle_event_returns_reason_for_unprocessable_request(self):
        with patch.object(
            planner_module,
            "_llm_plan_with_retries",
            return_value={
                "decision": {
                    "processable": False,
                    "reason": "No matching capability found.",
                    "task_shape": "lookup",
                    "presentation": {"format": "markdown"},
                },
                "steps": [],
                "attempt": 1,
                "validation_reason": "",
            },
        ):
            response = handle_event(
                types.SimpleNamespace(
                    event="user.ask",
                    payload={"question": "Book me a flight to NYC."},
                )
            )

        payload = next(item["payload"] for item in response["emits"] if item["event"] == "task.result")
        self.assertEqual(payload["detail"], "No matching capability found.")

    def test_parse_decision_accepts_ranked_plan_options(self):
        decision = _parse_decision(
            {
                "processable": True,
                "reason": "multiple viable approaches",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "shell_runner",
                        "task": "list files",
                        "instruction": {"operation": "run_command", "command": "find . -maxdepth 1 -type f"},
                    }
                ],
                "plan_options": [
                    {
                        "id": "option1",
                        "label": "Primary",
                        "reason": "use shell discovery",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "shell_runner",
                                "task": "list files",
                                "instruction": {"operation": "run_command", "command": "find . -maxdepth 1 -type f"},
                            }
                        ],
                    },
                    {
                        "id": "option2",
                        "label": "Fallback",
                        "reason": "use ripgrep file listing",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "shell_runner",
                                "task": "list files with rg",
                                "instruction": {"operation": "run_command", "command": "rg --files ."},
                            }
                        ],
                    },
                ],
            }
        )
        self.assertIsNotNone(decision)
        self.assertEqual(len(decision["plan_options"]), 2)
        self.assertEqual(decision["plan_options"][0]["id"], "option1")
        self.assertEqual(decision["plan_options"][1]["label"], "Fallback")

    def test_format_discovered_agents_includes_planning_hints_and_metadata(self):
        formatted = _format_discovered_agents(CAPABILITIES)
        self.assertIn("execution_model=llm_selects_local_execution", formatted)
        self.assertIn("deterministic_catalog_size=16", formatted)
        self.assertIn("cluster_aliases=[slurm, cluster]", formatted)
        self.assertIn("planning_hints[", formatted)
        self.assertIn("preferred_task_shapes=[count, lookup]", formatted)

    def test_capability_summary_uses_trigger_and_emit_contract_fields(self):
        capabilities = {
            "agents": [
                {
                    "name": "planner",
                    "description": "Planner",
                    "planning_hints": {
                        "keywords": ["route", "plan"],
                    },
                    "apis": [
                        {
                            "name": "plan_task",
                            "trigger_event": "user.ask",
                            "emits": ["task.plan"],
                            "request_contract": "agent_execution_request",
                            "result_contract": "agent_execution_result",
                            "planning_hints": {"instruction_operations": ["plan_task"]},
                        }
                    ],
                }
            ]
        }
        summary = _capability_summary(capabilities)
        self.assertEqual(summary["agents"][0]["trigger_events"], ["user.ask"])
        self.assertEqual(summary["agents"][0]["emits"], ["task.plan"])
        self.assertEqual(summary["agents"][0]["apis"][0]["trigger_event"], "user.ask")
        self.assertEqual(summary["agents"][0]["apis"][0]["emits"], ["task.plan"])

    def test_normalize_followup_shell_instruction_rewrites_single_brace_placeholder_to_prev(self):
        normalized = _normalize_followup_shell_instruction(
            {
                "operation": "run_command",
                "command": "docker inspect {container_id}",
            }
        )
        self.assertEqual(normalized["command"], "docker inspect {{prev}}")


if __name__ == "__main__":
    unittest.main()
