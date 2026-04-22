import copy
import os
import sys
import types
import unittest
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
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
    _derive_presentation,
    _derive_shell_command,
    _fallback_steps,
    _format_discovered_agents,
    _infer_task_shape,
    _llm_plan_with_retries,
    _normalize_task_shape,
    _parse_decision,
    _compound_fallback_steps,
    _conda_env_removal_steps,
    _normalize_followup_shell_instruction,
    _normalize_steps,
    _select_target_agent,
    _should_override_shell_command,
    _sql_fallback_steps_for_task,
    _split_compound_request,
    _step_semantic_drift,
    handle_event,
)


CAPABILITIES = {
    "agents": [
        {
            "name": "slurm_runner_cluster",
            "subscribes_to": ["task.plan"],
            "description": "Slurm cluster agent",
            "execution_model": "deterministic_first_with_llm_fallback",
            "deterministic_catalog_version": "v4-initial",
            "deterministic_catalog_size": 16,
            "deterministic_catalog_families": ["cluster", "queue", "history"],
            "cluster_name": "default",
            "cluster_aliases": ["slurm", "cluster"],
        },
        {
            "name": "sql_runner_mydb",
            "subscribes_to": ["task.plan"],
            "description": "SQL database agent",
            "database_name": "mydb",
            "database_aliases": ["mydb"],
            "execution_model": "deterministic_first_with_llm_fallback",
            "deterministic_catalog_version": "v4-initial",
            "deterministic_catalog_size": 11,
            "deterministic_catalog_families": ["schema", "table", "aggregate"],
        },
        {
            "name": "sql_runner_dicom_mock",
            "subscribes_to": ["task.plan"],
            "description": "SQL database agent",
            "database_name": "dicom_mock",
            "database_aliases": ["dicom_mock", "dicom mock"],
        },
        {
            "name": "shell_runner",
            "subscribes_to": ["task.plan"],
            "description": "Shell agent",
        }
    ]
}


class PlannerSemanticValidationTests(unittest.TestCase):
    def test_sql_fallback_uses_inspect_schema_for_schema_listing(self):
        steps = _sql_fallback_steps_for_task(
            "list all schemas in dicom_mock",
            "list all schemas in dicom_mock",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["target_agent"], "sql_runner_dicom_mock")
        self.assertEqual(steps[0]["instruction"]["operation"], "inspect_schema")
        self.assertEqual(steps[0]["instruction"]["focus"], "list all schemas in dicom_mock")

    def test_sql_table_count_request_stays_in_count_shape(self):
        question = "Count the tables in the dicom schema of the mydb database"
        steps = _sql_fallback_steps_for_task(question, question, CAPABILITIES)
        self.assertEqual(_infer_task_shape(question), "count")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["target_agent"], "sql_runner_mydb")
        self.assertEqual(steps[0]["instruction"]["operation"], "query_from_request")
        self.assertEqual(steps[0]["instruction"]["question"], question)

    def test_split_compound_request_on_repeated_count_clause(self):
        parts = _split_compound_request(
            "how many jobs are running on my slurm cluster and how many jobs are pending on my slurm cluster ?"
        )
        self.assertEqual(
            parts,
            [
                "how many jobs are running on my slurm cluster",
                "how many jobs are pending on my slurm cluster ?",
            ],
        )

    def test_compound_fallback_steps_expand_slurm_counts(self):
        steps = _compound_fallback_steps(
            "how many jobs are running on my slurm cluster and how many jobs are pending on my slurm cluster ?",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["target_agent"], "slurm_runner_cluster")
        self.assertEqual(steps[1]["target_agent"], "slurm_runner_cluster")
        self.assertEqual(steps[0]["instruction"]["question"], "how many jobs are running on my slurm cluster")
        self.assertEqual(steps[1]["instruction"]["question"], "how many jobs are pending on my slurm cluster ?")

    def test_compound_fallback_steps_expand_multi_domain_counts_with_difference(self):
        steps = _compound_fallback_steps(
            (
                "In the dicom_mock database count patients, and in the repository root count Python files, "
                "then report both counts and the difference."
            ),
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["target_agent"], "sql_runner_dicom_mock")
        self.assertEqual(steps[0]["instruction"]["question"], "In the dicom_mock database count patients")
        self.assertEqual(steps[1]["target_agent"], "shell_runner")
        self.assertIn("repository root", steps[1]["task"].lower())
        self.assertIn("wc -l", steps[1]["instruction"]["command"])
        self.assertEqual(steps[2]["target_agent"], "shell_runner")
        self.assertEqual(steps[2]["task"], "compute the absolute difference between the previous counts")
        self.assertEqual(steps[2]["depends_on"], ["step1", "step2"])

    def test_compound_fallback_steps_expand_multi_domain_counts_without_difference(self):
        steps = _compound_fallback_steps(
            (
                "Count pending Slurm jobs for vinith, count patients in the dicom_mock database, "
                "and count Python files in the repository root. Report all three counts in one answer."
            ),
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual([step["target_agent"] for step in steps], ["slurm_runner_cluster", "sql_runner_dicom_mock", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["question"], "Count pending Slurm jobs for vinith")
        self.assertEqual(steps[1]["instruction"]["question"], "count patients in the dicom_mock database")
        self.assertIn("repository root", steps[2]["task"].lower())
        self.assertIn("wc -l", steps[2]["instruction"]["command"])

    def test_split_compound_request_shared_verb_docker_inventory(self):
        parts = _split_compound_request("list all docker containers and docke r images on this machines")
        self.assertEqual(
            parts,
            [
                "list all docker containers on this machine",
                "list docker images on this machine",
            ],
        )

    def test_compound_fallback_steps_expand_shared_verb_docker_inventory(self):
        steps = _compound_fallback_steps(
            "list all docker containers and docke r images on this machines",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], "docker ps -a")
        self.assertEqual(steps[1]["instruction"]["command"], "docker images")

    def test_split_compound_request_expands_shared_verb_before_presentation_clause(self):
        parts = _split_compound_request(
            "Count all docker containers and docker images on this machine, then report both counts and the difference."
        )
        self.assertEqual(
            parts,
            [
                "Count all docker containers on this machine",
                "Count docker images on this machine",
                "report both counts and the difference",
            ],
        )

    def test_fallback_steps_return_same_family_shell_compound_steps(self):
        steps = _fallback_steps(
            "list all docker containers and docke r images on this machines",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], "docker ps -a")
        self.assertEqual(steps[1]["instruction"]["command"], "docker images")

    def test_conda_removal_workflow_expands_remove_confirm_and_final_list(self):
        steps = _conda_env_removal_steps(
            "remove conda environment named vinith, use -y and confirm it was removed, give me a final list of all env"
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], "conda env remove -n vinith -y")
        self.assertEqual(steps[0]["instruction"]["capture"]["mode"], "stdout_stripped")
        self.assertIn('"exists"', steps[1]["instruction"]["command"])
        self.assertEqual(steps[1]["instruction"]["capture"]["mode"], "json")
        self.assertEqual(steps[1]["depends_on"], ["step1"])
        self.assertEqual(steps[2]["instruction"]["command"], "conda env list")
        self.assertEqual(steps[2]["depends_on"], ["step2"])

    def test_fallback_steps_prefer_conda_removal_workflow_for_compound_remove_request(self):
        steps = _fallback_steps(
            "remove conda environment named vinith, use -y and confirm it was removed, give me a final list of all env",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["instruction"]["command"], "conda env remove -n vinith -y")
        self.assertEqual(steps[1]["instruction"]["capture"]["mode"], "json")
        self.assertEqual(steps[2]["instruction"]["command"], "conda env list")

    def test_fallback_steps_handle_missing_leading_r_in_remove(self):
        steps = _fallback_steps(
            "emove conda environment named vinith, use -y and confirm it was removed, give me a final list of all env",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["instruction"]["command"], "conda env remove -n vinith -y")

    def test_compound_fallback_steps_expand_git_branch_and_clean_state(self):
        steps = _compound_fallback_steps(
            "Show the current git branch and whether the working tree is clean.",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], "git branch --show-current")
        self.assertIn("git diff --quiet", steps[1]["instruction"]["command"])

    def test_compound_fallback_steps_expand_shared_scope_python_difference(self):
        steps = _compound_fallback_steps(
            "Using the shell, count Python files in the repository root and in agent_library/agents, then report both counts and the difference.",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], 'find . -maxdepth 1 -type f -name "*.py" | wc -l')
        self.assertEqual(
            steps[1]["instruction"]["command"],
            'find agent_library/agents -mindepth 1 -maxdepth 1 -type f -name "*.py" | wc -l',
        )
        self.assertEqual(steps[2]["depends_on"], ["step1", "step2"])

    def test_fallback_steps_use_explicit_root_markdown_inventory_commands(self):
        steps = _fallback_steps(
            "Using the shell in the repository root, list the Markdown files alphabetically and tell me the total count.",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertIn("markdown", steps[0]["task"].lower())
        self.assertEqual(
            steps[0]["instruction"]["command"],
            'find . -maxdepth 1 -type f -iname "*.md" -printf "%f\\n" | sort',
        )
        self.assertIn("count markdown files", steps[1]["task"].lower())
        self.assertEqual(steps[1]["instruction"]["command"], 'find . -maxdepth 1 -type f -iname "*.md" | wc -l')

    def test_fallback_steps_use_non_recursive_root_openwebui_inventory_commands(self):
        steps = _fallback_steps(
            "Using the shell in the repository root, count the files whose names contain openwebui and list them.",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(
            steps[0]["instruction"]["command"],
            'find . -maxdepth 1 -type f -iname "*openwebui*" -printf "%f\\n" | sort',
        )
        self.assertEqual(steps[1]["instruction"]["command"], 'find . -maxdepth 1 -type f -iname "*openwebui*" | wc -l')

    def test_fallback_steps_expand_shell_only_root_python_vs_markdown_difference(self):
        steps = _fallback_steps(
            "Using the shell in the repository root, count Python files and Markdown files, then report both counts and the difference.",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], 'find . -maxdepth 1 -type f -name "*.py" | wc -l')
        self.assertEqual(steps[1]["instruction"]["command"], 'find . -maxdepth 1 -type f -iname "*.md" | wc -l')
        self.assertEqual(steps[2]["depends_on"], ["step1", "step2"])
        self.assertIn("dependency_results", steps[2]["instruction"]["command"])

    def test_fallback_steps_use_deterministic_line_count_command(self):
        question = "Using the shell, how many lines are in VERSION_4_PRIMITIVE_CATALOG.md?"
        self.assertTrue(_should_override_shell_command(question.lower()))
        steps = _fallback_steps(question, CAPABILITIES)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["target_agent"], "shell_runner")
        self.assertEqual(steps[0]["command"], "wc -l < VERSION_4_PRIMITIVE_CATALOG.md")

    def test_fallback_steps_use_runtime_scope_for_inventory_requests(self):
        question = "Using the shell, under runtime list the Python files alphabetically and tell me the total count."
        steps = _fallback_steps(question, CAPABILITIES)
        self.assertEqual(len(steps), 2)
        self.assertEqual(
            steps[0]["instruction"]["command"],
            'find runtime -mindepth 1 -maxdepth 1 -type f -name "*.py" -printf "%f\\n" | sort',
        )
        self.assertEqual(steps[1]["instruction"]["command"], 'find runtime -mindepth 1 -maxdepth 1 -type f -name "*.py" | wc -l')

    def test_fallback_steps_use_deterministic_string_count_command(self):
        question = "Using the shell, in openwebui_gateway.py how many times does the string PlannerGateway appear?"
        self.assertTrue(_should_override_shell_command(question.lower()))
        steps = _fallback_steps(question, CAPABILITIES)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["target_agent"], "shell_runner")
        self.assertEqual(steps[0]["command"], "python3 -c 'import pathlib\nimport sys\n\npath = pathlib.Path(sys.argv[1])\nneedle = sys.argv[2]\nprint(path.read_text(encoding=\"utf-8\").count(needle))' openwebui_gateway.py PlannerGateway")

    def test_derive_shell_command_supports_reverse_string_count_phrasing(self):
        command = _derive_shell_command(
            "using the shell count how many times task.plan appears in runtime/engine.py"
        )
        self.assertEqual(
            command,
            "python3 -c 'import pathlib\nimport sys\n\npath = pathlib.Path(sys.argv[1])\nneedle = sys.argv[2]\nprint(path.read_text(encoding=\"utf-8\").count(needle))' runtime/engine.py task.plan",
        )

    def test_compound_fallback_steps_add_difference_for_generic_shell_count_pairs(self):
        steps = _compound_fallback_steps(
            (
                "Using the shell, in runtime/engine.py count how many times task.plan appears, and in "
                "openwebui_gateway.py count how many times PlannerGateway appears. Report both counts and the difference."
            ),
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["target_agent"], "shell_runner")
        self.assertEqual(steps[1]["target_agent"], "shell_runner")
        self.assertEqual(steps[2]["target_agent"], "shell_runner")
        self.assertIn("runtime/engine.py", steps[0]["instruction"]["command"])
        self.assertIn("openwebui_gateway.py", steps[1]["instruction"]["command"])
        self.assertIn("dependency_results", steps[2]["instruction"]["command"])
        self.assertEqual(steps[2]["depends_on"], ["step1", "step2"])

    def test_compound_fallback_steps_expand_line_count_and_markdown_token_difference(self):
        steps = _compound_fallback_steps(
            (
                "Using the shell, how many lines are in VERSION_4_PRIMITIVE_CATALOG.md, and across the Markdown files "
                "in the repository root how many times does the word graph appear? Report both counts and the difference."
            ),
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual([step["target_agent"] for step in steps], ["shell_runner", "shell_runner", "shell_runner"])
        self.assertEqual(steps[0]["instruction"]["command"], "wc -l < VERSION_4_PRIMITIVE_CATALOG.md")
        self.assertIn("graph", steps[1]["instruction"]["command"])
        self.assertEqual(steps[2]["depends_on"], ["step1", "step2"])

    def test_llm_plan_with_retries_requests_a_second_candidate_after_validation_failure(self):
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
                {
                    "id": "step3",
                    "target_agent": "shell_runner",
                    "task": "compute the absolute difference between the previous counts",
                    "instruction": {
                        "operation": "run_command",
                        "command": "python3 -c 'import json,sys; data=json.load(sys.stdin); print(abs(int(data[0]) - int(data[1])))'",
                        "capture": {"mode": "stdout_stripped"},
                    },
                    "depends_on": ["step1", "step2"],
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
                {"valid": False, "reason": "collapsed workflow", "issues": ["collapsed"], "goal_coverage": "partial", "decomposition": "collapsed", "user_action_alignment": "weak", "rewarded_paths": [], "disallowed_paths": ["step1"]},
                {"valid": True, "reason": "good workflow", "issues": [], "goal_coverage": "complete", "decomposition": "good", "user_action_alignment": "strong", "rewarded_paths": ["step1", "step2", "step3"], "disallowed_paths": []},
            ],
        ), patch.object(
            planner_module,
            "_debug_log",
        ):
            result = _llm_plan_with_retries(question, CAPABILITIES)

        self.assertTrue(result["steps"])
        self.assertEqual(result["steps"][0]["target_agent"], "sql_runner_mydb")
        self.assertEqual(result["steps"][1]["target_agent"], "shell_runner")
        self.assertEqual(result["steps"][2]["depends_on"], ["step1", "step2"])
        retry_mock.assert_called_once()

    def test_llm_plan_with_retries_reasks_for_collapsed_compound_boolean_followup(self):
        question = "list all conda environemtns and check if there is an environment named vinith ?"
        invalid_decision = {
            "processable": True,
            "reason": "first attempt",
            "task_shape": "lookup",
            "steps": [
                {
                    "id": "step1",
                    "target_agent": "shell_runner",
                    "task": "list all conda environments",
                    "instruction": {
                        "operation": "run_command",
                        "command": "conda env list",
                        "capture": {"mode": "stdout_stripped"},
                    },
                }
            ],
            "presentation": {"format": "markdown"},
        }
        valid_decision = {
            "processable": True,
            "reason": "second attempt",
            "task_shape": "lookup",
            "steps": [
                {
                    "id": "step1",
                    "target_agent": "shell_runner",
                    "task": "list all conda environments",
                    "instruction": {
                        "operation": "run_command",
                        "command": "conda env list",
                        "capture": {"mode": "stdout_stripped"},
                    },
                },
                {
                    "id": "step2",
                    "target_agent": "shell_runner",
                    "task": "check whether the conda environment named vinith exists",
                    "instruction": {
                        "operation": "run_command",
                        "command": "conda env list --json | python3 -c 'import json,sys,os; name=\"vinith\"; envs=json.load(sys.stdin).get(\"envs\", []); exists=any(os.path.basename(path.rstrip(\"/\")) == name for path in envs); print(json.dumps({\"exists\": exists, \"name\": name})); raise SystemExit(0 if exists else 1)'",
                        "capture": {"mode": "json"},
                        "allow_returncodes": [0, 1],
                    },
                    "depends_on": ["step1"],
                },
            ],
            "presentation": {"format": "markdown"},
        }

        with patch.object(planner_module, "_llm_decide", return_value=invalid_decision), patch.object(
            planner_module,
            "_llm_retry_decide",
            return_value=valid_decision,
        ) as retry_mock, patch.object(
            planner_module,
            "_llm_validate_plan_semantics",
            side_effect=[
                {"valid": False, "reason": "collapsed clause coverage", "issues": ["collapsed"], "goal_coverage": "partial", "decomposition": "collapsed", "user_action_alignment": "weak", "rewarded_paths": [], "disallowed_paths": ["step1"]},
                {"valid": True, "reason": "good decomposition", "issues": [], "goal_coverage": "complete", "decomposition": "good", "user_action_alignment": "strong", "rewarded_paths": ["step1", "step2"], "disallowed_paths": []},
            ],
        ), patch.object(
            planner_module,
            "_debug_log",
        ):
            result = _llm_plan_with_retries(question, CAPABILITIES)

        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][0]["instruction"]["command"], "conda env list")
        self.assertIn('"exists"', result["steps"][1]["instruction"]["command"])
        self.assertEqual(result["steps"][1]["depends_on"], ["step1"])
        retry_mock.assert_called_once()

    def test_handle_event_uses_valid_llm_plan_before_deterministic_fallback(self):
        planner_module.CAPABILITIES["agents"] = copy.deepcopy(CAPABILITIES["agents"])
        planner_module.CAPABILITIES["available_events"] = {"task.plan", "plan.progress", "task.result"}
        with patch.object(
            planner_module,
            "_llm_plan_with_retries",
            return_value={
                "decision": {
                    "processable": True,
                    "reason": "good routing",
                    "presentation": {"format": "plain"},
                    "task_shape": "count",
                },
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "shell_runner",
                        "task": "Using the shell, in openwebui_gateway.py how many times does the string PlannerGateway appear?",
                        "instruction": {
                            "operation": "run_command",
                            "command": "rg -o --fixed-strings 'PlannerGateway' openwebui_gateway.py | wc -l",
                            "capture": {"mode": "stdout_stripped"},
                        },
                    }
                ],
            },
        ):
            response = handle_event(
                types.SimpleNamespace(
                    event="user.ask",
                    payload={"question": "Using the shell, in openwebui_gateway.py how many times does the string PlannerGateway appear?"},
                )
            )

        task_plan = next(item for item in response["emits"] if item["event"] == "task.plan")
        step = task_plan["payload"]["steps"][0]
        self.assertEqual(step["target_agent"], "shell_runner")
        self.assertEqual(
            step["instruction"]["command"],
            "rg -o --fixed-strings 'PlannerGateway' openwebui_gateway.py | wc -l",
        )

    def test_handle_event_emits_llm_decomposed_shared_verb_shell_plan(self):
        request = types.SimpleNamespace(
            event="user.ask",
            payload={"question": "list all docker containers and docke r images on this machines"},
        )

        original_capabilities = copy.deepcopy(planner_module.CAPABILITIES)
        planner_module.CAPABILITIES.clear()
        planner_module.CAPABILITIES.update({"agents": copy.deepcopy(CAPABILITIES["agents"])})
        try:
            with patch(
                "agent_library.agents.llm_operations_planner._llm_plan_with_retries",
                return_value={
                    "decision": {
                        "processable": True,
                        "reason": "good decomposed docker plan",
                        "task_shape": "list",
                        "presentation": {"format": "markdown"},
                    },
                    "steps": [
                        {
                            "id": "step1",
                            "target_agent": "shell_runner",
                            "task": "list all docker containers on this machine",
                            "instruction": {
                                "operation": "run_command",
                                "command": "docker ps -a",
                            },
                        },
                        {
                            "id": "step2",
                            "target_agent": "shell_runner",
                            "task": "list docker images on this machine",
                            "instruction": {
                                "operation": "run_command",
                                "command": "docker images",
                            },
                        },
                    ],
                    "attempt": 2,
                    "validation_reason": "good decomposition",
                },
            ):
                response = handle_event(request)
        finally:
            planner_module.CAPABILITIES.clear()
            planner_module.CAPABILITIES.update(original_capabilities)

        task_plan = next(item["payload"] for item in response["emits"] if item["event"] == "task.plan")
        self.assertEqual(len(task_plan["steps"]), 2)
        self.assertEqual(task_plan["steps"][0]["target_agent"], "shell_runner")
        self.assertEqual(task_plan["steps"][0]["instruction"]["command"], "docker ps -a")
        self.assertEqual(task_plan["steps"][1]["instruction"]["command"], "docker images")

    def test_handle_event_returns_validation_failure_when_llm_exhausts_retries(self):
        request = types.SimpleNamespace(
            event="user.ask",
            payload={"question": "list all docker containers and docke r images on this machines"},
        )

        original_capabilities = copy.deepcopy(planner_module.CAPABILITIES)
        planner_module.CAPABILITIES.clear()
        planner_module.CAPABILITIES.update({"agents": copy.deepcopy(CAPABILITIES["agents"]), "available_events": {"task.result"}})
        try:
            with patch(
                "agent_library.agents.llm_operations_planner._llm_plan_with_retries",
                return_value={
                    "decision": {
                        "processable": True,
                        "reason": "bad condensed docker plan",
                        "task_shape": "list",
                        "presentation": {"format": "markdown"},
                    },
                    "steps": [],
                    "attempt": 3,
                    "validation_reason": "Planner kept collapsing the request into one broad step.",
                },
            ):
                response = handle_event(request)
        finally:
            planner_module.CAPABILITIES.clear()
            planner_module.CAPABILITIES.update(original_capabilities)

        payload = next(item["payload"] for item in response["emits"] if item["event"] == "task.result")
        self.assertEqual(payload["detail"], "Planner kept collapsing the request into one broad step.")

    def test_handle_event_emits_llm_decomposed_multi_domain_plan(self):
        request = types.SimpleNamespace(
            event="user.ask",
            payload={
                "question": (
                    "In the dicom_mock database count patients, and in the repository root count Python files, "
                    "then report both counts and the difference."
                )
            },
        )

        original_capabilities = copy.deepcopy(planner_module.CAPABILITIES)
        planner_module.CAPABILITIES.clear()
        planner_module.CAPABILITIES.update({"agents": copy.deepcopy(CAPABILITIES["agents"])})
        try:
            with patch(
                "agent_library.agents.llm_operations_planner._llm_plan_with_retries",
                return_value={
                    "decision": {
                        "processable": True,
                        "reason": "processable multi-domain count workflow",
                        "task_shape": "count",
                        "presentation": {"format": "markdown"},
                    },
                    "steps": [
                        {
                            "id": "step1",
                            "target_agent": "sql_runner_dicom_mock",
                            "task": "count patients in the dicom_mock database",
                            "instruction": {
                                "operation": "query_from_request",
                                "question": "count patients in the dicom_mock database",
                            },
                        },
                        {
                            "id": "step2",
                            "target_agent": "shell_runner",
                            "task": "count Python files in the repository root",
                            "instruction": {
                                "operation": "run_command",
                                "command": "find . -type f -name '*.py' | wc -l",
                            },
                        },
                        {
                            "id": "step3",
                            "target_agent": "shell_runner",
                            "task": "compute the absolute difference between the previous counts",
                            "instruction": {
                                "operation": "run_command",
                                "command": "python3 -c 'import json,sys; data=json.load(sys.stdin); print(abs(int(data[0]) - int(data[1])))'",
                            },
                            "depends_on": ["step1", "step2"],
                        },
                    ],
                    "attempt": 2,
                    "validation_reason": "good decomposition",
                },
            ):
                response = handle_event(request)
        finally:
            planner_module.CAPABILITIES.clear()
            planner_module.CAPABILITIES.update(original_capabilities)

        task_plan = next(item["payload"] for item in response["emits"] if item["event"] == "task.plan")
        self.assertEqual(len(task_plan["steps"]), 3)
        self.assertEqual(task_plan["steps"][0]["target_agent"], "sql_runner_dicom_mock")
        self.assertEqual(task_plan["steps"][1]["target_agent"], "shell_runner")
        self.assertEqual(task_plan["steps"][2]["target_agent"], "shell_runner")
        self.assertIn("difference", task_plan["steps"][2]["task"].lower())

    def test_detects_slurm_node_to_job_drift(self):
        self.assertTrue(
            _step_semantic_drift(
                "What are the names of nodes in my slurm cluster?",
                "slurm_runner_cluster",
                "show queued Slurm jobs for user vinith",
                {"operation": "query_from_request", "question": "show queued Slurm jobs for user vinith"},
            )
        )

    def test_normalize_steps_recovers_drifted_slurm_step_to_original_question(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "slurm_runner_cluster",
                "task": "show queued Slurm jobs for user vinith",
                "instruction": {"operation": "query_from_request", "question": "show queued Slurm jobs for user vinith"},
            }
        ]
        normalized = _normalize_steps("What are the names of nodes in my slurm cluster?", steps, CAPABILITIES)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["task"], "What are the names of nodes in my slurm cluster?")
        self.assertEqual(normalized[0]["instruction"]["operation"], "query_from_request")
        self.assertEqual(
            normalized[0]["instruction"]["question"],
            "What are the names of nodes in my slurm cluster?",
        )

    def test_normalize_steps_keeps_valid_slurm_node_step(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "slurm_runner_cluster",
                "task": "show node names in the Slurm cluster",
                "instruction": {"operation": "query_from_request", "question": "show node names in the Slurm cluster"},
            }
        ]
        normalized = _normalize_steps("What are the names of nodes in my slurm cluster?", steps, CAPABILITIES)
        self.assertEqual(normalized[0]["task"], "show node names in the Slurm cluster")
        self.assertEqual(
            normalized[0]["instruction"]["question"],
            "show node names in the Slurm cluster",
        )

    def test_normalize_steps_keeps_valid_sql_step(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "sql_runner_mydb",
                "task": "query how many patients have more than 20 studies in mydb",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "query how many patients have more than 20 studies in mydb",
                },
            }
        ]
        normalized = _normalize_steps(
            "what is the count of patients that have more than 20 studies in mydb",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(normalized[0]["instruction"]["question"], "query how many patients have more than 20 studies in mydb")

    def test_normalize_steps_recovers_compound_slurm_request_when_llm_drops_second_clause(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "slurm_runner_cluster",
                "task": "how many jobs are running on my slurm cluster",
                "instruction": {"operation": "query_from_request", "question": "how many jobs are running on my slurm cluster"},
            }
        ]
        normalized = _normalize_steps(
            "how many jobs are running on my slurm cluster and how many jobs are pending on my slurm cluster ?",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["instruction"]["question"], "how many jobs are running on my slurm cluster")
        self.assertEqual(normalized[1]["instruction"]["question"], "how many jobs are pending on my slurm cluster ?")

    def test_normalize_steps_recovers_multi_domain_count_workflow_when_llm_drops_other_agents(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "sql_runner_dicom_mock",
                "task": "Count pending Slurm jobs for vinith, count patients in the dicom_mock database, and count Python files in the repository root. Report all three counts in one answer.",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "Count pending Slurm jobs for vinith, count patients in the dicom_mock database, and count Python files in the repository root. Report all three counts in one answer.",
                },
            }
        ]
        normalized = _normalize_steps(
            "Count pending Slurm jobs for vinith, count patients in the dicom_mock database, and count Python files in the repository root. Report all three counts in one answer.",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(len(normalized), 3)
        self.assertEqual([step["target_agent"] for step in normalized], ["slurm_runner_cluster", "sql_runner_dicom_mock", "shell_runner"])

    def test_sql_fallback_declines_multi_domain_count_workflow(self):
        steps = _sql_fallback_steps_for_task(
            (
                "In the dicom_mock database count patients, and in the repository root count Python files, "
                "then report both counts and the difference."
            ),
            (
                "In the dicom_mock database count patients, and in the repository root count Python files, "
                "then report both counts and the difference."
            ),
            CAPABILITIES,
        )
        self.assertEqual(steps, [])

    def test_derive_shell_command_for_save_rows_task(self):
        command = _derive_shell_command("create a list of these patients and save it in patient.txt")
        self.assertIn("patient.txt", command)
        self.assertIn("json.dumps", command)
        self.assertIn("path.resolve", command)

    def test_normalize_steps_rewrites_count_then_save_sql_workflow_to_row_export(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "sql_runner_mydb",
                "task": "Query how many patients over 45 years of age with more than 20 studies exist in mydb",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "how many patients do i have over 45 years of age with more than 20 studies in mydb",
                },
            },
            {
                "id": "step2",
                "target_agent": "shell_runner",
                "task": "create a list of these patients and save it in patient.txt",
                "instruction": {
                    "operation": "run_command",
                    "command": "python3 -c \"bad command\"",
                    "input": {"$from": "step1.rows"},
                    "capture": {"mode": "stdout_stripped"},
                },
                "depends_on": ["step1"],
            },
        ]
        normalized = _normalize_steps(
            "how many patients do i have over 45 years of age with more than 20 studies in mydb, create a list of these users and save it in patient.txt",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(
            normalized[0]["instruction"]["question"],
            "how many patients do i have over 45 years of age with more than 20 studies in mydb. Return one row per matching result and include all relevant detail columns. Do not return only an aggregate count.",
        )
        self.assertIn("patient.txt", normalized[1]["instruction"]["command"])
        self.assertEqual(normalized[1]["instruction"]["input"], {"$from": "step1.rows"})

    def test_normalize_steps_appends_save_step_for_table_list_request(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "sql_runner_mydb",
                "task": "introspect the database schema to list all tables",
                "instruction": {
                    "operation": "inspect_schema",
                    "focus": "tables",
                },
            }
        ]
        normalized = _normalize_steps(
            "give me a list of all the tables in mydb and save this in a file named tables.txt and give me the final path of this file",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["instruction"]["operation"], "inspect_schema")
        self.assertEqual(normalized[1]["target_agent"], "shell_runner")
        self.assertIn("tables.txt", normalized[1]["instruction"]["command"])
        self.assertEqual(normalized[1]["instruction"]["input"], {"$from": "step1.rows"})
        self.assertEqual(normalized[1]["depends_on"], ["step1"])

    def test_normalize_followup_shell_instruction_rewrites_single_brace_placeholder_to_prev(self):
        normalized = _normalize_followup_shell_instruction(
            {"operation": "run_command", "command": "head -n 5 {file_path}"}
        )
        self.assertEqual(normalized["command"], "head -n 5 {{prev}}")

    def test_normalize_steps_rewrites_followup_shell_placeholder(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "shell_runner",
                "task": "find the newest shell script file",
                "instruction": {
                    "operation": "run_command",
                    "command": 'find . -type f -name "*.sh" -printf "%T@ %p\\n" | sort -nr | head -n 1 | cut -d" " -f2-',
                    "capture": {"mode": "stdout_first_line"},
                },
            },
            {
                "id": "step2",
                "target_agent": "shell_runner",
                "task": "show the first 5 lines of the newest shell script file",
                "instruction": {"operation": "run_command", "command": "head -n 5 {file_path}"},
                "depends_on": ["step1"],
            },
        ]
        normalized = _normalize_steps(
            "find the newest shell script and show the first 5 lines",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(normalized[1]["instruction"]["command"], "head -n 5 {{prev}}")

    def test_derive_presentation_enables_internal_steps_for_raw_stage_output_requests(self):
        presentation = _derive_presentation("list all docker containers and include raw outputs for each stage")
        self.assertTrue(presentation["include_internal_steps"])
        self.assertEqual(presentation["format"], "markdown_table")

    def test_infer_task_shape_treats_count_plus_details_sql_request_as_list(self):
        self.assertEqual(
            _infer_task_shape(
                "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
                target_agent="sql_runner_dicom_mock",
                presentation={"format": "markdown_table"},
            ),
            "list",
        )

    def test_normalize_task_shape_overrides_raw_count_for_count_plus_details_request(self):
        self.assertEqual(
            _normalize_task_shape(
                "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
                "count",
                target_agent="sql_runner_dicom_mock",
                presentation={"format": "markdown_table"},
            ),
            "list",
        )

    def test_sql_fallback_decomposes_count_plus_details_request(self):
        steps = _sql_fallback_steps_for_task(
            "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
            "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["target_agent"], "sql_runner_dicom_mock")
        self.assertIn("count", steps[0]["instruction"]["question"].lower())
        self.assertIn("list", steps[1]["instruction"]["question"].lower())
        self.assertIn("mrn", steps[1]["instruction"]["question"].lower())
        self.assertEqual(steps[1]["depends_on"], ["step1"])

    def test_normalize_steps_expands_single_sql_step_for_count_plus_details_request(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "sql_runner_mydb",
                "task": "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
                },
            }
        ]
        normalized = _normalize_steps(
            "count of all users in dicom_mock who have more than 10 studies and provide me their mrn and names and other details in the db",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["instruction"]["operation"], "query_from_request")
        self.assertIn("count", normalized[0]["instruction"]["question"].lower())
        self.assertIn("list", normalized[1]["instruction"]["question"].lower())

    def test_normalize_steps_repairs_top_level_directory_listing_command(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "shell_runner",
                "task": "list top-level directories in this repo",
                "instruction": {
                    "operation": "run_command",
                    "command": 'find . -maxdepth 1 -type d -not -path "./*" -printf "%f\\n"',
                },
            }
        ]
        normalized = _normalize_steps(
            "list top-level directories in this repo",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(
            normalized[0]["instruction"]["command"],
            'find . -mindepth 1 -maxdepth 1 -type d -printf "%f\\n" | sort',
        )

    def test_normalize_task_shape_overrides_raw_list_for_sql_export_request(self):
        self.assertEqual(
            _normalize_task_shape(
                "get a list of all patients over 45 having more than 2 studies and create a file with all their details and provide the location of the file to me so i can review.",
                "list",
                target_agent="sql_runner_mydb",
                presentation={"format": "markdown_table"},
            ),
            "save_artifact",
        )

    def test_normalize_steps_rewrites_single_shell_plan_into_sql_export_workflow(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "shell_runner",
                "task": "get a list of all patients over 45 having more than 2 studies and create a file with all their details and provide the location of the file to me so i can review.",
            }
        ]
        normalized = _normalize_steps(
            "get a list of all patients over 45 having more than 2 studies and create a file with all their details and provide the location of the file to me so i can review.",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["target_agent"], "sql_runner_mydb")
        self.assertEqual(normalized[0]["instruction"]["operation"], "query_from_request")
        self.assertIn("all relevant detail columns", normalized[0]["instruction"]["question"])
        self.assertEqual(normalized[1]["target_agent"], "shell_runner")
        self.assertIn("artifacts/exports/", normalized[1]["instruction"]["command"])
        self.assertEqual(normalized[1]["instruction"]["input"], {"$from": "step1.rows"})
        self.assertEqual(normalized[1]["depends_on"], ["step1"])

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

    def test_format_discovered_agents_includes_deterministic_metadata(self):
        formatted = _format_discovered_agents(CAPABILITIES)
        self.assertIn("execution_model=deterministic_first_with_llm_fallback", formatted)
        self.assertIn("deterministic_catalog_size=16", formatted)
        self.assertIn("cluster_aliases=[slurm, cluster]", formatted)
        self.assertIn("database_name=mydb", formatted)

    def test_format_discovered_agents_includes_planning_hints(self):
        capabilities = {
            "agents": [
                {
                    "name": "travel_agent",
                    "description": "Travel planner agent",
                    "capability_domains": ["travel"],
                    "action_verbs": ["book", "plan"],
                    "subscribes_to": ["task.plan"],
                    "apis": [
                        {
                            "name": "plan_trip",
                            "trigger_event": "task.plan",
                            "emits": ["task.result"],
                            "planning_hints": {
                                "instruction_operations": ["plan_trip"],
                            },
                        }
                    ],
                    "planning_hints": {
                        "keywords": ["flight", "hotel", "itinerary"],
                        "preferred_task_shapes": ["lookup"],
                        "routing_priority": 15,
                    },
                }
            ]
        }
        formatted = _format_discovered_agents(capabilities)
        self.assertIn("planning_hints[", formatted)
        self.assertIn("keywords=[flight, hotel, itinerary]", formatted)
        self.assertIn("instruction_operations=[plan_trip]", formatted)

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
        self.assertEqual(summary["agents"][0]["planning_hints"]["keywords"], ["plan", "route"])
        self.assertEqual(summary["agents"][0]["apis"][0]["trigger_event"], "user.ask")
        self.assertEqual(summary["agents"][0]["apis"][0]["emits"], ["task.plan"])
        self.assertEqual(summary["agents"][0]["apis"][0]["planning_hints"]["instruction_operations"], ["plan_task"])

    def test_infer_task_shape_count(self):
        self.assertEqual(
            _infer_task_shape("how many jobs are running on my slurm cluster?"),
            "count",
        )

    def test_infer_task_shape_slurm_node_count_and_state_is_lookup(self):
        self.assertEqual(
            _infer_task_shape("how many nodes are currently in my slurm cluster and what is their state ?"),
            "lookup",
        )

    def test_select_target_agent_prefers_slurm_for_node_inventory_summary(self):
        self.assertEqual(
            _select_target_agent("how many nodes are currently in my slurm cluster and what is their state ?", CAPABILITIES),
            "slurm_runner_cluster",
        )

    def test_select_target_agent_prefers_descriptor_planning_hints_for_custom_agent(self):
        capabilities = {
            "agents": [
                {
                    "name": "travel_agent",
                    "description": "Travel planning agent",
                    "capability_domains": ["travel"],
                    "action_verbs": ["book", "plan"],
                    "subscribes_to": ["task.plan"],
                    "apis": [
                        {
                            "name": "plan_trip",
                            "trigger_event": "task.plan",
                            "emits": ["task.result"],
                        }
                    ],
                    "planning_hints": {
                        "keywords": ["flight", "hotel", "airport", "itinerary", "travel"],
                        "preferred_task_shapes": ["lookup"],
                        "routing_priority": 25,
                    },
                },
                {
                    "name": "shell_runner",
                    "description": "Shell agent",
                    "subscribes_to": ["task.plan"],
                    "apis": [
                        {
                            "name": "run_command",
                            "trigger_event": "task.plan",
                            "emits": ["shell.result"],
                        }
                    ],
                    "planning_hints": {
                        "keywords": ["docker", "git", "file", "repo"],
                    },
                },
            ]
        }
        self.assertEqual(_select_target_agent("book a flight to NYC", capabilities), "travel_agent")

    def test_fallback_steps_use_single_slurm_step_for_node_inventory_summary(self):
        steps = _fallback_steps(
            "how many nodes are currently in my slurm cluster and what is their state ?",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["target_agent"], "slurm_runner_cluster")
        self.assertEqual(steps[0]["instruction"]["question"], "how many nodes are currently in my slurm cluster and what is their state ?")

    def test_normalize_steps_collapses_bad_two_step_node_inventory_plan_to_single_slurm_step(self):
        steps = [
            {
                "id": "step1",
                "target_agent": "sql_runner_mydb",
                "task": "count nodes are curently in my slurm cluster",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "count nodes are curently in my slurm cluster",
                },
            },
            {
                "id": "step2",
                "target_agent": "slurm_runner_cluster",
                "task": "what is their state ?",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "what is their state ?",
                },
            },
        ]
        normalized = _normalize_steps(
            "how many nodes are curently in my slurm cluster and what is their state ?",
            steps,
            CAPABILITIES,
        )
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["target_agent"], "slurm_runner_cluster")
        self.assertEqual(
            normalized[0]["instruction"]["question"],
            "how many nodes are curently in my slurm cluster and what is their state ?",
        )

    def test_infer_task_shape_save_artifact(self):
        self.assertEqual(
            _infer_task_shape("save these rows to results.txt and give me the final path"),
            "save_artifact",
        )

    def test_infer_task_shape_schema_summary(self):
        self.assertEqual(
            _infer_task_shape("show database schema and relationships"),
            "schema_summary",
        )

    def test_infer_task_shape_uses_lookup_for_compound_schema_and_count_workflow(self):
        self.assertEqual(
            _infer_task_shape(
                "In the dicom_mock database list the tables in the dicom schema alphabetically, and in the repository root using the shell count Markdown files. Report the table count, the Markdown count, and include the first few table names."
            ),
            "lookup",
        )

    def test_infer_task_shape_keeps_multi_domain_count_rows_workflow_out_of_list_shape(self):
        self.assertEqual(
            _infer_task_shape(
                "In the dicom_mock database count rows in dicom_tags, and in agent_library/specs using the shell count YAML spec files. Report both counts and the difference."
            ),
            "lookup",
        )

    def test_infer_task_shape_keeps_count_files_by_name_workflow_out_of_list_shape(self):
        self.assertEqual(
            _infer_task_shape(
                "In the mydb database count patients, and in the repository root using the shell count files whose names contain openwebui. Report both counts and the difference."
            ),
            "lookup",
        )

    def test_compound_fallback_steps_expand_mixed_db_and_string_count_workflow(self):
        steps = _compound_fallback_steps(
            (
                "In the mydb database count studies, and using the shell count how many times task.plan appears in "
                "runtime/engine.py. Report both counts and the difference."
            ),
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["target_agent"], "sql_runner_mydb")
        self.assertEqual(steps[0]["instruction"]["question"], "In the mydb database count studies")
        self.assertEqual(steps[1]["target_agent"], "shell_runner")
        self.assertEqual(
            steps[1]["instruction"]["command"],
            "python3 -c 'import pathlib\nimport sys\n\npath = pathlib.Path(sys.argv[1])\nneedle = sys.argv[2]\nprint(path.read_text(encoding=\"utf-8\").count(needle))' runtime/engine.py task.plan",
        )
        self.assertEqual(steps[2]["target_agent"], "shell_runner")
        self.assertEqual(steps[2]["depends_on"], ["step1", "step2"])


if __name__ == "__main__":
    unittest.main()
