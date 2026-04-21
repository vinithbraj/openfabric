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

from agent_library.agents.llm_operations_planner import (
    CAPABILITIES,
    _fallback_steps,
    _normalize_steps,
    _normalized_sql_query_question,
    handle_event,
)


TEST_CAPABILITIES = {
    "agents": [
        {
            "name": "slurm_runner_cluster",
            "subscribes_to": ["task.plan"],
            "emits": ["slurm.result", "task.result"],
            "metadata": {"template_agent": "slurm_runner"},
        },
        {
            "name": "sql_runner_dicom_mock",
            "subscribes_to": ["task.plan"],
            "emits": ["sql.result", "task.result"],
            "metadata": {"template_agent": "sql_runner"},
        },
        {
            "name": "shell_runner",
            "subscribes_to": ["task.plan"],
            "emits": ["shell.result", "task.result"],
        },
    ],
    "available_events": {"planner.replan.result", "plan.progress"},
}


class PlannerReplanGuardTests(unittest.TestCase):
    def setUp(self):
        self.original_capabilities = copy.deepcopy(CAPABILITIES)
        CAPABILITIES.clear()
        CAPABILITIES.update(copy.deepcopy(TEST_CAPABILITIES))

    def tearDown(self):
        CAPABILITIES.clear()
        CAPABILITIES.update(self.original_capabilities)

    def test_normalize_steps_contextualizes_slurm_followup_job_ids(self):
        steps = _normalize_steps(
            "In the Slurm cluster, how many pending jobs does vinith have, and list the job IDs.",
            [
                {
                    "id": "step1",
                    "target_agent": "slurm_runner_cluster",
                    "task": "In the Slurm cluster, how many pending jobs does vinith have",
                    "instruction": {
                        "operation": "query_from_request",
                        "question": "In the Slurm cluster, how many pending jobs does vinith have",
                    },
                },
                {
                    "id": "step2",
                    "target_agent": "slurm_runner_cluster",
                    "task": "list the job IDs",
                    "instruction": {
                        "operation": "query_from_request",
                        "question": "list the job IDs",
                    },
                },
            ],
            CAPABILITIES,
        )

        self.assertEqual(len(steps), 2)
        step2 = steps[1]
        self.assertEqual(step2["target_agent"], "slurm_runner_cluster")
        self.assertIn("pending", step2["task"].lower())
        self.assertIn("vinith", step2["task"].lower())
        self.assertIn("pending", step2["instruction"]["question"].lower())
        self.assertIn("vinith", step2["instruction"]["question"].lower())

    def test_normalize_steps_contextualizes_shell_followup_total_count(self):
        steps = _normalize_steps(
            "In the repository root, list the first five Python files alphabetically and tell me the total count.",
            [
                {
                    "id": "step1",
                    "target_agent": "shell_runner",
                    "task": "list the first five Python files alphabetically in the repository root",
                    "instruction": {
                        "operation": "run_command",
                        "command": 'find . -maxdepth 1 -type f -name "*.py" -print | sort | head -n 5',
                    },
                },
                {
                    "id": "step2",
                    "target_agent": "shell_runner",
                    "task": "tell me the total count",
                },
            ],
            CAPABILITIES,
        )

        self.assertEqual(len(steps), 2)
        step2 = steps[1]
        self.assertEqual(step2["target_agent"], "shell_runner")
        self.assertIn("count python files", step2["task"].lower())
        self.assertIn("repository root", step2["task"].lower())
        self.assertIn('find . -maxdepth 1 -type f -name "*.py" | wc -l', step2["instruction"]["command"])

    def test_single_shell_compound_step_recovers_to_list_and_count_steps(self):
        steps = _normalize_steps(
            "In the repository root, list the first five Python files alphabetically and tell me the total count.",
            [
                {
                    "id": "step1",
                    "target_agent": "shell_runner",
                    "task": "In the repository root, list the first five Python files alphabetically and tell me the total count.",
                }
            ],
            CAPABILITIES,
        )

        self.assertEqual(len(steps), 2)
        self.assertIn("first five python files", steps[0]["task"].lower())
        self.assertIn("count python files", steps[1]["task"].lower())
        self.assertIn('find . -maxdepth 1 -type f -name "*.py" | wc -l', steps[1]["instruction"]["command"])

    def test_fallback_steps_use_explicit_root_python_inventory_commands(self):
        steps = _fallback_steps(
            "In the repository root, list the first five Python files alphabetically and tell me the total count.",
            CAPABILITIES,
        )

        self.assertEqual(len(steps), 2)
        self.assertIn("first five python files", steps[0]["task"].lower())
        self.assertIn('find . -maxdepth 1 -type f -name "*.py" -printf "%f\\n" | sort | head -n 5', steps[0]["instruction"]["command"])
        self.assertIn("count python files", steps[1]["task"].lower())
        self.assertEqual(steps[1]["instruction"]["command"], 'find . -maxdepth 1 -type f -name "*.py" | wc -l')

    def test_normalize_steps_rewrites_difference_step_to_dependency_aggregate(self):
        steps = _normalize_steps(
            (
                "In the dicom_mock database count patients, and in the repository root count Python files, "
                "then report both counts and the difference."
            ),
            [
                {
                    "id": "step1",
                    "target_agent": "sql_runner_dicom_mock",
                    "task": "Count patients in the dicom_mock database",
                    "instruction": {
                        "operation": "query_from_request",
                        "question": "Count patients in the dicom_mock database",
                    },
                },
                {
                    "id": "step2",
                    "target_agent": "shell_runner",
                    "task": "Count Python files in the repository root",
                    "instruction": {
                        "operation": "run_command",
                        "command": 'find . -maxdepth 1 -type f -name "*.py" | wc -l',
                    },
                },
                {
                    "id": "step3",
                    "target_agent": "shell_runner",
                    "task": "count Python files in the repository root",
                    "instruction": {
                        "operation": "run_command",
                        "command": (
                            'python3 -c "import sys; dicom_count = int(sys.stdin.read().split()[0]); '
                            'py_count = int(sys.argv[1]); print(dicom_count - py_count)"'
                        ),
                    },
                    "depends_on": ["step1", "step2"],
                },
            ],
            CAPABILITIES,
        )

        self.assertEqual(len(steps), 3)
        step3 = steps[2]
        self.assertIn("difference", step3["task"].lower())
        self.assertIn("dependency_results", step3["instruction"]["command"])
        self.assertIn("abs(numbers[0] - numbers[1])", step3["instruction"]["command"])

    def test_normalized_sql_query_question_prefers_plain_count_task_over_distinct_drift(self):
        normalized = _normalized_sql_query_question(
            "Count patients in the dicom_mock database",
            "Count the number of distinct patients in the dicom_mock database",
        )
        self.assertEqual(normalized, "Count patients in the dicom_mock database")

    def test_workflow_replan_rejects_cross_domain_sql_candidate_for_slurm_question(self):
        request = types.SimpleNamespace(
            event="planner.replan.request",
            payload={
                "task": "In the Slurm cluster, how many pending jobs does vinith have, and list the job IDs.",
                "step_id": "__workflow__",
                "reason": "Workflow returned job information for multiple users instead of just vinith's pending jobs",
                "available_context": {
                    "last_steps": [
                        {
                            "id": "step1",
                            "target_agent": "slurm_runner_cluster",
                            "task": "In the Slurm cluster, how many pending jobs does vinith have",
                        }
                    ]
                },
            },
        )

        with patch(
            "agent_library.agents.llm_operations_planner._llm_replan",
            return_value={
                "replace_step_id": "__workflow__",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "sql_runner_dicom_mock",
                        "task": "List the qualifying rows with requested details",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "list the qualifying rows with requested details",
                        },
                    }
                ],
            },
        ):
            response = handle_event(request)

        emits = response["emits"]
        replan_payload = next(item["payload"] for item in emits if item["event"] == "planner.replan.result")
        self.assertTrue(replan_payload["steps"])
        self.assertEqual(replan_payload["steps"][0]["target_agent"], "slurm_runner_cluster")
        self.assertIn("pending", replan_payload["steps"][0]["instruction"]["question"].lower())
        self.assertIn("vinith", replan_payload["steps"][0]["instruction"]["question"].lower())


if __name__ == "__main__":
    unittest.main()
