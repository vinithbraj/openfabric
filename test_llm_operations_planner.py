import sys
import types
import unittest


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
sys.modules.setdefault("fastapi", fastapi_stub)

from agent_library.agents.llm_operations_planner import (
    _normalize_steps,
    _shell_retrieve_then_compute_steps,
    _step_semantic_drift,
)


CAPABILITIES = {
    "agents": [
        {
            "name": "shell_runner",
            "description": "Executes shell commands.",
            "subscribes_to": ["shell.exec", "task.plan"],
            "methods": [],
            "capability_domains": ["general_shell"],
            "action_verbs": ["run", "execute"],
            "emits": ["shell.result", "task.result"],
        },
        {
            "name": "slurm_runner_cluster",
            "subscribes_to": ["task.plan"],
            "description": "Slurm cluster agent",
        },
        {
            "name": "sql_runner_mydb",
            "subscribes_to": ["task.plan"],
            "description": "SQL database agent",
            "database_name": "mydb",
            "database_aliases": ["mydb"],
        },
    ]
}


class PlannerSemanticValidationTests(unittest.TestCase):
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


class PlannerDerivedStepTests(unittest.TestCase):
    def test_disk_space_question_decomposes_into_two_steps(self):
        steps = _shell_retrieve_then_compute_steps(
            "How much free space do I have on this machine and compute the size in GB?",
            CAPABILITIES,
        )
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["id"], "step1")
        self.assertEqual(steps[1]["depends_on"], ["step1"])
        self.assertIn("df -B1 /", steps[0]["command"])
        self.assertIn("{{step1}}", steps[1]["command"])


if __name__ == "__main__":
    unittest.main()
