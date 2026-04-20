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

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.agents.llm_operations_planner import (
    _derive_shell_command,
    _compound_fallback_steps,
    _normalize_steps,
    _split_compound_request,
    _step_semantic_drift,
)


CAPABILITIES = {
    "agents": [
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
        {
            "name": "shell_runner",
            "subscribes_to": ["task.plan"],
            "description": "Shell agent",
        }
    ]
}


class PlannerSemanticValidationTests(unittest.TestCase):
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

    def test_derive_shell_command_for_save_rows_task(self):
        command = _derive_shell_command("create a list of these patients and save it in patient.txt")
        self.assertIn("patient.txt", command)
        self.assertIn("PatientID", command)
        self.assertIn("PatientName", command)

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
            "how many patients do i have over 45 years of age with more than 20 studies in mydb. Return one row per matching patient and include PatientID and PatientName. Do not return only an aggregate count.",
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


if __name__ == "__main__":
    unittest.main()
