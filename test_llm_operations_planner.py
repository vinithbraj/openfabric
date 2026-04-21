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
    _derive_presentation,
    _derive_shell_command,
    _infer_task_shape,
    _normalize_task_shape,
    _parse_decision,
    _compound_fallback_steps,
    _normalize_followup_shell_instruction,
    _normalize_steps,
    _sql_fallback_steps_for_task,
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

    def test_infer_task_shape_count(self):
        self.assertEqual(
            _infer_task_shape("how many jobs are running on my slurm cluster?"),
            "count",
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


if __name__ == "__main__":
    unittest.main()
