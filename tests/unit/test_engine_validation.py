import copy
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)
jsonschema_stub = types.ModuleType("jsonschema")
jsonschema_stub.validate = lambda instance, schema: None
sys.modules.setdefault("jsonschema", jsonschema_stub)

from runtime.engine import Engine
from runtime.registry import ADAPTER_REGISTRY


class _ExecAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "task.plan":
            return []
        task = payload.get("task")
        if task == "primary attempt":
            return [("task.result", {"detail": "Primary execution finished.", "result": "partial answer"})]
        if task == "fallback attempt":
            return [("task.result", {"detail": "Fallback execution finished.", "result": "final answer"})]
        if task == "uncertain attempt":
            return [("task.result", {"detail": "Execution finished with ambiguous result.", "result": ""})]
        if task == "bad count step":
            return [("task.result", {"detail": "Step finished with state-only output.", "result": 'The state of the nodes is "idle".'})]
        if task == "fixed count and state step":
            return [("task.result", {"detail": "Recovered node inventory.", "result": "Total nodes: 3\nState idle: 2\nState mixed: 1"})]
        if task == "reduce raw shell output":
            return [
                (
                    "shell.result",
                    {
                        "detail": "Raw shell output ready.",
                        "command": "printf 'alpha\\nbeta\\n'",
                        "stdout": "alpha\nbeta\n",
                        "stderr": "",
                        "returncode": 0,
                        "reduction_request": {
                            "kind": "shell.local_reducer",
                            "task": "reduce raw shell output",
                            "source_command": "printf 'alpha\\nbeta\\n'",
                            "sample": "alpha\nbeta\n",
                        },
                    },
                )
            ]
        if task == "sql config failure step":
            return [
                (
                    "task.result",
                    {
                        "detail": "SQL task failed: RuntimeError: Unsupported SQL_AGENT_DSN scheme: unknown",
                        "status": "failed",
                        "error": "RuntimeError: Unsupported SQL_AGENT_DSN scheme: unknown",
                    },
                )
            ]
        return [("task.result", {"detail": "Unknown task", "status": "failed", "error": "unsupported"})]


class _CrashOnceExecAdapter:
    crash_counts = {}

    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "task.plan":
            return []
        if payload.get("task") == "resume after crash step":
            run_id = str(payload.get("run_id") or payload.get("original_task") or "resume_after_crash").strip()
            count = self.__class__.crash_counts.get(run_id, 0)
            if count == 0:
                self.__class__.crash_counts[run_id] = 1
                raise RuntimeError("simulated executor crash")
            return [("task.result", {"detail": "Resumed step finished.", "result": "resumed answer"})]
        return _ExecAdapter({}).handle(event_name, payload)


class _ValidatorAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "validation.request":
            return []
        if payload.get("validation_scope") == "step":
            if payload.get("step_task") == "bad count step":
                return [
                    (
                        "validation.result",
                        {
                            "valid": False,
                            "verdict": "invalid",
                            "reason": "The step returned only state text and missed the requested count.",
                            "retry_recommended": True,
                            "missing_requirements": ["count"],
                            "trace": ["Checked step output against the step task.", "Detected missing count evidence."],
                        },
                    )
                ]
            return [
                (
                    "validation.result",
                    {
                        "valid": True,
                        "verdict": "valid",
                        "reason": "The step output satisfies the step intent.",
                        "retry_recommended": False,
                        "missing_requirements": [],
                        "trace": ["Checked step output against the step task.", "Detected complete fulfillment."],
                    },
                )
            ]
        if payload.get("task") == "uncertain goal":
            return [
                (
                    "validation.result",
                    {
                        "valid": False,
                        "verdict": "uncertain",
                        "reason": "The result is too ambiguous to accept confidently.",
                        "retry_recommended": False,
                        "missing_requirements": ["narrower target"],
                        "trace": ["Validation could not determine whether the result satisfies the request."],
                    },
                )
            ]
        task_text = str(payload.get("task") or "").lower()
        result = payload.get("result")
        result_text = str(result)
        if (
            ("how many" in task_text and "state" in task_text) or "node inventory" in task_text
        ) and "total nodes:" in result_text.lower() and "state " in result_text.lower():
            return [
                (
                    "validation.result",
                    {
                        "valid": True,
                        "verdict": "valid",
                        "reason": "Recovered attempt now satisfies the combined count/state request.",
                        "retry_recommended": False,
                        "missing_requirements": [],
                        "trace": ["Checked the option output against the original task.", "Detected both count and state information."],
                    },
                )
            ]
        if "count the lines" in task_text and "reduced lines:" in result_text.lower():
            return [
                (
                    "validation.result",
                    {
                        "valid": True,
                        "verdict": "valid",
                        "reason": "Reduced shell output now satisfies the requested count.",
                        "retry_recommended": False,
                        "missing_requirements": [],
                        "trace": ["Checked the option output against the original task.", "Detected the reduced shell count."],
                    },
                )
            ]
        if payload.get("option_id") == "option1":
            return [
                (
                    "validation.result",
                    {
                        "valid": False,
                        "verdict": "invalid",
                        "reason": "Primary attempt only returned a partial answer.",
                        "retry_recommended": True,
                        "missing_requirements": ["final answer completeness"],
                        "trace": ["Checked the option output against the original task.", "Detected incomplete fulfillment."],
                    },
                )
            ]
        return [
            (
                "validation.result",
                {
                    "valid": True,
                    "verdict": "valid",
                    "reason": "Fallback attempt satisfies the task.",
                    "retry_recommended": False,
                    "missing_requirements": [],
                    "trace": ["Checked the option output against the original task.", "Detected complete fulfillment."],
                },
            )
        ]


class _RecorderAdapter:
    events = []

    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        self.__class__.events.append((event_name, copy.deepcopy(payload)))
        return []


class _ReducerAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "data.reduce":
            return []
        reduced_result = payload.get("existing_reduced_result")
        reduction_request = payload.get("reduction_request") if isinstance(payload.get("reduction_request"), dict) else {}
        if reduced_result in (None, "", [], {}):
            input_data = payload.get("input_data")
            if isinstance(reduction_request, dict) and reduction_request.get("kind") == "pass_through":
                reduced_result = input_data
            elif isinstance(input_data, str):
                reduced_result = f"Reduced lines: {len([line for line in input_data.splitlines() if line.strip()])}"
            elif isinstance(input_data, dict) and isinstance(input_data.get("rows"), list):
                reduced_result = f"Reduced rows: {len(input_data.get('rows', []))}"
        return [
            (
                "data.reduced",
                {
                    "step_id": payload.get("step_id"),
                    "task": payload.get("task", ""),
                    "original_task": payload.get("original_task", ""),
                    "target_agent": payload.get("target_agent", ""),
                    "source_event": payload.get("source_event", ""),
                    "detail": "Reduced step output.",
                    "reduced_result": reduced_result,
                    "strategy": (
                        "reduction_request"
                        if reduction_request
                        else "local_reduction_command" if payload.get("local_reduction_command") else "existing_reduced_result"
                    ),
                    "attempts": 1,
                    "local_reduction_command": payload.get("local_reduction_command"),
                },
            )
        ]


class _FallbackPlannerAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "planner.replan.request":
            return []
        if payload.get("step_id") == "step1":
            return [
                (
                    "planner.replan.result",
                    {
                        "replace_step_id": "step1",
                        "reason": "Use a repaired node inventory step.",
                        "steps": [
                            {
                                "id": "step1_1",
                                "target_agent": "executor",
                                "task": "fixed count and state step",
                                "instruction": {"operation": "run_command", "command": "fixed"},
                            }
                        ],
                    },
                )
            ]
        return [
            (
                "planner.replan.result",
                {
                    "replace_step_id": payload.get("step_id", "__workflow__"),
                    "reason": "Use the fallback workflow.",
                    "steps": [
                        {
                            "id": "step1",
                            "target_agent": "executor",
                            "task": "fallback attempt",
                            "instruction": {"operation": "run_command", "command": "fallback"},
                        }
                    ],
                },
            )
        ]


class _NoRecoveryPlannerAdapter:
    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "planner.replan.request":
            return []
        return [
            (
                "task.result",
                {
                    "detail": payload.get("reason") or "Planner could not further decompose the failed step.",
                    "status": "failed",
                    "error": payload.get("reason") or "Planner could not further decompose the failed step.",
                },
            )
        ]


TEST_SPEC = {
    "contracts": {
        "TaskPlan": {
            "type": "object",
            "required": ["task"],
            "properties": {
                "task": {"type": "string"},
                "run_id": {"type": "string"},
                "resume_run_id": {"type": "string"},
                "task_shape": {"type": "string"},
                "retry_budget": {"type": "object"},
                "steps": {"type": "array"},
                "plan_options": {"type": "array"},
            },
        },
        "TaskResult": {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {"type": "string"},
                "status": {"type": "string"},
                "error": {"type": "string"},
                "result": {},
            },
        },
        "ShellResult": {
            "type": "object",
            "properties": {
                "detail": {"type": "string"},
                "command": {"type": "string"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "returncode": {"type": "number"},
                "local_reduction_command": {"type": "string"},
                "reduction_request": {"type": "object"},
                "reduced_result": {},
            },
        },
        "ValidationRequest": {
            "type": "object",
            "required": ["task", "workflow_status"],
            "properties": {
                "task": {"type": "string"},
                "run_id": {"type": "string"},
                "task_shape": {"type": "string"},
                "validation_llm_budget_remaining": {"type": "number"},
                "workflow_status": {"type": "string"},
                "option_id": {"type": "string"},
                "steps": {"type": "array"},
                "result": {},
                "error": {"type": "string"},
                "node": {"type": "object"},
            },
        },
        "ValidationResult": {
            "type": "object",
            "required": ["valid", "reason", "retry_recommended"],
            "properties": {
                "valid": {"type": "boolean"},
                "verdict": {"type": "string"},
                "reason": {"type": "string"},
                "retry_recommended": {"type": "boolean"},
                "missing_requirements": {"type": "array"},
                "trace": {"type": "array"},
            },
        },
        "DataReduceRequest": {
            "type": "object",
            "required": ["task", "step_id", "source_event"],
            "properties": {
                "run_id": {"type": "string"},
                "attempt": {"type": "number"},
                "task": {"type": "string"},
                "original_task": {"type": "string"},
                "step_id": {"type": "string"},
                "target_agent": {"type": "string"},
                "source_event": {"type": "string"},
                "reduction_request": {"type": "object"},
                "local_reduction_command": {"type": "string"},
                "existing_reduced_result": {},
                "input_data": {},
                "source_value": {},
                "source_payload": {"type": "object"},
                "available_context": {"type": "object"},
                "node": {"type": "object"},
            },
        },
        "DataReducedResult": {
            "type": "object",
            "required": ["step_id", "source_event", "strategy"],
            "properties": {
                "step_id": {"type": "string"},
                "task": {"type": "string"},
                "original_task": {"type": "string"},
                "target_agent": {"type": "string"},
                "source_event": {"type": "string"},
                "detail": {"type": "string"},
                "reduced_result": {},
                "strategy": {"type": "string"},
                "attempts": {"type": "number"},
                "local_reduction_command": {"type": "string"},
                "error": {"type": "string"},
                "node": {"type": "object"},
            },
        },
        "ValidationProgress": {
            "type": "object",
            "required": ["stage", "task", "message"],
            "properties": {
                "stage": {"type": "string"},
                "task": {"type": "string"},
                "message": {"type": "string"},
            },
        },
        "PlannerReplanRequest": {
            "type": "object",
            "required": ["task", "step_id", "reason"],
            "properties": {
                "task": {"type": "string"},
                "task_shape": {"type": "string"},
                "step_id": {"type": "string"},
                "reason": {"type": "string"},
                "available_context": {"type": "object"},
                "partial_result": {},
                "presentation": {"type": "object"},
            },
        },
        "PlannerReplanResult": {
            "type": "object",
            "required": ["replace_step_id", "steps"],
            "properties": {
                "replace_step_id": {"type": "string"},
                "reason": {"type": "string"},
                "steps": {"type": "array"},
            },
        },
        "WorkflowResult": {
            "type": "object",
            "required": ["task", "status", "steps"],
            "properties": {
                "task": {"type": "string"},
                "run_id": {"type": "string"},
                "task_shape": {"type": "string"},
                "status": {"type": "string"},
                "steps": {"type": "array"},
                "result": {},
                "persistence": {"type": "object"},
                "graph": {"type": "object"},
            },
        },
        "ClarificationRequired": {
            "type": "object",
            "required": ["task", "detail"],
            "properties": {
                "task": {"type": "string"},
                "run_id": {"type": "string"},
                "task_shape": {"type": "string"},
                "detail": {"type": "string"},
                "question": {"type": "string"},
                "available_context": {"type": "object"},
                "missing_information": {"type": "array"},
                "persistence": {"type": "object"},
                "graph": {"type": "object"},
            },
        },
    },
    "events": {
        "task.plan": {"contract": "TaskPlan"},
        "task.result": {"contract": "TaskResult"},
        "shell.result": {"contract": "ShellResult"},
        "data.reduce": {"contract": "DataReduceRequest"},
        "data.reduced": {"contract": "DataReducedResult"},
        "validation.request": {"contract": "ValidationRequest"},
        "validation.result": {"contract": "ValidationResult"},
        "validation.progress": {"contract": "ValidationProgress"},
        "planner.replan.request": {"contract": "PlannerReplanRequest"},
        "planner.replan.result": {"contract": "PlannerReplanResult"},
        "workflow.result": {"contract": "WorkflowResult"},
        "clarification.required": {"contract": "ClarificationRequired"},
    },
    "agents": {
        "executor": {
            "runtime": {"adapter": "test_exec"},
            "subscribes_to": ["task.plan"],
            "emits": ["task.result"],
        },
        "validator": {
            "runtime": {"adapter": "test_validator"},
            "subscribes_to": ["validation.request"],
            "emits": ["validation.result"],
        },
        "data_reducer": {
            "runtime": {"adapter": "test_reducer"},
            "subscribes_to": ["data.reduce"],
            "emits": ["data.reduced"],
        },
        "fallback_planner": {
            "runtime": {"adapter": "test_fallback_planner"},
            "subscribes_to": ["planner.replan.request"],
            "emits": ["planner.replan.result"],
        },
        "recorder": {
            "runtime": {"adapter": "test_recorder"},
            "subscribes_to": ["validation.progress", "validation.request", "workflow.result", "planner.replan.request", "clarification.required", "data.reduce"],
            "emits": [],
        },
    },
}


class EngineValidationTests(unittest.TestCase):
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

    def test_task_plan_family_target_resolves_to_matching_sql_instance(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"] = {
            "sql_runner_mydb": {
                "runtime": {"adapter": "test_exec"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
                "metadata": {
                    "template_agent": "sql_runner",
                    "argument_name": "mydb",
                    "database_name": "mydb",
                    "database_aliases": ["mydb"],
                    "routing_priority": 10,
                },
            },
            "sql_runner_dicom_mock": {
                "runtime": {"adapter": "test_exec"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
                "metadata": {
                    "template_agent": "sql_runner",
                    "argument_name": "dicom_mock",
                    "database_name": "dicom_mock",
                    "database_aliases": ["dicom_mock", "dicom mock"],
                    "routing_priority": 50,
                },
            },
        }
        engine = Engine(spec)
        engine.setup()

        dicom_subscribers = engine._select_subscribers(
            "task.plan",
            {
                "target_agent": "sql_runner",
                "task": "list the tables in the dicom schema alphabetically",
                "original_task": "In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count.",
                "instruction": {"operation": "query_from_request", "question": "In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count."},
            },
        )
        mydb_subscribers = engine._select_subscribers(
            "task.plan",
            {
                "target_agent": "sql_runner",
                "task": "how many patients have more than 2 studies",
                "original_task": "In the mydb database, how many patients have more than 2 studies?",
                "instruction": {"operation": "query_from_request", "question": "In the mydb database, how many patients have more than 2 studies?"},
            },
        )

        self.assertEqual(dicom_subscribers, ["sql_runner_dicom_mock"])
        self.assertEqual(mydb_subscribers, ["sql_runner_mydb"])

    def test_build_agent_catalog_preserves_runtime_agent_name_when_metadata_has_template_name(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"] = {
            "sql_runner_mydb": {
                "runtime": {"adapter": "test_exec"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
                "metadata": {
                    "name": "sql_runner",
                    "template_agent": "sql_runner",
                    "database_name": "mydb",
                },
            },
        }
        engine = Engine(spec)

        catalog = engine._build_agent_catalog()

        self.assertEqual(catalog[0]["name"], "sql_runner_mydb")
        self.assertEqual(catalog[0]["template_agent"], "sql_runner")
        self.assertEqual(catalog[0]["database_name"], "mydb")

    def test_compact_sql_schema_table_result_preserves_small_table_inventory(self):
        engine = Engine(copy.deepcopy(TEST_SPEC))
        payload = {
            "detail": "Database tables listed.",
            "result": {
                "columns": ["schema", "table", "type"],
                "rows": [
                    {"schema": "dicom", "table": "dicom_tags", "type": "table"},
                    {"schema": "dicom", "table": "instances", "type": "table"},
                    {"schema": "dicom", "table": "patients", "type": "table"},
                    {"schema": "dicom", "table": "rtdose", "type": "table"},
                    {"schema": "dicom", "table": "rtplan", "type": "table"},
                    {"schema": "dicom", "table": "rtstruct", "type": "table"},
                    {"schema": "dicom", "table": "series", "type": "table"},
                    {"schema": "dicom", "table": "studies", "type": "table"},
                ],
                "row_count": 8,
            },
        }

        compact = engine._compact_event_payload("sql.result", payload)

        self.assertEqual(len(compact["result"]["rows"]), 8)
        self.assertNotIn("rows_note", compact["result"])

    def test_task_plan_family_target_resolves_to_matching_slurm_instance(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"] = {
            "slurm_runner_cluster": {
                "runtime": {"adapter": "test_exec"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
                "metadata": {
                    "template_agent": "slurm_runner",
                    "argument_name": "cluster",
                    "cluster_name": "default",
                    "cluster_aliases": ["slurm", "cluster", "hpc"],
                    "routing_priority": 25,
                },
            },
        }
        engine = Engine(spec)
        engine.setup()

        subscribers = engine._select_subscribers(
            "task.plan",
            {
                "target_agent": "slurm_runner",
                "task": "how many pending jobs does vinith have",
                "original_task": "In the Slurm cluster, how many pending jobs does vinith have, and list the job IDs.",
                "instruction": {"operation": "query_from_request", "question": "In the Slurm cluster, how many pending jobs does vinith have, and list the job IDs."},
            },
        )

        self.assertEqual(subscribers, ["slurm_runner_cluster"])

    def test_execute_single_workflow_step_records_resolved_concrete_target_agent(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"] = {
            "sql_runner_mydb": {
                "runtime": {"adapter": "test_exec"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
                "metadata": {
                    "template_agent": "sql_runner",
                    "argument_name": "mydb",
                    "database_name": "mydb",
                    "database_aliases": ["mydb"],
                    "routing_priority": 10,
                },
            },
            "sql_runner_dicom_mock": {
                "runtime": {"adapter": "test_exec"},
                "subscribes_to": ["task.plan"],
                "emits": ["task.result"],
                "metadata": {
                    "template_agent": "sql_runner",
                    "argument_name": "dicom_mock",
                    "database_name": "dicom_mock",
                    "database_aliases": ["dicom_mock", "dicom mock"],
                    "routing_priority": 50,
                },
            },
        }
        engine = Engine(spec)
        engine.setup()

        outcome = engine._execute_single_workflow_step(
            {
                "id": "step1",
                "target_agent": "sql_runner",
                "task": "list the tables in the dicom schema alphabetically",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count.",
                },
            },
            {
                "task": "In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count.",
                "task_shape": "schema_summary",
                "run_id": "run-test",
                "attempt": 1,
            },
            {"original_task": "In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count."},
            0,
        )

        self.assertEqual(outcome.get("target_agent"), "sql_runner_dicom_mock")
        self.assertEqual(outcome.get("step_payload", {}).get("target_agent"), "sql_runner_dicom_mock")

    def test_structured_step_result_exposes_shell_stdout_alias_for_followup_references(self):
        engine = Engine(TEST_SPEC)
        context = {"__step_results__": [], "__step_results_by_id__": {}}
        envelope = engine._structured_step_result(
            "step1",
            {"task": "show branch", "target_agent": "shell_runner"},
            "shell.result",
            {"command": "git branch --show-current", "stdout": "main\n", "stderr": "", "returncode": 0},
            "main",
            "completed",
            1.0,
        )
        engine._record_step_result(context, envelope)
        self.assertEqual(engine._resolve_reference_path("step1.stdout", context), "main")

    def test_structured_step_result_exposes_rows_alias_for_followup_references(self):
        engine = Engine(TEST_SPEC)
        context = {"__step_results__": [], "__step_results_by_id__": {}}
        envelope = engine._structured_step_result(
            "step1",
            {"task": "query rows", "target_agent": "sql_runner_mydb"},
            "task.result",
            {"detail": "Rows ready.", "result": {"rows": [{"name": "alice"}], "columns": ["name"], "row_count": 1}},
            {"rows": [{"name": "alice"}]},
            "completed",
            1.0,
        )
        engine._record_step_result(context, envelope)
        self.assertEqual(engine._resolve_reference_path("step1.rows.0.name", context), "alice")

    def test_recorded_context_aliases_preserve_full_rows_for_followup_references(self):
        engine = Engine(TEST_SPEC)
        rows = [{"schema": "flathr", "table": f"table_{index}", "type": "table"} for index in range(6)]
        payload = {
            "detail": "Database tables listed.",
            "sql": "",
            "result": {
                "rows": rows,
                "columns": ["schema", "table", "type"],
                "row_count": len(rows),
                "limit": len(rows),
            },
        }
        context = {"__step_results__": [], "__step_results_by_id__": {}}

        engine._record_context_value(context, "step1", "sql.result", payload, payload)
        envelope = engine._structured_step_result(
            "step1",
            {"task": "list tables", "target_agent": "sql_runner_mydb"},
            "sql.result",
            payload,
            payload,
            "completed",
            1.0,
        )
        engine._record_step_result(context, envelope)

        self.assertEqual(len(context["__step_results_by_id__"]["step1"]["rows"]), 5)
        self.assertEqual(len(engine._resolve_reference_path("step1.rows", context)), 6)
        self.assertEqual(engine._resolve_reference_path("step1.rows.5.table", context), "table_5")
        self.assertEqual(engine._resolve_reference_path("step1.result.rows.5.table", context), "table_5")

    def test_compact_event_payload_preserves_shell_output_for_workflow_grounding(self):
        engine = Engine(TEST_SPEC)
        compact = engine._compact_event_payload(
            "shell.result",
            {
                "command": "docker ps -a",
                "stdout": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db",
                "stderr": "",
                "returncode": 0,
            },
        )
        self.assertEqual(compact["command"], "docker ps -a")
        self.assertIn("postgres_db", compact["stdout"])
        self.assertIn("postgres_db", compact["stdout_excerpt"])

    def test_extract_result_value_uses_shell_detail_when_stdout_capture_is_empty(self):
        engine = Engine(TEST_SPEC)
        value = engine._extract_result_value(
            "shell.result",
            {
                "command": "conda env remove -n vinith -y",
                "stdout": "",
                "stderr": "EnvironmentLocationNotFound: Not a conda environment: /home/vinith/miniconda3/envs/vinith",
                "returncode": 0,
                "detail": "Conda environment `vinith` was already absent.",
                "normalized_result": "Conda environment `vinith` was already absent.",
            },
            {
                "instruction": {
                    "capture": {"mode": "stdout_stripped"},
                }
            },
        )
        self.assertEqual(value, "Conda environment `vinith` was already absent.")

    def test_infer_task_shape_treats_confirm_removed_as_boolean_check(self):
        engine = Engine(TEST_SPEC)
        shape = engine._infer_task_shape(
            "confirm the conda environment named vinith was removed",
            "list",
            "shell.result",
        )
        self.assertEqual(shape, "boolean_check")

    def test_boolean_check_accepts_structured_exists_payload(self):
        engine = Engine(TEST_SPEC)
        assessment = engine._assess_task_shape_result(
            "boolean_check",
            {"exists": False, "name": "vinith"},
            [],
        )
        self.assertEqual(assessment["verdict"], "valid")

    def test_engine_derives_fallback_only_after_validator_rejects_primary_attempt(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "produce the final answer",
                "task_shape": "lookup",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "primary attempt",
                        "instruction": {"operation": "run_command", "command": "primary"},
                    }
                ]
            },
        )
        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["task_shape"], "lookup")
        self.assertEqual(workflow["selected_option"]["id"], "option2")
        self.assertEqual(len(workflow["attempts"]), 2)
        self.assertTrue(workflow["validation"]["valid"])
        self.assertEqual(workflow["attempts"][0]["routing"]["action"], "replan_workflow")
        self.assertEqual(workflow["attempts"][0]["replan"]["derived_option_id"], "option2")
        self.assertEqual(workflow["attempts"][1]["option"]["derived_from_attempt"], 1)
        self.assertEqual(workflow["attempts"][1]["routing"]["action"], "accept_attempt")
        self.assertEqual(workflow["graph"]["kind"], "workflow_execution")
        self.assertEqual(workflow["graph"]["selected_option_id"], "option2")
        graph_node_ids = {item["node_id"] for item in workflow["graph"]["nodes"]}
        self.assertTrue(any(item.endswith(":attempt:1:step:step1") for item in graph_node_ids))
        self.assertTrue(any(item.endswith(":attempt:1:validator") for item in graph_node_ids))
        self.assertTrue(any(item.endswith(":attempt:2:validator") for item in graph_node_ids))
        self.assertTrue(any(item.endswith(":attempt:1:validator:router") for item in graph_node_ids))
        self.assertTrue(any(item.endswith(":attempt:1:validator:router:replan") for item in graph_node_ids))
        self.assertTrue(any(item.endswith(":attempt:2:validator:router") for item in graph_node_ids))
        graph_edges = {(item["source"], item["target"], item["relation"]) for item in workflow["graph"]["edges"]}
        root_node_id = workflow["graph"]["root_node_id"]
        self.assertIn(
            (f"{root_node_id}:attempt:1:validator:router:replan", f"{root_node_id}:attempt:2", "activates"),
            graph_edges,
        )

        validation_progress = [payload for event_name, payload in _RecorderAdapter.events if event_name == "validation.progress"]
        stages = [item["stage"] for item in validation_progress]
        self.assertIn("retrying", stages)
        self.assertIn("passed", stages)

        validation_requests = [
            payload
            for event_name, payload in _RecorderAdapter.events
            if event_name == "validation.request" and payload.get("validation_scope") != "step"
        ]
        self.assertEqual(validation_requests[0]["task_shape"], "lookup")
        self.assertEqual(
            validation_requests[0]["steps"],
            [{"result": "partial answer", "status": "completed"}],
        )
        self.assertEqual(validation_requests[0]["node"]["agent"], "validator")
        self.assertEqual(validation_requests[0]["node"]["role"], "validator")
        self.assertEqual(validation_requests[0]["node"]["scope"], "workflow")
        self.assertEqual(validation_requests[0]["node"]["status"], "pending")
        replan_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "planner.replan.request"]
        self.assertEqual(len(replan_requests), 1)
        self.assertEqual(replan_requests[0]["step_id"], "__workflow__")

    def test_engine_uses_next_existing_option_without_workflow_replan(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "produce the final answer",
                "task_shape": "lookup",
                "steps": [],
                "plan_options": [
                    {
                        "id": "option1",
                        "label": "Primary plan",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "executor",
                                "task": "primary attempt",
                                "instruction": {"operation": "run_command", "command": "primary"},
                            }
                        ],
                    },
                    {
                        "id": "option2",
                        "label": "Fallback plan",
                        "steps": [
                            {
                                "id": "step1",
                                "target_agent": "executor",
                                "task": "fallback attempt",
                                "instruction": {"operation": "run_command", "command": "fallback"},
                            }
                        ],
                    },
                ],
            },
        )

        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        self.assertEqual(workflow["selected_option"]["id"], "option2")
        self.assertEqual(workflow["attempts"][0]["routing"]["action"], "try_next_option")
        self.assertEqual(workflow["attempts"][1]["routing"]["action"], "accept_attempt")
        replan_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "planner.replan.request"]
        self.assertEqual(replan_requests, [])
        graph_edges = {(item["source"], item["target"], item["relation"]) for item in workflow["graph"]["edges"]}
        root_node_id = workflow["graph"]["root_node_id"]
        self.assertIn((f"{root_node_id}:attempt:1", f"{root_node_id}:attempt:2", "next_attempt"), graph_edges)

    def test_engine_escalates_to_clarification_after_uncertainty_threshold(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "uncertain goal",
                "task_shape": "lookup",
                "retry_budget": {
                    "max_attempts": 2,
                    "max_replans": 1,
                    "max_uncertain_attempts": 0,
                    "max_validation_llm_calls": 0,
                },
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "uncertain attempt",
                        "instruction": {"operation": "run_command", "command": "ambiguous"},
                    }
                ],
            },
        )
        clarification_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "clarification.required"]
        self.assertEqual(len(clarification_events), 1)
        clarification = clarification_events[0]
        self.assertEqual(clarification["task"], "uncertain goal")
        self.assertIn("narrower target", clarification.get("question", ""))
        self.assertEqual(clarification["attempts"][0]["routing"]["action"], "clarify")
        self.assertEqual(clarification["graph"]["status"], "needs_clarification")
        self.assertEqual(clarification["graph"]["statistics"]["clarification_count"], 1)
        clarification_node_ids = {item["node_id"] for item in clarification["graph"]["nodes"]}
        self.assertTrue(any(item.endswith(":validator:router:clarification") for item in clarification_node_ids))
        replan_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "planner.replan.request" and payload.get("task") == "uncertain goal"]
        self.assertEqual(len(replan_requests), 0)

    def test_engine_does_not_forward_internal_replan_task_result_side_channel(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"]["fallback_planner"]["runtime"]["adapter"] = "test_no_recovery_planner"
        spec["agents"]["recorder"]["subscribes_to"].append("task.result")
        engine = Engine(spec)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "produce the final answer",
                "task_shape": "lookup",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "primary attempt",
                        "instruction": {"operation": "run_command", "command": "primary"},
                    }
                ],
            },
        )
        recorded_task_results = [payload for event_name, payload in _RecorderAdapter.events if event_name == "task.result"]
        self.assertEqual(recorded_task_results, [])
        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        self.assertEqual(workflow_events[0]["status"], "failed")

    def test_engine_replans_step_when_completed_output_diverges_from_step_intent(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "show cluster node inventory",
                "task_shape": "lookup",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "bad count step",
                        "instruction": {"operation": "run_command", "command": "bad"},
                    }
                ],
            },
        )
        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        self.assertEqual(workflow["status"], "completed")
        self.assertEqual(workflow["result"], "Total nodes: 3\nState idle: 2\nState mixed: 1")
        self.assertEqual(workflow["steps"][0]["id"], "step1")
        self.assertEqual(workflow["steps"][0]["steps"][0]["task"], "fixed count and state step")
        self.assertEqual(workflow["steps"][0]["routing"]["action"], "replan_step")
        self.assertEqual(workflow["steps"][0]["replan"]["replace_step_id"], "step1")
        self.assertEqual(workflow["steps"][0]["steps"][0]["routing"]["action"], "accept_step")

        replan_requests = [
            payload
            for event_name, payload in _RecorderAdapter.events
            if event_name == "planner.replan.request" and payload.get("step_id") == "step1"
        ]
        self.assertEqual(len(replan_requests), 1)
        self.assertEqual(replan_requests[0]["failure_class"], "validation_failed")

        step_validation_requests = [
            payload
            for event_name, payload in _RecorderAdapter.events
            if event_name == "validation.request" and payload.get("validation_scope") == "step"
        ]
        self.assertEqual(len(step_validation_requests), 2)
        self.assertEqual(step_validation_requests[0]["step_task"], "bad count step")
        self.assertEqual(step_validation_requests[1]["step_task"], "fixed count and state step")
        self.assertEqual(step_validation_requests[0]["node"]["agent"], "validator")
        self.assertEqual(step_validation_requests[0]["node"]["role"], "validator")
        self.assertEqual(step_validation_requests[0]["node"]["step_id"], "step1")
        graph_node_ids = {item["node_id"] for item in workflow["graph"]["nodes"]}
        self.assertTrue(any(item.endswith(":step:step1:router") for item in graph_node_ids))
        self.assertTrue(any(item.endswith(":step:step1:router:replan") for item in graph_node_ids))

    def test_engine_routes_step_output_through_data_reducer_before_validation(self):
        engine = Engine(TEST_SPEC)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "count the lines in the shell output",
                "task_shape": "lookup",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "executor",
                        "task": "reduce raw shell output",
                        "instruction": {"operation": "run_command", "command": "reduce"},
                    }
                ],
            },
        )

        reduction_requests = [payload for event_name, payload in _RecorderAdapter.events if event_name == "data.reduce"]
        self.assertEqual(len(reduction_requests), 1)
        self.assertEqual(reduction_requests[0]["step_id"], "step1")
        self.assertEqual(reduction_requests[0]["source_event"], "shell.result")
        self.assertEqual(reduction_requests[0]["input_data"], "alpha\nbeta\n")
        self.assertEqual(reduction_requests[0]["reduction_request"]["kind"], "shell.local_reducer")
        self.assertEqual(reduction_requests[0]["node"]["agent"], "data_reducer")
        self.assertEqual(reduction_requests[0]["node"]["role"], "reducer")
        self.assertEqual(reduction_requests[0]["node"]["step_id"], "step1")

        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        step_payload = workflow["steps"][0]["payload"]
        self.assertEqual(step_payload["reduced_result"], "Reduced lines: 2")
        self.assertEqual(workflow["steps"][0]["routing"]["action"], "accept_step")
        self.assertEqual(workflow["graph"]["statistics"]["reducer_count"], 1)
        self.assertEqual(workflow["graph"]["statistics"]["router_count"], 2)
        self.assertTrue(any(item["node_id"].endswith(":reducer") for item in workflow["graph"]["nodes"]))

    def test_engine_fails_fast_on_non_retriable_sql_agent_configuration_error(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"]["sql_runner_dicom_mock"] = {
            "runtime": {"adapter": "test_exec"},
            "subscribes_to": ["task.plan"],
            "emits": ["task.result"],
            "metadata": {
                "template_agent": "sql_runner",
                "argument_name": "dicom_mock",
                "database_name": "dicom_mock",
            },
        }
        engine = Engine(spec)
        engine.setup()
        engine.emit(
            "task.plan",
            {
                "task": "count patients in dicom_mock",
                "task_shape": "count",
                "steps": [
                    {
                        "id": "step1",
                        "target_agent": "sql_runner_dicom_mock",
                        "task": "sql config failure step",
                        "instruction": {"operation": "query_from_request", "question": "count patients in dicom_mock"},
                    }
                ],
            },
        )

        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        workflow = workflow_events[0]
        self.assertEqual(workflow["status"], "failed")
        self.assertEqual(workflow["steps"][0]["routing"]["action"], "fail_run")
        self.assertEqual(workflow["steps"][0]["routing"]["failure_class"], "agent_configuration_failed")

        replan_requests = [
            payload
            for event_name, payload in _RecorderAdapter.events
            if event_name == "planner.replan.request" and payload.get("step_id") == "step1"
        ]
        self.assertEqual(replan_requests, [])

    def test_autostart_rejects_unhealthy_existing_sql_agent(self):
        spec = {
            "contracts": {},
            "events": {},
            "agents": {
                "sql_runner_dicom_mock": {
                    "runtime": {
                        "adapter": "http",
                        "endpoint": "http://127.0.0.1:8308/handle",
                        "autostart": {"app": "agent_library.agents.sql_runner:app"},
                    },
                    "metadata": {"template_agent": "sql_runner"},
                }
            },
        }
        engine = Engine(spec)
        with patch.object(engine, "_is_port_open", return_value=True), patch.object(
            engine,
            "_probe_existing_http_agent",
            return_value={"healthy": False, "detail": "Unsupported SQL_AGENT_DSN scheme: unknown"},
        ):
            with self.assertRaisesRegex(RuntimeError, "unhealthy configuration"):
                engine._autostart_http_services()

    def test_engine_replays_terminal_run_from_persisted_state(self):
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
        original = workflow_events[0]
        run_id = original["run_id"]
        state_path = engine.run_store.state_path(run_id)
        self.assertTrue(state_path.exists())

        with state_path.open("r", encoding="utf-8") as handle:
            persisted = json.load(handle)
        self.assertEqual(persisted["terminal_event"], "workflow.result")
        self.assertEqual(persisted["status"], "completed")

        _RecorderAdapter.events = []
        replay_engine = Engine(TEST_SPEC)
        replay_engine.setup()
        replay_engine.replay_run(run_id)

        replayed_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(replayed_events), 1)
        replayed = replayed_events[0]
        self.assertEqual(replayed["run_id"], run_id)
        self.assertEqual(replayed["result"], "final answer")
        self.assertEqual(replayed["persistence"]["status"], "completed")
        self.assertEqual(replayed["persistence"]["state_path"], str(state_path))

    def test_engine_resumes_run_after_persisted_step_crash(self):
        spec = copy.deepcopy(TEST_SPEC)
        spec["agents"]["executor"]["runtime"]["adapter"] = "test_crash_exec"
        engine = Engine(spec)
        engine.setup()
        run_id = "resume_after_crash_case"

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

        state_path = engine.run_store.state_path(run_id)
        timeline_path = engine.run_store.timeline_path(run_id)
        self.assertTrue(state_path.exists())
        self.assertTrue(timeline_path.exists())

        with state_path.open("r", encoding="utf-8") as handle:
            persisted = json.load(handle)
        self.assertEqual(persisted["status"], "running")
        self.assertEqual(persisted["current_attempt_index"], 1)
        self.assertEqual(persisted["attempts"][0]["workflow_state"]["inflight_step"]["task"], "resume after crash step")

        _RecorderAdapter.events = []
        resume_engine = Engine(spec)
        resume_engine.setup()
        resume_engine.resume_run(run_id)

        workflow_events = [payload for event_name, payload in _RecorderAdapter.events if event_name == "workflow.result"]
        self.assertEqual(len(workflow_events), 1)
        resumed = workflow_events[0]
        self.assertEqual(resumed["run_id"], run_id)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["result"], "resumed answer")
        self.assertEqual(resumed["persistence"]["status"], "completed")
        self.assertFalse(resumed["persistence"]["resumable"])
        self.assertEqual(resumed["persistence"]["state_path"], str(state_path))

        with state_path.open("r", encoding="utf-8") as handle:
            completed_state = json.load(handle)
        self.assertEqual(completed_state["terminal_event"], "workflow.result")
        self.assertEqual(completed_state["status"], "completed")

        with timeline_path.open("r", encoding="utf-8") as handle:
            stages = [json.loads(line)["stage"] for line in handle if line.strip()]
        self.assertIn("attempt_checkpoint", stages)
        self.assertIn("run_completed", stages)


if __name__ == "__main__":
    unittest.main()
