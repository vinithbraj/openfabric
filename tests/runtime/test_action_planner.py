from __future__ import annotations

import json
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, PlannerConfig, StepLog
from aor_runtime.runtime.action_planner import ActionPlan, ActionPlanCanonicalizer, ActionPlanValidator, LLMActionPlanner
from aor_runtime.runtime.error_normalization import normalize_planner_error
from aor_runtime.runtime.output_shape import scalar_field_for_tool
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.runtime.result_shape import validate_result_shape
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.runtime.sql_catalog import SqlColumnRef, SqlSchemaCatalog, SqlTableRef
from aor_runtime.tools.factory import build_tool_registry
from aor_runtime.tools.text_format import format_data


class FakeLLM:
    def __init__(self, responses: list[str | dict]) -> None:
        self.responses = [json.dumps(item) if isinstance(item, dict) else item for item in responses]
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def load_prompt(self, path: str | None, fallback: str) -> str:
        return fallback

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if not self.responses:
            raise AssertionError("LLM called more times than expected")
        return self.responses.pop(0)


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://example/db"},
        sql_default_database="dicom",
        **overrides,
    )


def _catalog() -> SqlSchemaCatalog:
    return SqlSchemaCatalog(
        database="dicom",
        dialect="postgresql",
        tables=[
            SqlTableRef(
                schema_name="flathr",
                table_name="Patient",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientID", primary_key=True),
                    SqlColumnRef(schema_name="flathr", table_name="Patient", column_name="PatientBirthDate"),
                ],
                primary_key_columns=["PatientID"],
            ),
            SqlTableRef(
                schema_name="flathr",
                table_name="Study",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyInstanceUID", primary_key=True),
                    SqlColumnRef(
                        schema_name="flathr",
                        table_name="Study",
                        column_name="PatientID",
                        foreign_key="flathr.Patient.PatientID",
                    ),
                ],
                primary_key_columns=["StudyInstanceUID"],
                foreign_keys=[
                    {
                        "constrained_columns": ["PatientID"],
                        "referred_schema": "flathr",
                        "referred_table": "Patient",
                        "referred_columns": ["PatientID"],
                    }
                ],
            ),
        ],
    )


def _patch_catalog(monkeypatch) -> None:
    catalog = _catalog()
    monkeypatch.setattr("aor_runtime.runtime.action_planner.get_all_sql_catalogs", lambda settings: [catalog])
    monkeypatch.setattr("aor_runtime.runtime.action_planner.get_sql_catalog", lambda settings, database=None: catalog)


def _dicom_relationship_catalog() -> SqlSchemaCatalog:
    return SqlSchemaCatalog(
        database="dicom",
        dialect="postgresql",
        tables=[
            SqlTableRef(
                schema_name="flathr",
                table_name="Study",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyInstanceUID", primary_key=True),
                    SqlColumnRef(schema_name="flathr", table_name="Study", column_name="StudyDescription"),
                ],
                primary_key_columns=["StudyInstanceUID"],
            ),
            SqlTableRef(
                schema_name="flathr",
                table_name="Series",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Series", column_name="SeriesInstanceUID", primary_key=True),
                    SqlColumnRef(schema_name="flathr", table_name="Series", column_name="StudyInstanceUID"),
                    SqlColumnRef(schema_name="flathr", table_name="Series", column_name="Modality"),
                    SqlColumnRef(schema_name="flathr", table_name="Series", column_name="SeriesDescription"),
                ],
                primary_key_columns=["SeriesInstanceUID"],
            ),
            SqlTableRef(
                schema_name="flathr",
                table_name="Instance",
                columns=[
                    SqlColumnRef(schema_name="flathr", table_name="Instance", column_name="SOPInstanceUID", primary_key=True),
                    SqlColumnRef(schema_name="flathr", table_name="Instance", column_name="SeriesInstanceUID"),
                ],
                primary_key_columns=["SOPInstanceUID"],
            ),
        ],
    )


def _patch_relationship_catalog(monkeypatch) -> None:
    catalog = _dicom_relationship_catalog()
    monkeypatch.setattr("aor_runtime.runtime.action_planner.get_all_sql_catalogs", lambda settings: [catalog])
    monkeypatch.setattr("aor_runtime.runtime.action_planner.get_sql_catalog", lambda settings, database=None: catalog)


def test_shell_scalar_field_uses_stdout_for_count_goals() -> None:
    assert scalar_field_for_tool("shell.exec", goal="How many processes are running on this machine?") == "stdout"
    assert scalar_field_for_tool("shell.exec", goal="What exit code did the command return?") == "returncode"


def test_shell_count_plan_cannot_return_command_status(tmp_path: Path) -> None:
    plan = ActionPlan.model_validate(
        {
            "goal": "How many processes are running on this machine?",
            "actions": [
                {
                    "id": "process_count",
                    "tool": "shell.exec",
                    "purpose": "Count processes.",
                    "inputs": {"command": "ps -eo pid= | wc -l"},
                    "output_binding": "process_count",
                    "expected_result_shape": {"kind": "scalar"},
                },
                {
                    "id": "return_result",
                    "tool": "runtime.return",
                    "purpose": "Return process count.",
                    "inputs": {"value": "$process_count.exit_code", "mode": "count"},
                    "depends_on": ["process_count"],
                    "output_binding": "runtime_return_result",
                    "expected_result_shape": {"kind": "scalar"},
                },
            ],
            "expected_final_shape": {"kind": "scalar"},
        }
    )

    validator = ActionPlanValidator(settings=_settings(tmp_path), tools=build_tool_registry(_settings(tmp_path)), allowed_tools=["shell.exec"])
    result = validator.validate(plan, goal="How many processes are running on this machine?")

    assert not result.valid
    assert any("stdout" in error and "exit status" in error for error in result.errors)


def test_shell_count_canonicalizer_returns_stdout() -> None:
    plan = ActionPlan.model_validate(
        {
            "goal": "How many processes are running on this machine?",
            "actions": [
                {
                    "id": "process_count",
                    "tool": "shell.exec",
                    "purpose": "Count processes.",
                    "inputs": {"command": "ps -eo pid= | wc -l"},
                    "output_binding": "process_count",
                    "expected_result_shape": {"kind": "scalar"},
                }
            ],
            "expected_final_shape": {"kind": "scalar"},
        }
    )

    result = ActionPlanCanonicalizer(goal="How many processes are running on this machine?").canonicalize(plan)
    final = result.plan.actions[-1]

    assert final.tool == "runtime.return"
    assert final.inputs["value"] == "$process_count.stdout"
    assert final.inputs["mode"] == "count"


def test_bare_sql_alias_canonicalizes_to_rows() -> None:
    plan = ActionPlan.model_validate(
        {
            "goal": "Show patient study counts.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "purpose": "Get patient study counts.",
                    "inputs": {"database": "dicom", "query": "SELECT 1 AS study_count"},
                    "output_binding": "patient_study_counts",
                    "expected_result_shape": {"kind": "table"},
                },
                {
                    "id": "format",
                    "tool": "text.format",
                    "purpose": "Format counts.",
                    "inputs": {"source": "patient_study_counts", "format": "markdown"},
                    "depends_on": ["query"],
                    "output_binding": "formatted_output",
                },
            ],
            "expected_final_shape": {"kind": "table"},
        }
    )

    result = ActionPlanCanonicalizer(goal="Show patient study counts.").canonicalize(plan)
    formatter = next(action for action in result.plan.actions if action.tool == "text.format")

    assert formatter.inputs["source"] == {"$ref": "patient_study_counts", "path": "rows"}


def test_invalid_dataflow_path_fails_before_execution(tmp_path: Path) -> None:
    plan = ActionPlan.model_validate(
        {
            "goal": "Show patient study counts.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "purpose": "Get patient study counts.",
                    "inputs": {"database": "dicom", "query": "SELECT 1 AS study_count"},
                    "output_binding": "patient_study_counts",
                    "expected_result_shape": {"kind": "table"},
                },
                {
                    "id": "format",
                    "tool": "text.format",
                    "purpose": "Format counts.",
                    "inputs": {"source": "$patient_study_counts.patient_study_counts", "format": "markdown"},
                    "depends_on": ["query"],
                    "output_binding": "formatted_output",
                },
            ],
            "expected_final_shape": {"kind": "table"},
        }
    )
    validator = ActionPlanValidator(settings=_settings(tmp_path), tools=build_tool_registry(_settings(tmp_path)), allowed_tools=["sql.query"])

    result = validator.validate(plan, goal="Show patient study counts.")

    assert not result.valid
    assert any("Invalid reference path" in error and "Suggested path: rows" in error for error in result.errors)


def test_task_planner_uses_action_planner_for_sql_count(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "Count patients in dicom.",
                "actions": [
                    {
                        "id": "query",
                        "tool": "sql.query",
                        "purpose": "Count patient rows.",
                        "inputs": {
                            "database": "dicom",
                            "query": 'SELECT COUNT(DISTINCT PatientID) AS patient_count FROM flathr."Patient"',
                        },
                        "output_binding": "patient_count_rows",
                        "expected_result_shape": {"kind": "scalar"},
                    },
                    {
                        "id": "format",
                        "tool": "text.format",
                        "purpose": "Format scalar count.",
                        "inputs": {"source": "$query.rows", "format": "txt"},
                        "depends_on": ["query"],
                        "output_binding": "formatted_count",
                    },
                    {
                        "id": "return",
                        "tool": "runtime.return",
                        "purpose": "Return count.",
                        "inputs": {"value": "$format.content", "mode": "text"},
                        "depends_on": ["format"],
                        "output_binding": "final",
                    },
                ],
                "expected_final_shape": {"kind": "scalar"},
                "notes": [],
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="count the number of patients in dicom",
        planner=PlannerConfig(),
        allowed_tools=["sql.query", "fs.read", "fs.write", "shell.exec", "python.exec"],
        input_payload={"task": "count the number of patients in dicom"},
    )

    assert planner.last_planning_mode == "validator_enforced_action_planner"
    assert [step.action for step in plan.steps] == ["sql.query", "text.format", "runtime.return"]
    assert plan.steps[0].args["query"] == 'SELECT COUNT(DISTINCT "PatientID") AS patient_count FROM flathr."Patient"'
    context = json.loads(llm.user_prompts[0])
    assert context["sql_schema"]["databases"][0]["tables"][0]["columns"][0]["name"] == "PatientID"
    assert "quote mixed-case identifiers" in json.dumps(context["sql_schema"])


def test_action_planner_normalizes_count_shape_and_completes_sql_plan(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "number of patients in dicom",
                "actions": [
                    {
                        "id": "query_patients",
                        "tool": "sql.query",
                        "purpose": "Count patients",
                        "inputs": {
                            "database": "dicom",
                            "query": "SELECT COUNT(DISTINCT PatientID) FROM flathr.Patient;",
                        },
                        "output_binding": "patient_count",
                        "expected_result_shape": {"kind": "scalar"},
                    }
                ],
                "expected_final_shape": {"kind": "count"},
                "notes": [],
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="number of patients in dicom",
        planner=PlannerConfig(),
        allowed_tools=["sql.query"],
        input_payload={"task": "number of patients in dicom"},
    )

    assert [step.action for step in plan.steps] == ["sql.query", "text.format", "runtime.return"]
    assert plan.steps[0].args["query"] == 'SELECT COUNT(DISTINCT "PatientID") FROM flathr."Patient"'
    assert plan.steps[1].args["source"] == {"$ref": "patient_count", "path": "rows"}
    assert planner.last_capability_metadata["normalized_action_plan"]["expected_final_shape"]["kind"] == "scalar"
    assert "Inserted text.format" in " ".join(planner.last_capability_metadata["canonicalization_repairs"])


def test_action_planner_normalizes_table_shape_synonyms(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "list patients",
                "actions": [
                    {
                        "id": "query",
                        "tool": "sql.query",
                        "inputs": {"database": "dicom", "query": 'SELECT "PatientID" FROM flathr."Patient"'},
                        "output_binding": "rows",
                        "expected_result_shape": {"kind": "records"},
                    }
                ],
                "expected_final_shape": "rows",
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(goal="list patients in dicom", planner=PlannerConfig(), allowed_tools=["sql.query"], input_payload={})

    assert [step.action for step in plan.steps] == ["sql.query", "text.format", "runtime.return"]
    assert planner.last_normalized_action_plan is not None
    assert planner.last_normalized_action_plan["actions"][0]["expected_result_shape"]["kind"] == "table"
    assert planner.last_normalized_action_plan["expected_final_shape"]["kind"] == "table"


def test_action_planner_rejects_raw_execution_plan_shape(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(
        llm=FakeLLM([{"steps": [{"id": 1, "action": "sql.query", "args": {"query": "SELECT 1"}}]}]),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    with pytest.raises(ValueError, match="ExecutionPlan shape"):
        planner.build_plan(goal="count patients", planner=PlannerConfig(), allowed_tools=["sql.query"], input_payload={})


def test_action_planner_completes_shell_plan_with_formatter_and_return(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "run pwd",
                    "actions": [
                        {
                            "id": "shell",
                            "tool": "shell.exec",
                            "inputs": {"command": "pwd"},
                            "output_binding": "shell_output",
                            "expected_result_shape": {"kind": "text"},
                        }
                    ],
                    "expected_final_shape": {"kind": "text"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(goal="run pwd", planner=PlannerConfig(), allowed_tools=["shell.exec"], input_payload={})

    assert [step.action for step in plan.steps] == ["shell.exec", "text.format", "runtime.return"]
    assert plan.steps[1].args["source"] == {"$ref": "shell_output", "path": "stdout"}


def test_action_planner_rejects_unrequested_shell_row_limit_for_all_processes(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "list all the cpu processes running in the computer",
                    "actions": [
                        {
                            "id": "processes",
                            "tool": "shell.exec",
                            "inputs": {"command": "ps aux --sort=-%cpu | head -n 20"},
                            "output_binding": "processes",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    with pytest.raises(ValueError, match="unrequested row limit"):
        planner.build_plan(
            goal="list all the cpu processes running in the computer",
            planner=PlannerConfig(),
            allowed_tools=["shell.exec"],
            input_payload={},
        )


def test_action_planner_rejects_unrequested_sql_row_limit_for_all_rows(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "list all patients in dicom",
                    "actions": [
                        {
                            "id": "patients",
                            "tool": "sql.query",
                            "inputs": {
                                "database": "dicom",
                                "query": 'SELECT "PatientID" FROM flathr."Patient" LIMIT 500',
                            },
                            "output_binding": "patients",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    with pytest.raises(ValueError, match="unrequested row limit"):
        planner.build_plan(
            goal="list all patients in dicom",
            planner=PlannerConfig(),
            allowed_tools=["sql.query"],
            input_payload={},
        )


def test_action_planner_allows_requested_top_shell_limit(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "show top 20 cpu processes",
                    "actions": [
                        {
                            "id": "processes",
                            "tool": "shell.exec",
                            "inputs": {"command": "ps aux --sort=-%cpu | head -n 20"},
                            "output_binding": "processes",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(goal="show top 20 cpu processes", planner=PlannerConfig(), allowed_tools=["shell.exec"], input_payload={})

    assert [step.action for step in plan.steps] == ["shell.exec", "text.format", "runtime.return"]
    assert plan.steps[0].args["command"] == "ps aux --sort=-%cpu | head -n 20"
    assert plan.steps[1].args["format"] == "markdown"


def test_action_planner_allows_explicit_shell_command_with_head(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "run ps aux --sort=-%cpu | head -n 20",
                    "actions": [
                        {
                            "id": "processes",
                            "tool": "shell.exec",
                            "inputs": {"command": "ps aux --sort=-%cpu | head -n 20"},
                            "output_binding": "processes",
                            "expected_result_shape": {"kind": "text"},
                        }
                    ],
                    "expected_final_shape": {"kind": "text"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(
        goal="run ps aux --sort=-%cpu | head -n 20",
        planner=PlannerConfig(),
        allowed_tools=["shell.exec"],
        input_payload={},
    )

    assert plan.steps[0].args["command"] == "ps aux --sort=-%cpu | head -n 20"


def test_action_planner_rejects_slurm_queue_for_system_process_goal(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "list all processing running in the system",
                    "actions": [
                        {
                            "id": "jobs",
                            "tool": "slurm.queue",
                            "inputs": {},
                            "output_binding": "jobs",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    with pytest.raises(ValueError, match="System process requests must use shell.exec"):
        planner.build_plan(
            goal="list all processing running in the system",
            planner=PlannerConfig(),
            allowed_tools=["slurm.queue", "shell.exec"],
            input_payload={},
        )


def test_action_planner_rejects_inline_json_return(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["slurm.metrics"])
    plan = ActionPlan.model_validate(
        {
            "goal": "status of slurm cluster",
            "actions": [
                {
                    "id": "metrics",
                    "tool": "slurm.metrics",
                    "inputs": {"metric_group": "cluster_summary"},
                    "output_binding": "metrics",
                },
                {
                    "id": "return",
                    "tool": "runtime.return",
                    "inputs": {"value": "$metrics.payload", "mode": "json"},
                    "depends_on": ["metrics"],
                    "output_binding": "final",
                },
            ],
        }
    )

    result = validator.validate(plan, goal="status of slurm cluster")

    assert result.valid is False
    assert "runtime.return mode=json is not user-facing" in "; ".join(result.errors)


def test_action_planner_completes_slurm_nodes_plan_with_formatter_and_return(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "list all nodes in slurm",
                    "actions": [
                        {
                            "id": "nodes",
                            "tool": "slurm.nodes",
                            "inputs": {},
                            "output_binding": "nodes_result",
                            "expected_result_shape": {"kind": "list"},
                        }
                    ],
                    "expected_final_shape": {"kind": "list"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(goal="list all nodes in slurm", planner=PlannerConfig(), allowed_tools=["slurm.nodes"], input_payload={})

    assert [step.action for step in plan.steps] == ["slurm.nodes", "text.format", "runtime.return"]
    assert plan.steps[1].args["source"] == {"$ref": "nodes_result", "path": "nodes"}


def test_action_planner_completes_slurm_status_plan_with_markdown_formatter(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "status of slurm cluster",
                    "actions": [
                        {
                            "id": "metrics",
                            "tool": "slurm.metrics",
                            "inputs": {"metric_group": "cluster_summary"},
                            "output_binding": "metrics_result",
                            "expected_result_shape": {"kind": "status"},
                        }
                    ],
                    "expected_final_shape": {"kind": "status"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(goal="status of slurm cluster", planner=PlannerConfig(), allowed_tools=["slurm.metrics"], input_payload={})

    assert [step.action for step in plan.steps] == ["slurm.metrics", "text.format", "runtime.return"]
    assert plan.steps[1].args["source"] == {"$ref": "metrics_result", "path": "payload"}
    assert plan.steps[1].args["format"] == "markdown"


def test_action_planner_applies_slurm_completed_state_obligation(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "average runtime in the totalseg partition for completed jobs",
                    "actions": [
                        {
                            "id": "avg_time",
                            "tool": "slurm.accounting_aggregate",
                            "inputs": {"partition": "totalseg", "metric": "average_elapsed"},
                            "output_binding": "avg_time_result",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(
        goal="average runtime in the totalseg partition for completed jobs",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate"],
        input_payload={},
    )

    assert plan.steps[0].action == "slurm.accounting_aggregate"
    assert plan.steps[0].args["state"] == "COMPLETED"
    assert plan.steps[0].args["include_all_states"] is False
    assert plan.steps[0].args["partition"] == "totalseg"
    assert any("state=COMPLETED" in repair for repair in planner.last_canonicalization_repairs)


def test_action_planner_repairs_slurm_count_plan_to_scalar_count_field(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "Give me just the count of pending jobs in the slurm cluster",
                    "actions": [
                        {
                            "id": "query_pending",
                            "tool": "slurm.queue",
                            "inputs": {"state": "pending"},
                            "output_binding": "pending_jobs",
                            "expected_result_shape": {"kind": "table"},
                        },
                        {
                            "id": "format_pending",
                            "tool": "text.format",
                            "inputs": {"source": "pending_jobs", "format": "json"},
                            "depends_on": ["query_pending"],
                            "output_binding": "formatted_jobs",
                        },
                    ],
                    "expected_final_shape": {"kind": "count"},
                }
            ]
        ),
        tools=build_tool_registry(settings),
        settings=settings,
    )

    plan = planner.build_plan(
        goal="Give me just the count of pending jobs in the slurm cluster",
        planner=PlannerConfig(),
        allowed_tools=["slurm.queue"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["slurm.queue", "runtime.return"]
    assert plan.steps[1].args["value"] == {"$ref": "pending_jobs", "path": "count"}
    assert plan.steps[1].args["mode"] == "count"
    assert "Repaired scalar final output" in " ".join(planner.last_canonicalization_repairs)


def test_action_planner_completes_sql_export_plan(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "export patients to patients.csv",
                "actions": [
                    {
                        "id": "query",
                        "tool": "sql.query",
                        "inputs": {"database": "dicom", "query": 'SELECT "PatientID" FROM flathr."Patient"'},
                        "output_binding": "rows",
                    }
                ],
                "expected_final_shape": {"kind": "file"},
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="export patients to patients.csv",
        planner=PlannerConfig(),
        allowed_tools=["sql.query", "fs.write"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["sql.query", "text.format", "fs.write", "runtime.return"]
    assert plan.steps[1].args["format"] == "csv"
    assert plan.steps[2].args["path"] == "patients.csv"
    assert plan.steps[2].args["content"] == {"$ref": "query_formatted", "path": "content"}


def test_action_planner_rejects_unknown_tool(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "bad",
            "actions": [{"id": "x", "tool": "os.delete_everything", "inputs": {}, "output_binding": "x"}],
        }
    )

    result = validator.validate(plan, goal="bad")

    assert result.valid is False
    assert "Unknown or disallowed tool" in "; ".join(result.errors)


def test_action_planner_rejects_unsafe_shell(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["shell.exec"])
    plan = ActionPlan.model_validate(
        {
            "goal": "run rm -rf /",
            "actions": [
                {
                    "id": "shell",
                    "tool": "shell.exec",
                    "inputs": {"command": "rm -rf /"},
                    "output_binding": "shell_out",
                },
                {
                    "id": "return",
                    "tool": "runtime.return",
                    "inputs": {"value": "$shell.stdout"},
                    "depends_on": ["shell"],
                    "output_binding": "final",
                },
            ],
        }
    )

    result = validator.validate(plan, goal="run rm -rf /")

    assert result.valid is False
    assert "Unsafe shell command" in "; ".join(result.errors)


def test_action_planner_rejects_export_without_formatter_reference(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query", "fs.write"])
    plan = ActionPlan.model_validate(
        {
            "goal": "export patients to patients.csv",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {"database": "dicom", "query": "SELECT 1 AS count_value"},
                    "output_binding": "rows",
                },
                {
                    "id": "write",
                    "tool": "fs.write",
                    "inputs": {"path": "patients.csv", "content": "placeholder"},
                    "depends_on": ["query"],
                    "output_binding": "written",
                },
                {
                    "id": "return",
                    "tool": "runtime.return",
                    "inputs": {"value": "$written.path"},
                    "depends_on": ["write"],
                    "output_binding": "final",
                },
            ],
        }
    )

    result = validator.validate(plan, goal="export patients to patients.csv")

    assert result.valid is False
    assert "fs.write content must reference upstream formatted output" in "; ".join(result.errors)
    assert "Export/save goals must include text.format" in "; ".join(result.errors)


def test_result_shape_validator_rejects_grouped_rows_for_count() -> None:
    history = [
        StepLog(
            step=ExecutionStep(
                id=1,
                action="sql.query",
                args={"query": 'SELECT p."PatientID", COUNT(*) FROM flathr."Patient" p GROUP BY p."PatientID"'},
                output="rows",
            ),
            result={"rows": [{"PatientID": "a", "count": 1}, {"PatientID": "b", "count": 1}]},
            success=True,
        )
    ]

    result = validate_result_shape("count of patients with more than 5 studies and over age 45", history)

    assert result.success is False
    assert "expected one numeric aggregate" in str(result.reason)
    assert result.metadata is not None
    assert result.metadata["failed_sql"].startswith("SELECT")


def test_action_planner_repairs_count_shape_failure_with_outer_count(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    failed_sql = (
        'SELECT p."PatientID", p."PatientBirthDate", COUNT(s."StudyInstanceUID") AS study_count '
        'FROM flathr."Patient" p '
        'JOIN flathr."Study" s ON p."PatientID" = s."PatientID" '
        'GROUP BY p."PatientID", p."PatientBirthDate" '
        'HAVING COUNT(s."StudyInstanceUID") > 5'
    )
    llm = FakeLLM(
        [
            {
                "goal": "count of patients with more than 5 studies and over age 45",
                "actions": [
                    {
                        "id": "query",
                        "tool": "sql.query",
                        "inputs": {"database": "dicom", "query": failed_sql},
                        "output_binding": "rows",
                        "expected_result_shape": {"kind": "table"},
                    }
                ],
                "expected_final_shape": {"kind": "count"},
            }
        ]
    )
    planner = LLMActionPlanner(llm=llm, tools=build_tool_registry(_settings(tmp_path)), settings=_settings(tmp_path))

    plan = planner.build_plan(
        goal="count of patients with more than 5 studies and over age 45",
        planner=PlannerConfig(),
        allowed_tools=["sql.query"],
        input_payload={},
        failure_context={
            "reason": "result_shape_failed",
            "shape_error": "Count request returned 500 rows; expected one numeric aggregate.",
            "failed_sql": failed_sql,
        },
    )

    query = plan.steps[0].args["query"]
    assert query.startswith("SELECT COUNT(*) AS count_value FROM (")
    assert failed_sql in query
    assert "Repaired grouped count SQL" in " ".join(planner.last_canonicalization_repairs)


def test_result_shape_validator_accepts_single_numeric_count() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"query": "SELECT COUNT(*) AS count_value FROM patients"}),
            result={"rows": [{"count_value": 42}]},
            success=True,
        )
    ]

    assert validate_result_shape("count the number of patients", history).success is True


def test_result_shape_validator_rejects_json_collection_for_non_sql_count() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={"state": "pending"}, output="pending_jobs"),
            result={"jobs": [{"job_id": "1", "state": "PENDING"}], "count": 136},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="text.format", args={"source": [{"job_id": "1"}], "format": "json"}, output="formatted_jobs"),
            result={"content": '[{"job_id": "1"}]', "format": "json"},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=3, action="runtime.return", args={"value": '[{"job_id": "1"}]'}, output="final"),
            result={"output": '[{"job_id": "1"}]', "value": '[{"job_id": "1"}]'},
            success=True,
        ),
    ]

    result = validate_result_shape("Give me just the count of pending jobs in the slurm cluster", history)

    assert result.success is False
    assert result.metadata is not None
    assert result.metadata["final_output_validation"] in {"count_not_scalar", "scalar_returned_collection", "raw_json_output"}


def test_result_shape_validator_rejects_raw_parseable_shell_table_for_list_prompt() -> None:
    stdout = "USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND\nroot 1 0.0 0.1 10 5 ? S 10:00 0:00 init"
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="shell.exec", args={"command": "ps aux"}, output="processes"),
            result={"stdout": stdout, "returncode": 0},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": stdout}, output="final"),
            result={"output": stdout, "value": stdout},
            success=True,
        ),
    ]

    result = validate_result_shape("list all cpu processes", history)

    assert result.success is False
    assert result.metadata is not None
    assert result.metadata["final_output_validation"] == "raw_parseable_table_output"


def test_sql_alias_validation_ignores_subquery_table_alias(tmp_path: Path) -> None:
    validator = RuntimeValidator(_settings(tmp_path))
    query = (
        'SELECT COUNT(*) AS count_value FROM ('
        'SELECT p."PatientID" FROM flathr."Patient" p'
        ') AS filtered_patients'
    )

    assert validator._extract_sql_aliases(query) == ["count_value"]


def test_action_planner_rewrites_bare_formatter_source_and_return_alias(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "show the top 10 patients with the most studies in dicom",
                "actions": [
                    {
                        "id": "query_counts",
                        "tool": "sql.query",
                        "inputs": {
                            "query": 'SELECT "PatientID", COUNT("StudyInstanceUID") AS study_count FROM flathr."Study" GROUP BY "PatientID" ORDER BY study_count DESC LIMIT 10'
                        },
                        "output_binding": "patient_study_counts",
                    },
                    {
                        "id": "format_counts",
                        "tool": "text.format",
                        "inputs": {"source": "patient_study_counts", "format": "markdown"},
                        "output_binding": "formatted_output",
                    },
                    {
                        "id": "return",
                        "tool": "runtime.return",
                        "inputs": {"value": "formatted_output", "mode": "text"},
                    },
                ],
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="show the top 10 patients with the most studies in dicom",
        planner=PlannerConfig(),
        allowed_tools=["sql.query"],
        input_payload={},
    )

    assert plan.steps[0].args["database"] == "dicom"
    assert plan.steps[1].args["source"] == {"$ref": "patient_study_counts", "path": "rows"}
    assert plan.steps[2].args["value"] == {"$ref": "formatted_output", "path": "content"}


def test_action_planner_uses_latest_sql_producer_for_final_formatter(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "count patients over age 45 in dicom",
                "actions": [
                    {
                        "id": "read_patients",
                        "tool": "sql.query",
                        "inputs": {"database": "dicom", "query": 'SELECT "PatientID", "PatientBirthDate" FROM flathr."Patient"'},
                        "output_binding": "patient_data",
                    },
                    {
                        "id": "format_patients",
                        "tool": "text.format",
                        "inputs": {"source": "patient_data", "format": "json"},
                        "output_binding": "formatted_patient_data",
                    },
                    {
                        "id": "count_patients",
                        "tool": "sql.query",
                        "inputs": {"database": "dicom", "query": 'SELECT COUNT(*) AS count_value FROM flathr."Patient"'},
                        "output_binding": "count_result",
                        "expected_result_shape": {"kind": "scalar"},
                    },
                ],
                "expected_final_shape": {"kind": "count"},
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="count patients over age 45 in dicom",
        planner=PlannerConfig(),
        allowed_tools=["sql.query"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["sql.query", "text.format", "runtime.return"]
    assert plan.steps[0].output == "count_result"
    assert plan.steps[1].args["source"] == {"$ref": "count_result", "path": "rows"}


def test_text_format_formats_sql_schema_catalog_as_markdown() -> None:
    content = format_data({"catalog": _catalog().model_dump()}, "markdown")["content"]

    assert "Patient" in content
    assert "PatientID" in content
    assert "| database | schema | table | columns |" in content


def test_result_shape_rejects_empty_and_literal_alias_finals() -> None:
    empty_history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"query": "SELECT 1"}, output="rows"),
            result={"rows": [{"value": 1}], "row_count": 1, "database": "dicom"},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": ""}, output="final"),
            result={"output": "", "value": ""},
            success=True,
        ),
    ]
    literal_history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"query": "SELECT 1"}, output="patient_data"),
            result={"rows": [{"value": 1}], "row_count": 1, "database": "dicom"},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": "patient_data"}, output="final"),
            result={"output": "patient_data", "value": "patient_data"},
            success=True,
        ),
    ]

    assert validate_result_shape("show patients", empty_history).metadata["final_output_validation"] == "empty_final_output"
    assert validate_result_shape("show patients", literal_history).metadata["final_output_validation"] == "literal_reference_output"


def test_result_shape_rejects_raw_json_for_non_json_status_prompt() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.metrics", args={"metric_group": "cluster_summary"}, output="metrics"),
            result={"metric_group": "cluster_summary", "payload": {"queue_count": 10}},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": {"queue_count": 10}, "mode": "json"}, output="final"),
            result={"output": '{"queue_count": 10}', "value": {"queue_count": 10}},
            success=True,
        ),
    ]

    result = validate_result_shape("status of slurm cluster", history)

    assert result.success is False
    assert result.metadata["final_output_validation"] == "raw_json_output"


def test_result_shape_rejects_raw_json_even_when_user_asks_for_json_inline() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.metrics", args={"metric_group": "cluster_summary"}, output="metrics"),
            result={"metric_group": "cluster_summary", "payload": {"queue_count": 10}},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": {"queue_count": 10}, "mode": "json"}, output="final"),
            result={"output": '{"queue_count": 10}', "value": {"queue_count": 10}},
            success=True,
        ),
    ]

    result = validate_result_shape("status of slurm cluster as json", history)

    assert result.success is False
    assert result.metadata["final_output_validation"] == "raw_json_output"


def test_sql_validator_rejects_unknown_catalog_column_before_execution(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "count series by body part examined in dicom",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {"database": "dicom", "query": 'SELECT "BodyPartExamined" FROM flathr."Series"'},
                    "output_binding": "rows",
                },
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$query.rows"}, "depends_on": ["query"]},
            ],
        }
    )

    result = validator.validate(plan, goal="count series by body part examined in dicom")

    assert result.valid is False
    assert "SQL references unknown column: BodyPartExamined" in "; ".join(result.errors)


def test_sql_validator_repairs_study_modality_to_series_join(tmp_path: Path, monkeypatch) -> None:
    _patch_relationship_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "Show the number of studies grouped by modality in dicom.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {
                        "database": "dicom",
                        "query": 'SELECT s."Modality", COUNT(*) AS study_count FROM flathr."Study" s GROUP BY s."Modality"',
                    },
                    "output_binding": "rows",
                },
                {"id": "format", "tool": "text.format", "inputs": {"source": "$query.rows"}, "depends_on": ["query"]},
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$format.content"}, "depends_on": ["format"]},
            ],
        }
    )

    result = validator.validate(plan, goal="Show the number of studies grouped by modality in dicom.")

    assert result.valid is True
    repaired = plan.actions[0].inputs["query"]
    assert 'JOIN "flathr"."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"' in repaired
    assert 'se."Modality" AS modality' in repaired
    assert 'COUNT(DISTINCT st."StudyInstanceUID") AS study_count' in repaired


def test_sql_modality_relationship_repair_is_limited_to_grouping_goals(tmp_path: Path, monkeypatch) -> None:
    _patch_relationship_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    original_query = (
        'SELECT st."StudyInstanceUID", COUNT(DISTINCT se."Modality") AS modality_count '
        'FROM flathr."Study" st JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID" '
        'GROUP BY st."StudyInstanceUID" ORDER BY modality_count DESC LIMIT 10'
    )
    plan = ActionPlan.model_validate(
        {
            "goal": "Show the top 10 studies with the highest number of distinct modalities in dicom.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {"database": "dicom", "query": original_query},
                    "output_binding": "rows",
                },
                {"id": "format", "tool": "text.format", "inputs": {"source": "$query.rows"}, "depends_on": ["query"]},
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$format.content"}, "depends_on": ["format"]},
            ],
        }
    )

    result = validator.validate(plan, goal="Show the top 10 studies with the highest number of distinct modalities in dicom.")

    assert result.valid is True
    assert "COUNT(DISTINCT se.\"Modality\") AS modality_count" in plan.actions[0].inputs["query"]


def test_action_planner_rewrites_generate_validate_explain_sql_to_nonexecuting_tool(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    llm = FakeLLM(
        [
            {
                "goal": "Generate a read-only SQL query to count studies per patient, validate it, and explain what it would return.",
                "actions": [
                    {
                        "id": "query",
                        "tool": "sql.query",
                        "inputs": {
                            "database": "dicom",
                            "query": 'SELECT "PatientID", COUNT("StudyInstanceUID") AS study_count FROM flathr."Study" GROUP BY "PatientID"',
                        },
                        "output_binding": "validated_sql",
                        "expected_result_shape": {"kind": "text"},
                    }
                ],
                "expected_final_shape": {"kind": "text"},
            }
        ]
    )
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="Generate a read-only SQL query to count studies per patient, validate it, and explain what it would return.",
        planner=PlannerConfig(),
        allowed_tools=["sql.query"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["sql.validate", "text.format", "runtime.return"]
    assert plan.steps[0].args["query"] == 'SELECT "PatientID", COUNT("StudyInstanceUID") AS study_count FROM flathr."Study" GROUP BY "PatientID"'


def test_sql_validator_repairs_min_max_avg_studies_per_patient_to_cte(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "Show min, max, and average number of studies per patient in dicom.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {"database": "dicom", "query": 'SELECT COUNT(*) FROM flathr."Study"'},
                    "output_binding": "rows",
                },
                {"id": "format", "tool": "text.format", "inputs": {"source": "$query.rows"}, "depends_on": ["query"]},
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$format.content"}, "depends_on": ["format"]},
            ],
        }
    )

    result = validator.validate(plan, goal="Show min, max, and average number of studies per patient in dicom.")

    assert result.valid is True
    repaired = plan.actions[0].inputs["query"]
    assert repaired.startswith("WITH grouped_counts AS")
    assert 'LEFT JOIN "flathr"."Study" r ON r."PatientID" = b."PatientID"' in repaired
    assert "MIN(related_count) AS min_studies_per_patient" in repaired
    assert "AVG(related_count) AS average_studies_per_patient" in repaired


def test_sql_validator_repairs_study_instance_count_through_series(tmp_path: Path, monkeypatch) -> None:
    _patch_relationship_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "Show the top 10 studies by total number of DICOM instances in dicom.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {
                        "database": "dicom",
                        "query": 'SELECT i."StudyInstanceUID", COUNT(*) AS instance_count FROM flathr."Instance" i GROUP BY i."StudyInstanceUID" ORDER BY instance_count DESC LIMIT 10',
                    },
                    "output_binding": "rows",
                },
                {"id": "format", "tool": "text.format", "inputs": {"source": "$query.rows"}, "depends_on": ["query"]},
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$format.content"}, "depends_on": ["format"]},
            ],
        }
    )

    result = validator.validate(plan, goal="Show the top 10 studies by total number of DICOM instances in dicom.")

    assert result.valid is True
    repaired = plan.actions[0].inputs["query"]
    assert 'JOIN "flathr"."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"' in repaired
    assert 'JOIN "flathr"."Instance" i ON i."SeriesInstanceUID" = se."SeriesInstanceUID"' in repaired
    assert 'i."StudyInstanceUID"' not in repaired
    assert "LIMIT 10" in repaired


def test_sql_validator_rejects_column_on_wrong_alias_before_execution(tmp_path: Path, monkeypatch) -> None:
    _patch_relationship_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "Show instance rows in dicom.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {"database": "dicom", "query": 'SELECT i."StudyInstanceUID" FROM flathr."Instance" i'},
                    "output_binding": "rows",
                },
                {"id": "format", "tool": "text.format", "inputs": {"source": "$query.rows"}, "depends_on": ["query"]},
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$format.content"}, "depends_on": ["format"]},
            ],
        }
    )

    result = validator.validate(plan, goal="Show instance rows in dicom.")

    assert result.valid is False
    assert "Column StudyInstanceUID is not on table flathr.Instance" in "; ".join(result.errors)


def test_sql_validator_reports_optional_missing_schema_concept_cleanly(tmp_path: Path, monkeypatch) -> None:
    _patch_relationship_catalog(monkeypatch)
    settings = _settings(tmp_path)
    validator = ActionPlanValidator(settings=settings, tools=build_tool_registry(settings), allowed_tools=["sql.query"])
    plan = ActionPlan.model_validate(
        {
            "goal": "Show all distinct body parts or anatomical regions if such a column exists in dicom, with counts.",
            "actions": [
                {
                    "id": "query",
                    "tool": "sql.query",
                    "inputs": {"database": "dicom", "query": 'SELECT "BodyPartExamined", COUNT(*) FROM flathr."Series" GROUP BY "BodyPartExamined"'},
                    "output_binding": "rows",
                },
                {"id": "format", "tool": "text.format", "inputs": {"source": "$query.rows"}, "depends_on": ["query"]},
                {"id": "return", "tool": "runtime.return", "inputs": {"value": "$format.content"}, "depends_on": ["format"]},
            ],
        }
    )

    result = validator.validate(plan, goal="Show all distinct body parts or anatomical regions if such a column exists in dicom, with counts.")

    assert result.valid is False
    assert "optional schema concept body part/anatomical region is not present" in "; ".join(result.errors)


def test_text_format_outputs_csv_and_markdown() -> None:
    rows = [{"PatientID": "A", "study_count": 3}, {"PatientID": "B", "study_count": 2}]

    csv_result = format_data(rows, "csv")
    markdown_result = format_data(rows, "markdown")

    assert csv_result["content"].splitlines()[0] == "PatientID,study_count"
    assert "| PatientID | study_count |" in markdown_result["content"]


def test_action_prompt_allows_raw_control_char_repair(tmp_path: Path, monkeypatch) -> None:
    _patch_catalog(monkeypatch)
    raw_with_newline = (
        '{"goal":"Count","actions":[{"id":"q","tool":"sql.query","inputs":'
        '{"database":"dicom","query":"SELECT COUNT(*) AS count_value\nFROM flathr.\\"Patient\\""},'
        '"output_binding":"rows"},{"id":"f","tool":"text.format","inputs":{"source":"$q.rows","format":"txt"},'
        '"depends_on":["q"],"output_binding":"formatted"},{"id":"r","tool":"runtime.return",'
        '"inputs":{"value":"$f.content"},"depends_on":["f"],"output_binding":"final"}]}'
    )
    settings = _settings(tmp_path)
    planner = LLMActionPlanner(llm=FakeLLM([raw_with_newline]), tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="count the number of patients in dicom",
        planner=PlannerConfig(),
        allowed_tools=["sql.query"],
        input_payload={},
    )

    assert [step.action for step in plan.steps] == ["sql.query", "text.format", "runtime.return"]


def test_action_plan_validation_error_is_user_safe() -> None:
    error = normalize_planner_error(
        error_type="ValidationError",
        detail="1 validation error for ActionPlan\nexpected_final_shape.kind\n  Input should be ...",
        llm_base_url="http://127.0.0.1:8000/v1",
    )

    assert error is not None
    assert error.message == "Planner produced an invalid action plan and the request was not executed."
    assert error.kind == "invalid_action_plan"
