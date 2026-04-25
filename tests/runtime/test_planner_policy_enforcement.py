from __future__ import annotations

import json
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.tools.sql import DatabaseSchema, SchemaInfo, TableSchema, ColumnSchema
from aor_runtime.runtime.decomposer import DEFAULT_DECOMPOSER_PROMPT
from aor_runtime.runtime.planner import DEFAULT_PLANNER_PROMPT, TaskPlanner
from aor_runtime.runtime.policies import (
    PlanContractViolation,
    classify_plan_violations,
    validate_plan_contract,
    validate_plan_efficiency,
)
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self, raw_responses: list[str]) -> None:
        self.raw_responses = raw_responses
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []
        self.models: list[str | None] = []
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

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
        if len(self.user_prompts) >= len(self.raw_responses):
            raise AssertionError("LLM called more times than expected")
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        self.models.append(model)
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.raw_responses[len(self.user_prompts) - 1]

    @property
    def call_count(self) -> int:
        return len(self.user_prompts)


def _serialize_responses(raw_response: str | dict | list[str | dict]) -> list[str]:
    responses = raw_response if isinstance(raw_response, list) else [raw_response]
    return [json.dumps(item) if isinstance(item, dict) else item for item in responses]


def _planner(tmp_path: Path, raw_response: str | dict | list[str | dict]) -> tuple[TaskPlanner, FakeLLM]:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    llm = FakeLLM(_serialize_responses(raw_response))
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    return planner, llm


def _planner_with_settings(
    tmp_path: Path, raw_response: str | dict | list[str | dict], **settings_overrides
) -> tuple[TaskPlanner, FakeLLM]:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **settings_overrides)
    llm = FakeLLM(_serialize_responses(raw_response))
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    return planner, llm


def _planner_config() -> PlannerConfig:
    return PlannerConfig(temperature=0.0)


def test_planner_injects_rendered_policies_into_context(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        {
            "steps": [
                {"id": 1, "action": "fs.write", "args": {"path": "notes.txt", "content": "hello"}},
                {"id": 2, "action": "fs.read", "args": {"path": "notes.txt"}},
            ]
        },
    )

    plan = planner.build_plan(
        goal="Create the file notes.txt with exact content hello",
        planner=_planner_config(),
        allowed_tools=["fs.write", "fs.read", "shell.exec", "python.exec"],
        input_payload={"task": "Create the file notes.txt with exact content hello"},
    )

    assert isinstance(plan, ExecutionPlan)
    assert llm.last_user_prompt is not None
    planner_context = json.loads(llm.last_user_prompt)
    assert "policies" in planner_context
    assert "filesystem_preference" in planner_context["policies"]
    assert planner.last_policies_used == ["filesystem_preference", "efficiency"]


def test_default_planner_prompt_prefers_sql_for_aggregation_and_counting() -> None:
    assert "aggregation, grouping, counting" in DEFAULT_PLANNER_PROMPT
    assert "Prefer pushing computation into SQL whenever possible." in DEFAULT_PLANNER_PROMPT


def test_simple_goal_uses_direct_planning_only(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        {"steps": [{"id": 1, "action": "fs.read", "args": {"path": "notes.txt"}}]},
    )

    plan = planner.build_plan(
        goal="Read notes.txt",
        planner=_planner_config(),
        allowed_tools=["fs.read"],
        input_payload={"task": "Read notes.txt"},
    )

    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == "direct"
    assert planner.last_high_level_plan is None
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1
    assert llm.models == [None]


def test_complex_goal_uses_decomposition_and_refinement(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [
            {"tasks": ["find matching txt files", "format the list as csv"]},
            {
                "steps": [
                    {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.txt"}, "output": "txt_matches"},
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["txt_matches"],
                        "output": "csv_result",
                        "args": {
                            "inputs": {"matches": {"$ref": "txt_matches", "path": "matches"}},
                            "code": "result = {'csv': ','.join(inputs['matches'])}",
                        },
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="find all *.txt files in this folder and provide list as csv",
        planner=_planner_config(),
        allowed_tools=["fs.find", "python.exec"],
        input_payload={"task": "find all *.txt files in this folder and provide list as csv"},
    )

    assert [step.action for step in plan.steps] == ["fs.find", "python.exec"]
    assert planner.last_planning_mode == "hierarchical"
    assert planner.last_high_level_plan == ["find matching txt files", "format the list as csv"]
    assert planner.last_llm_calls == 2
    assert llm.call_count == 2
    assert llm.models == [None, None]
    assert json.loads(llm.user_prompts[1])["high_level_plan"] == ["find matching txt files", "format the list as csv"]


def test_explicit_planner_model_override_is_forwarded_to_both_llm_stages(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [
            {"tasks": ["query patients", "return the count"]},
            {"steps": [{"id": 1, "action": "sql.query", "args": {"database": "dicom", "query": "SELECT COUNT(*) AS count FROM patient"}}]},
        ],
    )

    plan = planner.build_plan(
        goal="Query the number of patients in dicom and then return the count",
        planner=PlannerConfig(model="custom-model", temperature=0.0),
        allowed_tools=["sql.query"],
        input_payload={"task": "Query the number of patients in dicom and then return the count"},
    )

    assert isinstance(plan, ExecutionPlan)
    assert llm.models == ["custom-model", "custom-model"]


def test_empty_high_level_plan_fails(tmp_path: Path) -> None:
    planner, _ = _planner(tmp_path, [{"tasks": []}])

    with pytest.raises(ValueError, match="at least one task"):
        planner.build_plan(
            goal="find all txt files in this folder and provide list as csv",
            planner=_planner_config(),
            allowed_tools=["fs.find", "python.exec"],
            input_payload={"task": "find all txt files in this folder and provide list as csv"},
        )

    assert planner.last_error_stage == "decompose"
    assert planner.last_planning_mode == "hierarchical"
    assert planner.last_llm_calls == 1


def test_empty_execution_plan_fails(tmp_path: Path) -> None:
    planner, _ = _planner(tmp_path, {"steps": []})

    with pytest.raises(ValueError, match="Execution plan requires at least one step"):
        planner.build_plan(
            goal="Read notes.txt",
            planner=_planner_config(),
            allowed_tools=["fs.read"],
            input_payload={"task": "Read notes.txt"},
        )

    assert planner.last_error_stage == "direct"
    assert planner.last_planning_mode == "direct"


def test_planner_rejects_plan_that_exceeds_step_limit(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        {
            "steps": [
                {"id": index, "action": "fs.exists", "args": {"path": f"item_{index}.txt"}}
                for index in range(1, 14)
            ]
        },
    )

    with pytest.raises(ValueError, match="Plan too complex"):
        planner.build_plan(
            goal="Check whether many files exist",
            planner=_planner_config(),
            allowed_tools=["fs.exists"],
            input_payload={"task": "Check whether many files exist"},
        )


def test_planner_rejects_multiple_python_exec_steps(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        {
            "steps": [
                {"id": 1, "action": "python.exec", "args": {"code": "result = {'value': 1}"}, "output": "first"},
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["first"],
                    "args": {
                        "inputs": {"previous": {"$ref": "first", "path": "value"}},
                        "code": "result = {'value': inputs['previous'] + 1}",
                    },
                },
            ]
        },
    )

    plan = planner.build_plan(
        goal="Run multiple loops over local data",
        planner=_planner_config(),
        allowed_tools=["python.exec"],
        input_payload={"task": "Run multiple loops over local data"},
    )

    assert [step.action for step in plan.steps] == ["python.exec", "python.exec"]


def test_validate_plan_efficiency_accepts_compliant_plan() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.exists", "args": {"path": "notes.txt"}},
                {"id": 2, "action": "python.exec", "args": {"code": "result = {'value': 'ok'}"}},
            ]
        }
    )

    validate_plan_efficiency(plan)


def test_planner_accepts_delete_plan_with_fs_not_exists(tmp_path: Path) -> None:
    planner, _ = _planner_with_settings(
        tmp_path,
        {
            "steps": [
                {"id": 1, "action": "fs.exists", "args": {"path": "notes.txt"}},
                {"id": 2, "action": "shell.exec", "args": {"command": "rm notes.txt"}},
                {"id": 3, "action": "fs.not_exists", "args": {"path": "notes.txt"}},
            ]
        },
        available_nodes_raw="local",
        default_node="local",
    )

    plan = planner.build_plan(
        goal="Delete notes.txt",
        planner=_planner_config(),
        allowed_tools=["fs.exists", "fs.not_exists", "shell.exec"],
        input_payload={"task": "Delete notes.txt"},
    )

    assert [step.action for step in plan.steps] == ["fs.exists", "shell.exec", "fs.not_exists"]


def test_planner_injects_logical_nodes_without_gateway_details(tmp_path: Path) -> None:
    planner, llm = _planner_with_settings(
        tmp_path,
        {"steps": [{"id": 1, "action": "shell.exec", "args": {"node": "edge-1", "command": "uname -a"}}]},
        gateway_url="https://gateway.example.internal/exec",
        available_nodes_raw="edge-1,edge-2",
        default_node="edge-1",
    )

    plan = planner.build_plan(
        goal="Run uname -a on remote node edge-1",
        planner=_planner_config(),
        allowed_tools=["shell.exec"],
        input_payload={"task": "Run uname -a on remote node edge-1"},
    )

    assert isinstance(plan, ExecutionPlan)
    assert llm.last_user_prompt is not None
    planner_context = json.loads(llm.last_user_prompt)
    assert planner_context["nodes"]["available"] == ["edge-1", "edge-2"]
    assert planner_context["nodes"]["default"] == "edge-1"
    assert "gateway.example.internal" not in llm.last_user_prompt


def test_planner_uses_implicit_localhost_default_when_not_configured(tmp_path: Path) -> None:
    planner, llm = _planner_with_settings(
        tmp_path,
        {"steps": [{"id": 1, "action": "shell.exec", "args": {"command": "ls -1 | paste -sd, -"}}]},
    )

    plan = planner.build_plan(
        goal="return the current directory entries as a csv string",
        planner=_planner_config(),
        allowed_tools=["shell.exec"],
        input_payload={"task": "return the current directory entries as a csv string"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec"]
    assert llm.last_user_prompt is not None
    planner_context = json.loads(llm.last_user_prompt)
    assert planner_context["nodes"]["available"] == ["localhost"]
    assert planner_context["nodes"]["default"] == "localhost"


def test_planner_accepts_recursive_file_search_plan_with_fs_find(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [
            {"tasks": ["find matching txt files", "format the list as csv"]},
            {
                "steps": [
                    {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.txt"}, "output": "txt_matches"},
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["txt_matches"],
                        "output": "csv_result",
                        "args": {
                            "inputs": {"matches": {"$ref": "txt_matches", "path": "matches"}},
                            "code": "result = {'csv': ','.join(inputs['matches'])}",
                        },
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="find all *.txt files in this folder and provide list as csv",
        planner=_planner_config(),
        allowed_tools=["fs.find", "python.exec"],
        input_payload={"task": "find all *.txt files in this folder and provide list as csv"},
    )

    assert [step.action for step in plan.steps] == ["fs.find", "python.exec"]
    assert llm.system_prompts[-1] is not None
    assert "Use fs.find for recursive file discovery" in llm.system_prompts[-1]


def test_planner_accepts_total_file_size_plan_with_fs_size(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        {
            "steps": [
                {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.txt"}, "output": "txt_matches"},
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["txt_matches"],
                    "output": "size_summary",
                    "args": {
                        "inputs": {"files": {"$ref": "txt_matches", "path": "matches"}},
                        "code": "total_size = sum(fs.size(path) for path in inputs['files']); result = {'file_count': len(inputs['files']), 'total_size_bytes': total_size}"
                    },
                },
            ]
        },
    )

    plan = planner.build_plan(
        goal="compute the total size of all the txt files in this folder",
        planner=_planner_config(),
        allowed_tools=["fs.find", "fs.size", "python.exec"],
        input_payload={"task": "compute the total size of all the txt files in this folder"},
    )

    assert [step.action for step in plan.steps] == ["fs.find", "python.exec"]
    assert llm.last_system_prompt is not None
    assert "Use fs.size when the user asks for the size of a file" in llm.last_system_prompt


def test_planner_accepts_shell_plan_with_allowed_node(tmp_path: Path) -> None:
    planner, _ = _planner_with_settings(
        tmp_path,
        {"steps": [{"id": 1, "action": "shell.exec", "args": {"node": "edge-1", "command": "uname -a"}}]},
        available_nodes_raw="edge-1",
    )

    plan = planner.build_plan(
        goal="Run uname -a on remote node edge-1",
        planner=_planner_config(),
        allowed_tools=["shell.exec"],
        input_payload={"task": "Run uname -a on remote node edge-1"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec"]


def test_planner_rejects_shell_plan_with_disallowed_node(tmp_path: Path) -> None:
    planner, _ = _planner_with_settings(
        tmp_path,
        {"steps": [{"id": 1, "action": "shell.exec", "args": {"node": "edge-9", "command": "uname -a"}}]},
        available_nodes_raw="edge-1,edge-2",
    )

    with pytest.raises(ValueError, match="disallowed node"):
        planner.build_plan(
            goal="Run uname -a on remote node edge-9",
            planner=_planner_config(),
            allowed_tools=["shell.exec"],
            input_payload={"task": "Run uname -a on remote node edge-9"},
        )


def test_planner_rejects_invalid_data_dependency(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        {
            "steps": [
                {
                    "id": 1,
                    "action": "fs.write",
                    "input": ["missing_alias"],
                    "args": {"path": "notes.txt", "content": {"$ref": "missing_alias", "path": "csv"}},
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="Invalid data dependency"):
        planner.build_plan(
            goal="Write a file from previous data",
            planner=_planner_config(),
            allowed_tools=["fs.write"],
            input_payload={"task": "Write a file from previous data"},
        )


def test_planner_canonicalizes_repairable_dataflow_plan(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["query patient names", "format them as csv", "write the file"]},
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": "clinical_db", "query": "SELECT name FROM patients ORDER BY name"},
                        "output": "rows",
                    },
                    {
                        "id": 2,
                        "action": "python.exec",
                        "args": {
                            "inputs": {"rows": {"$ref": "rows", "path": "rows"}},
                            "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                        },
                        "output": "csv",
                    },
                    {
                        "id": 3,
                        "action": "fs.write",
                        "input": ["csv_result"],
                        "args": {
                            "path": "patients.csv",
                            "content": {"$ref": "csv", "path": "csv"},
                        },
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="Query patient names from clinical_db, format them as csv, and save them to patients.csv",
        planner=_planner_config(),
        allowed_tools=["sql.query", "python.exec", "fs.write"],
        input_payload={"task": "Query patient names from clinical_db, format them as csv, and save them to patients.csv"},
    )

    assert plan.steps[0].output == "step_1_rows"
    assert plan.steps[1].input == ["step_1_rows"]
    assert plan.steps[1].args["inputs"]["rows"]["$ref"] == "step_1_rows"
    assert plan.steps[2].input == ["step_2_data"]
    assert plan.steps[2].args["content"]["$ref"] == "step_2_data"
    assert planner.last_plan_canonicalized is True
    assert planner.last_plan_repairs


def test_planner_appends_text_readback_for_save_and_return_goal(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["query patient names", "format them as csv", "save them and return the csv"]},
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": "clinical_db", "query": "SELECT name FROM patients ORDER BY score DESC LIMIT 3"},
                        "output": "patient_rows",
                    },
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["patient_rows"],
                        "output": "patient_csv",
                        "args": {
                            "inputs": {"rows": {"$ref": "patient_rows", "path": "rows"}},
                            "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                        },
                    },
                    {
                        "id": 3,
                        "action": "fs.write",
                        "input": ["patient_csv"],
                        "args": {
                            "path": "outputs/top_patients.csv",
                            "content": {"$ref": "patient_csv", "path": "csv"},
                        },
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="Query the top 3 patients by score from clinical_db, save the result to outputs/top_patients.csv, and return it.",
        planner=_planner_config(),
        allowed_tools=["sql.query", "python.exec", "fs.write", "fs.read"],
        input_payload={"task": "Query the top 3 patients by score from clinical_db, save the result to outputs/top_patients.csv, and return it."},
    )

    assert [step.id for step in plan.steps] == [1, 2, 3, 4]
    assert plan.steps[-1].action == "fs.read"
    assert plan.steps[-1].args["path"] == "outputs/top_patients.csv"
    assert plan.steps[-1].args["__canonicalizer_added"] is True


def test_planner_keeps_already_valid_simple_plan_unchanged(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        {"steps": [{"id": 1, "action": "fs.read", "args": {"path": "notes.txt"}}]},
    )

    plan = planner.build_plan(
        goal="Read notes.txt",
        planner=_planner_config(),
        allowed_tools=["fs.read"],
        input_payload={"task": "Read notes.txt"},
    )

    assert plan.model_dump() == {"steps": [{"id": 1, "action": "fs.read", "args": {"path": "notes.txt"}, "input": [], "output": None}]}
    assert planner.last_plan_canonicalized is False
    assert planner.last_plan_repairs == []


def test_planner_rejects_placeholder_write_when_upstream_output_exists(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["query patient names", "write the csv file"]},
            {
                "steps": [
                    {"id": 1, "action": "sql.query", "output": "patient_rows", "args": {"database": "clinical_db", "query": "SELECT name FROM patients"}},
                    {"id": 2, "action": "fs.write", "args": {"path": "patients.csv", "content": "name1,name2,name3"}},
                ]
            },
        ],
    )

    with pytest.raises(ValueError, match="placeholder output"):
        planner.build_plan(
            goal="Query patient names from clinical_db and save them to patients.csv",
            planner=_planner_config(),
            allowed_tools=["sql.query", "fs.write"],
            input_payload={"task": "Query patient names from clinical_db and save them to patients.csv"},
        )


def test_planner_backfills_step_inputs_from_structured_refs(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["query the top patients", "format the names as csv", "write the csv file"]},
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": "clinical_db", "query": "SELECT name FROM patients ORDER BY score DESC LIMIT 3"},
                        "output": "patient_rows",
                    },
                    {
                        "id": 2,
                        "action": "python.exec",
                        "args": {
                            "inputs": {"rows": {"$ref": "patient_rows", "path": "rows"}},
                            "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                        },
                        "output": "patient_csv",
                    },
                    {
                        "id": 3,
                        "action": "fs.write",
                        "args": {
                            "path": "outputs/top_patients.csv",
                            "content": {"$ref": "patient_csv", "path": "csv"},
                        },
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="Query the top 3 patients by score from clinical_db, format the names as csv, and save the result to outputs/top_patients.csv.",
        planner=_planner_config(),
        allowed_tools=["sql.query", "python.exec", "fs.write"],
        input_payload={
            "task": "Query the top 3 patients by score from clinical_db, format the names as csv, and save the result to outputs/top_patients.csv."
        },
    )

    assert plan.steps[1].input == ["patient_rows"]
    assert plan.steps[2].input == ["patient_csv"]


def test_planner_backfills_python_inputs_from_prior_shell_output(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["list matching python files with shell", "format the list as csv"]},
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "shell.exec",
                        "args": {"command": "find src/aor_runtime/runtime -type f -name \"*.py\""},
                        "output": "py_files",
                    },
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["py_files"],
                        "args": {"code": "result = {'csv': ','.join(inputs['py_files'].splitlines())}"},
                        "output": "csv_result",
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="Using shell, list all .py files under src/aor_runtime/runtime and then return the list as a csv string.",
        planner=_planner_config(),
        allowed_tools=["shell.exec", "python.exec"],
        input_payload={"task": "Using shell, list all .py files under src/aor_runtime/runtime and then return the list as a csv string."},
    )

    assert plan.steps[1].input == ["py_files"]
    assert plan.steps[1].args["inputs"] == {"py_files": {"$ref": "py_files", "path": "stdout"}}
    assert "inputs['py_files'].splitlines()" in plan.steps[1].args["code"]
    assert "['stdout']" not in plan.steps[1].args["code"]


def test_planner_accepts_sql_rows_as_direct_list_values_in_python_exec(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["count studies", "return the count safely"]},
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": "clinical_db", "query": "SELECT COUNT(*) AS study_count FROM studies"},
                        "output": "study_rows",
                    },
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["study_rows"],
                        "output": "study_count",
                        "args": {
                            "inputs": {"rows": {"$ref": "study_rows", "path": "rows"}},
                            "code": "rows = inputs['rows']; result = {'count': rows[0]['study_count'] if rows else 0}",
                        },
                    },
                ]
            },
        ],
    )

    plan = planner.build_plan(
        goal="Count studies in clinical_db and return the count safely.",
        planner=_planner_config(),
        allowed_tools=["sql.query", "python.exec"],
        input_payload={"task": "Count studies in clinical_db and return the count safely."},
    )

    assert plan.steps[1].args["inputs"]["rows"] == {"$ref": "study_rows", "path": "rows"}
    assert "rows = inputs['rows']" in plan.steps[1].args["code"]
    assert "if rows else 0" in plan.steps[1].args["code"]
    assert "['rows']" not in plan.steps[1].args["code"].replace("inputs['rows']", "")
    assert plan.steps[1].output == "study_count"


def test_planner_rejects_when_explicit_shell_intent_is_ignored(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        [
            {"tasks": ["list matching python files", "format the list as csv"]},
            {
                "steps": [
                    {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.py"}, "output": "py_matches"},
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["py_matches"],
                        "args": {
                            "inputs": {"matches": {"$ref": "py_matches", "path": "matches"}},
                            "code": "result = {'csv': ','.join(inputs['matches'])}",
                        },
                    },
                ]
            },
        ],
    )

    with pytest.raises(ValueError, match="explicit tool request for shell\\.exec"):
        planner.build_plan(
            goal="Using shell, list all .py files here and return them as csv",
            planner=_planner_config(),
            allowed_tools=["fs.find", "python.exec", "shell.exec"],
            input_payload={"task": "Using shell, list all .py files here and return them as csv"},
        )


def test_planner_rejects_python_inputs_usage_without_args_inputs(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = {'csv': ','.join(inputs['rows'])}"},
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="must declare args\\.inputs"):
        planner.build_plan(
            goal="Using python, join rows as csv",
            planner=_planner_config(),
            allowed_tools=["python.exec"],
            input_payload={"task": "Using python, join rows as csv"},
        )


def test_planner_tracks_raw_output_and_error_type_for_malformed_json(tmp_path: Path) -> None:
    raw_response = """
    {
      "steps": [
        {
          "id": 1,
          "action": "fs.write",
          "args": {
            "path": "notes.txt",
            "content": "hello" + "world"
          }
        }
      ]
    }
    """
    planner, _ = _planner(tmp_path, raw_response)

    with pytest.raises(ValueError, match="Expecting ',' delimiter"):
        planner.build_plan(
            goal="Write notes.txt",
            planner=_planner_config(),
            allowed_tools=["fs.write"],
            input_payload={"task": "Write notes.txt"},
        )

    assert planner.last_error_type == "JSONDecodeError"
    assert planner.last_raw_output is not None
    assert '"content": "hello" + "world"' in planner.last_raw_output


def test_prompt_sources_include_execution_contract_rules() -> None:
    file_prompt = (Path(__file__).resolve().parents[2] / "prompts" / "planner_system.txt").read_text()
    decomposer_file_prompt = (Path(__file__).resolve().parents[2] / "prompts" / "decomposer_system.txt").read_text()

    required_snippets = [
        "If the task explicitly names a node, include that node in shell.exec args.",
        "Never invent node names outside the provided logical node list.",
        "If schema information includes a database dialect, generate SQL that is valid for that dialect.",
        "For PostgreSQL, do not use SQLite-only functions like strftime.",
        "Preserve full file paths exactly as provided in the goal or returned by tools.",
        "When fs.find returns matches, keep those returned match strings unchanged unless a downstream fs.size, fs.read, or fs.copy operation needs a concrete path",
        "Never strip directory prefixes, collapse to basenames, manually reconstruct alternate prefixes, or convert absolute paths into relative paths.",
        "If the user requests modifying SQL data or schema, do not generate a modifying sql.query plan.",
        "Return exactly what the user asked for: count requests return a number only, CSV requests return a CSV string only, and JSON requests return a JSON object only.",
        "Do not include extra text, file paths, acknowledgements, or wrapper objects in the final output unless the user explicitly asked for them.",
        "Output keys must match the user's requested keys exactly, including spelling and case.",
        "Every args value must be valid JSON as written.",
        "Prefer a direct shell.exec step for simple command-output formatting tasks.",
        "Use fs.not_exists to verify that a path is absent after deletion or cleanup.",
        "For text-content search across many files, prefer shell.exec with a portable find + grep command rather than python.exec file IO or fs.read loops.",
        "If the user specifies a file pattern such as *.txt or *.py, use find ROOT -type f -name \"<pattern>\" -exec grep -li -- \"needle\" {} + || true so the content search targets the requested files.",
        "If the user asks to search all files and does not specify a pattern, search all regular files with find ROOT -type f -exec grep -li -- \"needle\" {} + || true rather than inventing a restrictive extension filter.",
        "Use rg -l only as an optional faster variant when the shell environment is explicitly known to support rg; otherwise default to the portable find + grep form.",
        "Use SQL for aggregation, grouping, counting, filtering, joins, and histograms whenever the database can express the operation directly.",
        "For top-level or non-recursive file matching, prefer fs.list plus minimal filtering or formatting instead of fs.find.",
        "Do not use python.exec when sql.query, fs.*, or shell.exec can solve the task directly.",
        "In python.exec, code must be valid, minimal, and should only import json or re when an import is required.",
        "python.exec is a pure data transformation step: it must read from inputs[...], compute a transformed value, and assign the final value to result.",
        "python.exec must never call tools and must never perform side effects.",
        "all side-effecting work must appear as explicit non-Python tool steps in the plan.",
        "For fetch-and-extract tasks such as curl/fetch page + extract section/value, treat extraction as a shell-first pattern and prefer a single shell.exec step when shell can return the final extracted text directly.",
        "For web fetches in shell.exec, prefer curl -sL and prefer https:// for bare domains so redirects are followed and the fetched page is the canonical page.",
        "For simple HTML/text extraction, do not generate shell.exec -> python.exec or shell.exec -> shell.exec plans when one shell.exec pipeline can fetch once and return the requested extracted text directly.",
        "For simple HTML tag or section extraction such as <head>, <title>, or similar single-section extraction, use a single shell.exec pipeline and do not add python.exec unless shell cannot express the extraction cleanly.",
        "Do not generate multiple shell.exec steps that fetch the same URL twice for a single extraction task.",
        "For shell-based HTML/text extraction, prefer portable pipe-based commands and avoid process substitution like <(...).",
        "Use python.exec with re only when upstream shell or other tools already produced the full input text and the extraction is too awkward to express cleanly in shell.",
        "If python.exec uses re for extraction, result must be a plain string, list, dict, number, or boolean.",
        "Be correct first, minimal second, and efficient third.",
        "If a high_level_plan is provided in the planner context, refine it into executable steps",
        "If explicit_tool_intent is provided in the planner context, you MUST use those requested tools",
        "Every step that produces data for later use must declare an output alias.",
        "inputs dict",
    ]

    for snippet in required_snippets:
        assert snippet in DEFAULT_PLANNER_PROMPT
        assert snippet in file_prompt

    assert "break the user's goal into an ordered list of high-level tasks" in DEFAULT_DECOMPOSER_PROMPT.lower()
    assert "break the user's goal into an ordered list of high-level tasks" in decomposer_file_prompt.lower()


def test_prompt_sources_describe_resolved_python_input_shapes() -> None:
    file_prompt = (Path(__file__).resolve().parents[2] / "prompts" / "planner_system.txt").read_text()

    required_snippets = [
        "inputs[...] values are fully computed runtime results, not references, wrappers, or tool-response objects.",
        "inputs[...] is the value itself, not an object containing the value, and it does not contain implicit nested structure.",
        "sql.query -> list of dict rows, fs.find -> list of file path strings, fs.read -> string content, shell.exec -> stdout string, python.exec -> arbitrary resolved value.",
        'never access nested wrapper fields like ["stdout"], ["rows"], or ["content"]',
        "do not add defensive wrapper-detection logic or shape-probing branches for inputs[...]",
        "do not wrap inputs[...] into new containers unless the actual computation requires it",
        "do not rename inputs[...] unless needed for readability or a real transformation.",
        "shell.exec output passed through inputs[...] is a string; use .splitlines() when you need a list of lines.",
        "always handle empty SQL result lists safely before indexing.",
        "do not assume SQL result fields unless they were explicitly selected by the SQL query.",
        "downstream steps consume the python.exec output value directly",
        "If the user asks to return, list, show, or provide data, the final step must surface that data",
        "do not use os, subprocess, system calls, eval, exec, or direct fs.* / shell.exec(...) / sql.query(...) helper calls.",
        "\"code\": \"result = ','.join(inputs['py_files'].splitlines())\"",
        "\"code\": \"rows = inputs['rows']; result = rows[0]['study_count'] if rows else 0\"",
        "\"code\": \"import json; study_rows = inputs['study_rows']; series_rows = inputs['series_rows']; result = json.dumps({'studies': study_rows[0]['studies'] if study_rows else 0, 'series': series_rows[0]['series'] if series_rows else 0}, sort_keys=True)\"",
    ]

    for snippet in required_snippets:
        assert snippet in DEFAULT_PLANNER_PROMPT
        assert snippet in file_prompt


def test_prompt_sources_replace_weak_examples_with_path_and_scope_aware_ones() -> None:
    file_prompt = (Path(__file__).resolve().parents[2] / "prompts" / "planner_system.txt").read_text()

    required_snippets = [
        "count CT and MR series in dicom and return JSON with keys CT and MR",
        "find all top-level *.txt files under reports and return them as csv",
        "count studies and series in clinical_db, write a JSON summary to reports/summary.json, and return it",
        "find all *.txt files in this folder with the word vinith in their contents",
        "list all .py files with import in their contents",
        "list all files with vinith in their contents",
        "curl google.com and extract <head>",
        "fetch a page and extract title text",
        "curl example.com and extract <title>",
        "fetch a page, then extract a value with complex regex cleanup",
        "if the shell environment is explicitly known to support rg, find all *.txt files in this folder with the word vinith in their contents",
        "\"content\": {\"$ref\": \"patient_csv\"}",
        "\"code\": \"lines = inputs['text'].splitlines(); result = lines[1] if len(lines) > 1 else ''\"",
        "Anti-patterns:",
        "Invalid: a python.exec step that calls fs.write(...), fs.read(...), fs.copy(...), fs.find(...), fs.list(...), or fs.size(...).",
        "Invalid: a python.exec step that calls shell.exec(...).",
        "Invalid: a python.exec step that calls sql.query(...).",
    ]

    forbidden_snippets = [
        "f'inputs/{path}'",
        "lines = fs.read('notes.txt').splitlines()",
        "total_size = sum(fs.size(f'{root}/{match}') for match in matches)",
        "result = {'file_count': len(inputs['files']), 'total_size_bytes': sum(fs.size(path) for path in inputs['files'])}",
        "copied = []; [fs.copy(f'A/{name}', f'B/{name}') or copied.append(name) for name in inputs['entries'] if name.endswith('.txt')]",
        "\"content\": {\"$ref\": \"patient_csv\", \"path\": \"csv\"}",
        "open(f, 'r').read()",
    ]

    for snippet in required_snippets:
        assert snippet in DEFAULT_PLANNER_PROMPT
        assert snippet in file_prompt

    assert 'find . -type f -exec grep -li -- "vinith" {} + || true' in DEFAULT_PLANNER_PROMPT
    assert 'find . -type f -name "*.txt" -exec grep -li -- "vinith" {} + || true' in DEFAULT_PLANNER_PROMPT
    assert 'find . -type f -name "*.py" -exec grep -li -- "import" {} + || true' in DEFAULT_PLANNER_PROMPT
    assert 'rg -l -i --glob "*.txt" "vinith" .' in DEFAULT_PLANNER_PROMPT
    assert "curl -sL https://www.google.com | tr " in DEFAULT_PLANNER_PROMPT
    assert "sed -n 's:.*\\(<head[^>]*>.*</head>\\).*:\\1:p'" in DEFAULT_PLANNER_PROMPT
    assert "curl -sL https://example.com | tr " in DEFAULT_PLANNER_PROMPT
    assert "sed -n 's:.*<title[^>]*>\\([^<]*\\)</title>.*:\\1:p'" in DEFAULT_PLANNER_PROMPT
    assert "sed -n 's:.*\\(<title[^>]*>.*</title>\\).*:\\1:p'" in DEFAULT_PLANNER_PROMPT
    assert "import re; match = re.search" in DEFAULT_PLANNER_PROMPT
    assert "result = match.group(1) if match else ''" in DEFAULT_PLANNER_PROMPT
    assert 'find . -type f -exec grep -li -- \\"vinith\\" {} + || true' in file_prompt
    assert 'find . -type f -name \\"*.txt\\" -exec grep -li -- \\"vinith\\" {} + || true' in file_prompt
    assert 'find . -type f -name \\"*.py\\" -exec grep -li -- \\"import\\" {} + || true' in file_prompt
    assert 'rg -l -i --glob \\"*.txt\\" \\"vinith\\" .' in file_prompt
    assert "curl -sL https://www.google.com | tr " in file_prompt
    assert "sed -n 's:.*\\\\(<head[^>]*>.*</head>\\\\).*:\\\\1:p'" in file_prompt
    assert "curl -sL https://example.com | tr " in file_prompt
    assert "sed -n 's:.*<title[^>]*>\\\\([^<]*\\\\)</title>.*:\\\\1:p'" in file_prompt
    assert "sed -n 's:.*\\\\(<title[^>]*>.*</title>\\\\).*:\\\\1:p'" in file_prompt
    assert "import re; match = re.search" in file_prompt
    assert "result = match.group(1) if match else ''" in file_prompt

    for snippet in forbidden_snippets:
        assert snippet not in DEFAULT_PLANNER_PROMPT
        assert snippet not in file_prompt

    assert "<(curl -s" not in DEFAULT_PLANNER_PROMPT
    assert "<(curl -s" not in file_prompt


def test_classify_plan_violations_marks_modifying_sql_as_hard() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "DELETE FROM patient"},
                }
            ]
        }
    )

    violations = classify_plan_violations(plan, goal="Delete all patients from dicom")

    assert [violation.code for violation in violations.hard] == ["unsafe_sql"]
    assert violations.soft == []


def test_classify_plan_violations_marks_forbidden_python_import_as_hard() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "import os; result = 1"},
                }
            ]
        }
    )

    violations = classify_plan_violations(plan, goal="Return a number")

    assert any(violation.code == "forbidden_python_import" for violation in violations.hard)


def test_classify_plan_violations_allows_re_import() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "import re; result = bool(re.search('x', 'xyz'))"},
                }
            ]
        }
    )

    violations = classify_plan_violations(plan, goal="Return whether x exists")

    assert violations.hard == []


def test_classify_plan_violations_marks_open_as_hard() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = open('notes.txt').read()"},
                }
            ]
        }
    )

    violations = classify_plan_violations(plan, goal="Read notes.txt")

    assert any(violation.code == "forbidden_python_name" for violation in violations.hard)


def test_classify_plan_violations_marks_python_syntax_error_as_hard() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = ("},
                }
            ]
        }
    )

    violations = classify_plan_violations(plan, goal="Return a number")

    assert any(violation.code == "python_syntax_error" for violation in violations.hard)


def test_validate_plan_contract_rejects_nested_input_wrapper_access() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {
                        "inputs": {"rows": []},
                        "code": "result = inputs['rows']['count']",
                    },
                }
            ]
        }
    )

    with pytest.raises(PlanContractViolation, match="nested wrapper fields"):
        validate_plan_contract(plan, goal="Return the count")


def test_validate_plan_contract_flags_missing_result_assignment_as_soft() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "value = 1"},
                }
            ]
        }
    )

    with pytest.raises(PlanContractViolation) as exc_info:
        validate_plan_contract(plan, goal="Return the value")

    assert exc_info.value.tier == "soft"
    assert exc_info.value.code == "missing_result_assignment"


def test_validate_plan_contract_flags_unguarded_sql_row_index_as_soft() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "SELECT COUNT(*) AS patient_count FROM patient"},
                    "output": "patient_rows",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["patient_rows"],
                    "args": {
                        "inputs": {"rows": {"$ref": "patient_rows", "path": "rows"}},
                        "code": "rows = inputs['rows']; result = rows[0]['patient_count']",
                    },
                },
            ]
        }
    )

    with pytest.raises(PlanContractViolation) as exc_info:
        validate_plan_contract(plan, goal="Return the patient count")

    assert exc_info.value.tier == "soft"
    assert exc_info.value.code == "unguarded_sql_rows"


def test_planner_context_includes_database_dialect(tmp_path: Path, monkeypatch) -> None:
    planner, llm = _planner(
        tmp_path,
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {
                        "database": "dicom",
                        "query": "SELECT patient_id, name, dob FROM patient WHERE dob <= CURRENT_DATE - INTERVAL '45 years'",
                    },
                }
            ]
        },
    )

    def fake_schema(settings):
        return SchemaInfo(
            databases=[
                DatabaseSchema(
                    name="dicom",
                    dialect="postgresql",
                    tables=[
                        TableSchema(
                            name="patient",
                            columns=[ColumnSchema(name="patient_id", type="INTEGER"), ColumnSchema(name="name", type="TEXT"), ColumnSchema(name="dob", type="DATE")],
                        )
                    ],
                )
            ]
        )

    monkeypatch.setattr("aor_runtime.runtime.planner.get_schema", fake_schema)

    plan = planner.build_plan(
        goal="list all patients above 45 years of age in dicom",
        planner=_planner_config(),
        allowed_tools=["sql.query"],
        input_payload={"task": "list all patients above 45 years of age in dicom"},
    )

    assert isinstance(plan, ExecutionPlan)
    assert llm.last_user_prompt is not None
    planner_context = json.loads(llm.last_user_prompt)
    assert planner_context["schema"]["databases"][0]["name"] == "dicom"
    assert planner_context["schema"]["databases"][0]["dialect"] == "postgresql"
