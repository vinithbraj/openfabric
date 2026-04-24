from __future__ import annotations

import json
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.planner import DEFAULT_PLANNER_PROMPT, TaskPlanner
from aor_runtime.runtime.policies import validate_plan_efficiency
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self, raw_response: str) -> None:
        self.raw_response = raw_response
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
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.raw_response


def _planner(tmp_path: Path, raw_response: str | dict) -> tuple[TaskPlanner, FakeLLM]:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    serialized = json.dumps(raw_response) if isinstance(raw_response, dict) else raw_response
    llm = FakeLLM(serialized)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    return planner, llm


def _planner_with_settings(tmp_path: Path, raw_response: str | dict, **settings_overrides) -> tuple[TaskPlanner, FakeLLM]:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **settings_overrides)
    serialized = json.dumps(raw_response) if isinstance(raw_response, dict) else raw_response
    llm = FakeLLM(serialized)
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


def test_planner_rejects_plan_that_exceeds_step_limit(tmp_path: Path) -> None:
    planner, _ = _planner(
        tmp_path,
        {
            "steps": [
                {"id": index, "action": "fs.exists", "args": {"path": f"item_{index}.txt"}}
                for index in range(1, 10)
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
                {"id": 1, "action": "python.exec", "args": {"code": "result = {'value': 1}"}},
                {"id": 2, "action": "python.exec", "args": {"code": "result = {'value': 2}"}},
            ]
        },
    )

    with pytest.raises(ValueError, match="Too many python\\.exec steps"):
        planner.build_plan(
            goal="Run multiple loops over local data",
            planner=_planner_config(),
            allowed_tools=["python.exec"],
            input_payload={"task": "Run multiple loops over local data"},
        )


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
        {
            "steps": [
                {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.txt"}},
                {
                    "id": 2,
                    "action": "python.exec",
                    "args": {"code": "matches = fs.find('.', '*.txt'); result = {'csv': ','.join(matches)}"},
                },
            ]
        },
    )

    plan = planner.build_plan(
        goal="find all *.txt files in this folder and provide list as csv",
        planner=_planner_config(),
        allowed_tools=["fs.find", "python.exec"],
        input_payload={"task": "find all *.txt files in this folder and provide list as csv"},
    )

    assert [step.action for step in plan.steps] == ["fs.find", "python.exec"]
    assert llm.last_system_prompt is not None
    assert "Use fs.find for recursive file discovery" in llm.last_system_prompt


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


def test_prompt_sources_include_shell_helper_and_json_literal_rules() -> None:
    file_prompt = (Path(__file__).resolve().parents[2] / "prompts" / "planner_system.txt").read_text()

    required_snippets = [
        "If the task explicitly names a node, include that node in shell.exec args.",
        "Never invent node names outside the provided logical node list.",
        "shell.exec(...) returns an object with stdout, stderr, and returncode fields",
        "Every args value must be valid JSON as written.",
        "Prefer a direct shell.exec step for simple command-output formatting tasks.",
        "Use fs.not_exists to verify that a path is absent after deletion or cleanup.",
    ]

    for snippet in required_snippets:
        assert snippet in DEFAULT_PLANNER_PROMPT
        assert snippet in file_prompt
