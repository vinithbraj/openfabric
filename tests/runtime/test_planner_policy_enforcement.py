from __future__ import annotations

import json
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.runtime.policies import validate_plan_efficiency
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def load_prompt(self, path: str | None, fallback: str) -> str:
        return fallback

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.payload


def _planner(tmp_path: Path, payload: dict) -> tuple[TaskPlanner, FakeLLM]:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    llm = FakeLLM(payload)
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
