from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.planner import ACTIVE_PLANNING_MODE, DEFAULT_PLANNER_PROMPT, TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.loaded_prompts: list[str | None] = []
        self.user_prompts: list[str] = []

    def load_prompt(self, path: str | None, fallback: str) -> str:
        self.loaded_prompts.append(path)
        raise AssertionError("Legacy prompt files must not be loaded")

    def complete(self, *, system_prompt: str, user_prompt: str, **_: object) -> str:
        self.user_prompts.append(user_prompt)
        if not self.responses:
            raise AssertionError("LLM called more times than expected")
        return self.responses.pop(0)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def _read_file_action_plan(path: str = "notes.txt") -> dict:
    return {
        "goal": f"Read {path}",
        "actions": [
            {
                "id": "read_file",
                "tool": "fs.read",
                "purpose": "Read the requested file.",
                "inputs": {"path": path},
                "output_binding": "file_content",
                "expected_result_shape": {"kind": "text"},
            },
            {
                "id": "return_result",
                "tool": "runtime.return",
                "purpose": "Return the file content.",
                "inputs": {"value": {"$ref": "file_content", "path": "content"}, "mode": "text"},
                "depends_on": ["read_file"],
                "output_binding": "runtime_return_result",
                "expected_result_shape": {"kind": "text"},
            },
        ],
        "expected_final_shape": {"kind": "text"},
        "notes": [],
    }


def test_task_planner_always_uses_validator_enforced_action_planner(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        action_planner_enabled=False,
        legacy_execution_planner_enabled=True,
    )
    llm = FakeLLM([_read_file_action_plan()])
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="Read notes.txt",
        planner=PlannerConfig(prompt="legacy.txt", decomposer_prompt="legacy_decomposer.txt"),
        allowed_tools=["fs.read"],
        input_payload={"task": "Read notes.txt"},
    )

    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == ACTIVE_PLANNING_MODE
    assert planner.last_policies_used == [ACTIVE_PLANNING_MODE]
    assert planner.last_capability_name == "action_planner"
    assert planner.last_llm_calls == 1
    assert planner.last_raw_planner_llm_calls == 0
    assert llm.loaded_prompts == []


def test_legacy_prompt_files_are_retired_notes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    planner_prompt = (repo_root / "prompts" / "planner_system.txt").read_text()
    decomposer_prompt = (repo_root / "prompts" / "decomposer_system.txt").read_text()

    assert "Legacy planner prompt retired" in planner_prompt
    assert "Legacy decomposer prompt retired" in decomposer_prompt
    assert "ExecutionPlan schema" not in planner_prompt
    assert "HighLevelPlan schema" not in decomposer_prompt


def test_embedded_legacy_prompt_is_retired() -> None:
    assert "Legacy raw ExecutionPlan prompt retired" in DEFAULT_PLANNER_PROMPT
    assert "Use python.exec only" not in DEFAULT_PLANNER_PROMPT
