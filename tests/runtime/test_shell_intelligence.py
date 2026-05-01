from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import PlannerConfig
from aor_runtime.runtime.planner import ACTIVE_PLANNING_MODE, TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.call_count = 0

    def load_prompt(self, path: str | None, fallback: str) -> str:
        raise AssertionError("Legacy planner prompt should not be loaded")

    def complete(self, **_: object) -> str:
        self.call_count += 1
        if not self.responses:
            raise AssertionError("LLM called more times than expected")
        return self.responses.pop(0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        default_node="localhost",
        semantic_frame_mode="off",
    )


def _shell_plan(command: str) -> dict:
    return {
        "goal": "Run shell inspection.",
        "actions": [
            {
                "id": "inspect",
                "tool": "shell.exec",
                "purpose": "Inspect the local system.",
                "inputs": {"command": command},
                "output_binding": "inspection",
                "expected_result_shape": {"kind": "text"},
            },
            {
                "id": "return_result",
                "tool": "runtime.return",
                "purpose": "Return shell output.",
                "inputs": {"value": {"$ref": "inspection", "path": "stdout"}, "mode": "text"},
                "depends_on": ["inspect"],
                "output_binding": "runtime_return_result",
                "expected_result_shape": {"kind": "text"},
            },
        ],
        "expected_final_shape": {"kind": "text"},
        "notes": [],
    }


def _fs_glob_plan(path: str = ".") -> dict:
    return {
        "goal": "List files.",
        "actions": [
            {
                "id": "list_files",
                "tool": "fs.glob",
                "purpose": "List entries.",
                "inputs": {"path": path, "pattern": "*", "recursive": False},
                "output_binding": "entries",
                "expected_result_shape": {"kind": "table"},
            },
            {
                "id": "return_result",
                "tool": "runtime.return",
                "purpose": "Return entries.",
                "inputs": {"value": {"$ref": "entries", "path": "matches"}, "mode": "markdown"},
                "depends_on": ["list_files"],
                "output_binding": "runtime_return_result",
                "expected_result_shape": {"kind": "table"},
            },
        ],
        "expected_final_shape": {"kind": "table"},
        "notes": [],
    }


def _plan(tmp_path: Path, prompt: str, response: dict):
    settings = _settings(tmp_path)
    llm = FakeLLM([response])
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    plan = planner.build_plan(
        goal=prompt,
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.glob", "fs.find", "fs.aggregate", "fs.search_content", "shell.exec"],
        input_payload={"task": prompt},
    )
    return plan, planner, llm


def test_filesystem_listing_uses_action_planner_when_semantic_frame_disabled(tmp_path: Path) -> None:
    plan, planner, llm = _plan(tmp_path, "list all files in this folder", _fs_glob_plan())

    assert [step.action for step in plan.steps] == ["fs.glob", "text.format", "runtime.return"]
    assert plan.steps[0].args["path"] == "."
    assert planner.last_planning_mode == ACTIVE_PLANNING_MODE
    assert planner.last_capability_name == "action_planner"
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1


def test_disk_usage_still_gets_storage_command_canonicalization(tmp_path: Path) -> None:
    plan, planner, _ = _plan(tmp_path, "check disk usage", _shell_plan("ls"))

    assert [step.action for step in plan.steps] == ["shell.exec", "text.format", "runtime.return"]
    assert plan.steps[0].args["command"] == "df -h /"
    assert planner.last_planning_mode == ACTIVE_PLANNING_MODE


def test_explicit_read_only_command_uses_shell_exec(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "run ls -la", _shell_plan("ls -la"))

    assert [step.action for step in plan.steps] == ["shell.exec", "text.format", "runtime.return"]
    assert plan.steps[0].args["command"] == "ls -la"


def test_destructive_shell_action_is_rejected_by_validator(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM([_shell_plan("rm -rf /")])
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    try:
        planner.build_plan(
            goal="run rm -rf /",
            planner=PlannerConfig(temperature=0.0),
            allowed_tools=["shell.exec"],
            input_payload={"task": "run rm -rf /"},
        )
    except ValueError as exc:
        assert "Unsafe shell command" in str(exc) or "forbidden" in str(exc).lower()
    else:
        raise AssertionError("Expected destructive shell command to be rejected")
