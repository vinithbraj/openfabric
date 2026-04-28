from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


class FakeLLM:
    def __init__(self) -> None:
        self.call_count = 0

    def load_prompt(self, path: str | None, fallback: str) -> str:
        return fallback

    def complete(self, **_: object) -> str:
        self.call_count += 1
        raise AssertionError("LLM should not be called for deterministic shell intelligence tests")


def _planner(tmp_path: Path) -> tuple[TaskPlanner, FakeLLM]:
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        default_node="localhost",
        enable_llm_intent_extraction=True,
    )
    llm = FakeLLM()
    return TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings), llm


def _plan(tmp_path: Path, prompt: str) -> tuple[ExecutionPlan, TaskPlanner, FakeLLM]:
    planner, llm = _planner(tmp_path)
    plan = planner.build_plan(
        goal=prompt,
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.glob", "fs.find", "fs.aggregate", "fs.search_content", "shell.exec"],
        input_payload={"task": prompt},
    )
    return plan, planner, llm


def test_folder_listing_uses_native_filesystem_tool(tmp_path: Path) -> None:
    plan, planner, llm = _plan(tmp_path, "list all files in this folder")

    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[0].args["path"] == "."
    assert plan.steps[0].args["recursive"] is False
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0


def test_directory_listing_in_current_folder_uses_native_filesystem_tool(tmp_path: Path) -> None:
    plan, planner, llm = _plan(tmp_path, "list all directories in .")

    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[0].args["path"] == "."
    assert plan.steps[0].args["file_only"] is False
    assert plan.steps[0].args["dir_only"] is True
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0


def test_directory_listing_by_path_uses_native_filesystem_tool(tmp_path: Path) -> None:
    plan, planner, llm = _plan(tmp_path, "list all folders in ~/Desktop")

    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[0].args["path"] == "~/Desktop"
    assert plan.steps[0].args["recursive"] is False
    assert plan.steps[0].args["file_only"] is False
    assert plan.steps[0].args["dir_only"] is True
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0


def test_recursive_directory_listing_by_path_uses_native_filesystem_tool(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "find folders under ~/Desktop recursively")

    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[0].args["path"] == "~/Desktop"
    assert plan.steps[0].args["recursive"] is True
    assert plan.steps[0].args["file_only"] is False
    assert plan.steps[0].args["dir_only"] is True


def test_show_files_here_uses_native_filesystem_tool(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "show files here")

    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[0].args["path"] == "."


def test_disk_usage_uses_fixed_shell_template(tmp_path: Path) -> None:
    plan, planner, _ = _plan(tmp_path, "check disk usage")

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "df -h"
    assert planner.last_planning_mode == "deterministic_intent"


def test_port_usage_uses_fixed_shell_template(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "what process is using port 8310")

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "lsof -i :8310"


def test_explicit_read_only_command_uses_shell_exec(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "run ls -la")

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "ls -la"


def test_explicit_destructive_command_is_refused_before_execution(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "run rm -rf /")

    assert [step.action for step in plan.steps] == ["runtime.return"]
    assert "Request Not Executed" in plan.steps[0].args["value"]
    assert "shell.exec" not in [step.action for step in plan.steps]


def test_natural_language_mutation_is_refused_before_execution(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path, "kill process 123")

    assert [step.action for step in plan.steps] == ["runtime.return"]
    assert "Request Not Executed" in plan.steps[0].args["value"]
