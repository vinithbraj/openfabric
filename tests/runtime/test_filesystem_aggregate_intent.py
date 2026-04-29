from __future__ import annotations

from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.intent_classifier import classify_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import FileAggregateIntent


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


def test_classifier_matches_total_file_size_prompt() -> None:
    result = classify_intent("calculate total file size of all .mp4 files in /tmp/example")

    assert result.matched is True
    assert isinstance(result.intent, FileAggregateIntent)
    assert result.intent.path == "/tmp/example"
    assert result.intent.pattern == "*.mp4"
    assert result.intent.aggregate == "total_size"
    assert result.intent.output_mode == "text"
    assert result.intent.size_unit == "auto"


def test_classifier_matches_how_much_space_prompt() -> None:
    result = classify_intent("how much space do mp4 files take under /tmp/example")

    assert result.matched is True
    assert isinstance(result.intent, FileAggregateIntent)
    assert result.intent.pattern == "*.mp4"
    assert result.intent.recursive is True


def test_shell_framed_prompt_with_files_in_path_does_not_false_match_deterministically() -> None:
    result = classify_intent("using shell, how much space do mp4s take under /tmp/files/example")

    assert result.matched is False


def test_classifier_matches_total_bytes_bare_extension_prompt() -> None:
    result = classify_intent("total bytes for mp4 files under /tmp/example")

    assert result.matched is True
    assert isinstance(result.intent, FileAggregateIntent)
    assert result.intent.pattern == "*.mp4"
    assert result.intent.size_unit == "bytes"


def test_classifier_matches_count_and_total_size_prompt() -> None:
    result = classify_intent("count and total size of mp4 files in /tmp/example as json")

    assert result.matched is True
    assert isinstance(result.intent, FileAggregateIntent)
    assert result.intent.aggregate == "count_and_total_size"
    assert result.intent.output_mode == "json"


def test_classifier_distinguishes_top_level_vs_recursive() -> None:
    top_level = classify_intent("total size of top-level mp4 files in /tmp/example")
    recursive = classify_intent("recursively total size of mp4 files under /tmp/example")

    assert isinstance(top_level.intent, FileAggregateIntent)
    assert isinstance(recursive.intent, FileAggregateIntent)
    assert top_level.intent.recursive is False
    assert recursive.intent.recursive is True


def test_compile_file_aggregate_uses_fs_aggregate_and_runtime_return() -> None:
    settings = Settings(workspace_root=Path("/tmp"), run_store_path=Path("/tmp/runtime.db"))
    plan = compile_intent_to_plan(
        FileAggregateIntent(path="/tmp/example", pattern="*.mp4", recursive=True),
        ["fs.aggregate"],
        settings,
    )

    assert [step.action for step in plan.steps] == ["fs.aggregate", "runtime.return"]
    assert plan.steps[0].args["include_matches"] is False
    assert plan.steps[1].args["value"]["path"] == "summary_text"


def test_compile_count_and_total_size_json_keeps_matches() -> None:
    settings = Settings(workspace_root=Path("/tmp"), run_store_path=Path("/tmp/runtime.db"))
    plan = compile_intent_to_plan(
        FileAggregateIntent(
            path="/tmp/example",
            pattern="*.mp4",
            recursive=True,
            aggregate="count_and_total_size",
            output_mode="json",
        ),
        ["fs.aggregate"],
        settings,
    )

    assert [step.action for step in plan.steps] == ["fs.aggregate", "runtime.return"]
    assert plan.steps[0].args["include_matches"] is True
    assert "python.exec" not in [step.action for step in plan.steps]
    assert "shell.exec" not in [step.action for step in plan.steps]


@pytest.mark.skip(reason="LLM-exclusive runtime no longer supports zero-LLM natural-language execution")
def test_supported_prompt_uses_zero_llm_calls(tmp_path: Path) -> None:
    (tmp_path / "clip1.mp4").write_text("aaaa")
    (tmp_path / "clip2.mp4").write_text("bb")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "clip3.mp4").write_text("hello")
    engine = ExecutionEngine(Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", response_render_mode="raw"))

    state = engine.run_spec(
        str(SPEC_PATH),
        {"task": f"calculate total file size of all .mp4 files in {tmp_path}"},
    )

    assert state["metrics"]["llm_calls"] == 0
    assert state["final_output"]["content"] == "3 files, 11 bytes"
    planner_steps = state["plan"]["steps"]
    assert [step["action"] for step in planner_steps] == ["fs.aggregate", "runtime.return"]
