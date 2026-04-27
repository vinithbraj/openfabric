from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.failure_classifier import classify_failure, generate_prompt_suggestions


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


def test_ambiguous_meeting_notes_produces_ambiguous_file_reference_suggestions() -> None:
    goal = "Read the meeting notes and return line 2."
    error_type = classify_failure(goal)
    result = generate_prompt_suggestions(goal, error_type, context={"workspace_root": "/tmp/work"})

    assert error_type == "ambiguous_file_reference"
    assert result.error_type == "ambiguous_file_reference"
    assert any("meeting_notes.txt" in suggestion.suggested_prompt for suggestion in result.suggestions)


def test_file_task_with_no_path_produces_missing_file_path_suggestions() -> None:
    goal = "Count top-level .txt files and return the count only."
    error_type = classify_failure(goal)
    result = generate_prompt_suggestions(goal, error_type, context={"workspace_root": "/tmp/work"})

    assert error_type == "missing_file_path"
    assert result.error_type == "missing_file_path"
    assert any("/tmp/work" in suggestion.suggested_prompt for suggestion in result.suggestions)


def test_vague_sql_task_produces_ambiguous_database_suggestions() -> None:
    goal = "List all studies and save to studies.txt."
    error_type = classify_failure(goal, error=RuntimeError("planner boom"), metadata={"error_source": "sql"})
    result = generate_prompt_suggestions(goal, error_type)

    assert error_type == "ambiguous_database"
    assert result.error_type == "ambiguous_database"
    assert any("database" in suggestion.suggested_prompt.lower() for suggestion in result.suggestions)


def test_mutating_slurm_prompt_produces_unsupported_mutating_operation_suggestions() -> None:
    goal = "scancel job 12345"
    error_type = classify_failure(goal)
    result = generate_prompt_suggestions(goal, error_type)

    assert error_type == "unsupported_mutating_operation"
    assert any("SLURM" in suggestion.suggested_prompt for suggestion in result.suggestions)


def test_vague_output_formatting_produces_unsupported_output_shape_suggestions() -> None:
    goal = "List txt files in /tmp/work and make it nice."
    error_type = classify_failure(goal)
    result = generate_prompt_suggestions(goal, error_type)

    assert error_type == "unsupported_output_shape"
    assert any("CSV only" in suggestion.suggested_prompt or "JSON" in suggestion.suggested_prompt for suggestion in result.suggestions)


def test_tool_unavailable_error_produces_tool_unavailable_suggestions() -> None:
    goal = "Run pdftotext on report.pdf and return the text."
    error_type = classify_failure(
        goal,
        error=RuntimeError("pdftotext: command not found"),
        metadata={"reason": "tool_execution_failed", "failed_step": "shell.exec"},
    )
    result = generate_prompt_suggestions(goal, error_type, context={"workspace_root": "/tmp/work"})

    assert error_type == "tool_unavailable"
    assert result.error_type == "tool_unavailable"
    assert any("filesystem search" in suggestion.reason.lower() or "installed" in suggestion.suggested_prompt.lower() for suggestion in result.suggestions)


def test_file_aggregate_failure_produces_relevant_suggestions() -> None:
    goal = "calculate total file size of all .mp4 files in /home/vinith/Desktop/Workspace"
    error_type = classify_failure(
        goal,
        error=RuntimeError("python.exec must not call shell.exec()."),
        metadata={"error_kind": "contract", "reason": "planning_failed"},
    )
    result = generate_prompt_suggestions(goal, error_type, context={"workspace_root": "/home/vinith/Desktop/Workspace"})

    assert error_type == "file_aggregate_not_matched"
    assert result.error_type == "file_aggregate_not_matched"
    assert any(".mp4" in suggestion.suggested_prompt for suggestion in result.suggestions)
    assert any("total file size" in suggestion.suggested_prompt.lower() or "total size" in suggestion.suggested_prompt.lower() for suggestion in result.suggestions)


def test_llm_fallback_used_produces_deterministic_rewrite_suggestions() -> None:
    goal = "List all studies and save to studies.txt."
    error_type = classify_failure(goal, metadata={"status": "completed", "llm_calls": 1})
    result = generate_prompt_suggestions(goal, error_type, context={"outputs_dir": "/tmp/work/outputs"})

    assert error_type == "llm_fallback_used"
    assert result.error_type == "llm_fallback_used"
    assert any("database" in suggestion.suggested_prompt.lower() or "csv" in suggestion.suggested_prompt.lower() for suggestion in result.suggestions)


def test_successful_deterministic_output_does_not_include_suggestions_by_default(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n")
    engine = ExecutionEngine(Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db"))

    state = engine.run_spec(
        str(SPEC_PATH),
        {"task": f"Read line 2 from {target} and return only the line."},
    )

    assert state["final_output"]["content"] == "beta"
    assert "prompt_suggestions" not in state["final_output"]["metadata"]
    assert "failure_type" not in state["final_output"]["metadata"]


def test_successful_llm_fallback_keeps_content_but_records_suggestion_metadata(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "studies.txt"
    target.write_text("study-a\nstudy-b\n")
    engine = ExecutionEngine(Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db"))

    def fake_build_plan(**kwargs):
        engine.planner.last_policies_used = ["filesystem_preference"]
        engine.planner.last_high_level_plan = None
        engine.planner.last_planning_mode = "direct"
        engine.planner.last_llm_calls = 1
        engine.planner.last_error_stage = None
        engine.planner.last_plan_repairs = []
        engine.planner.last_plan_canonicalized = False
        engine.planner.last_original_execution_plan = None
        engine.planner.last_canonicalized_execution_plan = None
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": "fs.read", "args": {"path": str(target)}},
                ]
            }
        )

    monkeypatch.setattr(engine.planner, "build_plan", fake_build_plan)

    state = engine.run_spec(str(SPEC_PATH), {"task": "List all studies and save to studies.txt."})

    assert state["final_output"]["content"] == "study-a\nstudy-b"
    assert state["final_output"]["metadata"]["failure_type"] == "llm_fallback_used"
    assert state["final_output"]["metadata"]["suggestion_count"] >= 1
