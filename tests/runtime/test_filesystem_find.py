from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.executor import summarize_final_output
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry


def test_fs_find_returns_recursive_relative_matches(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    (tmp_path / "root.txt").write_text("root")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("child")
    (tmp_path / "nested" / "ignore.md").write_text("ignore")

    result = tools.invoke("fs.find", {"path": ".", "pattern": "*.txt"})

    assert result["matches"] == ["nested/child.txt", "root.txt"]


def test_validator_checks_fs_find_against_filesystem(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    validator = RuntimeValidator(settings)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("child")
    step = ExecutionStep(id=1, action="fs.find", args={"path": ".", "pattern": "*.txt"})
    log = StepLog(step=step, result=tools.invoke("fs.find", {"path": ".", "pattern": "*.txt"}), success=True)

    validation, checks = validator.validate([log], goal="find all txt files")

    assert validation.success is True
    assert checks[0]["detail"] == "file search matches filesystem"


def test_fs_find_final_output_lists_matches(tmp_path: Path) -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="fs.find", args={"path": ".", "pattern": "*.txt"}),
        result={"path": str(tmp_path), "pattern": "*.txt", "matches": ["nested/child.txt", "root.txt"]},
        success=True,
    )

    output = summarize_final_output("find all txt files", [log])

    assert output["content"] == "nested/child.txt\nroot.txt"
