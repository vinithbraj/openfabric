from __future__ import annotations

import os
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.executor import summarize_final_output
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry


def test_fs_aggregate_recursive_total_size(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    (tmp_path / "clip1.mp4").write_text("aaaa")
    (tmp_path / "clip2.mp4").write_text("bb")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "clip3.mp4").write_text("hello")

    result = tools.invoke("fs.aggregate", {"path": ".", "pattern": "*.mp4"})

    assert result["file_count"] == 3
    assert result["total_size_bytes"] == 11
    assert result["summary_text"] == "3 files, 11 bytes"


def test_fs_aggregate_non_recursive_total_size(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    (tmp_path / "clip1.mp4").write_text("aaaa")
    (tmp_path / "clip2.mp4").write_text("bb")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "clip3.mp4").write_text("hello")

    result = tools.invoke("fs.aggregate", {"path": ".", "pattern": "*.mp4", "recursive": False})

    assert result["file_count"] == 2
    assert result["total_size_bytes"] == 6
    assert [entry["relative_path"] for entry in result["matches"]] == ["clip1.mp4", "clip2.mp4"]


def test_fs_aggregate_pattern_excludes_other_extensions(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    (tmp_path / "clip1.mp4").write_text("aaaa")
    (tmp_path / "notes.txt").write_text("ignore")

    result = tools.invoke("fs.aggregate", {"path": ".", "pattern": "*.mp4"})

    assert result["file_count"] == 1
    assert result["total_size_bytes"] == 4
    assert [entry["name"] for entry in result["matches"]] == ["clip1.mp4"]


def test_fs_aggregate_empty_matches_returns_zeroes(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    (tmp_path / "notes.txt").write_text("ignore")

    result = tools.invoke("fs.aggregate", {"path": ".", "pattern": "*.mp4"})

    assert result["file_count"] == 0
    assert result["total_size_bytes"] == 0
    assert result["matches"] == []
    assert result["summary_text"] == "0 files, 0 bytes"


def test_fs_aggregate_expands_tilde_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    home_root = tmp_path / "videos"
    home_root.mkdir()
    (home_root / "clip1.mp4").write_text("aaaa")

    settings = Settings(workspace_root=tmp_path / "workspace", run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)

    result = tools.invoke("fs.aggregate", {"path": "~/videos", "pattern": "*.mp4"})

    assert result["file_count"] == 1
    assert result["total_size_bytes"] == 4
    assert result["path"] == str(home_root.resolve())


def test_fs_aggregate_matches_are_stably_sorted(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    (tmp_path / "zeta.mp4").write_text("z")
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "beta.mp4").write_text("bb")
    (tmp_path / "mid.mp4").write_text("mmm")

    result = tools.invoke("fs.aggregate", {"path": ".", "pattern": "*.mp4"})

    assert [entry["relative_path"] for entry in result["matches"]] == ["alpha/beta.mp4", "mid.mp4", "zeta.mp4"]


def test_fs_aggregate_skips_symlinks_that_resolve_outside_root(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.mp4").write_text("aaaa")
    outside = tmp_path / "outside.mp4"
    outside.write_text("bbbbb")
    os.symlink(outside, root / "linked-outside.mp4")

    result = tools.invoke("fs.aggregate", {"path": "root", "pattern": "*.mp4"})

    assert result["file_count"] == 1
    assert result["total_size_bytes"] == 4
    assert [entry["relative_path"] for entry in result["matches"]] == ["inside.mp4"]


def test_validator_checks_fs_aggregate_against_filesystem(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    tools = build_tool_registry(settings)
    validator = RuntimeValidator(settings)
    (tmp_path / "clip1.mp4").write_text("aaaa")
    (tmp_path / "clip2.mp4").write_text("bb")
    step = ExecutionStep(id=1, action="fs.aggregate", args={"path": ".", "pattern": "*.mp4"})
    log = StepLog(step=step, result=tools.invoke("fs.aggregate", {"path": ".", "pattern": "*.mp4"}), success=True)

    validation, checks = validator.validate([log], goal="calculate total file size of all .mp4 files in .")

    assert validation.success is True
    assert checks[0]["detail"] == "file aggregate matches filesystem"


def test_fs_aggregate_final_output_uses_summary_text(tmp_path: Path) -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="fs.aggregate", args={"path": ".", "pattern": "*.mp4"}),
        result={
            "path": str(tmp_path),
            "pattern": "*.mp4",
            "recursive": True,
            "file_count": 3,
            "total_size_bytes": 11,
            "matches": [],
            "summary_text": "3 files, 11 bytes",
            "display_size": "11 bytes",
        },
        success=True,
    )

    output = summarize_final_output("calculate total file size of all .mp4 files in .", [log])

    assert output["content"] == "3 files, 11 bytes"
