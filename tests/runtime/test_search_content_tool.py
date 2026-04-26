from __future__ import annotations

import os
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.executor import summarize_final_output
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")


def test_recursive_txt_search_finds_nested_txt_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "top.txt").write_text("cinnamon root\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("hello\ncinnamon nested\n")
    (tmp_path / "nested" / "ignore.md").write_text("cinnamon markdown\n")

    result = tools.invoke("fs.search_content", {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "recursive": True})

    assert result["matches"] == ["nested/child.txt", "top.txt"]
    assert [entry["relative_path"] for entry in result["entries"]] == ["nested/child.txt", "top.txt"]


def test_top_level_txt_search_excludes_nested_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "top.txt").write_text("cinnamon root\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("cinnamon nested\n")

    result = tools.invoke("fs.search_content", {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "recursive": False})

    assert result["matches"] == ["top.txt"]


def test_pattern_txt_excludes_markdown_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "keep.txt").write_text("cinnamon\n")
    (tmp_path / "skip.md").write_text("cinnamon\n")

    result = tools.invoke("fs.search_content", {"path": ".", "needle": "cinnamon", "pattern": "*.txt"})

    assert result["matches"] == ["keep.txt"]


def test_path_style_name_returns_basenames(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("cinnamon\n")

    result = tools.invoke(
        "fs.search_content",
        {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "recursive": True, "path_style": "name"},
    )

    assert result["matches"] == ["child.txt"]


def test_path_style_relative_returns_relative_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("cinnamon\n")

    result = tools.invoke(
        "fs.search_content",
        {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "recursive": True, "path_style": "relative"},
    )

    assert result["matches"] == ["nested/child.txt"]


def test_path_style_absolute_returns_absolute_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    target = tmp_path / "nested" / "child.txt"
    target.parent.mkdir()
    target.write_text("cinnamon\n")

    result = tools.invoke(
        "fs.search_content",
        {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "recursive": True, "path_style": "absolute"},
    )

    assert result["matches"] == [str(target)]


def test_case_insensitive_search_matches_different_casing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "spice.txt").write_text("Cinnamon roll\n")

    result = tools.invoke(
        "fs.search_content",
        {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "case_insensitive": True},
    )

    assert result["matches"] == ["spice.txt"]


def test_unreadable_or_binary_like_files_do_not_crash(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    readable = tmp_path / "good.txt"
    readable.write_text("cinnamon\n")
    binary_file = tmp_path / "bad.bin"
    binary_file.write_bytes(b"\x00\xff\x00cinnamon")
    unreadable = tmp_path / "locked.txt"
    unreadable.write_text("cinnamon\n")

    original_mode = unreadable.stat().st_mode
    try:
        if os.name == "posix" and os.geteuid() != 0:
            unreadable.chmod(0)
        result = tools.invoke("fs.search_content", {"path": ".", "needle": "cinnamon", "recursive": True})
    finally:
        if unreadable.exists():
            unreadable.chmod(original_mode)

    assert "good.txt" in result["matches"]
    assert "bad.bin" not in result["matches"]
    if os.name == "posix" and os.geteuid() != 0:
        assert "locked.txt" not in result["matches"]


def test_matched_lines_include_line_number_and_text(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "story.txt").write_text("hello\ncinnamon toast\nbye\ncinnamon tea\n")

    result = tools.invoke("fs.search_content", {"path": ".", "needle": "cinnamon", "pattern": "*.txt"})

    assert result["entries"][0]["matched_lines"] == [
        {"line_number": 2, "text": "cinnamon toast"},
        {"line_number": 4, "text": "cinnamon tea"},
    ]


def test_max_matches_limits_number_of_matching_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    (tmp_path / "a.txt").write_text("cinnamon\n")
    (tmp_path / "b.txt").write_text("cinnamon\n")
    (tmp_path / "c.txt").write_text("cinnamon\n")

    result = tools.invoke(
        "fs.search_content",
        {"path": ".", "needle": "cinnamon", "pattern": "*.txt", "max_matches": 2},
    )

    assert result["matches"] == ["a.txt", "b.txt"]
    assert [entry["relative_path"] for entry in result["entries"]] == ["a.txt", "b.txt"]


def test_validator_checks_search_content_against_filesystem(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = build_tool_registry(settings)
    validator = RuntimeValidator(settings)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child.txt").write_text("cinnamon\n")
    step = ExecutionStep(id=1, action="fs.search_content", args={"path": ".", "needle": "cinnamon", "pattern": "*.txt"})
    log = StepLog(step=step, result=tools.invoke("fs.search_content", {"path": ".", "needle": "cinnamon", "pattern": "*.txt"}), success=True)

    validation, checks = validator.validate([log], goal="find txt files containing cinnamon")

    assert validation.success is True
    assert checks[0]["detail"] == "search content matches filesystem"


def test_search_content_final_output_lists_matches(tmp_path: Path) -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="fs.search_content", args={"path": ".", "needle": "cinnamon", "pattern": "*.txt"}),
        result={
            "path": str(tmp_path),
            "needle": "cinnamon",
            "pattern": "*.txt",
            "recursive": True,
            "matches": ["nested/child.txt", "top.txt"],
            "entries": [],
        },
        success=True,
    )

    output = summarize_final_output("find txt files containing cinnamon", [log])

    assert output["content"] == "nested/child.txt\ntop.txt"
