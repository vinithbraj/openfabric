from __future__ import annotations

from pathlib import Path

import pytest

from gateway_agent.remote_runner import RemoteToolError, run_remote_operation


def test_remote_runner_lists_directory_entries(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    result = run_remote_operation("filesystem.list_directory", {"path": "."}, workspace_root=tmp_path)

    assert result["status"] == "success"
    names = {entry["name"] for entry in result["data_preview"]["entries"]}
    assert {"README.md", "subdir"} <= names


def test_remote_runner_reads_file_and_blocks_secrets(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello world", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")

    result = run_remote_operation(
        "filesystem.read_file",
        {"path": "README.md", "max_bytes": 5},
        workspace_root=tmp_path,
    )

    assert result["data_preview"]["content_preview"] == "hello"
    assert result["data_preview"]["truncated"] is True

    with pytest.raises(RemoteToolError):
        run_remote_operation("filesystem.read_file", {"path": ".env"}, workspace_root=tmp_path)


def test_remote_runner_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(RemoteToolError):
        run_remote_operation("filesystem.list_directory", {"path": "../outside"}, workspace_root=tmp_path)


def test_remote_runner_searches_files(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("print(1)", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.py").write_text("print(2)", encoding="utf-8")

    result = run_remote_operation(
        "filesystem.search_files",
        {"path": ".", "pattern": "*.py", "recursive": True},
        workspace_root=tmp_path,
    )

    assert result["data_preview"]["matches"] == ["nested/two.py", "one.py"]


def test_remote_runner_inspects_system_with_allowlisted_scope() -> None:
    result = run_remote_operation("shell.inspect_system", {"scope": "hostname"})

    assert result["status"] == "success"
    assert result["data_preview"]["scope"] == "hostname"
    assert isinstance(result["data_preview"]["facts"], list)
    assert result["metadata"]["fact_count"] >= 1
