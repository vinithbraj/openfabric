from __future__ import annotations

import socket
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


def test_remote_runner_lists_processes_and_checks_ports() -> None:
    process_result = run_remote_operation("shell.list_processes", {"pattern": "python", "limit": 5})

    assert process_result["status"] == "success"
    assert isinstance(process_result["data_preview"]["processes"], list)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        port = server.getsockname()[1]
        port_result = run_remote_operation("shell.check_port", {"port": port})
    finally:
        server.close()

    assert port_result["status"] == "success"
    assert port_result["data_preview"]["port"] == port
    assert port_result["metadata"]["listener_count"] >= 1


def test_remote_runner_rejects_arbitrary_shell_text_and_unknown_args() -> None:
    with pytest.raises(RemoteToolError):
        run_remote_operation("shell.which", {"program": "python3; rm -rf /"})

    with pytest.raises(RemoteToolError):
        run_remote_operation("shell.which", {"program": "python3", "command": "rm -rf /"})


def test_remote_runner_reads_pwd_and_git_status(tmp_path: Path) -> None:
    pwd_result = run_remote_operation("shell.pwd", {}, workspace_root=tmp_path)
    assert pwd_result["data_preview"]["cwd"] == str(tmp_path)

    result = run_remote_operation("shell.git_status", {"path": "."}, workspace_root=Path.cwd())
    assert result["status"] == "success"
    assert "status_lines" in result["data_preview"]
