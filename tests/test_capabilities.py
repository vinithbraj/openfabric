from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.capabilities.filesystem import (
    ListDirectoryCapability,
    ReadFileCapability,
    SearchFilesCapability,
)
from agent_runtime.capabilities.markdown import MarkdownRenderCapability
from agent_runtime.capabilities.python_data import TransformTableCapability
from agent_runtime.capabilities.shell import ShellInspectCapability
from agent_runtime.capabilities.sql import ReadQueryCapability
from agent_runtime.core.errors import SafetyError, ValidationError
from agent_runtime.execution.result_store import InMemoryResultStore
from gateway_agent.remote_runner import RemoteToolError, run_remote_operation


def _context(tmp_path: Path, store: InMemoryResultStore | None = None) -> dict[str, object]:
    return {
        "node_id": "node-test",
        "execution_context": {"workspace_root": str(tmp_path)},
        "result_store": store or InMemoryResultStore(),
    }


def test_filesystem_list_directory_runs_through_gateway_runner(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with pytest.raises(SafetyError):
        ListDirectoryCapability().execute({"path": "."}, _context(tmp_path))

    result = run_remote_operation("filesystem.list_directory", {"path": "."}, workspace_root=tmp_path)
    entries = result["data_preview"]["entries"]
    assert {entry["name"] for entry in entries} >= {"a.txt", "subdir"}
    assert {"name", "path", "type", "size", "modified_time"} <= set(entries[0])


def test_filesystem_read_file_runs_through_gateway_runner(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello world", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("SECRET=1", encoding="utf-8")

    with pytest.raises(SafetyError):
        ReadFileCapability().execute({"path": "README.md", "max_bytes": 5}, _context(tmp_path))

    result = run_remote_operation(
        "filesystem.read_file",
        {"path": "README.md", "max_bytes": 5},
        workspace_root=tmp_path,
    )

    assert result["data_preview"]["content_preview"] == "hello"
    assert result["data_preview"]["truncated"] is True

    with pytest.raises(RemoteToolError):
        run_remote_operation("filesystem.read_file", {"path": ".env"}, workspace_root=tmp_path)


def test_filesystem_search_files_runs_through_gateway_runner(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "two.txt").write_text("text", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "three.py").write_text("print(3)", encoding="utf-8")

    with pytest.raises(SafetyError):
        SearchFilesCapability().execute(
            {"path": ".", "pattern": "*.py", "recursive": True},
            _context(tmp_path),
        )

    result = run_remote_operation(
        "filesystem.search_files",
        {"path": ".", "pattern": "*.py", "recursive": True},
        workspace_root=tmp_path,
    )

    assert result["data_preview"]["matches"] == ["nested/three.py", "one.py"]


def test_shell_capability_runs_through_gateway_runner() -> None:
    with pytest.raises(SafetyError):
        ShellInspectCapability().execute({"scope": "hostname"}, {"node_id": "node-shell"})

    result = run_remote_operation("shell.inspect_system", {"scope": "hostname"})

    assert result["status"] == "success"
    assert result["data_preview"]["scope"] == "hostname"
    assert isinstance(result["data_preview"]["facts"], list)


def test_sql_read_query_accepts_structured_read_only_intent() -> None:
    result = ReadQueryCapability().execute(
        {
            "query_intent": {
                "template": "count_rows",
                "table": "patients",
                "filters": {"status": "active"},
            }
        },
        {"node_id": "node-sql"},
    )

    assert result.status == "success"
    assert result.data_preview["query_plan"]["template"] == "count_rows"
    assert result.data_preview["rows"] == []

    with pytest.raises(ValidationError):
        ReadQueryCapability().execute(
            {"query_intent": {"sql": "SELECT * FROM patients"}},
            {"node_id": "node-sql"},
        )


def test_python_data_transform_table_applies_deterministic_operations(tmp_path: Path) -> None:
    store = InMemoryResultStore()
    input_ref = store.put(
        [
            {"name": "b", "value": 2},
            {"name": "a", "value": 1},
            {"name": "c", "value": 3},
        ]
    )
    context = _context(tmp_path, store)

    head_result = TransformTableCapability().execute(
        {"input_ref": input_ref, "operation": "head", "parameters": {"count": 2}},
        context,
    )
    assert head_result.data_preview["row_count"] == 2

    aggregate_result = TransformTableCapability().execute(
        {
            "input_ref": input_ref,
            "operation": "aggregate",
            "parameters": {"metric": "sum", "column": "value"},
        },
        context,
    )
    assert aggregate_result.data_preview["summary"]["value"] == 6


def test_markdown_render_outputs_markdown_from_input_ref(tmp_path: Path) -> None:
    store = InMemoryResultStore()
    input_ref = store.put(
        {"rows": [{"name": "alice", "score": 10}, {"name": "bob", "score": 8}]}
    )
    context = _context(tmp_path, store)

    table_result = MarkdownRenderCapability().execute(
        {
            "input_ref": input_ref,
            "render_type": "table",
            "parameters": {"title": "Scores"},
        },
        context,
    )
    assert "| name | score |" in table_result.data_preview["markdown"]

    list_result = MarkdownRenderCapability().execute(
        {
            "input_ref": input_ref,
            "render_type": "summary",
            "parameters": {"title": "Summary"},
        },
        context,
    )
    assert "## Summary" in list_result.data_preview["markdown"]
