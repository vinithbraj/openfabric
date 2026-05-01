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
from agent_runtime.capabilities.sql import ReadQueryCapability
from agent_runtime.core.errors import ValidationError
from agent_runtime.execution.result_store import InMemoryResultStore


def _context(tmp_path: Path, store: InMemoryResultStore | None = None) -> dict[str, object]:
    return {
        "node_id": "node-test",
        "execution_context": {"workspace_root": str(tmp_path)},
        "result_store": store or InMemoryResultStore(),
    }


def test_filesystem_list_directory_returns_entry_records(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    result = ListDirectoryCapability().execute({"path": "."}, _context(tmp_path))

    assert result.status == "success"
    entries = result.data_preview["entries"]
    assert {entry["name"] for entry in entries} >= {"a.txt", "subdir"}
    assert {"name", "path", "type", "size", "modified_time"} <= set(entries[0])


def test_filesystem_read_file_truncates_and_blocks_secrets(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("hello world", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("SECRET=1", encoding="utf-8")

    result = ReadFileCapability().execute({"path": "README.md", "max_bytes": 5}, _context(tmp_path))

    assert result.data_preview["content_preview"] == "hello"
    assert result.data_preview["truncated"] is True

    with pytest.raises(ValidationError):
        ReadFileCapability().execute({"path": ".env"}, _context(tmp_path))


def test_filesystem_search_files_matches_pattern(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "two.txt").write_text("text", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "three.py").write_text("print(3)", encoding="utf-8")

    result = SearchFilesCapability().execute(
        {"path": ".", "pattern": "*.py", "recursive": True},
        _context(tmp_path),
    )

    assert result.status == "success"
    assert result.data_preview["matches"] == ["nested/three.py", "one.py"]


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
