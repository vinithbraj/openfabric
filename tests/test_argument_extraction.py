from __future__ import annotations

from typing import Any

from agent_runtime.capabilities import (
    CapabilityRegistry,
    ListDirectoryCapability,
    MarkdownRenderCapability,
    ReadFileCapability,
    ReadQueryCapability,
    SearchFilesCapability,
    TransformTableCapability,
)
from agent_runtime.core.types import CapabilityRef, TaskFrame
from agent_runtime.input_pipeline.argument_extraction import (
    ArgumentExtractionResult,
    extract_arguments,
)
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult


class FakeLLMClient:
    """Fake structured LLM client for argument extraction tests."""

    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.last_prompt = ""
        self.last_schema: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_prompt = prompt
        self.last_schema = schema
        for marker, payload in self.payloads.items():
            if marker in prompt:
                return dict(payload)
        raise AssertionError(f"no fake payload configured for prompt: {prompt}")


def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(ListDirectoryCapability())
    registry.register(ReadFileCapability())
    registry.register(SearchFilesCapability())
    registry.register(ReadQueryCapability())
    registry.register(TransformTableCapability())
    registry.register(MarkdownRenderCapability())
    return registry


def _task(
    task_id: str,
    description: str,
    semantic_verb: str,
    object_type: str,
    constraints: dict[str, Any] | None = None,
) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb=semantic_verb,
        object_type=object_type,
        intent_confidence=0.95,
        constraints=constraints or {},
        raw_evidence=description,
    )


def _selection(task_id: str, capability_id: str, operation_id: str) -> CapabilitySelectionResult:
    selected = CapabilityRef(
        capability_id=capability_id,
        operation_id=operation_id,
        confidence=0.95,
        reason="Capability was already selected for this task.",
    )
    return CapabilitySelectionResult(
        task_id=task_id,
        candidates=[selected],
        selected=selected,
        unresolved_reason=None,
    )


def _client() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "list all files in this folder": {
                "task_id": "task-list-files",
                "capability_id": "filesystem.list_directory",
                "operation_id": "list_directory",
                "arguments": {},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.93,
            },
            "read README.md": {
                "task_id": "task-read-readme",
                "capability_id": "filesystem.read_file",
                "operation_id": "read_file",
                "arguments": {"path": "README.md"},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.96,
            },
            "show top 10 rows from patients table": {
                "task_id": "task-top-ten",
                "capability_id": "sql.read_query",
                "operation_id": "read_query",
                "arguments": {
                    "query_intent": {"template": "select_rows", "table": "patients"}
                },
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.91,
            },
            "show top 10 rows": {
                "task_id": "task-missing-table",
                "capability_id": "sql.read_query",
                "operation_id": "read_query",
                "arguments": {},
                "missing_required_arguments": ["query_intent"],
                "assumptions": ["The prompt does not specify which table to read from."],
                "confidence": 0.58,
            },
            "read ../secrets.txt": {
                "task_id": "task-traversal",
                "capability_id": "filesystem.read_file",
                "operation_id": "read_file",
                "arguments": {"path": "../secrets.txt"},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.88,
            },
        }
    )


def test_extract_current_directory_path_as_workspace_root_token() -> None:
    client = _client()
    task = _task("task-list-files", "list all files in this folder", "read", "directory")

    results = extract_arguments(
        [task],
        [_selection("task-list-files", "filesystem.list_directory", "list_directory")],
        _registry(),
        client,
    )

    assert isinstance(results[0], ArgumentExtractionResult)
    assert results[0].arguments["path"] == "."
    assert "Do not generate shell commands." in client.last_prompt


def test_extract_readme_path() -> None:
    task = _task("task-read-readme", "read README.md", "read", "file")

    results = extract_arguments(
        [task],
        [_selection("task-read-readme", "filesystem.read_file", "read_file")],
        _registry(),
        _client(),
    )

    assert results[0].arguments["path"] == "README.md"
    assert results[0].missing_required_arguments == []


def test_extract_top_n_limit_for_table_like_read() -> None:
    task = _task("task-top-ten", "show top 10 rows from patients table", "read", "table")

    results = extract_arguments(
        [task],
        [_selection("task-top-ten", "sql.read_query", "read_query")],
        _registry(),
        _client(),
    )

    assert results[0].arguments["query_intent"] == {
        "template": "select_rows",
        "table": "patients",
    }
    assert results[0].arguments["limit"] == 10


def test_ambiguous_missing_table_name_is_rejected() -> None:
    task = _task("task-missing-table", "show top 10 rows", "read", "table")

    results = extract_arguments(
        [task],
        [_selection("task-missing-table", "sql.read_query", "read_query")],
        _registry(),
        _client(),
    )

    assert results[0].arguments["limit"] == 10
    assert "query_intent" in results[0].missing_required_arguments
    assert any("does not specify" in item.lower() for item in results[0].assumptions)


def test_path_traversal_outside_workspace_is_rejected() -> None:
    task = _task("task-traversal", "read ../secrets.txt", "read", "file")

    results = extract_arguments(
        [task],
        [_selection("task-traversal", "filesystem.read_file", "read_file")],
        _registry(),
        _client(),
    )

    assert "path" not in results[0].arguments
    assert "path" in results[0].missing_required_arguments
    assert any("outside the workspace root" in item.lower() for item in results[0].assumptions)
