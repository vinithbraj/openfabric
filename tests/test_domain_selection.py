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
from agent_runtime.core.types import TaskFrame
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult, select_capabilities


class FakeLLMClient:
    """Fake structured LLM client for capability selection tests."""

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


def _task(task_id: str, description: str, semantic_verb: str, object_type: str) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb=semantic_verb,
        object_type=object_type,
        intent_confidence=0.95,
        constraints={},
        raw_evidence=description,
    )


def _client() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "list files in current directory": {
                "task_id": "task-list-dir",
                "candidates": [
                    {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.96,
                        "reason": "Lists directory entries for a filesystem read task.",
                    }
                ],
                "selected": {
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "confidence": 0.96,
                    "reason": "Lists directory entries for a filesystem read task.",
                },
                "unresolved_reason": None,
            },
            "read README.md": {
                "task_id": "task-read-readme",
                "candidates": [
                    {
                        "capability_id": "filesystem.read_file",
                        "operation_id": "read_file",
                        "confidence": 0.97,
                        "reason": "Reads one file directly from the filesystem.",
                    }
                ],
                "selected": {
                    "capability_id": "filesystem.read_file",
                    "operation_id": "read_file",
                    "confidence": 0.97,
                    "reason": "Reads one file directly from the filesystem.",
                },
                "unresolved_reason": None,
            },
            "query patient count": {
                "task_id": "task-query-patient-count",
                "candidates": [
                    {
                        "capability_id": "sql.read_query",
                        "operation_id": "read_query",
                        "confidence": 0.92,
                        "reason": "A read-only SQL query best fits patient count analysis.",
                    }
                ],
                "selected": {
                    "capability_id": "sql.read_query",
                    "operation_id": "read_query",
                    "confidence": 0.92,
                    "reason": "A read-only SQL query best fits patient count analysis.",
                },
                "unresolved_reason": None,
            },
            "format result as markdown table": {
                "task_id": "task-render-markdown",
                "candidates": [
                    {
                        "capability_id": "markdown.render",
                        "operation_id": "render",
                        "confidence": 0.95,
                        "reason": "Rendering structured output as markdown matches the task directly.",
                    }
                ],
                "selected": {
                    "capability_id": "markdown.render",
                    "operation_id": "render",
                    "confidence": 0.95,
                    "reason": "Rendering structured output as markdown matches the task directly.",
                },
                "unresolved_reason": None,
            },
            "teleport the repository into another galaxy": {
                "task_id": "task-unknown",
                "candidates": [
                    {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.31,
                        "reason": "Weak filesystem guess only.",
                    }
                ],
                "selected": {
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "confidence": 0.31,
                    "reason": "Weak filesystem guess only.",
                },
                "unresolved_reason": None,
            },
        }
    )


def test_select_list_directory_capability() -> None:
    client = _client()
    task = _task("task-list-dir", "list files in current directory", "read", "filesystem")

    results = select_capabilities([task], _registry(), client)

    assert isinstance(results[0], CapabilitySelectionResult)
    assert results[0].selected is not None
    assert results[0].selected.capability_id == "filesystem.list_directory"
    assert "Do not produce commands, shell syntax, SQL, code, or executable syntax." in client.last_prompt


def test_select_read_file_capability() -> None:
    task = _task("task-read-readme", "read README.md", "read", "file")

    results = select_capabilities([task], _registry(), _client())

    assert results[0].selected is not None
    assert results[0].selected.capability_id == "filesystem.read_file"


def test_select_sql_query_capability() -> None:
    task = _task("task-query-patient-count", "query patient count", "analyze", "query")

    results = select_capabilities([task], _registry(), _client())

    assert results[0].selected is not None
    assert results[0].selected.capability_id == "sql.read_query"


def test_select_markdown_render_capability() -> None:
    task = _task("task-render-markdown", "format result as markdown table", "render", "markdown_table")

    results = select_capabilities([task], _registry(), _client())

    assert results[0].selected is not None
    assert results[0].selected.capability_id == "markdown.render"


def test_unknown_task_becomes_unresolved() -> None:
    task = _task("task-unknown", "teleport the repository into another galaxy", "execute", "unknown")

    results = select_capabilities([task], _registry(), _client())

    assert results[0].selected is None
    assert results[0].unresolved_reason is not None
    assert "below threshold" in results[0].unresolved_reason


def test_mismatched_semantic_verb_is_rejected_without_explicit_high_confidence_reason() -> None:
    client = FakeLLMClient(
        {
            "read README.md": {
                "task_id": "task-read-readme",
                "candidates": [
                    {
                        "capability_id": "markdown.render",
                        "operation_id": "render",
                        "confidence": 0.72,
                        "reason": "maybe useful",
                    }
                ],
                "selected": {
                    "capability_id": "markdown.render",
                    "operation_id": "render",
                    "confidence": 0.72,
                    "reason": "maybe useful",
                },
                "unresolved_reason": None,
            }
        }
    )
    task = _task("task-read-readme", "read README.md", "read", "file")

    results = select_capabilities([task], _registry(), client)

    assert results[0].selected is None
    assert results[0].unresolved_reason is not None
    assert "do not match task verb" in results[0].unresolved_reason
