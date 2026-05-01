from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_runtime.core.types import UserRequest
from agent_runtime.input_pipeline.decomposition import (
    DecompositionResult,
    PromptClassification,
    decompose_prompt,
)


class FakeLLMClient:
    """Fake LLM client that returns prompt-specific decomposition payloads."""

    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.last_prompt = ""
        self.last_schema: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_prompt = prompt
        self.last_schema = schema
        for raw_prompt, payload in self.payloads.items():
            if raw_prompt in prompt:
                return dict(payload)
        raise AssertionError(f"no fake payload configured for prompt: {prompt}")


def _classification(prompt_type: str) -> PromptClassification:
    return PromptClassification(
        prompt_type=prompt_type,
        requires_tools=True,
        likely_domains=["filesystem"],
        risk_level="low",
        needs_clarification=False,
        clarification_question=None,
        reason="test classification",
    )


def _client() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "list all files in this folder": {
                "tasks": [
                    {
                        "id": "task-find-files",
                        "description": "List all files in the current folder.",
                        "semantic_verb": "read",
                        "object_type": "filesystem",
                        "intent_confidence": 0.98,
                        "constraints": {"path": "."},
                        "dependencies": [],
                        "raw_evidence": "list all files in this folder",
                    }
                ],
                "global_constraints": {"path": "."},
                "unresolved_references": [],
                "assumptions": ["The current folder is the working directory."],
            },
            "Find all CSV files in this folder, merge them, and give me the top 10 rows as a markdown table.": {
                "tasks": [
                    {
                        "id": "task-find-csv",
                        "description": "Find CSV files in the current folder.",
                        "semantic_verb": "search",
                        "object_type": "filesystem",
                        "intent_confidence": 0.97,
                        "constraints": {"path": ".", "pattern": "*.csv"},
                        "dependencies": [],
                        "raw_evidence": "Find all CSV files in this folder",
                    },
                    {
                        "id": "task-read-csv",
                        "description": "Read the discovered CSV files.",
                        "semantic_verb": "read",
                        "object_type": "csv_files",
                        "intent_confidence": 0.96,
                        "constraints": {},
                        "dependencies": ["task-find-csv"],
                        "raw_evidence": "read CSV files",
                    },
                    {
                        "id": "task-merge-csv",
                        "description": "Merge the CSV file contents.",
                        "semantic_verb": "transform",
                        "object_type": "table",
                        "intent_confidence": 0.96,
                        "constraints": {},
                        "dependencies": ["task-read-csv"],
                        "raw_evidence": "merge them",
                    },
                    {
                        "id": "task-top-rows",
                        "description": "Select the top 10 rows from the merged result.",
                        "semantic_verb": "analyze",
                        "object_type": "table",
                        "intent_confidence": 0.95,
                        "constraints": {"row_limit": 10},
                        "dependencies": ["task-merge-csv"],
                        "raw_evidence": "top 10 rows",
                    },
                    {
                        "id": "task-render-markdown",
                        "description": "Render the selected rows as a markdown table.",
                        "semantic_verb": "render",
                        "object_type": "markdown_table",
                        "intent_confidence": 0.95,
                        "constraints": {"format": "markdown_table"},
                        "dependencies": ["task-top-rows"],
                        "raw_evidence": "as a markdown table",
                    },
                ],
                "global_constraints": {"path": ".", "row_limit": 10, "output_format": "markdown_table"},
                "unresolved_references": [],
                "assumptions": ["CSV files share mergeable columns."],
            },
            "read this CSV and summarize it": {
                "tasks": [
                    {
                        "id": "task-read-csv",
                        "description": "Read the referenced CSV file.",
                        "semantic_verb": "read",
                        "object_type": "csv_file",
                        "intent_confidence": 0.95,
                        "constraints": {},
                        "dependencies": [],
                        "raw_evidence": "read this CSV",
                    },
                    {
                        "id": "task-summarize-csv",
                        "description": "Summarize the CSV contents.",
                        "semantic_verb": "summarize",
                        "object_type": "table_summary",
                        "intent_confidence": 0.94,
                        "constraints": {},
                        "dependencies": ["task-read-csv"],
                        "raw_evidence": "summarize it",
                    },
                ],
                "global_constraints": {"output_format": "summary"},
                "unresolved_references": ["the exact CSV path is implicit"],
                "assumptions": ["The user has a single intended CSV file in context."],
            },
        }
    )


def test_decompose_simple_prompt() -> None:
    client = _client()

    result = decompose_prompt(
        UserRequest(raw_prompt="list all files in this folder"),
        _classification("simple_tool_task"),
        client,
    )

    assert isinstance(result, DecompositionResult)
    assert len(result.tasks) == 1
    assert result.tasks[0].description == "List all files in the current folder."
    assert result.global_constraints == {"path": "."}
    assert "Do not select tools." in client.last_prompt
    assert "Do not generate commands." in client.last_prompt
    assert "Do not generate SQL." in client.last_prompt
    assert "Do not generate shell syntax." in client.last_prompt


def test_decompose_compound_prompt_preserves_order_and_dependencies() -> None:
    result = decompose_prompt(
        UserRequest(
            raw_prompt="Find all CSV files in this folder, merge them, and give me the top 10 rows as a markdown table."
        ),
        _classification("compound_tool_task"),
        _client(),
    )

    assert [task.id for task in result.tasks] == [
        "task-find-csv",
        "task-read-csv",
        "task-merge-csv",
        "task-top-rows",
        "task-render-markdown",
    ]
    assert result.tasks[1].dependencies == ["task-find-csv"]
    assert result.tasks[2].dependencies == ["task-read-csv"]
    assert result.tasks[3].dependencies == ["task-merge-csv"]
    assert result.tasks[4].dependencies == ["task-top-rows"]
    assert result.global_constraints["row_limit"] == 10
    assert result.global_constraints["output_format"] == "markdown_table"


def test_decompose_multi_step_prompt_tracks_assumptions_and_unresolved_references() -> None:
    result = decompose_prompt(
        UserRequest(raw_prompt="read this CSV and summarize it"),
        _classification("compound_tool_task"),
        _client(),
    )

    assert len(result.tasks) == 2
    assert result.tasks[1].dependencies == ["task-read-csv"]
    assert result.unresolved_references == ["the exact CSV path is implicit"]
    assert result.assumptions == ["The user has a single intended CSV file in context."]


def test_decomposition_result_rejects_unknown_task_dependency() -> None:
    with pytest.raises(ValidationError, match="depends on unknown task id"):
        DecompositionResult(
            tasks=[
                {
                    "id": "task-a",
                    "description": "A",
                    "semantic_verb": "read",
                    "object_type": "filesystem",
                    "intent_confidence": 0.9,
                    "constraints": {},
                    "dependencies": ["missing-task"],
                }
            ],
            global_constraints={},
            unresolved_references=[],
            assumptions=[],
        )
