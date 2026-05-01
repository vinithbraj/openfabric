from __future__ import annotations

from typing import Any

import pytest

from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ActionDAG, ActionNode, ExecutionResult, ResultBundle, UserRequest
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.output_pipeline.display_selection import (
    DisplaySelectionInput,
    select_display_plan,
)
from agent_runtime.output_pipeline.orchestrator import compose_output


class FakeLLMClient:
    """Fake LLM client for display plan selection tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_prompt = ""
        self.last_schema: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_prompt = prompt
        self.last_schema = schema
        return dict(self.payload)


def _request(store: InMemoryResultStore | None = None, allow_full: bool = False) -> UserRequest:
    return UserRequest(
        raw_prompt="test prompt",
        safety_context={
            "result_store": store,
            "allow_full_output_access": allow_full,
        },
    )


def _dag(node_id: str = "node-1", capability_id: str = "filesystem.list_directory", operation_id: str = "list_directory") -> ActionDAG:
    return ActionDAG(
        dag_id="dag-1",
        nodes=[
            ActionNode(
                id=node_id,
                task_id="task-1",
                description="task",
                semantic_verb="read",
                capability_id=capability_id,
                operation_id=operation_id,
                arguments={},
                safety_labels=[],
            )
        ],
    )


def test_file_list_renders_as_markdown_table() -> None:
    bundle = ResultBundle(
        dag_id="dag-1",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-1",
                status="success",
                data_preview={
                    "entries": [
                        {"name": "README.md", "path": "README.md", "type": "file", "size": 10, "modified_time": "now"},
                        {"name": "src", "path": "src", "type": "directory", "size": 0, "modified_time": "now"},
                    ]
                },
            )
        ],
    )
    llm = FakeLLMClient(
        {
            "display_type": "table",
            "title": "Workspace Files",
            "sections": [
                {
                    "title": "Workspace Files",
                    "display_type": "table",
                    "source_node_id": "node-1",
                }
            ],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    output = compose_output(_request(), _dag(), bundle, llm)

    assert "## Workspace Files" in output
    assert "| name | path | type | size | modified_time |" in output
    assert "README.md" in output


def test_read_file_renders_as_code_block() -> None:
    bundle = ResultBundle(
        dag_id="dag-1",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-1",
                status="success",
                data_preview={"content_preview": "print('hello')\n", "truncated": False},
            )
        ],
    )
    llm = FakeLLMClient(
        {
            "display_type": "code_block",
            "title": "File Preview",
            "sections": [
                {
                    "title": "File Preview",
                    "display_type": "code_block",
                    "source_node_id": "node-1",
                    "parameters": {"language": "python"},
                }
            ],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    output = compose_output(_request(), _dag("node-1", "filesystem.read_file", "read_file"), bundle, llm)

    assert "```python" in output
    assert "print('hello')" in output


def test_process_rows_render_as_markdown_table() -> None:
    bundle = ResultBundle(
        dag_id="dag-1",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-1",
                status="success",
                data_preview={
                    "pattern": "python",
                    "processes": [
                        {"pid": 1, "command": "python3", "cpu_percent": 10.2, "memory_percent": 1.5},
                        {"pid": 2, "command": "uvicorn", "cpu_percent": 5.0, "memory_percent": 0.8},
                    ],
                    "truncated": False,
                },
            )
        ],
    )
    llm = FakeLLMClient(
        {
            "display_type": "table",
            "title": "Running Python Processes",
            "sections": [
                {
                    "title": "Running Python Processes",
                    "display_type": "table",
                    "source_node_id": "node-1",
                }
            ],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    output = compose_output(_request(), _dag("node-1", "shell.list_processes", "list_processes"), bundle, llm)

    assert "## Running Python Processes" in output
    assert "| pid | command | cpu_percent | memory_percent |" in output
    assert "python3" in output


def test_errors_render_cleanly() -> None:
    bundle = ResultBundle(
        dag_id="dag-1",
        status="error",
        results=[],
        safe_summary="Execution blocked.",
        metadata={"blocked_reasons": ["Shell execution is disabled."]},
    )

    output = compose_output(_request(), _dag(), bundle, FakeLLMClient({}))

    assert "Execution failed." in output
    assert "Shell execution is disabled." in output
    assert "Execution blocked." in output


def test_partial_failures_render_clearly() -> None:
    bundle = ResultBundle(
        dag_id="dag-1",
        status="partial",
        safe_summary="One node failed and one node was skipped.",
        results=[
            ExecutionResult(node_id="node-ok", status="success", data_preview={"markdown": "done"}),
            ExecutionResult(node_id="node-fail", status="error", error="simulated failure"),
            ExecutionResult(node_id="node-skip", status="skipped", error="Skipped because dependency failed."),
        ],
    )
    dag = ActionDAG(
        dag_id="dag-1",
        nodes=[
            ActionNode(
                id="node-ok",
                task_id="task-ok",
                description="ok",
                semantic_verb="read",
                capability_id="markdown.render",
                operation_id="render",
                arguments={},
                safety_labels=[],
            )
        ],
    )

    output = compose_output(_request(), dag, bundle, FakeLLMClient({}))

    assert "Partial results" in output
    assert "Completed:" in output
    assert "Errors:" in output
    assert "Skipped:" in output


def test_display_plan_cannot_reference_missing_data() -> None:
    selection_input = DisplaySelectionInput(
        original_prompt="list files",
        dag_summary={"dag_id": "dag-1"},
        result_summary={"status": "success"},
        safe_previews=[{"node_id": "node-1", "data_ref": "data-1", "preview": {"entries": []}}],
        available_display_types=["markdown", "table", "code_block", "multi_section", "json", "plain_text"],
    )
    llm = FakeLLMClient(
        {
            "display_type": "table",
            "title": "Broken Plan",
            "sections": [
                {
                    "title": "Broken Plan",
                    "display_type": "table",
                    "source_node_id": "missing-node",
                }
            ],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    with pytest.raises(ValidationError, match="missing node_id"):
        select_display_plan(selection_input, llm)
