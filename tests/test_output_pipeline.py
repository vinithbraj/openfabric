from __future__ import annotations

from typing import Any

import pytest

from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ActionDAG, ActionNode, DataRef, ExecutionResult, ResultBundle, UserRequest
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.output_pipeline.display_selection import (
    DisplaySelectionInput,
    select_display_plan,
)
from agent_runtime.output_pipeline.orchestrator import compose_output
from agent_runtime.output_pipeline.renderers import render_result_shape
from agent_runtime.output_pipeline.result_shapes import (
    AggregateResult,
    CapabilityListResult,
    DirectoryListingResult,
    normalize_execution_result,
)


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
                metadata={"capability_id": "shell.list_processes", "operation_id": "list_processes"},
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


def test_output_planning_sees_safe_previews_not_full_data() -> None:
    store = InMemoryResultStore()
    full_payload = {"blob": "x" * 5000}
    data_ref = store.put("node-1", full_payload, "object", {})
    bundle = ResultBundle(
        dag_id="dag-1",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-1",
                status="success",
                data_ref=DataRef.model_validate(data_ref.model_dump()),
                data_preview={"preview_text": "x" * 64, "truncated": True, "bytes": 5000},
            )
        ],
    )
    llm = FakeLLMClient(
        {
            "display_type": "plain_text",
            "title": "Preview",
            "sections": [
                {
                    "title": "Preview",
                    "display_type": "plain_text",
                    "source_node_id": "node-1",
                }
            ],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    compose_output(_request(store=store, allow_full=False), _dag(), bundle, llm)

    assert "preview_text" in llm.last_prompt
    assert '"blob"' not in llm.last_prompt
    assert "x" * 5000 not in llm.last_prompt


def test_list_directory_normalizes_to_directory_listing_result() -> None:
    shape = normalize_execution_result(
        ExecutionResult(
            node_id="node-1",
            status="success",
            data_preview={
                "path": ".",
                "entries": [
                    {"name": "README.md", "path": "README.md", "type": "file", "size": 10, "modified_time": "now"}
                ],
            },
            metadata={"capability_id": "filesystem.list_directory", "operation_id": "list_directory"},
        )
    )

    assert isinstance(shape, DirectoryListingResult)
    assert shape.path == "."
    assert shape.entries[0]["name"] == "README.md"


def test_aggregate_normalizes_to_aggregate_result() -> None:
    shape = normalize_execution_result(
        ExecutionResult(
            node_id="node-agg",
            status="success",
            data_preview={
                "operation": "sum",
                "field": "size",
                "value": 4777,
                "unit": "bytes",
                "row_count": 15,
                "used_count": 5,
                "skipped_count": 0,
                "label": "Total File Size",
            },
            metadata={"capability_id": "data.aggregate", "operation_id": "aggregate"},
        )
    )

    assert isinstance(shape, AggregateResult)
    assert shape.value == 4777
    assert shape.unit == "bytes"


def test_runtime_capabilities_normalize_to_capability_list_result() -> None:
    shape = normalize_execution_result(
        ExecutionResult(
            node_id="node-runtime",
            status="success",
            data_preview={
                "grouped_capabilities": {
                    "filesystem": [
                        {
                            "capability_id": "filesystem.list_directory",
                            "operation_id": "list_directory",
                            "name": "List Directory",
                            "description": "List files in a directory.",
                            "semantic_verbs": ["read"],
                            "object_types": ["filesystem.path"],
                            "read_only": True,
                            "risk_level": "low",
                        }
                    ]
                },
                "capability_count": 1,
            },
            metadata={
                "capability_id": "runtime.describe_capabilities",
                "operation_id": "describe_capabilities",
            },
        )
    )

    assert isinstance(shape, CapabilityListResult)
    assert "filesystem" in shape.grouped_capabilities
    assert shape.capability_count == 1


def test_aggregate_result_renders_scalar_value() -> None:
    rendered = render_result_shape(
        AggregateResult(
            node_id="node-agg",
            capability_id="data.aggregate",
            operation_id="aggregate",
            label="Total File Size",
            operation="sum",
            field="size",
            value=4777,
            unit="bytes",
            row_count=15,
            used_count=15,
            skipped_count=0,
        )
    )

    assert "Total File Size: 4777 bytes" in rendered
    assert "README.md" not in rendered


def test_directory_listing_result_renders_table() -> None:
    rendered = render_result_shape(
        DirectoryListingResult(
            node_id="node-list",
            capability_id="filesystem.list_directory",
            operation_id="list_directory",
            entries=[
                {"name": "README.md", "path": "README.md", "type": "file", "size": 10, "modified_time": "now"},
                {"name": "src", "path": "src", "type": "directory", "size": 0, "modified_time": "now"},
            ],
        )
    )

    assert "| name | path | type | size | modified_time |" in rendered
    assert "README.md" in rendered


def test_mixed_directory_and_aggregate_render_two_sections() -> None:
    request = UserRequest(raw_prompt="list all files in this directory and compute the total file size")
    dag = ActionDAG(
        dag_id="dag-mixed",
        nodes=[
            ActionNode(
                id="node-list",
                task_id="task-list",
                description="list files",
                semantic_verb="read",
                capability_id="filesystem.list_directory",
                operation_id="list_directory",
                arguments={},
                safety_labels=[],
            ),
            ActionNode(
                id="node-agg",
                task_id="task-agg",
                description="aggregate file size",
                semantic_verb="analyze",
                capability_id="data.aggregate",
                operation_id="aggregate",
                arguments={},
                depends_on=["node-list"],
                safety_labels=[],
            ),
        ],
        edges=[("node-list", "node-agg")],
    )
    bundle = ResultBundle(
        dag_id="dag-mixed",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-list",
                status="success",
                data_preview={
                    "entries": [
                        {"name": "README.md", "path": "README.md", "type": "file", "size": 10, "modified_time": "now"},
                        {"name": "src", "path": "src", "type": "directory", "size": 0, "modified_time": "now"},
                    ]
                },
                metadata={"capability_id": "filesystem.list_directory", "operation_id": "list_directory"},
            ),
            ExecutionResult(
                node_id="node-agg",
                status="success",
                data_preview={
                    "operation": "sum",
                    "field": "size",
                    "value": 10,
                    "unit": "bytes",
                    "row_count": 2,
                    "used_count": 1,
                    "skipped_count": 0,
                    "label": "Total File Size",
                },
                metadata={"capability_id": "data.aggregate", "operation_id": "aggregate"},
            ),
        ],
    )
    invalid_llm = FakeLLMClient(
        {
            "display_type": "multi_section",
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

    output = compose_output(request, dag, bundle, invalid_llm)

    assert "## Directory Listing" in output
    assert "Total File Size: 10 bytes" in output
    assert output.index("README.md") < output.index("Total File Size: 10 bytes")


def test_total_file_size_does_not_render_file_names() -> None:
    request = UserRequest(raw_prompt="total file size in this directory")
    bundle = ResultBundle(
        dag_id="dag-agg",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-agg",
                status="success",
                data_preview={
                    "operation": "sum",
                    "field": "size",
                    "value": 4777,
                    "unit": "bytes",
                    "row_count": 15,
                    "used_count": 5,
                    "skipped_count": 0,
                    "label": "Total File Size",
                },
                metadata={"capability_id": "data.aggregate", "operation_id": "aggregate"},
            )
        ],
    )
    invalid_llm = FakeLLMClient(
        {
            "display_type": "table",
            "title": "Broken Aggregate Plan",
            "sections": [{"display_type": "table", "source_node_id": "missing-node"}],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    output = compose_output(
        request,
        _dag("node-agg", "data.aggregate", "aggregate"),
        bundle,
        invalid_llm,
    )

    assert "Total File Size: 4777 bytes" in output
    assert "README.md" not in output


def test_deterministic_fallback_works_if_llm_display_plan_fails() -> None:
    bundle = ResultBundle(
        dag_id="dag-1",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-1",
                status="success",
                data_preview={
                    "entries": [
                        {"name": "README.md", "path": "README.md", "type": "file", "size": 10, "modified_time": "now"}
                    ]
                },
                metadata={"capability_id": "filesystem.list_directory", "operation_id": "list_directory"},
            )
        ],
    )
    invalid_llm = FakeLLMClient(
        {
            "display_type": "table",
            "title": "Broken Plan",
            "sections": [{"display_type": "table", "source_node_id": "missing-node"}],
            "constraints": {},
            "redaction_policy": "standard",
        }
    )

    output = compose_output(_request(), _dag(), bundle, invalid_llm)

    assert "| name | path | type | size | modified_time |" in output
    assert "README.md" in output


def test_no_raw_stack_traces_are_rendered() -> None:
    bundle = ResultBundle(
        dag_id="dag-err",
        status="error",
        safe_summary=(
            "Traceback (most recent call last):\n"
            "  File \"runtime.py\", line 10, in run\n"
            "ValueError: boom"
        ),
        results=[],
    )

    output = compose_output(_request(), _dag(), bundle, FakeLLMClient({}))

    assert "Traceback (most recent call last)" not in output
    assert "ValueError: boom" in output
