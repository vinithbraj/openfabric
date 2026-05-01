from __future__ import annotations

import pytest

from agent_runtime.capabilities import (
    CapabilityRegistry,
    ListDirectoryCapability,
    MarkdownRenderCapability,
    ReadFileCapability,
)
from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ActionDAG, ActionNode, CapabilityRef, InputRef, TaskFrame, UserRequest
from agent_runtime.input_pipeline.argument_extraction import ArgumentExtractionResult
from agent_runtime.input_pipeline.dag_builder import DAGBuilder, build_action_dag
from agent_runtime.input_pipeline.decomposition import DecompositionResult
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult


class DeleteFileCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="filesystem.delete_file",
        domain="filesystem",
        operation_id="delete_file",
        name="Delete File",
        description="Delete one file from disk.",
        semantic_verbs=["delete"],
        object_types=["file", "filesystem"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=["path"],
        optional_arguments=[],
        output_schema={"deleted": {"type": "boolean"}},
        risk_level="high",
        read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        examples=[{"arguments": {"path": "old.log"}}],
        safety_notes=["Destructive filesystem mutation."],
    )


def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(ListDirectoryCapability())
    registry.register(ReadFileCapability())
    registry.register(MarkdownRenderCapability())
    registry.register(DeleteFileCapability())
    return registry


def _request(registry: CapabilityRegistry) -> UserRequest:
    return UserRequest(
        raw_prompt="test request",
        safety_context={"capability_registry": registry},
    )


def _task(
    task_id: str,
    description: str,
    semantic_verb: str,
    object_type: str,
    dependencies: list[str] | None = None,
    requires_confirmation: bool = False,
) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb=semantic_verb,
        object_type=object_type,
        intent_confidence=0.95,
        constraints={},
        dependencies=dependencies or [],
        raw_evidence=description,
        requires_confirmation=requires_confirmation,
    )


def _selection(
    task_id: str,
    capability_id: str,
    operation_id: str,
    unresolved_reason: str | None = None,
) -> CapabilitySelectionResult:
    selected = None
    candidates = []
    if unresolved_reason is None:
        selected = CapabilityRef(
            capability_id=capability_id,
            operation_id=operation_id,
            confidence=0.95,
            reason="Selected capability matches the task.",
        )
        candidates = [selected]
    return CapabilitySelectionResult(
        task_id=task_id,
        candidates=candidates,
        selected=selected,
        unresolved_reason=unresolved_reason,
    )


def _arguments(
    task_id: str,
    capability_id: str,
    operation_id: str,
    arguments: dict[str, object],
    missing: list[str] | None = None,
) -> ArgumentExtractionResult:
    return ArgumentExtractionResult(
        task_id=task_id,
        capability_id=capability_id,
        operation_id=operation_id,
        arguments=arguments,
        missing_required_arguments=missing or [],
        assumptions=[],
        confidence=0.95,
    )


def test_builds_simple_compatibility_dag_from_task_frame() -> None:
    frame = TaskFrame(
        id="task-1",
        description="list files",
        semantic_verb="read",
        object_type="filesystem",
        intent_confidence=0.9,
        constraints={
            "capability_id": "filesystem.list_directory",
            "operation_id": "list_directory",
            "arguments": {"path": "."},
        },
    )

    dag = DAGBuilder().build(frame)

    assert len(dag.nodes) == 1
    assert dag.nodes[0].id == "node::task-1"
    assert dag.nodes[0].task_id == "task-1"
    assert dag.nodes[0].capability_id == "filesystem.list_directory"
    assert dag.nodes[0].operation_id == "list_directory"
    assert dag.nodes[0].arguments == {"path": "."}


def test_build_action_dag_preserves_linear_task_chain() -> None:
    registry = _registry()
    decomposition = DecompositionResult(
        tasks=[
            _task("task-find", "List files in current directory", "read", "directory"),
            _task("task-read", "Read README.md", "read", "file", dependencies=["task-find"]),
            _task(
                "task-render",
                "Render the file summary as markdown",
                "render",
                "markdown",
                dependencies=["task-read"],
            ),
        ],
        global_constraints={"path": ".", "format": "markdown"},
    )

    dag = build_action_dag(
        _request(registry),
        decomposition,
        [
            _selection("task-find", "filesystem.list_directory", "list_directory"),
            _selection("task-read", "filesystem.read_file", "read_file"),
            _selection("task-render", "markdown.render", "render"),
        ],
        [
            _arguments("task-find", "filesystem.list_directory", "list_directory", {"path": "."}),
            _arguments("task-read", "filesystem.read_file", "read_file", {"path": "README.md"}),
            _arguments(
                "task-render",
                "markdown.render",
                "render",
                {
                    "input_ref": "data-render",
                    "render_type": "summary",
                    "parameters": {"title": "README Summary"},
                },
            ),
        ],
    )

    assert [node.id for node in dag.nodes] == [
        "node::task-find",
        "node::task-read",
        "node::task-render",
    ]
    assert dag.nodes[1].depends_on == ["node::task-find"]
    assert dag.nodes[2].depends_on == ["node::task-read"]
    assert dag.edges == [
        ("node::task-find", "node::task-read"),
        ("node::task-read", "node::task-render"),
    ]
    assert dag.global_constraints == {"path": ".", "format": "markdown"}


def test_build_action_dag_supports_independent_parallel_tasks() -> None:
    registry = _registry()
    decomposition = DecompositionResult(
        tasks=[
            _task("task-list", "List files", "read", "directory"),
            _task("task-read", "Read README.md", "read", "file"),
        ],
        global_constraints={},
    )

    dag = build_action_dag(
        _request(registry),
        decomposition,
        [
            _selection("task-list", "filesystem.list_directory", "list_directory"),
            _selection("task-read", "filesystem.read_file", "read_file"),
        ],
        [
            _arguments("task-list", "filesystem.list_directory", "list_directory", {"path": "."}),
            _arguments("task-read", "filesystem.read_file", "read_file", {"path": "README.md"}),
        ],
    )

    assert len(dag.nodes) == 2
    assert dag.edges == []
    assert dag.nodes[0].depends_on == []
    assert dag.nodes[1].depends_on == []


def test_build_action_dag_rejects_missing_required_arguments() -> None:
    registry = _registry()
    decomposition = DecompositionResult(
        tasks=[_task("task-read", "Read README.md", "read", "file")],
        global_constraints={},
    )

    with pytest.raises(ValidationError, match="missing required arguments: path"):
        build_action_dag(
            _request(registry),
            decomposition,
            [_selection("task-read", "filesystem.read_file", "read_file")],
            [_arguments("task-read", "filesystem.read_file", "read_file", {}, missing=["path"])],
        )


def test_build_action_dag_rejects_unresolved_capability() -> None:
    registry = _registry()
    decomposition = DecompositionResult(
        tasks=[_task("task-unknown", "Teleport the repo", "execute", "unknown")],
        global_constraints={},
    )

    with pytest.raises(ValidationError, match="is unresolved"):
        build_action_dag(
            _request(registry),
            decomposition,
            [_selection("task-unknown", "unknown", "unknown", unresolved_reason="No safe capability matched.")],
            [_arguments("task-unknown", "unknown", "unknown", {})],
        )


def test_build_action_dag_normalizes_string_input_refs() -> None:
    registry = _registry()
    decomposition = DecompositionResult(
        tasks=[
            _task("task-source", "Read README.md", "read", "file"),
            _task("task-render", "Render README as markdown", "render", "markdown", dependencies=["task-source"]),
        ],
        global_constraints={},
    )

    dag = build_action_dag(
        _request(registry),
        decomposition,
        [
            _selection("task-source", "filesystem.read_file", "read_file"),
            _selection("task-render", "markdown.render", "render"),
        ],
        [
            _arguments("task-source", "filesystem.read_file", "read_file", {"path": "README.md"}),
            _arguments(
                "task-render",
                "markdown.render",
                "render",
                {"input_ref": "task-source.output", "render_type": "summary", "parameters": {}},
            ),
        ],
    )

    assert dag.nodes[1].arguments["input_ref"] == InputRef(source_node_id="node::task-source")


def test_input_ref_to_nonexistent_node_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown producer node"):
        ActionDAG(
            nodes=[
                ActionNode(
                    id="node-a",
                    task_id="task-a",
                    description="render output",
                    semantic_verb="render",
                    capability_id="markdown.render",
                    operation_id="render",
                    arguments={"input_ref": InputRef(source_node_id="node-missing")},
                    safety_labels=[],
                )
            ]
        )


def test_input_ref_to_non_dependency_node_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-dependency producer node"):
        ActionDAG(
            nodes=[
                ActionNode(
                    id="node-a",
                    task_id="task-a",
                    description="source",
                    semantic_verb="read",
                    capability_id="filesystem.read_file",
                    operation_id="read_file",
                    arguments={"path": "README.md"},
                    safety_labels=[],
                ),
                ActionNode(
                    id="node-b",
                    task_id="task-b",
                    description="other source",
                    semantic_verb="read",
                    capability_id="filesystem.read_file",
                    operation_id="read_file",
                    arguments={"path": "README.md"},
                    safety_labels=[],
                ),
                ActionNode(
                    id="node-c",
                    task_id="task-c",
                    description="render",
                    semantic_verb="render",
                    capability_id="markdown.render",
                    operation_id="render",
                    arguments={"input_ref": InputRef(source_node_id="node-a")},
                    depends_on=["node-b"],
                    safety_labels=[],
                ),
            ],
            edges=[("node-b", "node-c")],
        )


def test_delete_operation_marks_confirmation_required() -> None:
    registry = _registry()
    decomposition = DecompositionResult(
        tasks=[
            _task(
                "task-delete",
                "Delete old.log",
                "delete",
                "file",
                requires_confirmation=True,
            )
        ],
        global_constraints={},
    )

    dag = build_action_dag(
        _request(registry),
        decomposition,
        [_selection("task-delete", "filesystem.delete_file", "delete_file")],
        [_arguments("task-delete", "filesystem.delete_file", "delete_file", {"path": "old.log"})],
    )

    assert dag.requires_confirmation is True
    assert "requires-confirmation" in dag.nodes[0].safety_labels
