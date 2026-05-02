"""Core intermediate representation models for the agent runtime."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_runtime.core.ids import new_id

SemanticVerb = Literal[
    "read",
    "search",
    "create",
    "update",
    "delete",
    "transform",
    "analyze",
    "summarize",
    "compare",
    "execute",
    "render",
    "unknown",
]
RiskLevel = Literal["low", "medium", "high", "critical"]
ExecutionStatus = Literal["success", "error", "skipped"]
BundleStatus = Literal["success", "partial", "error", "confirmation_required"]
DisplayType = Literal[
    "plain_text",
    "markdown",
    "table",
    "json",
    "code_block",
    "file_link",
    "chart",
    "multi_section",
]
RedactionPolicy = Literal["none", "standard", "strict"]


class UserRequest(BaseModel):
    """Raw user request plus contexts available to input semantics."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: new_id("req"))
    raw_prompt: str
    user_context: dict[str, Any] = Field(default_factory=dict)
    session_context: dict[str, Any] = Field(default_factory=dict)
    safety_context: dict[str, Any] = Field(default_factory=dict)


class TaskFrame(BaseModel):
    """Typed semantic description of one user task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("task"))
    parent_id: str | None = None
    description: str
    semantic_verb: SemanticVerb
    object_type: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    constraints: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    raw_evidence: str | None = None
    requires_confirmation: bool = False
    risk_level: RiskLevel = "low"


class CapabilityRef(BaseModel):
    """Candidate mapping from a task to a capability operation."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str
    operation_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class DataRef(BaseModel):
    """Typed handle for one stored execution payload."""

    model_config = ConfigDict(extra="forbid")

    ref_id: str
    producer_node_id: str
    data_type: str
    preview: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputRef(BaseModel):
    """Typed reference from one node argument to another node's output."""

    model_config = ConfigDict(extra="forbid")

    source_node_id: str
    output_key: str | None = None
    expected_data_type: str | None = None


def _coerce_input_ref_values(value: Any) -> Any:
    """Recursively coerce explicit InputRef-shaped dictionaries."""

    if isinstance(value, InputRef):
        return value
    if isinstance(value, list):
        return [_coerce_input_ref_values(item) for item in value]
    if isinstance(value, dict):
        keys = set(value.keys())
        if "source_node_id" in keys and keys <= {"source_node_id", "output_key", "expected_data_type"}:
            return InputRef.model_validate(value)
        return {key: _coerce_input_ref_values(item) for key, item in value.items()}
    return value


def _iter_input_refs(value: Any) -> list[InputRef]:
    """Return every nested InputRef found within one argument value."""

    if isinstance(value, InputRef):
        return [value]
    if isinstance(value, list):
        refs: list[InputRef] = []
        for item in value:
            refs.extend(_iter_input_refs(item))
        return refs
    if isinstance(value, dict):
        refs = []
        for item in value.values():
            refs.extend(_iter_input_refs(item))
        return refs
    return []


class ActionNode(BaseModel):
    """Executable DAG node after capability selection."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("node"))
    task_id: str
    description: str
    semantic_verb: str
    capability_id: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    safety_labels: list[str] = Field(default_factory=list)
    dry_run: bool = False

    @model_validator(mode="before")
    @classmethod
    def coerce_argument_refs(cls, values: Any) -> Any:
        """Coerce explicit InputRef payloads nested inside node arguments."""

        if not isinstance(values, dict):
            return values
        arguments = values.get("arguments")
        if isinstance(arguments, dict):
            values = dict(values)
            values["arguments"] = {
                key: _coerce_input_ref_values(argument_value)
                for key, argument_value in arguments.items()
            }
        return values

    @property
    def is_unresolved(self) -> bool:
        """Return whether this node is intentionally unresolved."""

        labels = {label.strip().lower() for label in self.safety_labels}
        return "unresolved" in labels


class ActionDAG(BaseModel):
    """Validated action graph passed into execution."""

    model_config = ConfigDict(extra="forbid")

    dag_id: str = Field(default_factory=lambda: new_id("dag"))
    nodes: list[ActionNode] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    global_constraints: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    final_dag_hash: str | None = None
    execution_ready: bool = False
    safety_decision: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> "ActionDAG":
        """Enforce unique ids, valid dependencies, known capabilities, and acyclicity."""

        node_ids = [node.id for node in self.nodes]
        unique_node_ids = set(node_ids)
        if len(unique_node_ids) != len(node_ids):
            raise ValueError("ActionDAG node ids must be unique.")

        for node in self.nodes:
            if not node.capability_id.strip():
                raise ValueError(f"Action node {node.id} has an empty capability_id.")
            if node.capability_id.strip().lower() == "unknown" and not node.is_unresolved:
                raise ValueError(
                    f"Action node {node.id} has unknown capability_id but is not marked unresolved."
                )

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in unique_node_ids}
        for node in self.nodes:
            for dependency in node.depends_on:
                if dependency not in unique_node_ids:
                    raise ValueError(f"Action node {node.id} depends on unknown node {dependency}.")
                adjacency[dependency].add(node.id)

        for source, target in self.edges:
            if source not in unique_node_ids:
                raise ValueError(f"ActionDAG edge source {source} does not refer to an existing node.")
            if target not in unique_node_ids:
                raise ValueError(f"ActionDAG edge target {target} does not refer to an existing node.")
            adjacency[source].add(target)

        _assert_acyclic(adjacency)
        ancestors = _ancestor_map(adjacency)
        for node in self.nodes:
            for argument_key, argument_value in node.arguments.items():
                for input_ref in _iter_input_refs(argument_value):
                    if input_ref.source_node_id not in unique_node_ids:
                        raise ValueError(
                            f"Action node {node.id} references unknown producer node {input_ref.source_node_id}."
                        )
                    if input_ref.source_node_id not in ancestors.get(node.id, set()):
                        raise ValueError(
                            f"Action node {node.id} references non-dependency producer node {input_ref.source_node_id}."
                        )
        return self


def _assert_acyclic(adjacency: dict[str, set[str]]) -> None:
    """Raise ValueError if a directed adjacency map contains a cycle."""

    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in permanent:
            return
        if node_id in temporary:
            raise ValueError("ActionDAG must be acyclic.")
        temporary.add(node_id)
        for child_id in adjacency.get(node_id, set()):
            visit(child_id)
        temporary.remove(node_id)
        permanent.add(node_id)

    for candidate_id in adjacency:
        visit(candidate_id)


def _reverse_adjacency(adjacency: dict[str, set[str]]) -> dict[str, set[str]]:
    """Build reverse edges for one DAG adjacency map."""

    reverse: dict[str, set[str]] = {node_id: set() for node_id in adjacency}
    for source, children in adjacency.items():
        for child in children:
            reverse[child].add(source)
    return reverse


def _ancestor_map(adjacency: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return transitive ancestors for every node id in one DAG."""

    reverse = _reverse_adjacency(adjacency)
    cache: dict[str, set[str]] = {}

    def visit(node_id: str) -> set[str]:
        if node_id in cache:
            return cache[node_id]
        ancestors: set[str] = set()
        for parent_id in reverse.get(node_id, set()):
            ancestors.add(parent_id)
            ancestors.update(visit(parent_id))
        cache[node_id] = ancestors
        return ancestors

    return {node_id: visit(node_id) for node_id in adjacency}


class ExecutionResult(BaseModel):
    """Normalized result for one executed action node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    status: ExecutionStatus
    data_ref: DataRef | None = None
    data_preview: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultBundle(BaseModel):
    """Collection of execution results passed to output composition."""

    model_config = ConfigDict(extra="forbid")

    dag_id: str
    results: list[ExecutionResult] = Field(default_factory=list)
    status: BundleStatus
    safe_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DisplayPlan(BaseModel):
    """Typed display contract selected for a result bundle."""

    model_config = ConfigDict(extra="forbid")

    display_type: DisplayType
    title: str | None = None
    sections: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    redaction_policy: RedactionPolicy = "standard"


class RenderedOutput(BaseModel):
    """Final rendered response emitted to a client."""

    model_config = ConfigDict(extra="forbid")

    content: str
    display_plan: DisplayPlan
    metadata: dict[str, Any] = Field(default_factory=dict)
