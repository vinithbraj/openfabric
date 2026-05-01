"""Core intermediate representation models for the agent runtime."""

from __future__ import annotations

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
BundleStatus = Literal["success", "partial", "error"]
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


class ExecutionResult(BaseModel):
    """Normalized result for one executed action node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    status: ExecutionStatus
    data_ref: str | None = None
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
