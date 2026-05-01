from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_runtime.core.types import ActionDAG, ActionNode


def _node(node_id: str, *, depends_on: list[str] | None = None, capability_id: str = "filesystem") -> ActionNode:
    return ActionNode(
        id=node_id,
        task_id=f"task-{node_id}",
        description=f"node {node_id}",
        semantic_verb="read",
        capability_id=capability_id,
        operation_id="read",
        arguments={},
        depends_on=depends_on or [],
        safety_labels=[],
    )


def test_action_dag_accepts_valid_edges_and_dependencies() -> None:
    dag = ActionDAG(
        nodes=[_node("a"), _node("b", depends_on=["a"])],
        edges=[("a", "b")],
        global_constraints={"read_only": True},
    )

    assert dag.nodes[1].depends_on == ["a"]


def test_action_dag_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValidationError, match="node ids must be unique"):
        ActionDAG(nodes=[_node("same"), _node("same")])


def test_action_dag_rejects_invalid_dependency() -> None:
    with pytest.raises(ValidationError, match="depends on unknown node missing"):
        ActionDAG(nodes=[_node("a", depends_on=["missing"])])


def test_action_dag_rejects_invalid_edge_endpoint() -> None:
    with pytest.raises(ValidationError, match="edge target missing"):
        ActionDAG(nodes=[_node("a")], edges=[("a", "missing")])


def test_action_dag_rejects_cycles_from_dependencies() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        ActionDAG(nodes=[_node("a", depends_on=["b"]), _node("b", depends_on=["a"])])


def test_action_dag_rejects_cycles_from_edges() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        ActionDAG(nodes=[_node("a"), _node("b")], edges=[("a", "b"), ("b", "a")])


def test_action_dag_rejects_unknown_capability_unless_unresolved() -> None:
    with pytest.raises(ValidationError, match="unknown capability_id"):
        ActionDAG(nodes=[_node("a", capability_id="unknown")])


def test_action_dag_accepts_unknown_capability_when_unresolved() -> None:
    node = _node("a", capability_id="unknown")
    node.safety_labels.append("unresolved")

    dag = ActionDAG(nodes=[node])

    assert dag.nodes[0].capability_id == "unknown"
