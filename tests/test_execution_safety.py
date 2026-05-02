from __future__ import annotations

import pytest

from agent_runtime.capabilities import (
    CapabilityRegistry,
    ListDirectoryCapability,
    ReadQueryCapability,
    ShellCheckPortCapability,
    WriteFileCapability,
)
from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.config import RuntimeConfig
from agent_runtime.core.errors import SafetyError
from agent_runtime.core.types import ActionDAG, ActionNode, ExecutionResult
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.safety import SafetyDecision, evaluate_dag_safety
from agent_runtime.llm.reproducibility import hash_action_dag


def _ready_dag(dag: ActionDAG, allowed: bool = True) -> ActionDAG:
    prepared = dag.model_copy(
        update={
            "execution_ready": allowed,
            "safety_decision": {"allowed": allowed},
        }
    )
    return prepared.model_copy(update={"final_dag_hash": hash_action_dag(prepared)})


class EchoCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="echo.read",
        domain="generic",
        operation_id="read",
        name="Echo Read",
        description="Return the input.",
        semantic_verbs=["read"],
        object_types=["generic"],
        argument_schema={"value": {"type": "string"}},
        required_arguments=["value"],
        optional_arguments=[],
        output_schema={"value": {"type": "string"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"value": "hello"}}],
        safety_notes=[],
    )

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        validated = self.validate_arguments(arguments)
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview=dict(validated),
        )


class MutatingCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="echo.write",
        domain="generic",
        operation_id="write",
        name="Echo Write",
        description="Mutating placeholder.",
        semantic_verbs=["update"],
        object_types=["generic"],
        argument_schema={"value": {"type": "string"}},
        required_arguments=["value"],
        optional_arguments=[],
        output_schema={"value": {"type": "string"}},
        risk_level="medium",
        read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        examples=[{"arguments": {"value": "hello"}}],
        safety_notes=["Mutates placeholder state."],
    )


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
        execution_backend="gateway",
        backend_operation="filesystem.delete_file",
        risk_level="high",
        read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        examples=[{"arguments": {"path": "old.log"}}],
        safety_notes=["Destructive filesystem mutation."],
    )


def _engine() -> ExecutionEngine:
    registry = CapabilityRegistry()
    registry.register(EchoCapability())
    registry.register(MutatingCapability())
    return ExecutionEngine(registry)


def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(ListDirectoryCapability())
    registry.register(WriteFileCapability())
    registry.register(ReadQueryCapability())
    registry.register(ShellCheckPortCapability())
    registry.register(DeleteFileCapability())
    registry.register(EchoCapability())
    registry.register(MutatingCapability())
    return registry


def test_execution_engine_invokes_registered_capability() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-1",
                description="read value",
                semantic_verb="read",
                capability_id="echo.read",
                operation_id="read",
                arguments={"value": "hello"},
                safety_labels=[],
            )
        ]
    )

    bundle = _engine().execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "success"
    assert bundle.results[0].node_id == "node-1"
    assert bundle.results[0].data_preview == {"value": "hello"}


def test_execution_safety_blocks_mutating_action_by_default() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                task_id="task-1",
                description="write value",
                semantic_verb="update",
                capability_id="echo.write",
                operation_id="write",
                arguments={"value": "hello"},
                safety_labels=[],
            )
        ]
    )

    bundle = _engine().execute(_ready_dag(dag))

    assert bundle.status == "confirmation_required"
    assert bundle.metadata["confirmation_required"] is True


def test_safe_filesystem_read_is_allowed() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-fs-read",
                task_id="task-fs-read",
                description="list files",
                semantic_verb="read",
                capability_id="filesystem.list_directory",
                operation_id="list_directory",
                arguments={"path": "."},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig(gateway_url="http://gateway"))

    assert isinstance(decision, SafetyDecision)
    assert decision.allowed is True
    assert decision.requires_confirmation is False
    assert decision.sanitized_dag is not None
    assert decision.sanitized_dag.nodes[0].arguments["path"] == "."
    assert decision.sanitized_dag.nodes[0].arguments["limit"] == 1000


def test_unsafe_path_traversal_is_blocked() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-traversal",
                task_id="task-traversal",
                description="read parent secret",
                semantic_verb="read",
                capability_id="filesystem.list_directory",
                operation_id="list_directory",
                arguments={"path": "../secrets"},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig(gateway_url="http://gateway"))

    assert decision.allowed is False
    assert any("outside the workspace root" in reason.lower() for reason in decision.blocked_reasons)


def test_delete_requires_confirmation() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-delete",
                task_id="task-delete",
                description="delete old.log",
                semantic_verb="delete",
                capability_id="filesystem.delete_file",
                operation_id="delete_file",
                arguments={"path": "old.log"},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig(gateway_url="http://gateway"))

    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert decision.sanitized_dag is not None
    assert "requires-confirmation" in decision.sanitized_dag.nodes[0].safety_labels


def test_write_file_requires_confirmation_and_normalizes_path() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-write",
                task_id="task-write",
                description="save report to report.txt",
                semantic_verb="create",
                capability_id="filesystem.write_file",
                operation_id="write_file",
                arguments={"path": "./reports/report.txt", "format": "text", "content": "hello"},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig(gateway_url="http://gateway"))

    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert decision.sanitized_dag is not None
    assert decision.sanitized_dag.nodes[0].arguments["path"] == "reports/report.txt"


def test_shell_command_is_blocked_by_default() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-shell",
                task_id="task-shell",
                description="check one port",
                semantic_verb="read",
                capability_id="shell.check_port",
                operation_id="check_port",
                arguments={"port": 8310},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig(gateway_url="http://gateway"))

    assert decision.allowed is False
    assert any("shell execution is disabled" in reason.lower() for reason in decision.blocked_reasons)


def test_shell_command_is_allowed_when_policy_enables_it() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-shell-allowed",
                task_id="task-shell-allowed",
                description="check one port",
                semantic_verb="read",
                capability_id="shell.check_port",
                operation_id="check_port",
                arguments={"port": 8310},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(
        dag,
        _registry(),
        RuntimeConfig(gateway_url="http://gateway", allow_shell_execution=True),
    )

    assert decision.allowed is True
    assert decision.sanitized_dag is not None


def test_sql_read_is_allowed() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-sql-read",
                task_id="task-sql-read",
                description="select patients",
                semantic_verb="read",
                capability_id="sql.read_query",
                operation_id="read_query",
                arguments={"query_intent": {"template": "select_rows", "table": "patients"}},
                safety_labels=[],
            )
        ]
    )

    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig())

    assert decision.allowed is True
    assert decision.sanitized_dag is not None
    assert decision.sanitized_dag.nodes[0].arguments["limit"] == 100


def test_sql_write_is_blocked() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-sql-write",
                task_id="task-sql-write",
                description="update patients",
                semantic_verb="update",
                capability_id="sql.read_query",
                operation_id="read_query",
                arguments={"query_intent": {"template": "update_rows", "table": "patients"}},
                safety_labels=[],
            )
        ]
    )


    decision = evaluate_dag_safety(dag, _registry(), RuntimeConfig())

    assert decision.allowed is False
    assert any("sql write operations are blocked" in reason.lower() for reason in decision.blocked_reasons)
