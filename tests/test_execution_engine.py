from __future__ import annotations

from agent_runtime.capabilities import CapabilityRegistry
from agent_runtime.capabilities.base import BaseCapability, GatewayBackedCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.config import RuntimeConfig
from agent_runtime.core.types import ActionDAG, ActionNode, ExecutionResult, InputRef
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.llm.reproducibility import hash_action_dag


def _ready_dag(dag: ActionDAG, allowed: bool = True) -> ActionDAG:
    prepared = dag.model_copy(
        update={
            "execution_ready": allowed,
            "safety_decision": {"allowed": allowed},
        }
    )
    return prepared.model_copy(update={"final_dag_hash": hash_action_dag(prepared)})


class CountingReadCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="counting.read",
        domain="generic",
        operation_id="read",
        name="Counting Read",
        description="Return a tracked value.",
        semantic_verbs=["read"],
        object_types=["generic"],
        argument_schema={"value": {"type": "string"}},
        required_arguments=["value"],
        optional_arguments=["path"],
        output_schema={"value": {"type": "string"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"value": "hello"}}],
        safety_notes=[],
    )

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        self.calls += 1
        validated = self.validate_arguments(arguments)
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"value": validated["value"]},
            metadata={"calls": self.calls},
        )


class FailingCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="counting.fail",
        domain="generic",
        operation_id="read",
        name="Failing Read",
        description="Always fail.",
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

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        self.calls += 1
        raise RuntimeError("simulated failure")


class DeleteCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="generic.delete",
        domain="generic",
        operation_id="delete",
        name="Delete Value",
        description="Delete a generic value.",
        semantic_verbs=["delete"],
        object_types=["generic"],
        argument_schema={"value": {"type": "string"}},
        required_arguments=["value"],
        optional_arguments=[],
        output_schema={"deleted": {"type": "boolean"}},
        risk_level="high",
        read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        examples=[{"arguments": {"value": "old"}}],
        safety_notes=["Mutates state."],
    )

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        self.calls += 1
        validated = self.validate_arguments(arguments)
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"deleted": validated["value"]},
        )


class LargeOutputCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="generic.large",
        domain="generic",
        operation_id="read",
        name="Large Output",
        description="Return a large payload.",
        semantic_verbs=["read"],
        object_types=["generic"],
        argument_schema={},
        required_arguments=[],
        optional_arguments=[],
        output_schema={"blob": {"type": "string"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {}}],
        safety_notes=[],
    )

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"blob": "x" * 200},
        )


class GatewayListCapability(GatewayBackedCapability):
    manifest = CapabilityManifest(
        capability_id="filesystem.list_directory",
        domain="filesystem",
        operation_id="list_directory",
        name="Gateway List Directory",
        description="List directory via gateway.",
        semantic_verbs=["read"],
        object_types=["filesystem"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=["path"],
        optional_arguments=["limit"],
        output_schema={"entries": {"type": "array"}},
        execution_backend="gateway",
        backend_operation="filesystem.list_directory",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": "."}}],
        safety_notes=[],
    )


class FakeGatewayClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def invoke(self, *, node, capability, arguments, execution_context) -> ExecutionResult:
        self.calls.append(
            {
                "node_id": node.id,
                "capability_id": capability.manifest.capability_id,
                "arguments": dict(arguments),
                "execution_context": dict(execution_context),
            }
        )
        return ExecutionResult(
            node_id=node.id,
            status="success",
            data_preview={"entries": [{"name": "README.md", "path": "README.md"}]},
            metadata={"transport": "gateway"},
        )


class ProduceRowsCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="produce.rows",
        domain="generic",
        operation_id="read",
        name="Produce Rows",
        description="Produce tabular rows for downstream tests.",
        semantic_verbs=["read"],
        object_types=["table"],
        argument_schema={"rows": {"type": "array"}},
        required_arguments=["rows"],
        optional_arguments=[],
        output_schema={"rows": {"type": "array"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"rows": [{"value": 1}]}}],
        safety_notes=[],
    )

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        validated = self.validate_arguments(arguments)
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"rows": list(validated["rows"])},
            metadata={"data_type": "table"},
        )


class CountRowsCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="count.rows",
        domain="generic",
        operation_id="analyze",
        name="Count Rows",
        description="Count rows from resolved upstream input.",
        semantic_verbs=["analyze"],
        object_types=["table", "summary"],
        argument_schema={"input_ref": {"type": "string"}},
        required_arguments=["input_ref"],
        optional_arguments=[],
        output_schema={"value": {"type": "integer"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"input_ref": "node-a.output"}}],
        safety_notes=[],
    )

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        validated = self.validate_arguments(arguments)
        input_data = validated["input_ref"]
        if not isinstance(input_data, dict) or not isinstance(input_data.get("rows"), list):
            raise RuntimeError("resolved input_ref did not produce row data")
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"value": len(input_data["rows"])},
            metadata={"data_type": "scalar"},
        )


class FormatValueCapability(BaseCapability):
    manifest = CapabilityManifest(
        capability_id="format.value",
        domain="generic",
        operation_id="render",
        name="Format Value",
        description="Format one upstream scalar value.",
        semantic_verbs=["render"],
        object_types=["summary"],
        argument_schema={"input_ref": {"type": "string"}},
        required_arguments=["input_ref"],
        optional_arguments=[],
        output_schema={"text": {"type": "string"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"input_ref": "node-b.value"}}],
        safety_notes=[],
    )

    def execute(self, arguments: dict[str, object], context: dict[str, object]) -> ExecutionResult:
        validated = self.validate_arguments(arguments)
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"text": f"count={validated['input_ref']}"},
            metadata={"data_type": "text"},
        )


def _registry() -> tuple[CapabilityRegistry, CountingReadCapability, FailingCapability, DeleteCapability, LargeOutputCapability]:
    registry = CapabilityRegistry()
    counting = CountingReadCapability()
    failing = FailingCapability()
    deleting = DeleteCapability()
    large = LargeOutputCapability()
    registry.register(counting)
    registry.register(failing)
    registry.register(deleting)
    registry.register(large)
    return registry, counting, failing, deleting, large


def test_successful_linear_dag_executes_in_dependency_order() -> None:
    registry, counting, _, _, _ = _registry()
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-a",
                task_id="task-a",
                description="first read",
                semantic_verb="read",
                capability_id="counting.read",
                operation_id="read",
                arguments={"value": "a"},
                safety_labels=[],
            ),
            ActionNode(
                id="node-b",
                task_id="task-b",
                description="second read",
                semantic_verb="read",
                capability_id="counting.read",
                operation_id="read",
                arguments={"value": "b"},
                depends_on=["node-a"],
                safety_labels=[],
            ),
        ],
        edges=[("node-a", "node-b")],
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "success"
    assert [result.node_id for result in bundle.results] == ["node-a", "node-b"]
    assert counting.calls == 2


def test_failed_dependency_causes_downstream_skip() -> None:
    registry, counting, failing, _, _ = _registry()
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-fail",
                task_id="task-fail",
                description="fail first",
                semantic_verb="read",
                capability_id="counting.fail",
                operation_id="read",
                arguments={"value": "x"},
                safety_labels=[],
            ),
            ActionNode(
                id="node-after",
                task_id="task-after",
                description="read after failure",
                semantic_verb="read",
                capability_id="counting.read",
                operation_id="read",
                arguments={"value": "after"},
                depends_on=["node-fail"],
                safety_labels=[],
            ),
        ],
        edges=[("node-fail", "node-after")],
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True, "stop_on_error": False})

    assert bundle.status == "error"
    assert failing.calls == 1
    assert counting.calls == 0
    assert bundle.results[0].status == "error"
    assert bundle.results[1].status == "skipped"


def test_blocked_dag_does_not_execute() -> None:
    registry, counting, _, _, _ = _registry()
    engine = ExecutionEngine(registry, RuntimeConfig(workspace_root="."))
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-blocked",
                task_id="task-blocked",
                description="blocked traversal",
                semantic_verb="read",
                capability_id="counting.read",
                operation_id="read",
                arguments={"value": "x", "path": "../secret.txt"},
                safety_labels=[],
            )
        ]
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "error"
    assert counting.calls == 0
    assert bundle.metadata["blocked_reasons"]


def test_confirmation_required_dag_does_not_execute_without_confirmation() -> None:
    registry, _, _, deleting, _ = _registry()
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-delete",
                task_id="task-delete",
                description="delete value",
                semantic_verb="delete",
                capability_id="generic.delete",
                operation_id="delete",
                arguments={"value": "old"},
                safety_labels=[],
            )
        ]
    )

    bundle = engine.execute(_ready_dag(dag), {})

    assert bundle.status == "confirmation_required"
    assert bundle.metadata["confirmation_required"] is True
    assert deleting.calls == 0


def test_large_output_is_stored_by_reference() -> None:
    registry, _, _, _, large = _registry()
    store = InMemoryResultStore()
    engine = ExecutionEngine(
        registry,
        RuntimeConfig(max_output_preview_bytes=64),
        store,
    )
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-large",
                task_id="task-large",
                description="read large output",
                semantic_verb="read",
                capability_id="generic.large",
                operation_id="read",
                arguments={},
                safety_labels=[],
            )
        ]
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "success"
    result = bundle.results[0]
    assert result.data_ref is not None
    assert result.data_preview is not None
    assert result.data_preview["truncated"] is True
    assert store.get(result.data_ref.ref_id) == {"blob": "x" * 200}


def test_gateway_backed_node_executes_through_gateway_client() -> None:
    registry = CapabilityRegistry()
    registry.register(GatewayListCapability())
    gateway_client = FakeGatewayClient()
    engine = ExecutionEngine(
        registry,
        RuntimeConfig(workspace_root=".", gateway_url="http://gateway"),
        gateway_client=gateway_client,
    )
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-gateway",
                task_id="task-gateway",
                description="list files remotely",
                semantic_verb="read",
                capability_id="filesystem.list_directory",
                operation_id="list_directory",
                arguments={"path": "."},
                safety_labels=[],
            )
        ]
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "success"
    assert bundle.results[0].metadata["transport"] == "gateway"
    assert gateway_client.calls[0]["capability_id"] == "filesystem.list_directory"


def test_node_b_consumes_node_a_output_successfully() -> None:
    registry = CapabilityRegistry()
    registry.register(ProduceRowsCapability())
    registry.register(CountRowsCapability())
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-a",
                task_id="task-a",
                description="produce rows",
                semantic_verb="read",
                capability_id="produce.rows",
                operation_id="read",
                arguments={"rows": [{"value": 1}, {"value": 2}]},
                safety_labels=[],
            ),
            ActionNode(
                id="node-b",
                task_id="task-b",
                description="count rows",
                semantic_verb="analyze",
                capability_id="count.rows",
                operation_id="analyze",
                arguments={"input_ref": InputRef(source_node_id="node-a", expected_data_type="table")},
                depends_on=["node-a"],
                safety_labels=[],
            ),
        ],
        edges=[("node-a", "node-b")],
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "success"
    assert bundle.results[1].data_preview == {"value": 2}


def test_node_c_consumes_node_b_output_successfully() -> None:
    registry = CapabilityRegistry()
    registry.register(ProduceRowsCapability())
    registry.register(CountRowsCapability())
    registry.register(FormatValueCapability())
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-a",
                task_id="task-a",
                description="produce rows",
                semantic_verb="read",
                capability_id="produce.rows",
                operation_id="read",
                arguments={"rows": [{"value": 1}, {"value": 2}, {"value": 3}]},
                safety_labels=[],
            ),
            ActionNode(
                id="node-b",
                task_id="task-b",
                description="count rows",
                semantic_verb="analyze",
                capability_id="count.rows",
                operation_id="analyze",
                arguments={"input_ref": InputRef(source_node_id="node-a", expected_data_type="table")},
                depends_on=["node-a"],
                safety_labels=[],
            ),
            ActionNode(
                id="node-c",
                task_id="task-c",
                description="format count",
                semantic_verb="render",
                capability_id="format.value",
                operation_id="render",
                arguments={"input_ref": InputRef(source_node_id="node-b", output_key="value", expected_data_type="scalar")},
                depends_on=["node-b"],
                safety_labels=[],
            ),
        ],
        edges=[("node-a", "node-b"), ("node-b", "node-c")],
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True})

    assert bundle.status == "success"
    assert bundle.results[2].data_preview == {"text": "count=3"}


def test_input_ref_to_failed_node_causes_dependent_skip() -> None:
    registry = CapabilityRegistry()
    registry.register(FailingCapability())
    registry.register(CountRowsCapability())
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-fail",
                task_id="task-fail",
                description="fail first",
                semantic_verb="read",
                capability_id="counting.fail",
                operation_id="read",
                arguments={"value": "x"},
                safety_labels=[],
            ),
            ActionNode(
                id="node-b",
                task_id="task-b",
                description="count rows",
                semantic_verb="analyze",
                capability_id="count.rows",
                operation_id="analyze",
                arguments={"input_ref": InputRef(source_node_id="node-fail", expected_data_type="table")},
                depends_on=["node-fail"],
                safety_labels=[],
            ),
        ],
        edges=[("node-fail", "node-b")],
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True, "stop_on_error": False})

    assert bundle.results[0].status == "error"
    assert bundle.results[1].status == "skipped"


def test_input_ref_data_type_mismatch_is_rejected() -> None:
    registry = CapabilityRegistry()
    registry.register(ProduceRowsCapability())
    registry.register(CountRowsCapability())
    engine = ExecutionEngine(registry)
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-a",
                task_id="task-a",
                description="produce rows",
                semantic_verb="read",
                capability_id="produce.rows",
                operation_id="read",
                arguments={"rows": [{"value": 1}]},
                safety_labels=[],
            ),
            ActionNode(
                id="node-b",
                task_id="task-b",
                description="count rows",
                semantic_verb="analyze",
                capability_id="count.rows",
                operation_id="analyze",
                arguments={"input_ref": InputRef(source_node_id="node-a", expected_data_type="scalar")},
                depends_on=["node-a"],
                safety_labels=[],
            ),
        ],
        edges=[("node-a", "node-b")],
    )

    bundle = engine.execute(_ready_dag(dag), {"confirmation": True, "stop_on_error": False})

    assert bundle.status == "partial"
    assert bundle.results[1].status == "error"
    assert "expected data_type" in (bundle.results[1].error or "")
