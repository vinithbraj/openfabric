"""Execution pipeline engine."""

from __future__ import annotations

from collections import deque
from typing import Any

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.config import RuntimeConfig
from agent_runtime.core.types import ActionDAG, ActionNode, ExecutionResult, ResultBundle
from agent_runtime.execution.errors import ExecutionError
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.execution.safety import (
    SafetyDecision,
    SafetyPolicy,
    evaluate_dag_safety,
)


def _coerce_safety_policy(safety_policy: RuntimeConfig | SafetyPolicy | dict[str, Any] | None) -> SafetyPolicy:
    """Normalize execution safety input into a SafetyPolicy instance."""

    if isinstance(safety_policy, SafetyPolicy):
        return safety_policy
    if isinstance(safety_policy, RuntimeConfig):
        return SafetyPolicy(safety_policy)
    if isinstance(safety_policy, dict):
        return SafetyPolicy(RuntimeConfig.model_validate(safety_policy))
    return SafetyPolicy(RuntimeConfig())


def _topological_sort(dag: ActionDAG) -> list[ActionNode]:
    """Return DAG nodes in dependency order while preserving stable input order."""

    node_by_id = {node.id: node for node in dag.nodes}
    adjacency: dict[str, set[str]] = {node.id: set() for node in dag.nodes}
    indegree: dict[str, int] = {node.id: 0 for node in dag.nodes}

    for node in dag.nodes:
        for dependency in node.depends_on:
            if node.id not in adjacency[dependency]:
                adjacency[dependency].add(node.id)
                indegree[node.id] += 1

    for source, target in dag.edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1

    stable_order = {node.id: index for index, node in enumerate(dag.nodes)}
    queue = deque(
        sorted(
            [node_id for node_id, degree in indegree.items() if degree == 0],
            key=lambda node_id: stable_order[node_id],
        )
    )
    ordered_ids: list[str] = []

    while queue:
        node_id = queue.popleft()
        ordered_ids.append(node_id)
        for child_id in sorted(adjacency[node_id], key=lambda item: stable_order[item]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)

    if len(ordered_ids) != len(dag.nodes):
        raise ExecutionError("Unable to topologically sort DAG.")

    return [node_by_id[node_id] for node_id in ordered_ids]


def _dependency_ids(node: ActionNode, dag: ActionDAG) -> set[str]:
    """Return direct dependency ids for one node from both node and edge declarations."""

    dependencies = set(node.depends_on)
    dependencies.update(source for source, target in dag.edges if target == node.id)
    return dependencies


class ExecutionEngine:
    """Execute a typed DAG through registered capabilities."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        safety_policy: RuntimeConfig | SafetyPolicy | dict[str, Any] | None = None,
        result_store: InMemoryResultStore | None = None,
    ) -> None:
        self.registry = registry
        self.safety_policy = _coerce_safety_policy(safety_policy)
        self.result_store = result_store or InMemoryResultStore()

    def _safety_bundle(
        self,
        dag: ActionDAG,
        decision: SafetyDecision,
        safe_summary: str,
        metadata: dict[str, Any],
    ) -> ResultBundle:
        """Return a normalized error bundle for blocked or gated execution."""

        return ResultBundle(
            dag_id=dag.dag_id,
            results=[],
            status="error",
            safe_summary=safe_summary,
            metadata={
                "blocked_reasons": list(decision.blocked_reasons),
                "warnings": list(decision.warnings),
                "requires_confirmation": decision.requires_confirmation,
                **metadata,
            },
        )

    def _normalize_result(
        self,
        node: ActionNode,
        raw_result: ExecutionResult,
    ) -> ExecutionResult:
        """Store large payloads by reference and emit only safe previews."""

        preview_source = raw_result.data_preview
        data_ref = raw_result.data_ref
        safe_preview = preview_source

        if preview_source is not None:
            safe_preview = self.result_store.preview(
                preview_source,
                self.safety_policy.config.max_output_preview_bytes,
            )
            if safe_preview != preview_source or (
                isinstance(safe_preview, dict) and safe_preview.get("truncated") is True
            ):
                data_ref = self.result_store.put(preview_source)

        normalized_metadata = {
            "capability_id": node.capability_id,
            "operation_id": node.operation_id,
            **dict(raw_result.metadata),
        }

        normalized = ExecutionResult(
            node_id=node.id,
            status=raw_result.status,
            data_ref=data_ref,
            data_preview=safe_preview,
            error=raw_result.error,
            metadata=normalized_metadata,
        )
        self.result_store.add(normalized)
        return normalized

    def execute(self, dag: ActionDAG, context: dict[str, Any] | None = None) -> ResultBundle:
        """Execute a DAG in dependency order and return a normalized bundle."""

        context = dict(context or {})
        confirmation_granted = bool(context.get("confirmation", False))
        effective_config = self.safety_policy.config.model_copy(
            update={
                "confirmation_granted": (
                    self.safety_policy.config.confirmation_granted or confirmation_granted
                ),
                "allow_mutating_capabilities": (
                    self.safety_policy.config.allow_mutating_capabilities or confirmation_granted
                ),
            }
        )
        decision = evaluate_dag_safety(dag, self.registry, effective_config)
        if not decision.allowed:
            return self._safety_bundle(
                dag,
                decision,
                safe_summary="Execution blocked by safety policy.",
                metadata={"confirmation_required": False},
            )

        if decision.requires_confirmation and not confirmation_granted:
            return self._safety_bundle(
                dag,
                decision,
                safe_summary="Execution requires confirmation before proceeding.",
                metadata={"confirmation_required": True},
            )

        executable_dag = decision.sanitized_dag or dag
        ordered_nodes = _topological_sort(executable_dag)
        stop_on_error = bool(context.get("stop_on_error", self.safety_policy.config.stop_on_error))
        runtime_policy = SafetyPolicy(effective_config)

        results: list[ExecutionResult] = []
        node_status: dict[str, str] = {}
        hard_stop = False

        for node in ordered_nodes:
            dependency_statuses = {
                dependency_id: node_status.get(dependency_id)
                for dependency_id in _dependency_ids(node, executable_dag)
            }
            if hard_stop:
                skipped = ExecutionResult(
                    node_id=node.id,
                    status="skipped",
                    error="Skipped because stop_on_error halted downstream execution.",
                    metadata={
                        "capability_id": node.capability_id,
                        "operation_id": node.operation_id,
                    },
                )
                results.append(skipped)
                node_status[node.id] = skipped.status
                self.result_store.add(skipped)
                continue

            if any(status in {"error", "skipped"} for status in dependency_statuses.values()):
                skipped = ExecutionResult(
                    node_id=node.id,
                    status="skipped",
                    error="Skipped because one or more dependencies did not succeed.",
                    metadata={
                        "capability_id": node.capability_id,
                        "operation_id": node.operation_id,
                        "dependency_statuses": dependency_statuses,
                    },
                )
                results.append(skipped)
                node_status[node.id] = skipped.status
                self.result_store.add(skipped)
                continue

            capability = self.registry.get(node.capability_id)
            runtime_policy.assert_allowed(capability, node.operation_id)
            try:
                raw_result = capability.execute(
                    node.arguments,
                    {
                        "node_id": node.id,
                        "task_id": node.task_id,
                        "execution_context": context,
                        "result_store": self.result_store,
                        "config": self.safety_policy.config,
                    },
                )
            except Exception as exc:
                errored = ExecutionResult(
                    node_id=node.id,
                    status="error",
                    error=str(exc),
                    metadata={
                        "capability_id": node.capability_id,
                        "operation_id": node.operation_id,
                    },
                )
                results.append(errored)
                node_status[node.id] = errored.status
                self.result_store.add(errored)
                if stop_on_error:
                    hard_stop = True
                continue

            normalized = self._normalize_result(node, raw_result)
            results.append(normalized)
            node_status[node.id] = normalized.status
            if normalized.status == "error" and stop_on_error:
                hard_stop = True

        if results and all(result.status == "success" for result in results):
            bundle_status = "success"
            safe_summary = "All DAG nodes executed successfully."
        elif any(result.status == "error" for result in results):
            bundle_status = "partial" if any(
                result.status == "success" for result in results
            ) else "error"
            safe_summary = "Execution completed with one or more errors."
        elif any(result.status == "skipped" for result in results):
            bundle_status = "partial"
            safe_summary = "Execution completed with skipped downstream nodes."
        else:
            bundle_status = "error"
            safe_summary = "No executable results were produced."

        return ResultBundle(
            dag_id=executable_dag.dag_id,
            results=results,
            status=bundle_status,
            safe_summary=safe_summary,
            metadata={
                "warnings": list(decision.warnings),
                "confirmation_required": False,
                "stop_on_error": stop_on_error,
            },
        )
