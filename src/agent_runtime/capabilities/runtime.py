"""Internal runtime self-introspection capabilities."""

from __future__ import annotations

from typing import Any

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.types import ExecutionResult


def _manifest_record(manifest: CapabilityManifest, include_details: bool) -> dict[str, Any]:
    """Convert one capability manifest into a safe introspection record."""

    record: dict[str, Any] = {
        "capability_id": manifest.capability_id,
        "operation_id": manifest.operation_id,
        "name": manifest.name,
        "read_only": manifest.read_only,
        "risk_level": manifest.risk_level,
    }
    if include_details:
        record.update(
            {
                "description": manifest.description,
                "semantic_verbs": list(manifest.semantic_verbs),
                "object_types": list(manifest.object_types),
                "output_object_types": list(manifest.output_object_types),
                "output_fields": list(manifest.output_fields),
                "output_affordances": list(manifest.output_affordances),
            }
        )
    return record


def _runtime_state_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime state snapshot from capability execution context."""

    payload = dict(context or {})
    runtime_state = payload.get("runtime_state")
    if isinstance(runtime_state, dict):
        return runtime_state
    nested = payload.get("execution_context")
    if isinstance(nested, dict):
        nested_state = nested.get("runtime_state")
        if isinstance(nested_state, dict):
            return nested_state
    return {}


class RuntimeDescribeCapabilitiesCapability(BaseCapability):
    """Describe the registered runtime capabilities safely from the registry."""

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry
        self.manifest = CapabilityManifest(
            capability_id="runtime.describe_capabilities",
            domain="runtime",
            operation_id="describe_capabilities",
            name="Describe Runtime Capabilities",
            description="Describe currently registered runtime capabilities grouped by domain.",
            semantic_verbs=["read", "summarize"],
            object_types=["runtime.capabilities", "capabilities", "tools", "registry"],
            argument_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "include_details": {"type": "boolean"},
                },
            },
            required_arguments=[],
            optional_arguments=["domain", "include_details"],
            output_schema={"type": "object"},
            execution_backend="internal",
            backend_operation="runtime.describe_capabilities",
            risk_level="low",
            read_only=True,
            mutates_state=False,
            requires_confirmation=False,
            examples=[
                {"prompt": "what are my capabilities?"},
                {"prompt": "list registered capabilities"},
            ],
            safety_notes=[
                "Exposes only registry metadata.",
                "Does not reveal backend command strings or gateway internals.",
            ],
        )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        validated = self.validate_arguments(arguments)
        domain_filter = str(validated.get("domain") or "").strip().lower() or None
        include_details = bool(validated.get("include_details", False))

        manifests = self.registry.list_manifests()
        if domain_filter is not None:
            manifests = [manifest for manifest in manifests if manifest.domain.lower() == domain_filter]
        manifests = sorted(manifests, key=lambda manifest: (manifest.domain.lower(), manifest.capability_id))

        grouped: dict[str, list[dict[str, Any]]] = {}
        rows: list[dict[str, Any]] = []
        for manifest in manifests:
            record = _manifest_record(manifest, include_details)
            grouped.setdefault(manifest.domain, []).append(record)
            rows.append({"domain": manifest.domain, **record})

        grouped_payload: dict[str, list[Any]]
        if include_details:
            grouped_payload = grouped
        else:
            grouped_payload = {
                domain: [record["capability_id"] for record in records]
                for domain, records in grouped.items()
            }

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "title": "Runtime Capabilities",
                "domain_filter": domain_filter,
                "include_details": include_details,
                "domains": sorted(grouped),
                "capability_count": len(rows),
                "rows": rows if include_details else [],
                "grouped_capabilities": grouped_payload,
            },
            metadata={"data_type": "table"},
        )


class RuntimeDescribePipelineCapability(BaseCapability):
    """Describe the current runtime pipeline stages and control boundaries."""

    manifest = CapabilityManifest(
        capability_id="runtime.describe_pipeline",
        domain="runtime",
        operation_id="describe_pipeline",
        name="Describe Runtime Pipeline",
        description="Describe the current runtime pipeline stages and indicate which are LLM-assisted or deterministic.",
        semantic_verbs=["read", "summarize"],
        object_types=["runtime.pipeline", "pipeline", "architecture"],
        argument_schema={"type": "object", "properties": {}},
        required_arguments=[],
        optional_arguments=[],
        output_schema={"type": "object"},
        execution_backend="internal",
        backend_operation="runtime.describe_pipeline",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {"prompt": "show pipeline"},
            {"prompt": "how does this runtime work?"},
        ],
        safety_notes=["Exposes only pipeline metadata."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        _ = self.validate_arguments(arguments)
        rows = [
            {
                "stage": "prompt_classification",
                "type": "llm_assisted",
                "description": "Classify the prompt and decide whether tools are required.",
            },
            {
                "stage": "task_decomposition",
                "type": "llm_assisted",
                "description": "Break the prompt into atomic user-meaningful tasks.",
            },
            {
                "stage": "decomposition_critique",
                "type": "llm_assisted_optional",
                "description": "Advisory critique of the proposed decomposition before task normalization.",
            },
            {
                "stage": "verb_assignment",
                "type": "llm_assisted",
                "description": "Assign semantic verbs, object types, and risk hints to tasks.",
            },
            {
                "stage": "capability_selection",
                "type": "llm_assisted_plus_deterministic_validation",
                "description": "Rank candidate capabilities and validate them against the registry.",
            },
            {
                "stage": "argument_extraction",
                "type": "llm_assisted_plus_deterministic_validation",
                "description": "Extract typed arguments and normalize them against capability schemas.",
            },
            {
                "stage": "dag_construction",
                "type": "deterministic",
                "description": "Build the typed ActionDAG and wire dependencies and input references.",
            },
            {
                "stage": "dag_review",
                "type": "llm_assisted_optional",
                "description": "Advisory review of sanitized DAG metadata without execution authority.",
            },
            {
                "stage": "safety_evaluation",
                "type": "deterministic",
                "description": "Apply safety policy, confirmation rules, and gateway/backend checks.",
            },
            {
                "stage": "execution",
                "type": "deterministic",
                "description": "Execute trusted DAG nodes locally, internally, or through the gateway.",
            },
            {
                "stage": "failure_repair",
                "type": "llm_assisted_optional",
                "description": "Optional one-shot repair proposal after a safe failure, subject to validation.",
            },
            {
                "stage": "output_composition",
                "type": "llm_assisted_plus_deterministic_rendering",
                "description": "Select a display plan from safe previews and render the final response deterministically.",
            },
        ]
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "title": "Runtime Pipeline",
                "rows": rows,
                "stage_count": len(rows),
            },
            metadata={"data_type": "table"},
        )


class RuntimeShowLastPlanCapability(BaseCapability):
    """Return a safe summary of the last planned request."""

    manifest = CapabilityManifest(
        capability_id="runtime.show_last_plan",
        domain="runtime",
        operation_id="show_last_plan",
        name="Show Last Plan",
        description="Show a safe summary of the last prompt, tasks, selected capabilities, DAG nodes, and safety decision.",
        semantic_verbs=["read", "summarize"],
        object_types=["runtime.plan", "plan", "dag"],
        argument_schema={"type": "object", "properties": {}},
        required_arguments=[],
        optional_arguments=[],
        output_schema={"type": "object"},
        execution_backend="internal",
        backend_operation="runtime.show_last_plan",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {"prompt": "show last plan"},
            {"prompt": "show the DAG"},
        ],
        safety_notes=["Shows only sanitized planning metadata and no raw tool outputs."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        _ = self.validate_arguments(arguments)
        runtime_state = _runtime_state_from_context(context)
        snapshot = runtime_state.get("last_plan")
        if not isinstance(snapshot, dict):
            return ExecutionResult(
                node_id=str(context.get("node_id") or ""),
                status="success",
                data_preview={
                    "message": "No previous plan is available yet.",
                    "available": False,
                },
                metadata={"data_type": "summary"},
            )

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "available": True,
                "prompt": snapshot.get("prompt"),
                "request_id": snapshot.get("request_id"),
                "task_count": snapshot.get("task_count"),
                "tasks": snapshot.get("tasks", []),
                "selected_capabilities": snapshot.get("selected_capabilities", []),
                "dag_id": snapshot.get("dag_id"),
                "rows": snapshot.get("rows", []),
                "safety_decision": snapshot.get("safety_decision", {}),
            },
            metadata={"data_type": "table"},
        )


class RuntimeExplainLastFailureCapability(BaseCapability):
    """Return a safe explanation of the last runtime failure."""

    manifest = CapabilityManifest(
        capability_id="runtime.explain_last_failure",
        domain="runtime",
        operation_id="explain_last_failure",
        name="Explain Last Failure",
        description="Explain the last runtime failure category and safe reason.",
        semantic_verbs=["read", "summarize"],
        object_types=["runtime.failure", "failure", "error"],
        argument_schema={"type": "object", "properties": {}},
        required_arguments=[],
        optional_arguments=[],
        output_schema={"type": "object"},
        execution_backend="internal",
        backend_operation="runtime.explain_last_failure",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {"prompt": "why did that fail?"},
            {"prompt": "explain the last error"},
        ],
        safety_notes=["Exposes only safe failure category and reason."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        _ = self.validate_arguments(arguments)
        runtime_state = _runtime_state_from_context(context)
        snapshot = runtime_state.get("last_failure")
        if not isinstance(snapshot, dict):
            return ExecutionResult(
                node_id=str(context.get("node_id") or ""),
                status="success",
                data_preview={
                    "message": "No previous failure has been recorded yet.",
                    "available": False,
                },
                metadata={"data_type": "summary"},
            )

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "available": True,
                "category": snapshot.get("category"),
                "stage": snapshot.get("stage"),
                "reason": snapshot.get("reason"),
                "request_id": snapshot.get("request_id"),
                "prompt": snapshot.get("prompt"),
                "metadata": snapshot.get("metadata", {}),
            },
            metadata={"data_type": "summary"},
        )
