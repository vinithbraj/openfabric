"""Optional LLM-assisted repair after safe execution failure."""

from __future__ import annotations

from typing import Any

from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.errors import CapabilityNotFoundError, ValidationError
from agent_runtime.core.types import ActionDAG, ActionNode, ResultBundle, UserRequest
from agent_runtime.execution.safety import evaluate_dag_safety
from agent_runtime.llm.proposals import FailureRepairProposal
from agent_runtime.llm.structured_call import structured_call


def _find_failed_node(bundle: ResultBundle, dag: ActionDAG) -> tuple[ActionNode, dict[str, Any]] | None:
    """Return the first failed node plus its normalized result metadata."""

    node_by_id = {node.id: node for node in dag.nodes}
    for result in bundle.results:
        if result.status == "error" and result.node_id in node_by_id:
            return node_by_id[result.node_id], result.model_dump(mode="json")
    return None


def _build_failure_repair_prompt(
    user_request: UserRequest,
    node: ActionNode,
    failed_result: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Build the failure-repair prompt for one failed node."""

    schema = model_json_schema(FailureRepairProposal)
    return "\n".join(
        [
            "You are proposing one safe repair for a failed execution node in an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "You may propose corrected arguments, an alternate capability, skipping the node, or asking for clarification.",
            "You may not bypass safety policy.",
            "Original user prompt:",
            user_request.raw_prompt,
            "Failed node metadata:",
            str(
                {
                    "node_id": node.id,
                    "task_id": node.task_id,
                    "description": node.description,
                    "capability_id": node.capability_id,
                    "operation_id": node.operation_id,
                    "arguments": node.arguments,
                    "safety_labels": node.safety_labels,
                }
            ),
            "Failed execution result:",
            str(failed_result),
            "Capability manifest:",
            str(manifest),
            "JSON schema:",
            str(schema),
        ]
    )


def _apply_repaired_arguments(node: ActionNode, validated_arguments: dict[str, Any], dag: ActionDAG) -> ActionDAG:
    """Return a DAG copy with one node's arguments replaced."""

    updated_nodes = []
    for existing in dag.nodes:
        if existing.id == node.id:
            updated_nodes.append(existing.model_copy(update={"arguments": validated_arguments}))
        else:
            updated_nodes.append(existing)
    return dag.model_copy(update={"nodes": updated_nodes})


def attempt_failure_repair(
    *,
    user_request: UserRequest,
    dag: ActionDAG,
    result_bundle: ResultBundle,
    registry: CapabilityRegistry,
    llm_client,
    safety_config,
) -> tuple[ActionDAG | None, dict[str, Any]]:
    """Attempt one safe repaired DAG after a failed execution."""

    located = _find_failed_node(result_bundle, dag)
    if located is None:
        return None, {"attempted": False, "reason": "No failed node was available for repair."}

    failed_node, failed_result = located
    try:
        capability = registry.get(failed_node.capability_id)
    except CapabilityNotFoundError:
        return None, {"attempted": False, "reason": "Failed node capability is not registered."}

    prompt = _build_failure_repair_prompt(
        user_request,
        failed_node,
        failed_result,
        {
            "capability_id": capability.manifest.capability_id,
            "operation_id": capability.manifest.operation_id,
            "required_arguments": capability.manifest.required_arguments,
            "optional_arguments": capability.manifest.optional_arguments,
            "read_only": capability.manifest.read_only,
            "risk_level": capability.manifest.risk_level,
        },
    )
    proposal = structured_call(llm_client, prompt, FailureRepairProposal)

    metadata: dict[str, Any] = {
        "attempted": True,
        "proposal": proposal.model_dump(mode="json"),
    }

    if proposal.ask_for_clarification:
        metadata["clarification_required"] = True
        return None, metadata

    if proposal.skip_node:
        metadata["rejected"] = "Skip-node repair is advisory only and is not auto-applied."
        return None, metadata

    if proposal.alternate_capability_id:
        metadata["rejected"] = "Alternate-capability repair is not auto-applied in this runtime."
        return None, metadata

    repaired_arguments = dict(failed_node.arguments)
    repaired_arguments.update(dict(proposal.corrected_arguments))

    if any(key in repaired_arguments for key in {"command", "cmd", "shell_command"}):
        metadata["rejected"] = "Failure repair may not introduce arbitrary shell command arguments."
        return None, metadata

    validated_arguments = capability.validate_arguments(repaired_arguments)
    repaired_dag = _apply_repaired_arguments(failed_node, validated_arguments, dag)
    decision = evaluate_dag_safety(repaired_dag, registry, safety_config)
    if not decision.allowed:
        metadata["rejected"] = "Failure repair was rejected by deterministic safety validation."
        metadata["blocked_reasons"] = list(decision.blocked_reasons)
        return None, metadata

    return repaired_dag, metadata
