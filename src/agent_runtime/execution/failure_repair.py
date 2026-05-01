"""Optional LLM-assisted repair after safe execution failure."""

from __future__ import annotations

from typing import Any

from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.errors import CapabilityNotFoundError
from agent_runtime.core.types import ActionDAG, ActionNode, CapabilityRef, ResultBundle, TaskFrame, UserRequest
from agent_runtime.execution.safety import evaluate_dag_safety
from agent_runtime.input_pipeline.capability_fit import finalize_capability_fit
from agent_runtime.llm.proposals import CapabilityFitProposal, FailureRepairProposal
from agent_runtime.llm.structured_call import structured_call


def _find_failed_node(bundle: ResultBundle, dag: ActionDAG) -> tuple[ActionNode, dict[str, Any]] | None:
    """Return the first failed node plus its normalized result metadata."""

    node_by_id = {node.id: node for node in dag.nodes}
    for result in bundle.results:
        if result.status == "error" and result.node_id in node_by_id:
            return node_by_id[result.node_id], result.model_dump(mode="json")
    return None


def _sanitize_failed_result(failed_result: dict[str, Any]) -> dict[str, Any]:
    """Return a safe, bounded view of one failed execution result."""

    metadata = {
        key: value
        for key, value in dict(failed_result.get("metadata") or {}).items()
        if key not in {"stdout", "stderr", "traceback", "raw_output", "data_preview", "data_ref"}
    }
    error_text = str(failed_result.get("error") or "").strip()
    return {
        "node_id": failed_result.get("node_id"),
        "status": failed_result.get("status"),
        "error": error_text[:400],
        "metadata": metadata,
    }


def _build_failure_repair_prompt(
    *,
    original_prompt: str,
    node: ActionNode,
    sanitized_error: dict[str, Any],
    manifest: dict[str, Any],
    previous_arguments: dict[str, Any],
) -> str:
    """Build the failure-repair prompt for one failed node."""

    schema = model_json_schema(FailureRepairProposal)
    return "\n".join(
        [
            "You are proposing one safe repair for a failed execution node in an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "You may propose corrected arguments, an alternate capability, skipping the node, or asking the user for clarification.",
            "You may not bypass safety policy, confirmation requirements, or deterministic validation.",
            "Original user prompt:",
            original_prompt,
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
            "Sanitized failure:",
            str(sanitized_error),
            "Previous arguments:",
            str(previous_arguments),
            "Capability manifest:",
            str(manifest),
            "JSON schema:",
            str(schema),
        ]
    )


def propose_failure_repair_with_llm(
    original_prompt: str,
    failed_node: ActionNode,
    manifest: dict[str, Any],
    sanitized_error: dict[str, Any],
    previous_arguments: dict[str, Any],
    llm_client,
) -> FailureRepairProposal:
    """Ask the LLM for one advisory repair proposal for a failed node."""

    prompt = _build_failure_repair_prompt(
        original_prompt=original_prompt,
        node=failed_node,
        sanitized_error=sanitized_error,
        manifest=manifest,
        previous_arguments=previous_arguments,
    )
    proposal = structured_call(llm_client, prompt, FailureRepairProposal)
    if proposal.failed_node_id is None:
        proposal = proposal.model_copy(update={"failed_node_id": failed_node.id})
    return proposal


def _apply_repaired_node(node: ActionNode, *, dag: ActionDAG, updates: dict[str, Any]) -> ActionDAG:
    """Return a DAG copy with one node's trusted fields replaced."""

    updated_nodes = []
    for existing in dag.nodes:
        if existing.id == node.id:
            updated_nodes.append(existing.model_copy(update=updates))
        else:
            updated_nodes.append(existing)
    return dag.model_copy(update={"nodes": updated_nodes})


def _contains_arbitrary_shell_text(arguments: dict[str, Any]) -> bool:
    """Return whether repaired arguments try to smuggle raw shell command text."""

    if any(key in arguments for key in {"command", "cmd", "shell_command"}):
        return True

    suspicious_tokens = ("&&", "||", ";", "|", ">", "<", "$(", "`", "\n")
    for value in arguments.values():
        if isinstance(value, str) and any(token in value for token in suspicious_tokens):
            return True
    return False


def _task_from_failed_node(node: ActionNode, manifest_domain: str) -> TaskFrame:
    """Construct a minimal semantic task for deterministic fit validation."""

    return TaskFrame(
        id=node.task_id,
        description=node.description,
        semantic_verb=node.semantic_verb,
        object_type=manifest_domain,
        intent_confidence=1.0,
        constraints={},
        dependencies=[],
        raw_evidence=node.description,
        requires_confirmation=False,
        risk_level="low",
    )


def _alternate_fit_proposal(
    *,
    task: TaskFrame,
    candidate: CapabilityRef,
    candidate_domain: str,
    proposal: FailureRepairProposal,
) -> CapabilityFitProposal:
    """Build a minimal fit proposal for deterministic alternate-capability validation."""

    return CapabilityFitProposal(
        task_id=task.id,
        candidate_capability_id=candidate.capability_id,
        candidate_operation_id=candidate.operation_id,
        fits=True,
        confidence=proposal.confidence,
        primary_failure_mode=None,
        semantic_reason=proposal.reason,
        domain_reason=proposal.reason,
        object_type_reason=proposal.reason,
        argument_reason=proposal.reason,
        risk_reason=proposal.reason,
        missing_capability_description=None,
        suggested_domain=candidate_domain,
        suggested_object_type=task.object_type,
        requires_clarification=False,
        clarification_question=None,
    )


def attempt_failure_repair(
    *,
    user_request: UserRequest,
    dag: ActionDAG,
    result_bundle: ResultBundle,
    registry: CapabilityRegistry,
    llm_client,
    safety_config,
    repair_attempt_count: int = 0,
    max_repair_attempts: int = 1,
) -> tuple[ActionDAG | None, dict[str, Any]]:
    """Attempt one safe repaired DAG after a failed execution."""

    if repair_attempt_count >= max_repair_attempts:
        return None, {
            "attempted": False,
            "rejected": "Repair retry limit reached.",
            "retry_limit_enforced": True,
        }

    located = _find_failed_node(result_bundle, dag)
    if located is None:
        return None, {"attempted": False, "reason": "No failed node was available for repair."}

    failed_node, failed_result = located
    try:
        capability = registry.get(failed_node.capability_id)
    except CapabilityNotFoundError:
        return None, {"attempted": False, "reason": "Failed node capability is not registered."}

    manifest_summary = {
        "capability_id": capability.manifest.capability_id,
        "operation_id": capability.manifest.operation_id,
        "required_arguments": capability.manifest.required_arguments,
        "optional_arguments": capability.manifest.optional_arguments,
        "read_only": capability.manifest.read_only,
        "risk_level": capability.manifest.risk_level,
    }
    sanitized_error = _sanitize_failed_result(failed_result)
    try:
        proposal = propose_failure_repair_with_llm(
            original_prompt=user_request.raw_prompt,
            failed_node=failed_node,
            manifest=manifest_summary,
            sanitized_error=sanitized_error,
            previous_arguments=dict(failed_node.arguments),
            llm_client=llm_client,
        )
    except Exception:
        return None, {
            "attempted": False,
            "rejected": "LLM failure repair proposal was unavailable.",
        }

    metadata: dict[str, Any] = {
        "attempted": True,
        "proposal": proposal.model_dump(mode="json"),
        "repair_attempt_count": repair_attempt_count + 1,
    }

    if proposal.failed_node_id and proposal.failed_node_id != failed_node.id:
        metadata["rejected"] = "Repair proposal referenced a different failed node."
        return None, metadata

    if proposal.proposed_action == "ask_user":
        metadata["clarification_required"] = True
        metadata["user_message"] = proposal.user_message
        return None, metadata

    if proposal.proposed_action == "skip_with_explanation":
        metadata["rejected"] = "Skip-node repair is advisory only and is not auto-applied."
        metadata["user_message"] = proposal.user_message
        return None, metadata

    corrected_arguments = dict(failed_node.arguments)
    corrected_arguments.update(dict(proposal.corrected_arguments))
    if _contains_arbitrary_shell_text(corrected_arguments):
        metadata["rejected"] = "Failure repair may not introduce arbitrary shell command arguments."
        return None, metadata

    if proposal.proposed_action == "alternate_capability":
        alternate_id = str(proposal.alternate_capability_id or "").strip()
        if not alternate_id:
            metadata["rejected"] = "Alternate-capability repair did not provide an alternate capability id."
            return None, metadata
        try:
            alternate_capability = registry.get(alternate_id)
        except CapabilityNotFoundError:
            metadata["rejected"] = "Alternate-capability repair referenced an unknown capability."
            return None, metadata

        task = _task_from_failed_node(failed_node, alternate_capability.manifest.domain)
        candidate = CapabilityRef(
            capability_id=alternate_capability.manifest.capability_id,
            operation_id=alternate_capability.manifest.operation_id,
            confidence=proposal.confidence,
            reason=proposal.reason,
        )
        fit_decision = finalize_capability_fit(
            task,
            candidate,
            alternate_capability.manifest,
            _alternate_fit_proposal(
                task=task,
                candidate=candidate,
                candidate_domain=alternate_capability.manifest.domain,
                proposal=proposal,
            ),
            classification_context={
                "original_prompt": user_request.raw_prompt,
                "likely_domains": [alternate_capability.manifest.domain],
            },
        )
        metadata["fit_decision"] = fit_decision.model_dump(mode="json")
        if not fit_decision.is_fit:
            metadata["rejected"] = "Alternate-capability repair did not pass deterministic capability fit."
            return None, metadata
        if alternate_capability.manifest.mutates_state or alternate_capability.manifest.requires_confirmation:
            metadata["rejected"] = "Alternate-capability repair may not introduce mutating or confirmation-gated execution."
            return None, metadata
        try:
            validated_arguments = alternate_capability.validate_arguments(corrected_arguments)
        except Exception as exc:
            metadata["rejected"] = f"Alternate-capability repair failed argument validation: {exc}"
            return None, metadata
        repaired_dag = _apply_repaired_node(
            failed_node,
            dag=dag,
            updates={
                "capability_id": alternate_capability.manifest.capability_id,
                "operation_id": alternate_capability.manifest.operation_id,
                "arguments": validated_arguments,
            },
        )
    else:
        if proposal.proposed_action != "retry_with_arguments":
            metadata["rejected"] = f"Unsupported failure repair action: {proposal.proposed_action}"
            return None, metadata
        try:
            validated_arguments = capability.validate_arguments(corrected_arguments)
        except Exception as exc:
            metadata["rejected"] = f"Failure repair arguments did not validate: {exc}"
            return None, metadata
        repaired_dag = _apply_repaired_node(
            failed_node,
            dag=dag,
            updates={"arguments": validated_arguments},
        )

    decision = evaluate_dag_safety(repaired_dag, registry, safety_config)
    if not decision.allowed:
        metadata["rejected"] = "Failure repair was rejected by deterministic safety validation."
        metadata["blocked_reasons"] = list(decision.blocked_reasons)
        return None, metadata
    if decision.requires_confirmation:
        metadata["rejected"] = "Failure repair may not bypass confirmation requirements."
        metadata["blocked_reasons"] = list(decision.blocked_reasons)
        return None, metadata

    return repaired_dag, metadata
