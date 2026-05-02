"""Advisory LLM review of trusted planning artifacts."""

from __future__ import annotations

from typing import Any

from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ActionDAG, UserRequest
from agent_runtime.llm.proposals import DAGReviewProposal
from agent_runtime.llm.structured_call import structured_call


def _sanitize_argument_preview(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, JSON-safe preview of node arguments for review."""

    preview: dict[str, Any] = {}
    for key, value in dict(arguments).items():
        text = str(value)
        if len(text) > 200:
            text = text[:200] + "..."
        preview[key] = text
    return preview


def _sanitized_dag_metadata(dag: ActionDAG) -> dict[str, Any]:
    """Return a metadata-only DAG view suitable for advisory LLM review."""

    return {
        "dag_id": dag.dag_id,
        "node_count": len(dag.nodes),
        "requires_confirmation": dag.requires_confirmation,
        "global_constraints": dict(dag.global_constraints),
        "edges": list(dag.edges),
        "nodes": [
            {
                "node_id": node.id,
                "task_id": node.task_id,
                "description": node.description,
                "capability_id": node.capability_id,
                "operation_id": node.operation_id,
                "semantic_verb": node.semantic_verb,
                "argument_preview": _sanitize_argument_preview(node.arguments),
                "depends_on": list(node.depends_on),
            }
            for node in dag.nodes
        ],
    }


def _build_capability_summaries(
    dag: ActionDAG,
    registry: CapabilityRegistry | None = None,
) -> list[dict[str, Any]]:
    """Return safe capability summaries for the DAG review prompt."""

    summaries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for node in dag.nodes:
        key = (node.capability_id, node.operation_id)
        if key in seen:
            continue
        seen.add(key)
        summary = {
            "capability_id": node.capability_id,
            "operation_id": node.operation_id,
        }
        if registry is not None:
            try:
                capability = registry.get(node.capability_id)
            except Exception:
                capability = None
            if capability is not None:
                summary.update(
                    {
                        "domain": capability.manifest.domain,
                        "description": capability.manifest.description,
                        "semantic_verbs": list(capability.manifest.semantic_verbs),
                        "object_types": list(capability.manifest.object_types),
                        "output_object_types": list(capability.manifest.output_object_types),
                        "output_fields": list(capability.manifest.output_fields),
                        "output_affordances": list(capability.manifest.output_affordances),
                        "required_arguments": list(capability.manifest.required_arguments),
                        "optional_arguments": list(capability.manifest.optional_arguments),
                    }
                )
        summaries.append(summary)
    return summaries


def _build_dag_review_prompt(
    original_prompt: str,
    sanitized_dag: dict[str, Any],
    capability_summaries: list[dict[str, Any]],
) -> str:
    """Build a strict JSON-only prompt for advisory DAG review."""

    schema = model_json_schema(DAGReviewProposal)
    return "\n".join(
        [
            "You are reviewing a sanitized action DAG for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "You are advisory only. Do not assume you can bypass runtime validation or safety.",
            "Use only the original prompt, sanitized DAG metadata, capability summaries, and dependency graph.",
            "Original user prompt:",
            original_prompt,
            "Sanitized DAG metadata:",
            str(sanitized_dag),
            "Capability summaries:",
            str(capability_summaries),
            "JSON schema:",
            str(schema),
        ]
    )


def _validate_dag_review(review: DAGReviewProposal, dag: ActionDAG) -> DAGReviewProposal:
    """Validate that advisory review references only known DAG nodes."""

    node_ids = {node.id for node in dag.nodes}
    unknown_suspicious_nodes = [node_id for node_id in review.suspicious_nodes if node_id not in node_ids]
    if unknown_suspicious_nodes:
        raise ValidationError(
            f"DAG review referenced unknown suspicious nodes: {', '.join(unknown_suspicious_nodes)}"
        )
    if review.recommended_repair:
        for field_name in ("node_id", "source_node_id", "target_node_id", "producer_node_id", "consumer_node_id"):
            field_value = review.recommended_repair.get(field_name)
            if field_value is not None and str(field_value) not in node_ids:
                raise ValidationError(
                    f"DAG review recommended repair for unknown node via {field_name}: {field_value}"
                )
    return review


def review_dag_with_llm(
    original_prompt: str,
    sanitized_dag: dict[str, Any],
    capability_summaries: list[dict[str, Any]],
    llm_client,
) -> DAGReviewProposal:
    """Run advisory DAG review on metadata only."""

    prompt = _build_dag_review_prompt(
        original_prompt=original_prompt,
        sanitized_dag=sanitized_dag,
        capability_summaries=capability_summaries,
    )
    try:
        return structured_call(llm_client, prompt, DAGReviewProposal)
    except Exception:
        return DAGReviewProposal(
            missing_user_intents=[],
            suspicious_nodes=[],
            dependency_warnings=[],
            dataflow_warnings=[],
            output_expectation_warnings=[],
            recommended_repair=None,
            confidence=0.0,
        )


def review_action_dag(
    user_request: UserRequest,
    dag: ActionDAG,
    llm_client,
    registry: CapabilityRegistry | None = None,
) -> DAGReviewProposal:
    """Run advisory DAG review using metadata only."""

    try:
        review = review_dag_with_llm(
            original_prompt=user_request.raw_prompt,
            sanitized_dag=_sanitized_dag_metadata(dag),
            capability_summaries=_build_capability_summaries(dag, registry),
            llm_client=llm_client,
        )
        return _validate_dag_review(review, dag)
    except Exception:
        return DAGReviewProposal(
            missing_user_intents=[],
            suspicious_nodes=[],
            dependency_warnings=[],
            dataflow_warnings=[],
            output_expectation_warnings=[],
            recommended_repair=None,
            confidence=0.0,
        )
