"""Advisory LLM review of trusted planning artifacts."""

from __future__ import annotations

from typing import Any

from pydantic.json_schema import model_json_schema

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


def _build_dag_review_prompt(user_request: UserRequest, dag: ActionDAG) -> str:
    """Build a strict JSON-only prompt for advisory DAG review."""

    schema = model_json_schema(DAGReviewProposal)
    return "\n".join(
        [
            "You are reviewing a sanitized action DAG for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "You are advisory only. Do not assume you can bypass runtime validation or safety.",
            "Use only the original prompt and sanitized DAG metadata.",
            "Original user prompt:",
            user_request.raw_prompt,
            "Sanitized DAG metadata:",
            str(_sanitized_dag_metadata(dag)),
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
        target_node_id = review.recommended_repair.get("node_id")
        if target_node_id is not None and str(target_node_id) not in node_ids:
            raise ValidationError(
                f"DAG review recommended repair for unknown node: {target_node_id}"
            )
    return review


def review_action_dag(
    user_request: UserRequest,
    dag: ActionDAG,
    llm_client,
) -> DAGReviewProposal:
    """Run advisory DAG review using metadata only."""

    prompt = _build_dag_review_prompt(user_request, dag)
    try:
        review = structured_call(llm_client, prompt, DAGReviewProposal)
        return _validate_dag_review(review, dag)
    except Exception:
        return DAGReviewProposal(
            missing_user_intents=[],
            suspicious_nodes=[],
            dependency_warnings=[],
            output_expectation_warnings=[],
            recommended_repair=None,
            confidence=0.0,
        )
