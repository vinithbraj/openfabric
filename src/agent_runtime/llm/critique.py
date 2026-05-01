"""LLM critique helpers for semantic planning proposals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import model_json_schema

from agent_runtime.core.types import UserRequest
from agent_runtime.llm.structured_call import structured_call
from agent_runtime.llm.proposals import TaskDecompositionProposal


class DecompositionCritique(BaseModel):
    """Structured critique of a decomposition proposal."""

    model_config = ConfigDict(extra="forbid")

    missing_user_intents: list[str] = Field(default_factory=list)
    hallucinated_tasks: list[str] = Field(default_factory=list)
    dependency_warnings: list[str] = Field(default_factory=list)
    unsafe_task_warnings: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    recommended_repair: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, values: Any) -> Any:
        """Accept older critique payloads and normalize them to the new schema."""

        if not isinstance(values, dict):
            return values
        normalized = dict(values)
        if "missing_user_intents" not in normalized:
            normalized["missing_user_intents"] = list(normalized.get("missing_tasks") or [])
        if "dependency_warnings" not in normalized:
            normalized["dependency_warnings"] = list(normalized.get("dependency_issues") or [])
        if "unsafe_task_warnings" not in normalized:
            normalized["unsafe_task_warnings"] = list(normalized.get("unsafe_tasks") or [])
        if "recommended_repair" not in normalized:
            overly_broad = list(normalized.get("overly_broad_tasks") or [])
            if overly_broad:
                normalized["recommended_repair"] = {
                    "narrow_tasks": overly_broad,
                    "reason": "Some proposed tasks are too broad and should be decomposed more precisely.",
                }
        for legacy_key in ("missing_tasks", "dependency_issues", "unsafe_tasks", "overly_broad_tasks"):
            normalized.pop(legacy_key, None)
        return normalized


def _build_decomposition_critique_prompt(
    original_prompt: str,
    proposal: TaskDecompositionProposal | dict[str, Any],
    registry_summary: Any,
) -> str:
    """Build the critique prompt for one decomposition proposal."""

    schema = model_json_schema(DecompositionCritique)
    proposal_payload = (
        proposal.model_dump(mode="json") if hasattr(proposal, "model_dump") else dict(proposal)
    )
    return "\n".join(
        [
            "You are critiquing a proposed task decomposition for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "You are advisory only. Do not assume you can execute, mutate, or bypass runtime validation.",
            "Use only the original user prompt, the proposed tasks, global constraints, and available capability domains.",
            "Answer whether any user-requested tasks are missing, hallucinated, unsafe, have bad dependencies, or contain unresolved references.",
            "Original user prompt:",
            original_prompt,
            "Registry/domain summary:",
            str(registry_summary),
            "Proposed decomposition:",
            str(proposal_payload),
            "JSON schema:",
            str(schema),
        ]
    )


def critique_decomposition_with_llm(
    original_prompt: str,
    decomposition: TaskDecompositionProposal | dict[str, Any],
    registry_summary: Any,
    llm_client,
) -> DecompositionCritique:
    """Critique one decomposition proposal through a structured LLM call."""

    prompt = _build_decomposition_critique_prompt(
        original_prompt,
        decomposition,
        registry_summary,
    )
    try:
        return structured_call(llm_client, prompt, DecompositionCritique)
    except Exception:
        return DecompositionCritique(
            missing_user_intents=[],
            hallucinated_tasks=[],
            dependency_warnings=[],
            unsafe_task_warnings=[],
            unresolved_references=[],
            recommended_repair=None,
            confidence=0.0,
        )


def critique_decomposition(
    user_request: UserRequest,
    proposal: TaskDecompositionProposal,
    available_domains: list[str],
    llm_client,
) -> DecompositionCritique:
    """Compatibility wrapper around the decomposition critique stage."""

    registry_summary = sorted(
        {
            str(domain).strip().lower()
            for domain in available_domains
            if str(domain or "").strip()
        }
    )
    return critique_decomposition_with_llm(
        user_request.raw_prompt,
        proposal,
        registry_summary,
        llm_client,
    )


def critique_requires_repair(critique: DecompositionCritique) -> bool:
    """Return whether critique findings justify one repair attempt."""

    return any(
        (
            critique.missing_user_intents,
            critique.hallucinated_tasks,
            critique.dependency_warnings,
            critique.unsafe_task_warnings,
            critique.unresolved_references,
            critique.recommended_repair,
        )
    )
