"""LLM critique helpers for semantic planning proposals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import model_json_schema

from agent_runtime.core.types import UserRequest
from agent_runtime.llm.structured_call import structured_call
from agent_runtime.llm.proposals import TaskDecompositionProposal


class DecompositionCritique(BaseModel):
    """Structured critique of a decomposition proposal."""

    model_config = ConfigDict(extra="forbid")

    missing_tasks: list[str] = Field(default_factory=list)
    hallucinated_tasks: list[str] = Field(default_factory=list)
    dependency_issues: list[str] = Field(default_factory=list)
    overly_broad_tasks: list[str] = Field(default_factory=list)
    unsafe_tasks: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def _build_decomposition_critique_prompt(
    user_request: UserRequest,
    proposal: TaskDecompositionProposal,
    available_domains: list[str],
) -> str:
    """Build the critique prompt for one decomposition proposal."""

    schema = model_json_schema(DecompositionCritique)
    return "\n".join(
        [
            "You are critiquing a proposed task decomposition for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "Use only the original user prompt, the proposed tasks, global constraints, and available capability domains.",
            "Answer whether tasks are missing, hallucinated, too broad, unsafe, or have unresolved references/dependency issues.",
            "Original user prompt:",
            user_request.raw_prompt,
            "Available capability domains:",
            str(sorted({domain.strip().lower() for domain in available_domains if domain})),
            "Proposed decomposition:",
            str(proposal.model_dump()),
            "JSON schema:",
            str(schema),
        ]
    )


def critique_decomposition(
    user_request: UserRequest,
    proposal: TaskDecompositionProposal,
    available_domains: list[str],
    llm_client,
) -> DecompositionCritique:
    """Critique one decomposition proposal through a structured LLM call."""

    prompt = _build_decomposition_critique_prompt(user_request, proposal, available_domains)
    try:
        return structured_call(llm_client, prompt, DecompositionCritique)
    except Exception:
        return DecompositionCritique(
            missing_tasks=[],
            hallucinated_tasks=[],
            dependency_issues=[],
            overly_broad_tasks=[],
            unsafe_tasks=[],
            unresolved_references=[],
            confidence=0.0,
        )


def critique_requires_repair(critique: DecompositionCritique) -> bool:
    """Return whether critique findings justify one repair attempt."""

    return any(
        (
            critique.missing_tasks,
            critique.hallucinated_tasks,
            critique.dependency_issues,
            critique.overly_broad_tasks,
            critique.unsafe_tasks,
            critique.unresolved_references,
        )
    )
