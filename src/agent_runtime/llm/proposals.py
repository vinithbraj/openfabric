"""Untrusted proposal models and helpers for structured LLM planning stages."""

from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class PromptClassificationProposal(BaseModel):
    """Untrusted LLM proposal for prompt classification."""

    model_config = ConfigDict(extra="forbid")

    prompt_type: str
    requires_tools: bool
    likely_domains: list[str] = Field(default_factory=list)
    risk_level: str
    needs_clarification: bool
    clarification_question: str | None = None
    reason: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)


class TaskFrameProposal(BaseModel):
    """Untrusted LLM proposal for one task frame."""

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_id: str | None = None
    description: str
    semantic_verb: str = "unknown"
    object_type: str = "unknown"
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    constraints: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    raw_evidence: str | None = None
    requires_confirmation: bool = False
    risk_level: str = "low"


class TaskDecompositionProposal(BaseModel):
    """Untrusted LLM proposal for prompt decomposition."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskFrameProposal] = Field(default_factory=list)
    global_constraints: dict[str, Any] = Field(default_factory=dict)
    unresolved_references: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class VerbAssignmentItemProposal(BaseModel):
    """Untrusted LLM proposal for one task verb assignment."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    semantic_verb: str
    object_type: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    risk_level: str
    requires_confirmation: bool = False


class VerbAssignmentProposal(BaseModel):
    """Untrusted LLM proposal for task verb assignment."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[VerbAssignmentItemProposal] = Field(default_factory=list)


class CapabilityRefProposal(BaseModel):
    """Untrusted LLM proposal for one capability candidate."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str
    operation_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    assumptions: list[str] = Field(default_factory=list)


class CapabilitySelectionProposal(BaseModel):
    """Untrusted LLM proposal for capability selection."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidates: list[CapabilityRefProposal] = Field(default_factory=list)
    selected: CapabilityRefProposal | None = None
    unresolved_reason: str | None = None


class ArgumentExtractionProposal(BaseModel):
    """Untrusted LLM proposal for capability arguments."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    capability_id: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required_arguments: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DAGReviewProposal(BaseModel):
    """Untrusted LLM proposal for DAG review."""

    model_config = ConfigDict(extra="forbid")

    missing_user_intents: list[str] = Field(default_factory=list)
    suspicious_nodes: list[str] = Field(default_factory=list)
    dependency_warnings: list[str] = Field(default_factory=list)
    output_expectation_warnings: list[str] = Field(default_factory=list)
    recommended_repair: dict[str, Any] | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class FailureRepairProposal(BaseModel):
    """Untrusted LLM proposal for post-failure repair."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    corrected_arguments: dict[str, Any] = Field(default_factory=dict)
    alternate_capability_id: str | None = None
    alternate_operation_id: str | None = None
    skip_node: bool = False
    ask_for_clarification: bool = False
    clarification_question: str | None = None
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class DisplayPlanProposal(BaseModel):
    """Untrusted LLM proposal for output display planning."""

    model_config = ConfigDict(extra="forbid")

    display_type: str
    title: str | None = None
    sections: list[dict[str, Any]] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    redaction_policy: str = "standard"


class StructuredProposalAttempt(BaseModel, Generic[ModelT]):
    """One raw/parsed structured proposal attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate_index: int
    raw_llm_response: Any = None
    parsed_proposal: Any = None
    validation_errors: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether the attempt parsed cleanly into the requested model."""

        return self.parsed_proposal is not None and not self.validation_errors


def _proposal_prompt(
    system_prompt: str,
    user_payload: Any,
    candidate_index: int,
    total_candidates: int,
) -> str:
    """Build one JSON-only prompt for an N-best structured proposal attempt."""

    payload = json.dumps(user_payload, sort_keys=True, default=str, ensure_ascii=True)
    return "\n".join(
        [
            system_prompt,
            "Return JSON only.",
            "Do not return markdown fences.",
            f"Candidate index: {candidate_index + 1} of {total_candidates}.",
            "User payload:",
            payload,
        ]
    )


def collect_n_best_structured_attempts(
    llm_client,
    system_prompt: str,
    user_payload: Any,
    output_model: type[ModelT],
    n: int,
) -> list[StructuredProposalAttempt[ModelT]]:
    """Collect N-best raw/parsed structured proposal attempts."""

    total = max(1, int(n))
    schema = output_model.model_json_schema()
    attempts: list[StructuredProposalAttempt[ModelT]] = []
    for index in range(total):
        prompt = _proposal_prompt(system_prompt, user_payload, index, total)
        raw_response = llm_client.complete_json(prompt, schema)
        try:
            parsed = output_model.model_validate(raw_response)
            attempts.append(
                StructuredProposalAttempt[ModelT](
                    candidate_index=index,
                    raw_llm_response=raw_response,
                    parsed_proposal=parsed,
                )
            )
        except PydanticValidationError as exc:
            attempts.append(
                StructuredProposalAttempt[ModelT](
                    candidate_index=index,
                    raw_llm_response=raw_response,
                    validation_errors=[str(exc)],
                )
            )
    return attempts


def generate_n_best_structured_proposals(
    llm_client,
    system_prompt: str,
    user_payload: Any,
    output_model: type[ModelT],
    n: int,
) -> list[ModelT]:
    """Return the valid parsed structured proposals from N-best generation."""

    attempts = collect_n_best_structured_attempts(
        llm_client=llm_client,
        system_prompt=system_prompt,
        user_payload=user_payload,
        output_model=output_model,
        n=n,
    )
    proposals: list[ModelT] = []
    for attempt in attempts:
        if attempt.parsed_proposal is not None and not attempt.validation_errors:
            proposals.append(attempt.parsed_proposal)
    return proposals
