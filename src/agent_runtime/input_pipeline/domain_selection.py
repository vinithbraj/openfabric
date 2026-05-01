"""Capability selection for typed task frames."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import CapabilityNotFoundError
from agent_runtime.core.types import CapabilityRef, TaskFrame
from agent_runtime.input_pipeline.plan_selection import CandidateEvaluation, select_best_candidate
from agent_runtime.llm.proposals import CapabilitySelectionProposal, collect_n_best_structured_attempts
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)


class CapabilitySelectionResult(BaseModel):
    """Ranked candidate capabilities for one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidates: list[CapabilityRef] = Field(default_factory=list)
    selected: CapabilityRef | None = None
    unresolved_reason: str | None = None


class _SelectionResponse(BaseModel):
    """Internal structured response returned by the LLM for one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidates: list[CapabilityRef] = Field(default_factory=list)
    selected: CapabilityRef | None = None
    unresolved_reason: str | None = None


class DomainSelector:
    """Small deterministic fallback for coarse domain guessing."""

    def select(self, prompt: str) -> str:
        """Return a domain hint without invoking tools."""

        lowered = prompt.strip().lower()
        if "sql" in lowered or "database" in lowered or "patient" in lowered:
            return "sql"
        if "shell" in lowered or "process" in lowered:
            return "shell"
        if "python" in lowered or "dataframe" in lowered:
            return "python_data"
        if "markdown" in lowered or "table" in lowered:
            return "presentation"
        return "filesystem"


def _build_selection_prompt(task: TaskFrame, llm_manifest: list[dict[str, object]]) -> str:
    """Build the strict JSON-only prompt for one-task capability selection."""

    schema = model_json_schema(_SelectionResponse)
    return "\n".join(
        [
            "You are selecting capability candidates for one task frame.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable syntax.",
            "Rank the best capability candidates for the task.",
            "Use only the provided compact capability manifest.",
            "Task frame:",
            str(
                {
                    "task_id": task.id,
                    "description": task.description,
                    "semantic_verb": task.semantic_verb,
                    "object_type": task.object_type,
                    "constraints": task.constraints,
                }
            ),
            "Compact capability manifest:",
            str(llm_manifest),
            "JSON schema:",
            str(schema),
        ]
    )


def _reason_is_explicit(reason: str) -> bool:
    """Return whether a selection reason is detailed enough to justify a mismatch."""

    words = [word for word in str(reason or "").strip().split() if word]
    return len(words) >= 5


def _verb_matches(task: TaskFrame, manifest: CapabilityManifest) -> bool:
    """Return whether the task verb is declared on the manifest."""

    return task.semantic_verb in {verb.strip().lower() for verb in manifest.semantic_verbs}


def _validate_candidate(
    task: TaskFrame,
    candidate: CapabilityRef,
    registry: CapabilityRegistry,
) -> tuple[CapabilityRef | None, str | None]:
    """Validate one selected candidate against registry and semantic rules."""

    try:
        capability = registry.get(candidate.capability_id)
    except CapabilityNotFoundError:
        return None, f"Selected capability does not exist: {candidate.capability_id}."

    manifest = capability.manifest
    if candidate.operation_id != manifest.operation_id:
        return None, (
            f"Selected operation does not match capability manifest for {candidate.capability_id}: "
            f"{candidate.operation_id} != {manifest.operation_id}."
        )

    if candidate.confidence < 0.60:
        return None, (
            f"Selected capability confidence is below threshold for task {task.id}: "
            f"{candidate.confidence:.2f} < 0.60."
        )

    if not _verb_matches(task, manifest):
        if not (candidate.confidence >= 0.85 and _reason_is_explicit(candidate.reason)):
            return None, (
                f"Selected capability semantic verbs do not match task verb {task.semantic_verb} "
                f"for {candidate.capability_id}."
            )

    if manifest.requires_confirmation:
        task.requires_confirmation = True

    return candidate, None


def _proposal_candidate_to_trusted(candidate) -> CapabilityRef:
    """Convert one untrusted capability candidate proposal into the trusted model."""

    return CapabilityRef.model_validate(
        {
            "capability_id": candidate.capability_id,
            "operation_id": candidate.operation_id,
            "confidence": candidate.confidence,
            "reason": candidate.reason,
        }
    )


def select_capabilities(
    tasks: list[TaskFrame],
    registry: CapabilityRegistry,
    llm_client,
    n_best: int = 3,
    trace: PlanningTrace | None = None,
) -> list[CapabilitySelectionResult]:
    """Select ranked capability candidates for task frames via a structured LLM call."""

    llm_manifest = registry.export_llm_manifest()
    results: list[CapabilitySelectionResult] = []

    for task in tasks:
        prompt = _build_selection_prompt(task, llm_manifest)
        attempts = collect_n_best_structured_attempts(
            llm_client=llm_client,
            system_prompt=prompt,
            user_payload={
                "task_id": task.id,
                "description": task.description,
                "semantic_verb": task.semantic_verb,
                "object_type": task.object_type,
            },
            output_model=CapabilitySelectionProposal,
            n=n_best,
        )
        evaluations: list[CandidateEvaluation[CapabilitySelectionResult]] = []
        for attempt in attempts:
            evaluation = CandidateEvaluation[CapabilitySelectionResult](
                raw_response=attempt.raw_llm_response,
                validation_errors=list(attempt.validation_errors),
            )
            if attempt.parsed_proposal is None or attempt.validation_errors:
                evaluations.append(evaluation)
                continue
            response = attempt.parsed_proposal
            ranked_candidates = sorted(
                [
                    _proposal_candidate_to_trusted(candidate)
                    for candidate in response.candidates
                ],
                key=lambda item: item.confidence,
                reverse=True,
            )
            selected_candidate = response.selected
            validated_selected: CapabilityRef | None = None
            unresolved_reason = response.unresolved_reason
            if selected_candidate is not None:
                validated_selected, validation_error = _validate_candidate(
                    task,
                    _proposal_candidate_to_trusted(selected_candidate),
                    registry,
                )
                if validation_error:
                    unresolved_reason = validation_error
                    evaluation.rejection_reasons.append(validation_error)
            if validated_selected is None and not unresolved_reason and not ranked_candidates:
                unresolved_reason = f"No capability candidates were returned for task {task.id}."
                evaluation.rejection_reasons.append(unresolved_reason)
            evaluation.proposal = CapabilitySelectionResult(
                task_id=task.id,
                candidates=ranked_candidates,
                selected=validated_selected,
                unresolved_reason=unresolved_reason,
            )
            evaluation.confidence = (
                float(validated_selected.confidence) if validated_selected is not None else 0.0
            )
            evaluation.capability_compatibility = 1 if validated_selected is not None else 0
            if validated_selected is not None:
                manifest = registry.get(validated_selected.capability_id).manifest
                evaluation.risk_score = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(
                    manifest.risk_level,
                    0,
                )
            evaluation.assumption_count = sum(
                len(getattr(candidate, "assumptions", [])) for candidate in response.candidates
            )
            evaluation.unresolved_count = 0 if unresolved_reason is None else 1
            evaluations.append(evaluation)

        selected = select_best_candidate(evaluations)
        if selected is None or selected.proposal is None:
            results.append(
                CapabilitySelectionResult(
                    task_id=task.id,
                    candidates=[],
                    selected=None,
                    unresolved_reason=f"No capability candidates were returned for task {task.id}.",
                )
            )
            continue

        if isinstance(trace, PlanningTrace):
            model_name, temperature = llm_client_metadata(llm_client)
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="capability_selection",
                    request_id=str(trace.request_id),
                    model_name=model_name,
                    llm_temperature=temperature,
                    prompt_template_id="capability_selection",
                    raw_llm_response=[attempt.raw_llm_response for attempt in attempts],
                    parsed_proposal=[
                        attempt.parsed_proposal.model_dump(mode="json")
                        if attempt.parsed_proposal is not None
                        else None
                        for attempt in attempts
                    ],
                    validation_errors=[
                        error for evaluation in evaluations for error in evaluation.validation_errors
                    ],
                    selected_candidate=selected.proposal.model_dump(mode="json"),
                    rejection_reasons=[
                        reason for evaluation in evaluations for reason in evaluation.rejection_reasons
                    ],
                ),
            )

        results.append(selected.proposal)

    return results
