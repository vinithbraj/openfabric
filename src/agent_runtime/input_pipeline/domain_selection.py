"""Capability selection for typed task frames."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.semantic_compatibility import (
    canonical_domain,
    canonical_object_family,
    canonical_semantic_verb,
    domains_compatible,
    has_hard_cross_domain_conflict,
    object_types_compatible,
    semantic_verbs_compatible,
)
from agent_runtime.core.errors import CapabilityNotFoundError
from agent_runtime.core.types import CapabilityRef, TaskFrame
from agent_runtime.input_pipeline.decomposition import (
    _looks_like_system_tool_prompt,
    _runtime_introspection_capability_id,
)
from agent_runtime.input_pipeline.plan_selection import CandidateEvaluation, select_best_candidate
from agent_runtime.llm.proposals import (
    CapabilityCandidateEvaluationProposal,
    CapabilitySelectionProposal,
    collect_n_best_structured_attempts,
)
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
    """Internal structured response returned by the LLM for one shortlist judgment."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    evaluations: list[CapabilityCandidateEvaluationProposal] = Field(default_factory=list)
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


_SHORTLIST_SIZE = 5


def _normalized_likely_domains(classification_context: dict[str, Any] | None) -> list[str]:
    """Return canonical likely domains from classification context."""

    likely_domains = (classification_context or {}).get("likely_domains")
    if not isinstance(likely_domains, list):
        return []
    normalized: list[str] = []
    for value in likely_domains:
        canonical = canonical_domain(value)
        if canonical is not None and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _manifest_risk_score(manifest: CapabilityManifest) -> int:
    """Return a comparable risk score for one manifest."""

    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(manifest.risk_level, 0)


def _exact_object_match(task: TaskFrame, manifest: CapabilityManifest) -> bool:
    """Return whether the task object family exactly matches one manifest object family."""

    task_object = canonical_object_family(task.object_type)
    manifest_objects = {
        canonical_object_family(value)
        for value in manifest.object_types
    }
    return task_object is not None and task_object in manifest_objects


def _preferred_capability_bias(task: TaskFrame, manifest: CapabilityManifest) -> int:
    """Return a deterministic bias for highly-obvious intent/capability pairings."""

    lowered = str(task.description or "").strip().lower()
    object_family = canonical_object_family(task.object_type)
    if manifest.capability_id == "runtime.describe_capabilities" and object_family == "runtime.capabilities":
        return 10
    if manifest.capability_id == "runtime.describe_pipeline" and object_family == "runtime.pipeline":
        return 10
    if manifest.capability_id == "system.memory_status" and object_family == "system.memory":
        return 10
    if manifest.capability_id == "system.disk_usage" and object_family == "system.disk":
        return 10
    if manifest.capability_id == "system.cpu_load" and object_family == "system.cpu":
        return 10
    if manifest.capability_id == "system.uptime" and object_family == "system.uptime":
        return 10
    if manifest.capability_id == "system.environment_summary" and object_family == "system.environment":
        return 10
    if manifest.capability_id == "shell.git_status" and object_family == "git.repository":
        return 10
    if manifest.capability_id == "filesystem.write_file" and task.semantic_verb in {"create", "update", "render"}:
        save_markers = ("save", "write", "export", "persist", "report", "file", "disk")
        if object_family in {"filesystem.file", "report", "document", "markdown", "json"}:
            return 10
        if any(marker in lowered for marker in save_markers):
            return 8
    if manifest.capability_id == "filesystem.list_directory" and any(marker in lowered for marker in ("list files", "this directory", "this folder", "current directory")):
        return 6
    if manifest.capability_id == "filesystem.read_file" and any(marker in lowered for marker in ("read file", "readme", "open file")):
        return 6
    return 0


def _build_shortlist(
    task: TaskFrame,
    registry: CapabilityRegistry,
    classification_context: dict[str, Any] | None = None,
    shortlist_size: int = _SHORTLIST_SIZE,
) -> list[CapabilityManifest]:
    """Return a deterministic ranked shortlist of candidate manifests for one task."""

    likely_domains = _normalized_likely_domains(classification_context)
    task_domain = canonical_domain(canonical_object_family(task.object_type))
    task_verb = canonical_semantic_verb(task.semantic_verb)

    scored: list[tuple[tuple[int, int, int, int, int, int, int, str], CapabilityManifest]] = []
    for manifest in registry.list_manifests():
        if has_hard_cross_domain_conflict(
            task_domain,
            manifest.domain,
            task.object_type,
            list(manifest.object_types),
            likely_domains,
        ):
            continue
        domain_match = domains_compatible(
            task_domain,
            manifest.domain,
            likely_domains,
            task.object_type,
            list(manifest.object_types),
        )
        object_match = object_types_compatible(task.object_type, list(manifest.object_types), likely_domains)
        verb_match = semantic_verbs_compatible(task_verb, list(manifest.semantic_verbs))
        exact_object = _exact_object_match(task, manifest)
        bias = _preferred_capability_bias(task, manifest)
        safe_read_alignment = 1 if (task_verb not in {"create", "update", "delete"} and manifest.read_only) else 0
        if not any((domain_match, object_match, verb_match, bias > 0)):
            continue
        score = (
            bias,
            1 if exact_object else 0,
            1 if object_match else 0,
            1 if domain_match else 0,
            1 if verb_match else 0,
            safe_read_alignment,
            -_manifest_risk_score(manifest),
            manifest.capability_id,
        )
        scored.append((score, manifest))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [manifest for _, manifest in scored[:shortlist_size]]


def _shortlist_candidate_confidence(rank: int) -> float:
    """Return one stable heuristic confidence for a shortlist position."""

    return max(0.60, 0.96 - (rank * 0.06))


def _build_selection_prompt(task: TaskFrame, shortlist_manifest: list[dict[str, object]]) -> str:
    """Build the strict JSON-only prompt for one-task shortlist judging."""

    shortlist_json = json.dumps(shortlist_manifest, sort_keys=True, default=str, ensure_ascii=True)
    return "\n".join(
        [
            "You are selecting capability candidates for one task frame.",
            "Do not produce commands, shell syntax, SQL, code, or executable syntax.",
            "The runtime has already built a deterministic shortlist of candidate capabilities.",
            "Evaluate every shortlisted candidate with fits: true or false.",
            "Do not invent new capability ids or operation ids.",
            "Use only the provided shortlisted candidates.",
            "Judge semantic fit, domain fit, object-type fit, argument plausibility, and risk.",
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
            "Deterministic capability shortlist:",
            shortlist_json,
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


def _evaluation_to_trusted_candidate(
    evaluation: CapabilityCandidateEvaluationProposal,
    *,
    candidate_reason_prefix: str = "",
) -> CapabilityRef:
    """Convert one candidate evaluation into a trusted selected-candidate reference."""

    reason = str(evaluation.reason or "").strip()
    if candidate_reason_prefix and reason:
        reason = f"{candidate_reason_prefix} {reason}".strip()
    elif candidate_reason_prefix:
        reason = candidate_reason_prefix
    return CapabilityRef.model_validate(
        {
            "capability_id": evaluation.capability_id,
            "operation_id": evaluation.operation_id,
            "confidence": evaluation.confidence,
            "reason": reason,
        }
    )


def _runtime_introspection_override(
    task: TaskFrame,
    registry: CapabilityRegistry,
) -> CapabilitySelectionResult | None:
    """Return a deterministic runtime capability selection for introspection prompts."""

    capability_id = _runtime_introspection_capability_id(task.description)
    if capability_id is None:
        return None
    capability = registry.get(capability_id)
    selected = CapabilityRef(
        capability_id=capability.manifest.capability_id,
        operation_id=capability.manifest.operation_id,
        confidence=1.0,
        reason="Deterministically routed to runtime self-introspection capability.",
    )
    return CapabilitySelectionResult(
        task_id=task.id,
        candidates=[selected],
        selected=selected,
        unresolved_reason=None,
    )


def _system_capability_override(
    task: TaskFrame,
    registry: CapabilityRegistry,
) -> CapabilitySelectionResult | None:
    """Return a deterministic system capability selection for obvious system-inspection prompts."""

    lowered = str(task.description or "").strip().lower()
    if not lowered or not _looks_like_system_tool_prompt(lowered):
        return None

    return None


def _validate_shortlist_evaluations(
    task: TaskFrame,
    shortlist: list[CapabilityManifest],
    response: CapabilitySelectionProposal,
) -> tuple[dict[tuple[str, str], CapabilityCandidateEvaluationProposal], list[str]]:
    """Validate that one LLM shortlist judgment refers only to the runtime shortlist."""

    errors: list[str] = []
    shortlist_keys = {
        (manifest.capability_id, manifest.operation_id): manifest for manifest in shortlist
    }
    evaluations: dict[tuple[str, str], CapabilityCandidateEvaluationProposal] = {}
    if response.task_id != task.id:
        errors.append(f"Shortlist evaluation task_id mismatch: {response.task_id} != {task.id}.")
    for evaluation in response.evaluations:
        key = (evaluation.capability_id, evaluation.operation_id)
        if key not in shortlist_keys:
            errors.append(
                f"Evaluated capability is not in the deterministic shortlist: {evaluation.capability_id}."
            )
            continue
        if key in evaluations:
            errors.append(f"Duplicate evaluation for capability {evaluation.capability_id}.")
            continue
        evaluations[key] = evaluation
    if not evaluations:
        errors.append(f"No shortlisted capabilities were evaluated for task {task.id}.")
    return evaluations, errors


def _select_from_shortlist(
    task: TaskFrame,
    shortlist: list[CapabilityManifest],
    evaluation_map: dict[tuple[str, str], CapabilityCandidateEvaluationProposal],
    classification_context: dict[str, Any] | None = None,
) -> CapabilityRef | None:
    """Return the best candidate from one evaluated shortlist."""

    likely_domains = _normalized_likely_domains(classification_context)
    task_domain = canonical_domain(canonical_object_family(task.object_type))
    task_object = canonical_object_family(task.object_type)
    accepted: list[tuple[tuple[int, int, int, float, int, int, str], CapabilityRef]] = []
    fallback: list[tuple[tuple[int, int, int, int, str], CapabilityRef]] = []

    for manifest in shortlist:
        key = (manifest.capability_id, manifest.operation_id)
        evaluation = evaluation_map.get(key)
        if evaluation is None:
            continue
        domain_match = domains_compatible(
            task_domain,
            manifest.domain,
            likely_domains,
            task.object_type,
            list(manifest.object_types),
        )
        object_match = object_types_compatible(task.object_type, list(manifest.object_types), likely_domains)
        exact_object = _exact_object_match(task, manifest)
        hard_conflict = has_hard_cross_domain_conflict(
            task_domain,
            manifest.domain,
            task.object_type,
            list(manifest.object_types),
            likely_domains,
        )
        missing_args = len(evaluation.missing_arguments_likely)
        trusted = _evaluation_to_trusted_candidate(evaluation)
        if evaluation.fits and evaluation.confidence >= 0.60 and not hard_conflict:
            accepted.append(
                (
                    (
                        1 if exact_object else 0,
                        1 if object_match else 0,
                        1 if domain_match else 0,
                        float(evaluation.confidence),
                        -_manifest_risk_score(manifest),
                        -missing_args,
                        manifest.capability_id,
                    ),
                    trusted,
                )
            )
        elif (
            task_object is not None
            and exact_object
            and domain_match
            and semantic_verbs_compatible(task.semantic_verb, list(manifest.semantic_verbs))
            and not hard_conflict
            and evaluation.confidence < 0.85
        ):
            fallback.append(
                (
                    (
                        1 if exact_object else 0,
                        1 if object_match else 0,
                        1 if domain_match else 0,
                        -_manifest_risk_score(manifest),
                        manifest.capability_id,
                    ),
                    trusted.model_copy(
                        update={
                            "reason": (
                                "Deterministic shortlist fallback selected this candidate after inconclusive binary judging. "
                                + trusted.reason
                            ).strip()
                        }
                    ),
                )
            )

    if accepted:
        accepted.sort(key=lambda item: item[0], reverse=True)
        return accepted[0][1]
    if fallback:
        fallback.sort(key=lambda item: item[0], reverse=True)
        return fallback[0][1]
    return None


def select_capabilities(
    tasks: list[TaskFrame],
    registry: CapabilityRegistry,
    llm_client,
    classification_context: dict[str, Any] | None = None,
    n_best: int = 3,
    trace: PlanningTrace | None = None,
) -> list[CapabilitySelectionResult]:
    """Select ranked capability candidates via deterministic shortlist plus binary LLM judging."""

    results: list[CapabilitySelectionResult] = []

    for task in tasks:
        override = _runtime_introspection_override(task, registry)
        if override is not None:
            if isinstance(trace, PlanningTrace):
                append_trace_entry(
                    trace,
                    PlanningTraceEntry(
                        stage="capability_selection",
                        request_id=str(trace.request_id),
                        prompt_template_id="capability_selection",
                        parsed_proposal=None,
                        selected_candidate=override.model_dump(mode="json"),
                        deterministic_normalizations=["runtime_introspection_override"],
                    ),
                )
            results.append(override)
            continue
        shortlist = _build_shortlist(task, registry, classification_context, shortlist_size=_SHORTLIST_SIZE)
        shortlist_candidates = [
            CapabilityRef(
                capability_id=manifest.capability_id,
                operation_id=manifest.operation_id,
                confidence=_shortlist_candidate_confidence(index),
                reason="Deterministic shortlist candidate from canonical semantic compatibility.",
            )
            for index, manifest in enumerate(shortlist)
        ]
        shortlist_manifest = [
            {
                "capability_id": manifest.capability_id,
                "operation_id": manifest.operation_id,
                "domain": manifest.domain,
                "description": manifest.description,
                "semantic_verbs": list(manifest.semantic_verbs),
                "object_types": list(manifest.object_types),
                "required_arguments": list(manifest.required_arguments),
                "optional_arguments": list(manifest.optional_arguments),
                "risk_level": manifest.risk_level,
                "read_only": manifest.read_only,
                "examples": list(manifest.examples),
            }
            for manifest in shortlist
        ]
        if not shortlist:
            results.append(
                CapabilitySelectionResult(
                    task_id=task.id,
                    candidates=[],
                    selected=None,
                    unresolved_reason=f"No deterministic shortlist candidates were available for task {task.id}.",
                )
            )
            continue

        prompt = _build_selection_prompt(task, shortlist_manifest)
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
            evaluation_map, validation_errors = _validate_shortlist_evaluations(task, shortlist, response)
            evaluation.rejection_reasons.extend(validation_errors)
            validated_selected = _select_from_shortlist(
                task,
                shortlist,
                evaluation_map,
                classification_context,
            )
            unresolved_reason = response.unresolved_reason
            if validated_selected is not None:
                validated_selected, validation_error = _validate_candidate(
                    task,
                    validated_selected,
                    registry,
                )
                if validation_error:
                    unresolved_reason = validation_error
                    evaluation.rejection_reasons.append(validation_error)
            if validated_selected is None and not unresolved_reason:
                unresolved_reason = (
                    evaluation.rejection_reasons[0]
                    if evaluation.rejection_reasons
                    else f"No shortlisted capability was accepted for task {task.id}."
                )
                evaluation.rejection_reasons.append(unresolved_reason)
            evaluation.proposal = CapabilitySelectionResult(
                task_id=task.id,
                candidates=shortlist_candidates,
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
                selected_key = (validated_selected.capability_id, validated_selected.operation_id)
                selected_evaluation = evaluation_map.get(selected_key)
                evaluation.missing_required_count = (
                    len(selected_evaluation.missing_arguments_likely)
                    if selected_evaluation is not None
                    else 0
                )
            evaluation.assumption_count = sum(
                len(item.missing_arguments_likely) for item in response.evaluations
            )
            evaluation.unresolved_count = 0 if unresolved_reason is None else 1
            evaluations.append(evaluation)

        selected = select_best_candidate(evaluations)
        if selected is None or selected.proposal is None:
            results.append(
                CapabilitySelectionResult(
                    task_id=task.id,
                    candidates=shortlist_candidates,
                    selected=None,
                    unresolved_reason=f"No shortlisted capability was accepted for task {task.id}.",
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
                    deterministic_normalizations=["deterministic_capability_shortlist"],
                ),
            )

        results.append(selected.proposal)

    return results
