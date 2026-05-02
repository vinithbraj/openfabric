"""LLM-assisted capability-fit validation between selected tools and user intent."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import CapabilityNotFoundError
from agent_runtime.core.semantic_compatibility import (
    canonical_domain,
    canonical_object_family,
    canonical_semantic_verb,
    domains_compatible as canonical_domains_compatible,
    has_hard_cross_domain_conflict,
    normalize_token,
    object_types_compatible as canonical_object_types_compatible,
    semantic_verbs_compatible as canonical_semantic_verbs_compatible,
)
from agent_runtime.core.types import CapabilityRef, TaskFrame
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult
from agent_runtime.input_pipeline.plan_selection import CandidateEvaluation, select_best_candidate
from agent_runtime.llm.proposals import CapabilityFitProposal
from agent_runtime.llm.structured_call import StructuredCallDiagnostics, StructuredCallError, structured_call
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)

CapabilityFitStatus = Literal[
    "fit",
    "semantic_mismatch",
    "domain_mismatch",
    "object_type_mismatch",
    "missing_required_arguments",
    "unsupported_capability_gap",
    "ambiguous",
    "rejected",
]

_FIT_ACCEPTING_STATUSES = {"fit", "missing_required_arguments"}
_MUTATING_VERBS = {"create", "update", "delete"}


class CapabilityFitDecision(BaseModel):
    """Trusted runtime decision about whether a capability truly fits a task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidate_capability_id: str | None = None
    candidate_operation_id: str | None = None
    status: CapabilityFitStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    llm_proposal: CapabilityFitProposal | None = None
    deterministic_rejections: list[str] = Field(default_factory=list)
    missing_capability_description: str | None = None
    suggested_domain: str | None = None
    suggested_object_type: str | None = None
    normalized_task_domain: str | None = None
    normalized_likely_domains: list[str] = Field(default_factory=list)
    normalized_task_object_type: str | None = None
    normalized_manifest_domain: str | None = None
    normalized_manifest_object_types: list[str] = Field(default_factory=list)
    llm_diagnostics: StructuredCallDiagnostics | None = None
    llm_failed_structurally: bool = False
    requires_clarification: bool = False
    clarification_question: str | None = None

    @property
    def is_fit(self) -> bool:
        """Return whether the task may proceed to argument extraction/execution."""

        return self.status == "fit"


class CapabilityGapDescription(BaseModel):
    """Trusted user-facing explanation of a missing runtime capability."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    understood_user_intent: str
    missing_capability_description: str
    suggested_domain: str | None = None
    suggested_object_type: str | None = None
    suggested_capability_id: str | None = None
    user_facing_message: str
    confidence: float = Field(ge=0.0, le=1.0)


class OutputContractResolution(BaseModel):
    """Trusted resolution showing a task is already satisfied by upstream output."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    producer_task_id: str
    producer_capability_id: str
    producer_operation_id: str
    matched_output_object_types: list[str] = Field(default_factory=list)
    matched_output_fields: list[str] = Field(default_factory=list)
    matched_output_affordances: list[str] = Field(default_factory=list)
    resolution_source: Literal["deterministic", "llm_overlap_review"] = "deterministic"
    llm_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str


class OutputOverlapReviewProposal(BaseModel):
    """Strict boolean LLM proposal for overlap between a task and upstream outputs."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    producer_task_id: str
    satisfied_from_output: bool
    confidence: float = Field(ge=0.0, le=1.0)

def infer_task_domain(task: TaskFrame, classification_context: dict[str, Any] | None = None) -> str | None:
    """Infer the canonical task domain from constraints, object type, and likely domains."""

    context = dict(classification_context or {})
    explicit_domain = canonical_domain(task.constraints.get("domain"))
    if explicit_domain is not None:
        return explicit_domain

    object_domain = canonical_domain(canonical_object_family(task.object_type))
    if object_domain is not None:
        return object_domain

    likely_domains = context.get("likely_domains")
    if isinstance(likely_domains, list):
        for domain in likely_domains:
            normalized = canonical_domain(domain)
            if normalized is not None:
                return normalized
    return None


def normalized_likely_domains(classification_context: dict[str, Any] | None = None) -> list[str]:
    """Return canonical likely domains from classification context."""

    context = dict(classification_context or {})
    likely_domains = context.get("likely_domains")
    if not isinstance(likely_domains, list):
        return []
    normalized: list[str] = []
    for domain in likely_domains:
        canonical = canonical_domain(domain)
        if canonical is not None and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def domains_compatible(
    task_domain: str | None,
    manifest_domain: str | None,
    likely_domains: list[str] | None = None,
    task_object_type: str | None = None,
    manifest_object_types: list[str] | None = None,
) -> bool:
    """Return whether task and manifest domains are canonically compatible."""

    return canonical_domains_compatible(
        task_domain,
        manifest_domain,
        likely_domains,
        task_object_type,
        manifest_object_types or [],
    )


def semantic_verbs_compatible(task_verb: str, manifest_verbs: list[str]) -> bool:
    """Return whether one task verb is semantically compatible with manifest verbs."""

    return canonical_semantic_verbs_compatible(task_verb, manifest_verbs)


def object_types_compatible(
    task_object_type: str | None,
    manifest_object_types: list[str],
    context_domains: list[str] | None = None,
) -> bool:
    """Return whether one task object family is compatible with manifest object types."""

    return canonical_object_types_compatible(task_object_type, manifest_object_types, context_domains)


def has_hard_domain_mismatch(
    task: TaskFrame,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None = None,
) -> bool:
    """Return whether task and manifest have a hard cross-domain conflict."""

    return has_hard_cross_domain_conflict(
        infer_task_domain(task, classification_context),
        manifest.domain,
        task.object_type,
        list(manifest.object_types),
        normalized_likely_domains(classification_context),
    )


def has_hard_object_type_mismatch(task: TaskFrame, manifest: CapabilityManifest) -> bool:
    """Return whether object families are fundamentally incompatible."""

    return (
        not object_types_compatible(task.object_type, list(manifest.object_types), [])
        and has_hard_cross_domain_conflict(
            canonical_domain(canonical_object_family(task.object_type)),
            manifest.domain,
            task.object_type,
            list(manifest.object_types),
            [],
        )
    )


def _fit_prompt(
    task: TaskFrame,
    candidate: CapabilityRef,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None = None,
) -> str:
    """Build the JSON-only capability-fit assessment prompt."""

    likely_domains = normalized_likely_domains(classification_context)
    payload = {
        "original_prompt_preview": str((classification_context or {}).get("original_prompt") or "")[:400],
        "task": {
            "task_id": task.id,
            "description": task.description,
            "semantic_verb": task.semantic_verb,
            "object_type": task.object_type,
            "constraints_summary": task.constraints,
        },
        "classification_context": {"likely_domains": likely_domains},
        "candidate_capability": {
            "capability_id": candidate.capability_id,
            "operation_id": candidate.operation_id,
            "candidate_confidence": candidate.confidence,
            "selection_reason": candidate.reason,
        },
        "candidate_manifest": {
            "capability_id": manifest.capability_id,
            "operation_id": manifest.operation_id,
            "domain": manifest.domain,
            "description": manifest.description,
            "semantic_verbs": manifest.semantic_verbs,
            "object_types": manifest.object_types,
            "output_object_types": manifest.output_object_types,
            "output_fields": manifest.output_fields,
            "output_affordances": manifest.output_affordances,
            "required_arguments": manifest.required_arguments,
            "optional_arguments": manifest.optional_arguments,
            "risk_level": manifest.risk_level,
            "read_only": manifest.read_only,
            "mutates_state": manifest.mutates_state,
            "requires_confirmation": manifest.requires_confirmation,
            "examples": manifest.examples,
        },
    }
    payload_json = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return "\n".join(
        [
            "You are assessing whether a selected capability truly fits a task in an intelligent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, Python, or executable plans.",
            "Do not use shell command templates, backend command strings, gateway internals, secrets, raw execution outputs, or full file contents.",
            "For this candidate capability, decide whether it fits the task.",
            "Answer with fits: true or false, confidence, optional primary_failure_mode, and structured reasons.",
            "Do not invent status labels outside the schema.",
            "Do not accept a candidate merely because it shares a generic verb like read, search, or show.",
            "Consider semantic meaning, domain, object type, required arguments, and risk.",
            "Declared output metadata may help explain what a capability produces, but do not treat output fields as a license to ignore core semantic mismatches.",
            "If a domain label appears to be a synonym, explain that.",
            "If a different capability would fit better, name it if present.",
            "Payload:",
            payload_json,
        ]
    )


def assess_capability_fit_with_llm(
    task: TaskFrame,
    candidate: CapabilityRef,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None,
    llm_client,
) -> tuple[CapabilityFitProposal, StructuredCallDiagnostics | None]:
    """Ask the LLM whether a candidate capability is a true semantic fit."""

    prompt = _fit_prompt(task, candidate, manifest, classification_context)
    diagnostics: StructuredCallDiagnostics | None = None
    try:
        proposal = structured_call(llm_client, prompt, CapabilityFitProposal)
    except StructuredCallError as exc:
        diagnostics = exc.diagnostics
        proposal = CapabilityFitProposal(
            task_id=task.id,
            candidate_capability_id=candidate.capability_id,
            candidate_operation_id=candidate.operation_id,
            fits=False,
            confidence=0.0,
            primary_failure_mode="ambiguous",
            semantic_reason="The runtime could not obtain a structured capability-fit assessment.",
            domain_reason="",
            object_type_reason="",
            argument_reason="",
            risk_reason="",
            better_capability_id=None,
            missing_capability_description=None,
            suggested_domain=infer_task_domain(task, classification_context),
            suggested_object_type=task.object_type,
            requires_clarification=False,
            clarification_question=None,
        )
    if proposal.task_id != task.id:
        proposal = proposal.model_copy(update={"task_id": task.id})
    if proposal.candidate_capability_id != candidate.capability_id:
        proposal = proposal.model_copy(update={"candidate_capability_id": candidate.capability_id})
    if proposal.candidate_operation_id != candidate.operation_id:
        proposal = proposal.model_copy(update={"candidate_operation_id": candidate.operation_id})
    return proposal, diagnostics


def finalize_capability_fit(
    task: TaskFrame,
    candidate: CapabilityRef,
    manifest: CapabilityManifest,
    llm_fit: CapabilityFitProposal,
    llm_diagnostics: StructuredCallDiagnostics | None = None,
    classification_context: dict[str, Any] | None = None,
) -> CapabilityFitDecision:
    """Apply deterministic runtime rules to one untrusted capability-fit proposal."""

    task_domain = infer_task_domain(task, classification_context)
    likely_domains = normalized_likely_domains(classification_context)
    manifest_domain = canonical_domain(manifest.domain)
    task_object_type = canonical_object_family(task.object_type)
    manifest_object_types = [
        object_type
        for object_type in (canonical_object_family(value) for value in manifest.object_types)
        if object_type is not None
    ]
    llm_fits = bool(llm_fit.fits)
    llm_failure_mode = str(llm_fit.primary_failure_mode or "").strip().lower() or None
    deterministic_rejections: list[str] = []
    reasons: list[str] = []

    if candidate.capability_id != manifest.capability_id:
        deterministic_rejections.append("candidate capability_id does not match manifest capability_id")
    if candidate.operation_id != manifest.operation_id:
        deterministic_rejections.append("candidate operation_id does not match manifest operation_id")
    if canonical_semantic_verb(task.semantic_verb) in _MUTATING_VERBS and not manifest.mutates_state:
        deterministic_rejections.append("mutating task cannot be satisfied by a non-mutating capability")
    if manifest.capability_id.strip().lower() == "unknown":
        deterministic_rejections.append("unknown capability cannot be accepted")
    if has_hard_cross_domain_conflict(
        task_domain,
        manifest.domain,
        task.object_type,
        list(manifest.object_types),
        likely_domains,
    ):
        deterministic_rejections.append(
            f"hard cross-domain conflict between {task_object_type or task_domain} and {manifest_object_types[0] if manifest_object_types else manifest_domain}"
        )
    if not semantic_verbs_compatible(task.semantic_verb, list(manifest.semantic_verbs)):
        deterministic_rejections.append("semantic verb incompatible with candidate capability")
    if not canonical_domains_compatible(
        task_domain,
        manifest.domain,
        likely_domains,
        task.object_type,
        list(manifest.object_types),
    ):
        deterministic_rejections.append("task domain is incompatible with candidate capability domain")
    if not object_types_compatible(task.object_type, list(manifest.object_types), likely_domains):
        deterministic_rejections.append("task object type is incompatible with candidate capability object types")
    if manifest.requires_confirmation and not task.requires_confirmation:
        deterministic_rejections.append("candidate capability requires confirmation that is not present on the task")

    reasons.extend(
        reason
        for reason in [
            llm_fit.semantic_reason,
            llm_fit.domain_reason,
            llm_fit.object_type_reason,
            llm_fit.argument_reason,
            llm_fit.risk_reason,
            candidate.reason,
        ]
        if str(reason or "").strip()
    )
    if llm_diagnostics is not None:
        reasons.append(
            f"Capability-fit LLM response failed with {llm_diagnostics.error_kind}: {llm_diagnostics.error_message}"
        )

    deterministic_strong = (
        semantic_verbs_compatible(task.semantic_verb, list(manifest.semantic_verbs))
        and not any(reason.startswith("hard cross-domain conflict") for reason in deterministic_rejections)
        and (
            canonical_domains_compatible(
                task_domain,
                manifest.domain,
                likely_domains,
                task.object_type,
                list(manifest.object_types),
            )
            or object_types_compatible(task.object_type, list(manifest.object_types), likely_domains)
        )
    )
    llm_high_conf_reject = (
        llm_fits is False
        and llm_failure_mode in {
            "semantic_mismatch",
            "domain_mismatch",
            "object_type_mismatch",
            "unsupported_capability_gap",
            "rejected",
        }
        and llm_fit.confidence >= 0.85
    )

    status: CapabilityFitStatus = "fit"
    if any(reason.startswith("hard cross-domain conflict") for reason in deterministic_rejections):
        status = "domain_mismatch"
    elif "semantic verb incompatible with candidate capability" in deterministic_rejections:
        status = "semantic_mismatch"
    elif "task domain is incompatible with candidate capability domain" in deterministic_rejections:
        status = "domain_mismatch"
    elif "task object type is incompatible with candidate capability object types" in deterministic_rejections:
        status = "object_type_mismatch"
    elif llm_fits and llm_fit.confidence >= 0.70 and deterministic_strong:
        status = "fit"
    elif deterministic_strong and (
        llm_fits
        or llm_failure_mode in {"ambiguous", "missing_required_arguments"}
        or llm_fit.confidence < 0.70
    ):
        status = "fit"
        if not llm_fits:
            reasons.append("Deterministic compatibility was strong enough to accept the candidate despite low-confidence LLM ambiguity.")
    elif llm_high_conf_reject and not deterministic_strong:
        status = llm_failure_mode  # type: ignore[assignment]
    elif llm_fits and llm_fit.confidence >= 0.70 and not deterministic_rejections:
        status = "fit"
    elif llm_failure_mode == "missing_required_arguments" and deterministic_strong:
        status = "fit"
        reasons.append("The candidate semantically fits, but some arguments may still need extraction.")
    elif llm_failure_mode in {
        "semantic_mismatch",
        "domain_mismatch",
        "object_type_mismatch",
        "missing_required_arguments",
        "unsupported_capability_gap",
        "ambiguous",
        "rejected",
    }:
        status = llm_failure_mode  # type: ignore[assignment]
    else:
        status = "ambiguous"

    suggested_domain = canonical_domain(llm_fit.suggested_domain) or task_domain
    suggested_object_type = canonical_object_family(llm_fit.suggested_object_type) or task_object_type or task.object_type
    missing_description = llm_fit.missing_capability_description
    if status != "fit" and not missing_description:
        inferred_domain = suggested_domain or "unknown"
        missing_description = (
            f"A capability for {canonical_semantic_verb(task.semantic_verb)} over {suggested_object_type or task.object_type} in the {inferred_domain} domain "
            f"is not available or does not fit safely."
        )

    final_confidence = min(candidate.confidence, llm_fit.confidence)
    if status == "fit" and llm_fits is False and llm_failure_mode in {"ambiguous", "missing_required_arguments"} and deterministic_strong:
        final_confidence = candidate.confidence

    return CapabilityFitDecision(
        task_id=task.id,
        candidate_capability_id=candidate.capability_id,
        candidate_operation_id=candidate.operation_id,
        status=status,
        confidence=final_confidence,
        reasons=reasons or [f"Capability fit status: {status}."],
        llm_proposal=llm_fit,
        deterministic_rejections=deterministic_rejections,
        missing_capability_description=missing_description,
        suggested_domain=suggested_domain,
        suggested_object_type=suggested_object_type,
        normalized_task_domain=task_domain,
        normalized_likely_domains=likely_domains,
        normalized_task_object_type=task_object_type,
        normalized_manifest_domain=manifest_domain,
        normalized_manifest_object_types=manifest_object_types,
        llm_diagnostics=llm_diagnostics,
        llm_failed_structurally=llm_diagnostics is not None,
        requires_clarification=bool(llm_fit.requires_clarification),
        clarification_question=llm_fit.clarification_question,
    )


def _gap_from_task(
    task: TaskFrame,
    decision: CapabilityFitDecision,
) -> CapabilityGapDescription:
    """Construct one trusted user-facing capability gap description."""

    missing_description = (
        decision.missing_capability_description
        or f"The runtime lacks a compatible capability for {task.description}."
    )
    suggested_capability_id = (
        decision.llm_proposal.better_capability_id
        if decision.llm_proposal is not None and decision.llm_proposal.better_capability_id
        else None
    )
    user_facing_message = (
        f"I understood the request as '{task.description}', but this runtime does not currently "
        f"have a compatible capability to do that safely. {missing_description}"
    )
    if decision.requires_clarification and decision.clarification_question:
        user_facing_message = f"{user_facing_message} Clarification may help: {decision.clarification_question}"
    return CapabilityGapDescription(
        task_id=task.id,
        understood_user_intent=task.description,
        missing_capability_description=missing_description,
        suggested_domain=decision.suggested_domain,
        suggested_object_type=decision.suggested_object_type,
        suggested_capability_id=suggested_capability_id,
        user_facing_message=user_facing_message,
        confidence=decision.confidence,
    )


def _output_overlap_prompt(
    task: TaskFrame,
    producer_task: TaskFrame,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None = None,
) -> str:
    """Build the JSON-only prompt for reviewing overlap with upstream outputs."""

    payload = {
        "original_prompt_preview": str((classification_context or {}).get("original_prompt") or "")[:400],
        "downstream_task": {
            "task_id": task.id,
            "description": task.description,
            "semantic_verb": task.semantic_verb,
            "object_type": task.object_type,
        },
        "dependency_relation": {
            "depends_on_task_id": producer_task.id,
            "dependency_description": producer_task.description,
        },
        "upstream_task": {
            "task_id": producer_task.id,
            "description": producer_task.description,
            "semantic_verb": producer_task.semantic_verb,
            "object_type": producer_task.object_type,
        },
        "upstream_capability_manifest": {
            "capability_id": manifest.capability_id,
            "operation_id": manifest.operation_id,
            "domain": manifest.domain,
            "description": manifest.description,
            "semantic_verbs": manifest.semantic_verbs,
            "object_types": manifest.object_types,
            "output_object_types": manifest.output_object_types,
            "output_fields": manifest.output_fields,
            "output_affordances": manifest.output_affordances,
            "read_only": manifest.read_only,
            "mutates_state": manifest.mutates_state,
            "requires_confirmation": manifest.requires_confirmation,
        },
    }
    payload_json = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return "\n".join(
        [
            "You are reviewing whether a downstream task is already satisfied by upstream declared outputs.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, Python, or executable plans.",
            "Decide whether the downstream task can be satisfied entirely from the upstream capability's declared outputs without any additional execution.",
            "Answer only with satisfied_from_output: true or false, confidence, task_id, and producer_task_id.",
            "Do not invent undeclared outputs, hidden side effects, or implicit filesystem lookups.",
            "If the downstream task would require any new action, new computation, new I/O, or any state mutation, answer false.",
            "Payload:",
            payload_json,
        ]
    )


def review_output_overlap_with_llm(
    task: TaskFrame,
    producer_task: TaskFrame,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None,
    llm_client,
) -> tuple[OutputOverlapReviewProposal, StructuredCallDiagnostics | None]:
    """Ask the LLM for a strict boolean overlap judgment on upstream outputs."""

    prompt = _output_overlap_prompt(task, producer_task, manifest, classification_context)
    diagnostics: StructuredCallDiagnostics | None = None
    try:
        proposal = structured_call(llm_client, prompt, OutputOverlapReviewProposal)
    except StructuredCallError as exc:
        diagnostics = exc.diagnostics
        proposal = OutputOverlapReviewProposal(
            task_id=task.id,
            producer_task_id=producer_task.id,
            satisfied_from_output=False,
            confidence=0.0,
        )
    if proposal.task_id != task.id:
        proposal = proposal.model_copy(update={"task_id": task.id})
    if proposal.producer_task_id != producer_task.id:
        proposal = proposal.model_copy(update={"producer_task_id": producer_task.id})
    return proposal, diagnostics


def _normalized_output_object_types(manifest: CapabilityManifest) -> list[str]:
    """Return canonical output object families for one manifest."""

    return [
        family
        for family in (canonical_object_family(value) for value in manifest.output_object_types)
        if family is not None
    ]


def _normalized_output_fields(manifest: CapabilityManifest) -> list[str]:
    """Return normalized output field names for one manifest."""

    fields: list[str] = []
    for value in manifest.output_fields:
        normalized = normalize_token(value)
        if normalized is not None and normalized not in fields:
            fields.append(normalized)
    return fields


def _normalized_output_affordances(manifest: CapabilityManifest) -> list[str]:
    """Return normalized output affordance names for one manifest."""

    affordances: list[str] = []
    for value in manifest.output_affordances:
        normalized = normalize_token(value)
        if normalized is not None and normalized not in affordances:
            affordances.append(normalized)
    return affordances


def _task_requests_absolute_path(task: TaskFrame) -> bool:
    """Return whether a task explicitly asks for an absolute/full path."""

    description = normalize_token(task.description)
    if description is None:
        return False
    return "full_path" in description or "absolute_path" in description


def _task_requests_path(task: TaskFrame) -> bool:
    """Return whether a task asks for path-like metadata."""

    description = normalize_token(task.description)
    if description is None:
        return False
    return (
        "path" in description
        or "where_it_was_saved" in description
        or "saved_location" in description
        or "saved_to" in description
    )


def _path_contract_evidence(
    manifest: CapabilityManifest,
) -> tuple[list[str], list[str], list[str]]:
    """Return path-like output evidence declared by one manifest."""

    output_object_types = _normalized_output_object_types(manifest)
    output_fields = _normalized_output_fields(manifest)
    output_affordances = _normalized_output_affordances(manifest)

    matched_output_object_types = [
        item for item in output_object_types if item == "filesystem.path"
    ]
    matched_output_fields = [
        item for item in output_fields if item in {"path", "absolute_path"}
    ]
    matched_output_affordances = [
        (
            "returns.absolute_path"
            if item == "returns_absolute_path"
            else "returns.relative_path"
        )
        for item in output_affordances
        if item in {"returns_absolute_path", "returns_relative_path"}
    ]
    return matched_output_object_types, matched_output_fields, matched_output_affordances


def _resolve_task_from_upstream_output(
    task: TaskFrame,
    producer_task: TaskFrame,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None = None,
) -> OutputContractResolution | None:
    """Return a resolution when one upstream capability output already satisfies a task."""

    if canonical_semantic_verb(task.semantic_verb) in _MUTATING_VERBS | {"execute"}:
        return None

    likely_domains = normalized_likely_domains(classification_context)
    output_object_types = _normalized_output_object_types(manifest)
    output_fields = _normalized_output_fields(manifest)
    output_affordances = _normalized_output_affordances(manifest)

    matched_output_object_types: list[str] = []
    matched_output_fields: list[str] = []
    matched_output_affordances: list[str] = []

    if output_object_types and object_types_compatible(task.object_type, output_object_types, likely_domains):
        matched_output_object_types = output_object_types

    wants_absolute_path = _task_requests_absolute_path(task)
    wants_any_path = wants_absolute_path or _task_requests_path(task)
    if wants_absolute_path and (
        "absolute_path" in output_fields or "returns_absolute_path" in output_affordances
    ):
        if "absolute_path" in output_fields:
            matched_output_fields.append("absolute_path")
        if "returns_absolute_path" in output_affordances:
            matched_output_affordances.append("returns.absolute_path")
    elif wants_any_path and (
        "path" in output_fields
        or "absolute_path" in output_fields
        or "returns_relative_path" in output_affordances
        or "returns_absolute_path" in output_affordances
    ):
        if "path" in output_fields:
            matched_output_fields.append("path")
        if "absolute_path" in output_fields:
            matched_output_fields.append("absolute_path")
        if "returns_relative_path" in output_affordances:
            matched_output_affordances.append("returns.relative_path")
        if "returns_absolute_path" in output_affordances:
            matched_output_affordances.append("returns.absolute_path")

    if not matched_output_object_types and not matched_output_fields and not matched_output_affordances:
        return None

    producer_label = f"{manifest.capability_id} via {manifest.operation_id}"
    reason_bits: list[str] = []
    if matched_output_object_types:
        reason_bits.append(
            "its declared output object types include "
            + ", ".join(f"`{item}`" for item in matched_output_object_types)
        )
    if matched_output_fields:
        reason_bits.append(
            "it returns output fields "
            + ", ".join(f"`{item}`" for item in matched_output_fields)
        )
    if matched_output_affordances:
        reason_bits.append(
            "it advertises output affordances "
            + ", ".join(f"`{item}`" for item in matched_output_affordances)
        )
    reason = (
        f"Task `{task.id}` can be satisfied from upstream task `{producer_task.id}` because `{producer_label}` already returns the requested metadata; "
        + "; ".join(reason_bits)
        + "."
    )
    return OutputContractResolution(
        task_id=task.id,
        producer_task_id=producer_task.id,
        producer_capability_id=manifest.capability_id,
        producer_operation_id=manifest.operation_id,
        matched_output_object_types=matched_output_object_types,
        matched_output_fields=matched_output_fields,
        matched_output_affordances=matched_output_affordances,
        resolution_source="deterministic",
        reason=reason,
    )


def _resolve_task_from_upstream_output_with_llm(
    task: TaskFrame,
    producer_task: TaskFrame,
    manifest: CapabilityManifest,
    classification_context: dict[str, Any] | None,
    llm_client,
    trace: PlanningTrace | None = None,
) -> OutputContractResolution | None:
    """Ask the LLM about semantic overlap, then validate that overlap deterministically."""

    if llm_client is None:
        return None
    if canonical_semantic_verb(task.semantic_verb) in _MUTATING_VERBS | {"execute"}:
        return None

    path_object_types, path_fields, path_affordances = _path_contract_evidence(manifest)
    if not path_object_types and not path_fields and not path_affordances:
        return None

    review, diagnostics = review_output_overlap_with_llm(
        task,
        producer_task,
        manifest,
        classification_context,
        llm_client,
    )
    if isinstance(trace, PlanningTrace):
        model_name, temperature = llm_client_metadata(llm_client)
        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="output_contract_overlap_review",
                request_id=str(trace.request_id),
                model_name=model_name,
                llm_temperature=temperature,
                prompt_template_id="output_contract_overlap_review",
                raw_llm_response=(
                    diagnostics.model_dump(mode="json")
                    if diagnostics is not None
                    else review.model_dump(mode="json")
                ),
                parsed_proposal=review.model_dump(mode="json"),
                llm_diagnostics=(
                    diagnostics.model_dump(mode="json")
                    if diagnostics is not None
                    else None
                ),
            )
        )

    if diagnostics is not None or not review.satisfied_from_output:
        return None

    reason_bits: list[str] = []
    if path_object_types:
        reason_bits.append(
            "the producer declares output object types "
            + ", ".join(f"`{item}`" for item in path_object_types)
        )
    if path_fields:
        reason_bits.append(
            "the producer returns output fields "
            + ", ".join(f"`{item}`" for item in path_fields)
        )
    if path_affordances:
        reason_bits.append(
            "the producer advertises output affordances "
            + ", ".join(f"`{item}`" for item in path_affordances)
        )

    reason = (
        f"Task `{task.id}` can be satisfied from upstream task `{producer_task.id}` because the overlap reviewer judged the downstream task to be fully satisfied by declared upstream outputs, "
        "and deterministic validation confirmed path-like output metadata on the producer; "
        + "; ".join(reason_bits)
        + "."
    )
    return OutputContractResolution(
        task_id=task.id,
        producer_task_id=producer_task.id,
        producer_capability_id=manifest.capability_id,
        producer_operation_id=manifest.operation_id,
        matched_output_object_types=path_object_types,
        matched_output_fields=path_fields,
        matched_output_affordances=path_affordances,
        resolution_source="llm_overlap_review",
        llm_confidence=review.confidence,
        reason=reason,
    )


def resolve_tasks_from_output_contracts(
    tasks: list[TaskFrame],
    fit_decisions: list[CapabilityFitDecision],
    selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
    classification_context: dict[str, Any] | None = None,
    llm_client=None,
    trace: PlanningTrace | None = None,
) -> list[OutputContractResolution]:
    """Resolve non-executable tasks that are already satisfied by upstream outputs."""

    task_by_id = {task.id: task for task in tasks}
    decision_by_task = {decision.task_id: decision for decision in fit_decisions}
    selection_by_task = {selection.task_id: selection for selection in selections}
    resolutions: list[OutputContractResolution] = []

    for task in tasks:
        decision = decision_by_task.get(task.id)
        if decision is None:
            continue
        if not task.dependencies:
            continue
        for dependency_id in task.dependencies:
            producer_task = task_by_id.get(dependency_id)
            producer_decision = decision_by_task.get(dependency_id)
            producer_selection = selection_by_task.get(dependency_id)
            if producer_task is None or producer_decision is None or not producer_decision.is_fit:
                continue
            if producer_selection is None or producer_selection.selected is None:
                continue
            try:
                producer_manifest = registry.get(producer_selection.selected.capability_id).manifest
            except CapabilityNotFoundError:
                continue
            resolution = _resolve_task_from_upstream_output(
                task,
                producer_task,
                producer_manifest,
                classification_context=classification_context,
            )
            if resolution is None and not decision.is_fit:
                resolution = _resolve_task_from_upstream_output_with_llm(
                    task,
                    producer_task,
                    producer_manifest,
                    classification_context,
                    llm_client,
                    trace=trace,
                )
            if resolution is not None:
                resolutions.append(resolution)
                break
    return resolutions


def assess_capability_fit(
    tasks: list[TaskFrame],
    selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
    classification_context: dict[str, Any] | None,
    llm_client,
    trace: PlanningTrace | None = None,
) -> tuple[list[CapabilityFitDecision], list[CapabilityGapDescription]]:
    """Assess candidate capabilities and produce trusted fit decisions and gaps."""

    selection_by_task = {selection.task_id: selection for selection in selections}
    decisions: list[CapabilityFitDecision] = []
    gaps: list[CapabilityGapDescription] = []

    for task in tasks:
        selection = selection_by_task.get(task.id)
        if selection is None:
            decision = CapabilityFitDecision(
                task_id=task.id,
                status="unsupported_capability_gap",
                confidence=0.0,
                reasons=["No capability selection result was available for this task."],
                candidate_capability_id=None,
                candidate_operation_id=None,
                llm_proposal=None,
                deterministic_rejections=["no capability selection result"],
                missing_capability_description=f"No capability candidates were available for '{task.description}'.",
                suggested_domain=infer_task_domain(task, classification_context),
                suggested_object_type=task.object_type,
            )
            decisions.append(decision)
            gaps.append(_gap_from_task(task, decision))
            continue

        ordered_candidates: list[CapabilityRef] = []
        if selection.selected is not None:
            ordered_candidates.append(selection.selected)
        for candidate in selection.candidates:
            if all(existing.capability_id != candidate.capability_id or existing.operation_id != candidate.operation_id for existing in ordered_candidates):
                ordered_candidates.append(candidate)

        evaluations: list[CandidateEvaluation[CapabilityFitDecision]] = []
        raw_responses: list[Any] = []
        parsed_proposals: list[Any] = []

        if not ordered_candidates:
            decision = CapabilityFitDecision(
                task_id=task.id,
                candidate_capability_id=None,
                candidate_operation_id=None,
                status="unsupported_capability_gap",
                confidence=0.0,
                reasons=[
                    selection.unresolved_reason
                    or "No selected or ranked capability candidates were available."
                ],
                llm_proposal=None,
                deterministic_rejections=["no capability candidates"],
                missing_capability_description=selection.unresolved_reason
                or f"No capability candidates fit '{task.description}'.",
                suggested_domain=infer_task_domain(task, classification_context),
                suggested_object_type=task.object_type,
            )
            decisions.append(decision)
            gaps.append(_gap_from_task(task, decision))
            if isinstance(trace, PlanningTrace):
                append_trace_entry(
                    trace,
                    PlanningTraceEntry(
                        stage="capability_fit",
                        request_id=str(trace.request_id),
                        prompt_template_id="capability_fit",
                        selected_candidate=decision.model_dump(mode="json"),
                        rejection_reasons=list(decision.deterministic_rejections),
                    ),
                )
            continue

        for candidate in ordered_candidates:
            evaluation = CandidateEvaluation[CapabilityFitDecision]()
            try:
                manifest = registry.get(candidate.capability_id).manifest
            except CapabilityNotFoundError:
                decision = CapabilityFitDecision(
                    task_id=task.id,
                    candidate_capability_id=candidate.capability_id,
                    candidate_operation_id=candidate.operation_id,
                    status="rejected",
                    confidence=0.0,
                    reasons=[f"Candidate capability is not registered: {candidate.capability_id}."],
                    llm_proposal=None,
                    deterministic_rejections=["unknown capability"],
                    missing_capability_description=f"No registered capability exists for {candidate.capability_id}.",
                    suggested_domain=infer_task_domain(task, classification_context),
                    suggested_object_type=task.object_type,
                )
                evaluation.proposal = decision
                evaluation.rejection_reasons.extend(decision.deterministic_rejections)
                evaluations.append(evaluation)
                continue

            llm_fit, llm_diagnostics = assess_capability_fit_with_llm(
                task,
                candidate,
                manifest,
                classification_context,
                llm_client,
            )
            raw_responses.append(
                llm_diagnostics.model_dump(mode="json")
                if llm_diagnostics is not None
                else llm_fit.model_dump(mode="json")
            )
            parsed_proposals.append(llm_fit.model_dump(mode="json"))
            decision = finalize_capability_fit(
                task,
                candidate,
                manifest,
                llm_fit,
                llm_diagnostics,
                classification_context,
            )
            evaluation.proposal = decision
            evaluation.confidence = decision.confidence
            evaluation.capability_compatibility = 1 if decision.status == "fit" else 0
            evaluation.risk_score = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(
                manifest.risk_level,
                0,
            )
            evaluation.assumption_count = 1 if decision.requires_clarification else 0
            evaluation.unresolved_count = 0 if decision.status == "fit" else 1
            if not decision.is_fit:
                evaluation.rejection_reasons.extend(decision.deterministic_rejections or [decision.status])
            evaluations.append(evaluation)

        selected = select_best_candidate(evaluations)
        selected_decision = selected.proposal if selected is not None else None
        if selected_decision is None:
            selected_decision = CapabilityFitDecision(
                task_id=task.id,
                status="unsupported_capability_gap",
                confidence=0.0,
                reasons=["No capability-fit decisions were produced."],
                candidate_capability_id=None,
                candidate_operation_id=None,
                llm_proposal=None,
                deterministic_rejections=["no capability-fit decisions"],
                missing_capability_description=f"No capability could be validated for '{task.description}'.",
                suggested_domain=infer_task_domain(task, classification_context),
                suggested_object_type=task.object_type,
            )

        decisions.append(selected_decision)
        if not selected_decision.is_fit:
            gaps.append(_gap_from_task(task, selected_decision))

        if isinstance(trace, PlanningTrace):
            model_name, temperature = llm_client_metadata(llm_client)
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="capability_fit",
                    request_id=str(trace.request_id),
                    model_name=model_name,
                    llm_temperature=temperature,
                    prompt_template_id="capability_fit",
                    raw_llm_response=raw_responses,
                    parsed_proposal=parsed_proposals,
                    selected_candidate=selected_decision.model_dump(mode="json"),
                    llm_diagnostics=(
                        selected_decision.llm_diagnostics.model_dump(mode="json")
                        if selected_decision.llm_diagnostics is not None
                        else None
                    ),
                    rejection_reasons=[
                        reason for evaluation in evaluations for reason in evaluation.rejection_reasons
                    ],
                ),
            )

    return decisions, gaps


__all__ = [
    "CapabilityFitDecision",
    "CapabilityFitProposal",
    "CapabilityFitStatus",
    "CapabilityGapDescription",
    "OutputContractResolution",
    "assess_capability_fit",
    "assess_capability_fit_with_llm",
    "domains_compatible",
    "finalize_capability_fit",
    "has_hard_domain_mismatch",
    "has_hard_object_type_mismatch",
    "infer_task_domain",
    "object_types_compatible",
    "review_output_overlap_with_llm",
    "resolve_tasks_from_output_contracts",
    "semantic_verbs_compatible",
]
