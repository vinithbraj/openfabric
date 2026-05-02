"""Semantic verb assignment for decomposed task frames."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.semantic_compatibility import (
    canonical_domain,
    canonical_object_family,
    match_object_family_from_text,
    object_domain,
)
from agent_runtime.core.types import RiskLevel, SemanticVerb, TaskFrame
from agent_runtime.llm.proposals import VerbAssignmentProposal
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)
from agent_runtime.llm.structured_call import structured_call

VERB_VOCABULARY: tuple[str, ...] = (
    "read",
    "search",
    "create",
    "update",
    "delete",
    "transform",
    "analyze",
    "summarize",
    "compare",
    "execute",
    "render",
    "unknown",
)


class TaskVerbAssignment(BaseModel):
    """Typed LLM response for one task-frame verb assignment."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    semantic_verb: SemanticVerb
    object_type: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    requires_confirmation: bool = False


class TaskVerbAssignmentResult(BaseModel):
    """Collection of verb assignments returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[TaskVerbAssignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_task_ids(self) -> "TaskVerbAssignmentResult":
        """Ensure each task id appears at most once in the response."""

        task_ids = [assignment.task_id for assignment in self.assignments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task verb assignments must reference unique task ids.")
        return self


class VerbClassifier:
    """Small deterministic fallback used by the placeholder orchestrator."""

    def classify(self, prompt: str) -> str:
        """Return a conservative verb guess without calling an LLM."""

        lowered = prompt.strip().lower()
        if lowered.startswith("list"):
            return "read"
        if lowered.startswith("count"):
            return "analyze"
        if lowered.startswith("search"):
            return "search"
        if lowered.startswith("delete") or lowered.startswith("remove"):
            return "delete"
        if lowered.startswith("run"):
            return "execute"
        if lowered.startswith("summarize"):
            return "summarize"
        return "read"


def _known_object_type_vocabulary(
    registry: CapabilityRegistry,
    likely_domains: list[str] | None = None,
) -> list[str]:
    """Return a runtime-owned object-type vocabulary derived from registered manifests."""

    known_types = {"unknown"}
    for manifest in registry.list_manifests():
        for object_type in [*manifest.object_types, *manifest.output_object_types]:
            canonical = canonical_object_family(object_type)
            if canonical is not None:
                known_types.add(canonical)

    normalized_likely_domains = {
        canonical_domain(domain)
        for domain in (likely_domains or [])
        if canonical_domain(domain) is not None
    }
    if not normalized_likely_domains:
        return sorted(known_types)

    prioritized = [
        object_type
        for object_type in sorted(known_types)
        if object_type == "unknown" or object_domain(object_type) in normalized_likely_domains
    ]
    remainder = [object_type for object_type in sorted(known_types) if object_type not in prioritized]
    return prioritized + remainder


def _normalize_assigned_object_type(
    raw_object_type: str,
    task: TaskFrame,
    allowed_object_types: list[str],
    semantic_verb: str,
) -> str:
    """Normalize one assigned object type into the runtime-owned vocabulary."""

    allowed = set(allowed_object_types)
    combined_text = " ".join(
        part
        for part in (
            raw_object_type,
            task.description,
            task.raw_evidence or "",
        )
        if str(part).strip()
    )
    lowered_combined = combined_text.lower()
    if "filesystem.path" in allowed and (
        "full path" in lowered_combined
        or "absolute path" in lowered_combined
        or "file path" in lowered_combined
        or "saved path" in lowered_combined
        or "where it was saved" in lowered_combined
    ):
        return "filesystem.path"
    if (
        semantic_verb in {"create", "update"}
        and "filesystem.file" in allowed
        and (
            " file" in f" {lowered_combined}"
            or ".txt" in lowered_combined
            or ".md" in lowered_combined
            or ".json" in lowered_combined
            or "save" in lowered_combined
            or "write" in lowered_combined
        )
    ):
        return "filesystem.file"
    inferred = match_object_family_from_text(
        combined_text,
        allowed_object_types,
    )
    canonical = canonical_object_family(raw_object_type)
    if canonical in allowed:
        if (
            inferred in allowed
            and inferred is not None
            and object_domain(inferred) == object_domain(canonical)
            and "." not in canonical
            and "." in inferred
        ):
            return inferred
        return canonical
    if inferred in allowed:
        return inferred
    return "unknown"


def _build_assignment_prompt(
    tasks: list[TaskFrame],
    registry: CapabilityRegistry,
    likely_domains: list[str] | None = None,
) -> str:
    """Build the strict JSON-only prompt for semantic verb assignment."""

    schema = model_json_schema(TaskVerbAssignmentResult)
    object_type_vocabulary = _known_object_type_vocabulary(registry, likely_domains)
    task_lines = [
        f"- task_id={task.id}; description={task.description}; current_constraints={task.constraints}"
        for task in tasks
    ]
    return "\n".join(
        [
            "You are assigning semantic verbs to existing task frames.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not generate commands, shell syntax, SQL, code, or executable plans.",
            "Assign exactly one semantic verb from this controlled vocabulary:",
            ", ".join(VERB_VOCABULARY),
            "Choose object_type from this runtime-owned controlled vocabulary only:",
            ", ".join(object_type_vocabulary),
            "Do not invent new object_type labels. Return the closest matching object_type from the provided list, or unknown if nothing fits.",
            f"Likely domains: {', '.join(likely_domains or []) or 'unknown'}",
            "Also assign object_type, intent_confidence, risk_level, and requires_confirmation.",
            "Only classify task meaning. Do not choose tools or implementation details.",
            "Task list:",
            *task_lines,
            "JSON schema:",
            str(schema),
        ]
    )


def _post_process_assignment(task: TaskFrame) -> TaskFrame:
    """Apply deterministic risk and confirmation rules after LLM assignment."""

    dry_run = bool(task.constraints.get("dry_run"))
    elevated_risk = bool(task.constraints.get("elevated_risk"))
    low_risk_verbs = {"read", "search", "analyze", "summarize", "render"}
    if task.semantic_verb == "delete" and not dry_run:
        task.requires_confirmation = True
    if task.semantic_verb in {"create", "update"} and not dry_run:
        task.requires_confirmation = True
    if task.semantic_verb in low_risk_verbs and not elevated_risk:
        task.risk_level = "low"
        task.requires_confirmation = False
    if task.semantic_verb == "execute" and task.risk_level == "low":
        task.risk_level = "medium"
    if task.semantic_verb == "execute" and bool(task.constraints.get("read_only_capability")):
        task.risk_level = "low"
    return task


def assign_semantic_verbs(
    tasks: list[TaskFrame],
    llm_client,
    registry: CapabilityRegistry,
    likely_domains: list[str] | None = None,
    trace: PlanningTrace | None = None,
) -> list[TaskFrame]:
    """Assign semantic verbs and semantic metadata to task frames through an LLM."""

    if not tasks:
        return []

    allowed_object_types = _known_object_type_vocabulary(registry, likely_domains)
    prompt = _build_assignment_prompt(tasks, registry, likely_domains)
    proposal = structured_call(llm_client, prompt, VerbAssignmentProposal)
    assignment_result = TaskVerbAssignmentResult.model_validate(
        {
            "assignments": [
                {
                    "task_id": assignment.task_id,
                    "semantic_verb": assignment.semantic_verb,
                    "object_type": assignment.object_type,
                    "intent_confidence": assignment.intent_confidence,
                    "risk_level": assignment.risk_level,
                    "requires_confirmation": assignment.requires_confirmation,
                }
                for assignment in proposal.assignments
            ]
        }
    )
    assignments = {assignment.task_id: assignment for assignment in assignment_result.assignments}

    updated_tasks: list[TaskFrame] = []
    deterministic_normalizations: list[str] = []
    for task in tasks:
        assignment = assignments.get(task.id)
        if assignment is None:
            raise ValueError(f"Missing semantic verb assignment for task {task.id}.")
        normalized_object_type = _normalize_assigned_object_type(
            assignment.object_type,
            task,
            allowed_object_types,
            assignment.semantic_verb,
        )
        updated_task = task.model_copy(
            update={
                "semantic_verb": assignment.semantic_verb,
                "object_type": normalized_object_type,
                "intent_confidence": assignment.intent_confidence,
                "risk_level": assignment.risk_level,
                "requires_confirmation": assignment.requires_confirmation,
            }
        )
        processed_task = _post_process_assignment(updated_task)
        if processed_task != updated_task or normalized_object_type != assignment.object_type:
            deterministic_normalizations.append(task.id)
        updated_tasks.append(processed_task)
    if isinstance(trace, PlanningTrace):
        model_name, temperature = llm_client_metadata(llm_client)
        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="semantic_verb_assignment",
                request_id=str(trace.request_id),
                model_name=model_name,
                llm_temperature=temperature,
                prompt_template_id="semantic_verb_assignment",
                raw_llm_response=proposal.model_dump(mode="json"),
                parsed_proposal=proposal.model_dump(mode="json"),
                selected_candidate=assignment_result.model_dump(mode="json"),
                deterministic_normalizations=deterministic_normalizations,
            ),
        )
    return updated_tasks
