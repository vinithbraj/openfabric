"""Argument extraction for selected capability tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import CapabilityNotFoundError
from agent_runtime.core.types import InputRef, TaskFrame
from agent_runtime.input_pipeline.dataflow_planning import ValidatedDataflowPlan
from agent_runtime.input_pipeline.capability_fit import CapabilityFitDecision
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult
from agent_runtime.input_pipeline.plan_selection import CandidateEvaluation, select_best_candidate
from agent_runtime.llm.proposals import ArgumentExtractionProposal, collect_n_best_structured_attempts
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)


class ArgumentExtractionResult(BaseModel):
    """Normalized argument payload for one selected capability."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    capability_id: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required_arguments: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class _ArgumentExtractionResponse(BaseModel):
    """Structured LLM response for capability argument extraction."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    capability_id: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required_arguments: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ArgumentExtractor:
    """Compatibility placeholder used by the lightweight scaffold orchestrator."""

    def extract(self, prompt: str) -> list[object]:
        """Return no inferred arguments until the orchestrator is upgraded."""

        _ = prompt
        return []


def _build_argument_prompt(
    task: TaskFrame,
    manifest: CapabilityManifest,
    fixed_arguments: dict[str, Any] | None = None,
) -> str:
    """Build the strict JSON-only prompt for one argument extraction task."""

    schema = model_json_schema(_ArgumentExtractionResponse)
    capability_accepts_sql = "query" in (
        set(manifest.required_arguments) | set(manifest.optional_arguments)
    )
    global_constraints = task.constraints.get("global_constraints", task.constraints)
    return "\n".join(
        [
            "You are extracting typed arguments for one already-selected capability.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not generate shell commands.",
            (
                "Do not generate raw SQL unless the capability explicitly accepts a SQL string argument."
                if not capability_accepts_sql
                else "This capability accepts a SQL string argument named query. Only include SQL if the prompt explicitly supports it."
            ),
            "Do not invent file paths, table names, database names, or external resources.",
            "If required information is missing, leave the argument absent and list it in missing_required_arguments.",
            *(
                [
                    "These arguments are already fixed by validated dataflow planning and must be preserved exactly:",
                    str(fixed_arguments),
                ]
                if fixed_arguments
                else ["No arguments are prefilled by validated dataflow planning."]
            ),
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
            "Global constraints:",
            str(global_constraints),
            "Selected capability manifest:",
            str(
                {
                    "capability_id": manifest.capability_id,
                    "operation_id": manifest.operation_id,
                    "domain": manifest.domain,
                    "description": manifest.description,
                    "argument_schema": manifest.argument_schema,
                    "required_arguments": manifest.required_arguments,
                    "optional_arguments": manifest.optional_arguments,
                    "examples": manifest.examples,
                }
            ),
            "JSON schema:",
            str(schema),
        ]
    )


def _normalize_path_phrase(value: str) -> str:
    """Normalize common current-directory phrases into the workspace root token."""

    normalized = str(value or "").strip().lower()
    if normalized in {"this folder", "current folder", "current directory", "here"}:
        return "."
    return str(value)


def _normalize_workspace_path(value: Any, workspace_root: Path) -> tuple[str | None, str | None]:
    """Normalize a path argument and reject traversal outside the workspace root."""

    if not isinstance(value, str) or not value.strip():
        return None, "Path value must be a non-empty string."

    raw_value = _normalize_path_phrase(value).strip()
    if raw_value == ".":
        return ".", None

    candidate = Path(raw_value)
    resolved = (workspace_root / candidate).resolve(strict=False) if not candidate.is_absolute() else candidate.resolve(strict=False)

    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError:
        return None, f"Path {value!r} resolves outside the workspace root."

    normalized = relative.as_posix()
    return normalized or ".", None


def _extract_limit_from_text(text: str) -> int | None:
    """Extract an explicit row limit from common prompt phrasings."""

    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\blimit\s+(\d+)\b",
    ]
    lowered = str(text or "").lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def _supports_argument(manifest: CapabilityManifest, argument_name: str) -> bool:
    """Return whether a capability manifest allows one argument name."""

    allowed = set(manifest.required_arguments) | set(manifest.optional_arguments)
    return argument_name in allowed


def _is_table_like_read(task: TaskFrame, manifest: CapabilityManifest) -> bool:
    """Return whether the task/capability pair represents a row-oriented read."""

    object_type = task.object_type.strip().lower()
    if manifest.capability_id == "sql.read_query":
        return any(token in object_type for token in {"table", "row", "dataset", "query", "records"})
    if not manifest.read_only:
        return False
    if task.semantic_verb not in {"read", "search", "analyze"}:
        return False
    manifest_objects = {value.strip().lower() for value in manifest.object_types}
    return bool({"table", "dataset", "records"} & manifest_objects) or any(
        token in object_type for token in {"table", "row", "dataset", "records"}
    )


def _apply_deterministic_normalization(
    task: TaskFrame,
    manifest: CapabilityManifest,
    response: _ArgumentExtractionResponse,
    workspace_root: Path,
    fixed_arguments: dict[str, Any] | None = None,
) -> ArgumentExtractionResult:
    """Normalize extracted arguments and compute deterministic missing fields."""

    allowed_arguments = set(manifest.required_arguments) | set(manifest.optional_arguments)
    arguments = {
        key: value for key, value in dict(response.arguments).items() if key in allowed_arguments
    }
    assumptions = list(response.assumptions)
    missing_required = set(response.missing_required_arguments)

    if _supports_argument(manifest, "path"):
        if "path" not in arguments:
            inferred_path = _extract_current_directory_path(task.description)
            if inferred_path is not None:
                arguments["path"] = inferred_path
        elif isinstance(arguments["path"], str):
            arguments["path"] = _normalize_path_phrase(arguments["path"])

    if _supports_argument(manifest, "limit") and "limit" not in arguments:
        explicit_limit = _extract_limit_from_text(task.description)
        if explicit_limit is not None:
            arguments["limit"] = explicit_limit
        elif _is_table_like_read(task, manifest):
            arguments["limit"] = 100

    for key in list(arguments):
        if key == "path" or key.endswith("_path"):
            normalized_path, error = _normalize_workspace_path(arguments[key], workspace_root)
            if error:
                arguments.pop(key, None)
                if key in manifest.required_arguments:
                    missing_required.add(key)
                assumptions.append(error)
            elif normalized_path is not None:
                arguments[key] = normalized_path

    for required_argument in manifest.required_arguments:
        if required_argument not in arguments:
            missing_required.add(required_argument)

    if fixed_arguments:
        arguments.update(fixed_arguments)
        for required_argument in fixed_arguments:
            missing_required.discard(required_argument)

    return ArgumentExtractionResult(
        task_id=task.id,
        capability_id=manifest.capability_id,
        operation_id=manifest.operation_id,
        arguments=arguments,
        missing_required_arguments=sorted(missing_required),
        assumptions=assumptions,
        confidence=response.confidence,
    )


def _extract_current_directory_path(description: str) -> str | None:
    """Return the workspace-root token when the task clearly refers to the current directory."""

    lowered = str(description or "").lower()
    markers = ("this folder", "current folder", "current directory", "here")
    if any(marker in lowered for marker in markers):
        return "."
    return None


def _fixed_arguments_for_task(
    task_id: str,
    dataflow_plan: ValidatedDataflowPlan | None,
) -> dict[str, InputRef]:
    """Return validated dataflow-prefilled InputRef arguments for one task."""

    if dataflow_plan is None:
        return {}
    fixed: dict[str, InputRef] = {}
    for ref in dataflow_plan.refs:
        if ref.consumer_task_id == task_id:
            fixed[ref.consumer_argument_name] = ref.input_ref
    return fixed


def extract_arguments(
    tasks: list[TaskFrame],
    selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
    llm_client,
    n_best: int = 3,
    trace: PlanningTrace | None = None,
    dataflow_plan: ValidatedDataflowPlan | None = None,
    capability_fit_decisions: list[CapabilityFitDecision] | None = None,
) -> list[ArgumentExtractionResult]:
    """Extract and normalize capability arguments for selected task frames."""

    task_by_id = {task.id: task for task in tasks}
    selection_by_task = {selection.task_id: selection for selection in selections}
    fit_by_task = {decision.task_id: decision for decision in (capability_fit_decisions or [])}
    workspace_root = Path.cwd().resolve()
    results: list[ArgumentExtractionResult] = []

    for task in tasks:
        fit_decision = fit_by_task.get(task.id)
        if fit_decision is not None and not fit_decision.is_fit:
            if isinstance(trace, PlanningTrace):
                append_trace_entry(
                    trace,
                    PlanningTraceEntry(
                        stage="argument_extraction_skipped",
                        request_id=str(trace.request_id),
                        prompt_template_id="argument_extraction",
                        selected_candidate={
                            "task_id": task.id,
                            "reason": fit_decision.status,
                            "capability_fit_decision": fit_decision.model_dump(mode="json"),
                        },
                        rejection_reasons=list(fit_decision.deterministic_rejections),
                    ),
                )
            continue
        selection = selection_by_task.get(task.id)
        if selection is None or selection.selected is None:
            results.append(
                ArgumentExtractionResult(
                    task_id=task.id,
                    capability_id="unknown",
                    operation_id="unknown",
                    arguments={},
                    missing_required_arguments=[],
                    assumptions=[
                        selection.unresolved_reason if selection is not None and selection.unresolved_reason else "No capability was selected for this task."
                    ],
                    confidence=0.0,
                )
            )
            continue

        try:
            capability = registry.get(selection.selected.capability_id)
        except CapabilityNotFoundError:
            results.append(
                ArgumentExtractionResult(
                    task_id=task.id,
                    capability_id=selection.selected.capability_id,
                    operation_id=selection.selected.operation_id,
                    arguments={},
                    missing_required_arguments=[],
                    assumptions=[f"Selected capability is not registered: {selection.selected.capability_id}."],
                    confidence=0.0,
                )
            )
            continue

        manifest = capability.manifest
        fixed_arguments = _fixed_arguments_for_task(task.id, dataflow_plan)
        prompt = _build_argument_prompt(task_by_id[task.id], manifest, fixed_arguments)
        attempts = collect_n_best_structured_attempts(
            llm_client=llm_client,
            system_prompt=prompt,
            user_payload={
                "task_id": task.id,
                "description": task.description,
                "constraints": task.constraints,
                "capability_id": manifest.capability_id,
                "operation_id": manifest.operation_id,
            },
            output_model=ArgumentExtractionProposal,
            n=n_best,
        )
        evaluations: list[CandidateEvaluation[ArgumentExtractionResult]] = []
        for attempt in attempts:
            evaluation = CandidateEvaluation[ArgumentExtractionResult](
                raw_response=attempt.raw_llm_response,
                validation_errors=list(attempt.validation_errors),
            )
            if attempt.parsed_proposal is None or attempt.validation_errors:
                evaluations.append(evaluation)
                continue
            proposal = attempt.parsed_proposal
            response = _ArgumentExtractionResponse.model_validate(proposal.model_dump(mode="json"))
            normalized = _apply_deterministic_normalization(
                task,
                manifest,
                response,
                workspace_root,
                fixed_arguments,
            )
            evaluation.proposal = normalized
            evaluation.confidence = normalized.confidence
            evaluation.missing_required_count = len(normalized.missing_required_arguments)
            evaluation.assumption_count = len(normalized.assumptions)
            evaluations.append(evaluation)
        selected = select_best_candidate(evaluations)
        if selected is None or selected.proposal is None:
            results.append(
                ArgumentExtractionResult(
                    task_id=task.id,
                    capability_id=manifest.capability_id,
                    operation_id=manifest.operation_id,
                    arguments={},
                    missing_required_arguments=list(manifest.required_arguments),
                    assumptions=["No valid argument extraction proposal was accepted."],
                    confidence=0.0,
                )
            )
            continue

        if isinstance(trace, PlanningTrace):
            model_name, temperature = llm_client_metadata(llm_client)
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="argument_extraction",
                    request_id=str(trace.request_id),
                    model_name=model_name,
                    llm_temperature=temperature,
                    prompt_template_id="argument_extraction",
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
