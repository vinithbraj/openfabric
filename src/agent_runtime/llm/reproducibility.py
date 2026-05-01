"""Tracing, hashing, and replay utilities for planning reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlanningTraceEntry(BaseModel):
    """One LLM/runtime decision record within a planning trace."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    request_id: str
    model_name: str = "unknown"
    prompt_template_id: str
    prompt_template_version: str = "v1"
    capability_manifest_version: str | None = None
    schema_version: str = "v1"
    llm_temperature: float | None = None
    raw_llm_response: Any = None
    parsed_proposal: Any = None
    llm_diagnostics: Any = None
    validation_errors: list[str] = Field(default_factory=list)
    selected_candidate: Any = None
    rejection_reasons: list[str] = Field(default_factory=list)
    deterministic_normalizations: list[str] = Field(default_factory=list)
    safety_decision: dict[str, Any] | None = None
    final_dag_hash: str | None = None


class PlanningTrace(BaseModel):
    """Replayable structured record of what the runtime planned and accepted."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    raw_prompt: str = ""
    model_name: str = "unknown"
    prompt_template_versions: dict[str, str] = Field(default_factory=dict)
    capability_manifest_hash: str | None = None
    capability_manifest_version: str | None = None
    schema_version: str = "v1"
    llm_temperature: float | None = None

    prompt_classification_raw: Any = None
    prompt_classification_validated: dict[str, Any] | None = None

    decomposition_raw: Any = None
    decomposition_validated: dict[str, Any] | None = None

    verb_assignment_raw: Any = None
    verb_assignment_validated: dict[str, Any] | None = None

    capability_candidates_by_task: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    capability_shortlist_evaluations_by_task: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    capability_fit_proposals_by_task: dict[str, Any] = Field(default_factory=dict)
    capability_fit_decisions_by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)
    capability_fit_diagnostics_by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selected_capability_by_task: dict[str, dict[str, Any] | None] = Field(default_factory=dict)
    rejected_capabilities_by_task: dict[str, list[str]] = Field(default_factory=dict)
    capability_gaps_by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)

    dataflow_plan_raw: Any = None
    dataflow_plan_validated: dict[str, Any] | None = None
    dataflow_rejected_refs: list[dict[str, Any]] = Field(default_factory=list)

    argument_extraction_raw_by_task: dict[str, Any] = Field(default_factory=dict)
    argument_extraction_validated_by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)
    skipped_argument_extraction_by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)

    dag_raw: dict[str, Any] | None = None
    dag_validated: dict[str, Any] | None = None
    validated_dag: dict[str, Any] | None = None
    final_dag_hash: str | None = None

    safety_decision: dict[str, Any] | None = None
    execution_ready: bool = False

    deterministic_normalizations: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    user_facing_errors: list[str] = Field(default_factory=list)

    entries: list[PlanningTraceEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _stable_json(value: Any) -> str:
    """Serialize one structure deterministically for hashing and persistence."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_prompt_payload(payload: Any) -> str:
    """Return a stable hash for one prompt payload."""

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def hash_capability_manifest(manifest: Any) -> str:
    """Return a stable hash for one capability manifest payload."""

    return hashlib.sha256(_stable_json(manifest).encode("utf-8")).hexdigest()


def hash_action_dag(dag: Any) -> str:
    """Return a stable hash for one trusted action DAG payload."""

    if hasattr(dag, "model_dump"):
        payload = dag.model_dump(mode="json")
    else:
        payload = dag
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("final_dag_hash", None)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def save_planning_trace(trace: PlanningTrace, path: str | Path) -> Path:
    """Persist one planning trace as JSON."""

    target = Path(path)
    target.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_planning_trace(path: str | Path) -> PlanningTrace:
    """Load one planning trace from JSON."""

    return PlanningTrace.model_validate_json(Path(path).read_text(encoding="utf-8"))


def llm_client_metadata(llm_client) -> tuple[str, float | None]:
    """Extract model metadata from an LLM client when available."""

    model_name = str(getattr(llm_client, "model", "unknown") or "unknown")
    temperature = getattr(llm_client, "temperature", None)
    try:
        parsed_temperature = None if temperature is None else float(temperature)
    except (TypeError, ValueError):
        parsed_temperature = None
    return model_name, parsed_temperature


def _extend_unique(target: list[str], values: list[str]) -> None:
    """Append strings to a list without duplicating existing entries."""

    for value in values:
        if value not in target:
            target.append(value)


def _set_if_task(trace_mapping: dict[str, Any], task_id: str | None, value: Any) -> None:
    """Set one task-scoped trace mapping when the task id exists."""

    if task_id:
        trace_mapping[str(task_id)] = value


def append_trace_entry(trace: PlanningTrace, entry: PlanningTraceEntry) -> None:
    """Append one structured entry and project it into high-level trace fields."""

    trace.entries.append(entry)
    trace.prompt_template_versions[entry.prompt_template_id] = entry.prompt_template_version
    if entry.capability_manifest_version:
        trace.capability_manifest_hash = entry.capability_manifest_version
        trace.capability_manifest_version = entry.capability_manifest_version
    if entry.model_name and entry.model_name != "unknown":
        trace.model_name = entry.model_name
    if entry.llm_temperature is not None:
        trace.llm_temperature = entry.llm_temperature
    _extend_unique(trace.deterministic_normalizations, list(entry.deterministic_normalizations))
    _extend_unique(trace.validation_errors, list(entry.validation_errors))

    stage = entry.stage
    selected = entry.selected_candidate
    parsed = entry.parsed_proposal

    if stage == "prompt_classification":
        trace.prompt_classification_raw = entry.raw_llm_response
        if isinstance(selected, dict):
            trace.prompt_classification_validated = dict(selected)
        return

    if stage == "task_decomposition":
        trace.decomposition_raw = entry.raw_llm_response
        if isinstance(selected, dict):
            trace.decomposition_validated = dict(selected)
        return

    if stage == "task_decomposition_critique":
        critiques = trace.metadata.setdefault("decomposition_critiques", [])
        critiques.append(
            {
                "raw": entry.raw_llm_response,
                "parsed": entry.parsed_proposal,
                "selected_candidate": entry.selected_candidate,
            }
        )
        return

    if stage == "semantic_verb_assignment":
        trace.verb_assignment_raw = entry.raw_llm_response
        if isinstance(selected, dict):
            trace.verb_assignment_validated = dict(selected)
        return

    if stage == "capability_selection":
        task_id = None
        if isinstance(selected, dict):
            task_id = selected.get("task_id")
            candidates = selected.get("candidates")
            if isinstance(candidates, list):
                _set_if_task(trace.capability_candidates_by_task, str(task_id) if task_id else None, candidates)
            _set_if_task(
                trace.selected_capability_by_task,
                str(task_id) if task_id else None,
                selected.get("selected"),
            )
            if entry.rejection_reasons:
                _set_if_task(
                    trace.rejected_capabilities_by_task,
                    str(task_id) if task_id else None,
                    list(entry.rejection_reasons),
                )
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and item.get("task_id") is not None:
                    evaluations = item.get("evaluations")
                    if isinstance(evaluations, list):
                        _set_if_task(
                            trace.capability_shortlist_evaluations_by_task,
                            str(item.get("task_id")),
                            evaluations,
                        )
        elif isinstance(parsed, dict) and parsed.get("task_id") is not None:
            evaluations = parsed.get("evaluations")
            if isinstance(evaluations, list):
                _set_if_task(
                    trace.capability_shortlist_evaluations_by_task,
                    str(parsed.get("task_id")),
                    evaluations,
                )
        return

    if stage == "capability_fit":
        task_id = None
        if isinstance(selected, dict):
            task_id = selected.get("task_id")
            _set_if_task(
                trace.capability_fit_decisions_by_task,
                str(task_id) if task_id else None,
                dict(selected),
            )
            if selected.get("status") != "fit":
                gap_payload = {
                    "task_id": task_id,
                    "status": selected.get("status"),
                    "missing_capability_description": selected.get("missing_capability_description"),
                    "suggested_domain": selected.get("suggested_domain"),
                    "suggested_object_type": selected.get("suggested_object_type"),
                    "clarification_question": selected.get("clarification_question"),
                }
                _set_if_task(trace.capability_gaps_by_task, str(task_id) if task_id else None, gap_payload)
            diagnostics = selected.get("llm_diagnostics")
            if isinstance(diagnostics, dict):
                _set_if_task(
                    trace.capability_fit_diagnostics_by_task,
                    str(task_id) if task_id else None,
                    diagnostics,
                )
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    _set_if_task(
                        trace.capability_fit_proposals_by_task,
                        str(item.get("task_id")) if item.get("task_id") is not None else None,
                        item,
                    )
        elif isinstance(parsed, dict):
            _set_if_task(
                trace.capability_fit_proposals_by_task,
                str(parsed.get("task_id")) if parsed.get("task_id") is not None else None,
                parsed,
            )
        if task_id and isinstance(entry.llm_diagnostics, dict):
            _set_if_task(
                trace.capability_fit_diagnostics_by_task,
                str(task_id),
                entry.llm_diagnostics,
            )
        if task_id and entry.rejection_reasons:
            existing = trace.rejected_capabilities_by_task.get(str(task_id), [])
            trace.rejected_capabilities_by_task[str(task_id)] = [*existing, *entry.rejection_reasons]
        return

    if stage == "dataflow_planning":
        trace.dataflow_plan_raw = entry.raw_llm_response
        if isinstance(selected, dict):
            trace.dataflow_plan_validated = dict(selected)
            rejected = selected.get("rejected_refs")
            if isinstance(rejected, list):
                trace.dataflow_rejected_refs = [dict(item) for item in rejected if isinstance(item, dict)]
        return

    if stage == "argument_extraction":
        task_id = None
        if isinstance(selected, dict):
            task_id = selected.get("task_id")
            _set_if_task(
                trace.argument_extraction_validated_by_task,
                str(task_id) if task_id else None,
                dict(selected),
            )
        if isinstance(entry.raw_llm_response, list):
            for raw_item in entry.raw_llm_response:
                if isinstance(raw_item, dict) and raw_item.get("task_id") is not None:
                    _set_if_task(
                        trace.argument_extraction_raw_by_task,
                        str(raw_item.get("task_id")),
                        raw_item,
                    )
        elif isinstance(entry.raw_llm_response, dict):
            _set_if_task(
                trace.argument_extraction_raw_by_task,
                str(entry.raw_llm_response.get("task_id")) if entry.raw_llm_response.get("task_id") is not None else None,
                entry.raw_llm_response,
            )
        return

    if stage == "argument_extraction_skipped":
        if isinstance(selected, dict):
            _set_if_task(
                trace.skipped_argument_extraction_by_task,
                str(selected.get("task_id")) if selected.get("task_id") is not None else None,
                dict(selected),
            )
        return

    if stage == "dag_construction":
        if isinstance(entry.raw_llm_response, dict):
            trace.dag_raw = dict(entry.raw_llm_response)
        if isinstance(selected, dict):
            trace.dag_validated = dict(selected)
            trace.validated_dag = dict(selected)
        if entry.final_dag_hash:
            trace.final_dag_hash = entry.final_dag_hash
        return

    if stage == "dag_review":
        reviews = trace.metadata.setdefault("dag_reviews", [])
        reviews.append(
            {
                "raw": entry.raw_llm_response,
                "parsed": entry.parsed_proposal,
                "selected_candidate": entry.selected_candidate,
                "rejection_reasons": list(entry.rejection_reasons),
            }
        )
        return

    if stage == "safety_evaluation":
        trace.safety_decision = dict(entry.safety_decision or {})
        trace.final_dag_hash = entry.final_dag_hash or trace.final_dag_hash
        trace.execution_ready = bool((entry.safety_decision or {}).get("allowed", False))
        return

    if stage == "failure_repair":
        repairs = trace.metadata.setdefault("failure_repairs", [])
        repairs.append(
            {
                "raw": entry.raw_llm_response,
                "parsed": entry.parsed_proposal,
                "selected_candidate": entry.selected_candidate,
                "rejection_reasons": list(entry.rejection_reasons),
            }
        )
        return


def replay_from_validated_dag(
    trace: PlanningTrace,
    execution_engine,
    context: dict[str, Any] | None = None,
) -> Any:
    """Replay one validated DAG through the execution engine without any LLM calls."""

    from agent_runtime.core.errors import ValidationError
    from agent_runtime.core.types import ActionDAG

    dag_payload = trace.dag_validated or trace.validated_dag
    if dag_payload is None:
        raise ValidationError("PlanningTrace does not contain a validated DAG for replay.")

    dag = ActionDAG.model_validate(dag_payload)
    replay_hash = hash_action_dag(dag)
    if not trace.final_dag_hash or replay_hash != trace.final_dag_hash:
        raise ValidationError("Validated DAG hash does not match the recorded final_dag_hash.")

    replay_context = dict(context or {})
    trusted_replay = bool(replay_context.pop("trusted_replay", False))
    if trusted_replay:
        dag = dag.model_copy(
            update={
                "execution_ready": True,
                "final_dag_hash": trace.final_dag_hash,
            }
        )
    return execution_engine.execute(dag, replay_context)


__all__ = [
    "PlanningTrace",
    "PlanningTraceEntry",
    "append_trace_entry",
    "hash_action_dag",
    "hash_capability_manifest",
    "hash_prompt_payload",
    "llm_client_metadata",
    "load_planning_trace",
    "replay_from_validated_dag",
    "save_planning_trace",
]
