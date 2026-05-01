"""Tracing and hashing utilities for planning reproducibility."""

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
    validation_errors: list[str] = Field(default_factory=list)
    selected_candidate: Any = None
    rejection_reasons: list[str] = Field(default_factory=list)
    deterministic_normalizations: list[str] = Field(default_factory=list)
    safety_decision: dict[str, Any] | None = None
    final_dag_hash: str | None = None


class PlanningTrace(BaseModel):
    """Replayable trace of LLM proposals and deterministic runtime decisions."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    entries: list[PlanningTraceEntry] = Field(default_factory=list)
    capability_manifest_version: str | None = None
    schema_version: str = "v1"
    validated_dag: dict[str, Any] | None = None
    final_dag_hash: str | None = None
    safety_decision: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _stable_json(value: Any) -> str:
    """Serialize a structure deterministically for hashing and persistence."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_prompt_payload(payload: Any) -> str:
    """Return a stable hash for a prompt payload."""

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def hash_capability_manifest(manifest: Any) -> str:
    """Return a stable hash for the compact capability manifest."""

    return hashlib.sha256(_stable_json(manifest).encode("utf-8")).hexdigest()


def hash_action_dag(dag: Any) -> str:
    """Return a stable hash for a trusted action DAG."""

    if hasattr(dag, "model_dump"):
        payload = dag.model_dump(mode="json")
    else:
        payload = dag
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("final_dag_hash", None)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def save_planning_trace(trace: PlanningTrace, path: str | Path) -> Path:
    """Persist a planning trace as JSON."""

    target = Path(path)
    target.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_planning_trace(path: str | Path) -> PlanningTrace:
    """Load a planning trace from JSON."""

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


def append_trace_entry(trace: PlanningTrace, entry: PlanningTraceEntry) -> None:
    """Append one structured entry to a planning trace."""

    trace.entries.append(entry)
