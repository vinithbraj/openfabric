"""Pydantic schemas for capability manifests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CapabilityManifest(BaseModel):
    """Declarative capability metadata exposed to the runtime and the LLM."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str
    domain: str
    operation_id: str
    name: str
    description: str
    semantic_verbs: list[str] = Field(default_factory=list)
    object_types: list[str] = Field(default_factory=list)
    argument_schema: dict[str, Any] = Field(default_factory=dict)
    required_arguments: list[str] = Field(default_factory=list)
    optional_arguments: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    execution_backend: Literal["gateway", "local", "internal"] = "local"
    backend_operation: str | None = None
    risk_level: Literal["low", "medium", "high", "critical"]
    read_only: bool
    mutates_state: bool
    requires_confirmation: bool
    examples: list[dict[str, Any]] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
