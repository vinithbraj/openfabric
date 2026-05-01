"""Typed repair boundary for future LLM-assisted correction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RepairRequest(BaseModel):
    """Compact repair input that excludes raw operational data."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    error_code: str
    message: str


class RepairDecision(BaseModel):
    """Placeholder repair decision emitted by a future typed repair call."""

    model_config = ConfigDict(extra="forbid")

    can_repair: bool = False
    notes: list[str] = Field(default_factory=list)
