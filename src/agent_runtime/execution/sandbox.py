"""Sandbox placeholder for future isolated execution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SandboxContext(BaseModel):
    """Describes the execution sandbox chosen for a capability call."""

    model_config = ConfigDict(extra="forbid")

    name: str = "in_process_placeholder"
    isolated: bool = False
