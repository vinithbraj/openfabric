from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    ok: bool = True
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    status: Literal["continue", "complete", "blocked", "failed"] = "complete"
    summary: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    reasoning: str = ""


class AgentRunResult(BaseModel):
    agent_name: str
    status: Literal["completed", "failed", "blocked"] = "completed"
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    tool_results: list[ToolResult] = Field(default_factory=list)
    iterations: int = 0
    score: float | None = None
    error: str | None = None
    raw_actions: list[dict[str, Any]] = Field(default_factory=list)


class RouterDecision(BaseModel):
    selected: str
    rationale: str = ""
    confidence: float | None = None


class RunEvent(BaseModel):
    run_id: str
    node: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class RunSummary(BaseModel):
    run_id: str
    spec_name: str
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
