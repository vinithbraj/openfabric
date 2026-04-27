from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionStep(BaseModel):
    id: int
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    input: list[str] = Field(default_factory=list)
    output: str | None = None


class ExecutionPlan(BaseModel):
    steps: list[ExecutionStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_steps(self) -> "ExecutionPlan":
        if not self.steps:
            raise ValueError("Execution plan requires at least one step.")
        expected_id = 1
        seen_ids: set[int] = set()
        for step in self.steps:
            if step.id in seen_ids:
                raise ValueError(f"Duplicate step id {step.id}.")
            seen_ids.add(step.id)
            if step.id != expected_id:
                raise ValueError("Step ids must be sequential starting at 1.")
            expected_id += 1
        return self


class HighLevelPlan(BaseModel):
    tasks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tasks(self) -> "HighLevelPlan":
        if not self.tasks:
            raise ValueError("High-level plan requires at least one task.")
        normalized: list[str] = []
        for task in self.tasks:
            text = str(task).strip()
            if not text:
                raise ValueError("High-level plan tasks must be non-empty.")
            normalized.append(text)
        self.tasks = normalized
        return self


class ToolSpec(BaseModel):
    name: str
    description: str
    arguments_schema: dict[str, Any]


class StepLog(BaseModel):
    step: ExecutionStep
    result: dict[str, Any] = Field(default_factory=dict)
    success: bool
    error: str | None = None
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str = Field(default_factory=utc_now_iso)


class ValidationResult(BaseModel):
    success: bool
    reason: str | None = None


class PlannerConfig(BaseModel):
    model: str | None = None
    prompt: str | None = None
    decomposer_prompt: str | None = None
    temperature: float = 0.0


class RuntimePolicy(BaseModel):
    max_retries: int = 2


class FinalOutput(BaseModel):
    content: str = ""
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    llm_calls: int = 0
    llm_intent_calls: int = 0
    raw_planner_llm_calls: int = 0
    latency_ms: float = 0.0
    steps_executed: int = 0
    retries: int = 0


AgentTriggerType = Literal["manual", "timer", "external"]


class AgentSession(BaseModel):
    id: str
    spec_name: str
    spec_path: str
    goal: str
    input: dict[str, Any] = Field(default_factory=dict)
    compiled_spec: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    status: str
    current_trigger: AgentTriggerType = "manual"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


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


class RuntimeStatus(BaseModel):
    status: Literal["planning", "executing", "validating", "retrying", "completed", "failed"]
    detail: str = ""


PlanStep = ExecutionStep
