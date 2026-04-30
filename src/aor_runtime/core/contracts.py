"""OpenFABRIC Runtime Module: aor_runtime.core.contracts

Purpose:
    Define shared execution-plan, step-log, and runtime contract models.

Responsibilities:
    Define stable data structures and helpers that keep planning, execution, validation, and persistence aligned.

Data flow / Interfaces:
    Exports dataclasses, models, and utility functions consumed by runtime, tools, API, and tests.

Boundaries:
    Keeps shared primitives dependency-light and free of domain-specific execution policy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    """Utc now iso for the surrounding runtime workflow.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.contracts.utc_now_iso.
    """
    return datetime.now(timezone.utc).isoformat()


class ExecutionStep(BaseModel):
    """Represent execution step within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ExecutionStep.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.ExecutionStep and related tests.
    """
    id: int
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    input: list[str] = Field(default_factory=list)
    output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def separate_internal_metadata(self) -> "ExecutionStep":
        """Move runtime-internal semantic keys out of executable tool args.

        Inputs:
            Uses this step's args and metadata fields after normal model validation.

        Returns:
            The same step with sanitized args and runtime-only metadata preserved.

        Used by:
            ExecutionPlan creation, dataflow resolution, and tests that load old plans with legacy internal args.
        """
        if not any(str(key).startswith("__semantic_") for key in self.args):
            return self
        clean_args = dict(self.args)
        clean_metadata = dict(self.metadata)
        projection = clean_args.pop("__semantic_projection", None)
        for key in list(clean_args):
            if str(key).startswith("__semantic_"):
                clean_args.pop(key, None)
        if projection is not None and "semantic_projection" not in clean_metadata:
            clean_metadata["semantic_projection"] = projection
        self.args = clean_args
        self.metadata = clean_metadata
        return self


class ExecutionPlan(BaseModel):
    """Represent execution plan within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ExecutionPlan.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.ExecutionPlan and related tests.
    """
    steps: list[ExecutionStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_steps(self) -> "ExecutionPlan":
        """Validate validate steps invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by shared runtime contracts through ExecutionPlan.validate_steps calls and related tests.
        """
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
    """Represent high level plan within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by HighLevelPlan.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.HighLevelPlan and related tests.
    """
    tasks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tasks(self) -> "HighLevelPlan":
        """Validate validate tasks invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by shared runtime contracts through HighLevelPlan.validate_tasks calls and related tests.
        """
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
    """Represent tool spec within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolSpec.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.ToolSpec and related tests.
    """
    name: str
    description: str
    arguments_schema: dict[str, Any]


class StepLog(BaseModel):
    """Represent step log within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by StepLog.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.StepLog and related tests.
    """
    step: ExecutionStep
    result: dict[str, Any] = Field(default_factory=dict)
    success: bool
    error: str | None = None
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str = Field(default_factory=utc_now_iso)


class ValidationResult(BaseModel):
    """Represent validation result within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ValidationResult.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.ValidationResult and related tests.
    """
    success: bool
    reason: str | None = None


class PlannerConfig(BaseModel):
    """Represent planner config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlannerConfig.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.PlannerConfig and related tests.
    """
    model: str | None = None
    prompt: str | None = None
    decomposer_prompt: str | None = None
    temperature: float = 0.0


class RuntimePolicy(BaseModel):
    """Represent runtime policy within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimePolicy.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.RuntimePolicy and related tests.
    """
    max_retries: int = 2


class FinalOutput(BaseModel):
    """Represent final output within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FinalOutput.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.FinalOutput and related tests.
    """
    content: str = ""
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """Represent run metrics within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RunMetrics.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.RunMetrics and related tests.
    """
    llm_calls: int = 0
    llm_intent_calls: int = 0
    raw_planner_llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    latency_ms: float = 0.0
    steps_executed: int = 0
    retries: int = 0


AgentTriggerType = Literal["manual", "timer", "external"]


class AgentSession(BaseModel):
    """Represent agent session within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by AgentSession.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.AgentSession and related tests.
    """
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
    """Represent run event within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RunEvent.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.RunEvent and related tests.
    """
    run_id: str
    node: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class RunSummary(BaseModel):
    """Represent run summary within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RunSummary.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.RunSummary and related tests.
    """
    run_id: str
    spec_name: str
    status: str
    input: dict[str, Any] = Field(default_factory=dict)
    final_state: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class RuntimeStatus(BaseModel):
    """Represent runtime status within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeStatus.

    Data flow / Interfaces:
        Instances are created and consumed by shared runtime contracts code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.core.contracts.RuntimeStatus and related tests.
    """
    status: Literal["planning", "executing", "validating", "retrying", "completed", "failed"]
    detail: str = ""


PlanStep = ExecutionStep
