"""Prompt classification and typed task decomposition interfaces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import model_json_schema

from agent_runtime.core.types import TaskFrame, UserRequest
from agent_runtime.llm.structured_call import structured_call


class PromptClassification(BaseModel):
    """Typed prompt classification produced by the first input-pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    prompt_type: Literal[
        "simple_question",
        "simple_tool_task",
        "compound_tool_task",
        "complex_workflow",
        "ambiguous",
        "unsupported",
    ]
    requires_tools: bool
    likely_domains: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"]
    needs_clarification: bool
    clarification_question: str | None = None
    reason: str


class DecompositionResult(BaseModel):
    """Typed decomposition output for a user prompt."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskFrame] = Field(default_factory=list)
    global_constraints: dict[str, Any] = Field(default_factory=dict)
    unresolved_references: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> "DecompositionResult":
        """Ensure task dependencies point at task ids within the same result."""

        task_ids = {task.id for task in self.tasks}
        for task in self.tasks:
            for dependency in task.dependencies:
                if dependency not in task_ids:
                    raise ValueError(
                        f"Task {task.id} depends on unknown task id {dependency}."
                    )
        return self


class Decomposer:
    """Minimal decomposer that keeps a prompt as one task."""

    def decompose(self, request: UserRequest) -> DecompositionResult:
        """Return the request prompt as a single undecomposed task."""

        return DecompositionResult(
            tasks=[
                TaskFrame(
                    description=request.raw_prompt,
                    semantic_verb="unknown",
                    object_type="unknown",
                    intent_confidence=0.0,
                    constraints={},
                    raw_evidence=request.raw_prompt,
                )
            ]
        )


def _classification_schema_description() -> str:
    """Return a short human-readable summary of the classification schema."""

    return (
        "PromptClassification fields: "
        "prompt_type, requires_tools, likely_domains, risk_level, "
        "needs_clarification, clarification_question, reason."
    )


def _decomposition_schema_description() -> str:
    """Return a short human-readable summary of the decomposition schema."""

    return (
        "DecompositionResult fields: tasks, global_constraints, "
        "unresolved_references, assumptions. "
        "Each task is a TaskFrame with id, description, semantic_verb, "
        "object_type, intent_confidence, constraints, dependencies, "
        "raw_evidence, requires_confirmation, risk_level."
    )


def _build_classification_prompt(user_request: UserRequest) -> str:
    """Build the strict JSON-only prompt sent to the LLM classifier."""

    schema = model_json_schema(PromptClassification)
    return "\n".join(
        [
            "You are classifying a user prompt for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "Only classify the prompt. Do not solve it.",
            _classification_schema_description(),
            "JSON schema:",
            str(schema),
            "User prompt:",
            user_request.raw_prompt,
        ]
    )


def classify_prompt(user_request: UserRequest, llm_client) -> PromptClassification:
    """Classify a user prompt through a strict structured LLM call."""

    prompt = _build_classification_prompt(user_request)
    return structured_call(llm_client, prompt, PromptClassification)


def _build_decomposition_prompt(
    user_request: UserRequest,
    classification: PromptClassification,
) -> str:
    """Build the strict JSON-only prompt sent to the LLM decomposer."""

    schema = model_json_schema(DecompositionResult)
    return "\n".join(
        [
            "You are decomposing a user prompt into atomic user-meaningful tasks.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not select tools.",
            "Do not generate commands.",
            "Do not generate SQL.",
            "Do not generate shell syntax.",
            "Break compound prompts into atomic user-meaningful tasks.",
            "Preserve ordering and dependencies.",
            "Extract global constraints such as limits, date ranges, output preferences, paths, table names, row limits, and safety constraints.",
            "Use task dependencies to express ordering.",
            "Only describe meaning. Do not solve the task.",
            _decomposition_schema_description(),
            "Prompt classification context:",
            str(classification.model_dump()),
            "JSON schema:",
            str(schema),
            "User prompt:",
            user_request.raw_prompt,
        ]
    )


def decompose_prompt(
    user_request: UserRequest,
    classification: PromptClassification,
    llm_client,
) -> DecompositionResult:
    """Decompose a prompt into ordered atomic tasks through a structured LLM call."""

    prompt = _build_decomposition_prompt(user_request, classification)
    return structured_call(llm_client, prompt, DecompositionResult)
