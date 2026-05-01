"""Semantic verb assignment for decomposed task frames."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import model_json_schema

from agent_runtime.core.types import RiskLevel, SemanticVerb, TaskFrame
from agent_runtime.llm.structured_call import structured_call

VERB_VOCABULARY: tuple[str, ...] = (
    "read",
    "search",
    "create",
    "update",
    "delete",
    "transform",
    "analyze",
    "summarize",
    "compare",
    "execute",
    "render",
    "unknown",
)


class TaskVerbAssignment(BaseModel):
    """Typed LLM response for one task-frame verb assignment."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    semantic_verb: SemanticVerb
    object_type: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    requires_confirmation: bool = False


class TaskVerbAssignmentResult(BaseModel):
    """Collection of verb assignments returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[TaskVerbAssignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_task_ids(self) -> "TaskVerbAssignmentResult":
        """Ensure each task id appears at most once in the response."""

        task_ids = [assignment.task_id for assignment in self.assignments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task verb assignments must reference unique task ids.")
        return self


class VerbClassifier:
    """Small deterministic fallback used by the placeholder orchestrator."""

    def classify(self, prompt: str) -> str:
        """Return a conservative verb guess without calling an LLM."""

        lowered = prompt.strip().lower()
        if lowered.startswith("list"):
            return "read"
        if lowered.startswith("count"):
            return "analyze"
        if lowered.startswith("search"):
            return "search"
        if lowered.startswith("delete") or lowered.startswith("remove"):
            return "delete"
        if lowered.startswith("run"):
            return "execute"
        if lowered.startswith("summarize"):
            return "summarize"
        return "read"


def _build_assignment_prompt(tasks: list[TaskFrame]) -> str:
    """Build the strict JSON-only prompt for semantic verb assignment."""

    schema = model_json_schema(TaskVerbAssignmentResult)
    task_lines = [
        f"- task_id={task.id}; description={task.description}; current_constraints={task.constraints}"
        for task in tasks
    ]
    return "\n".join(
        [
            "You are assigning semantic verbs to existing task frames.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not generate commands, shell syntax, SQL, code, or executable plans.",
            "Assign exactly one semantic verb from this controlled vocabulary:",
            ", ".join(VERB_VOCABULARY),
            "Also assign object_type, intent_confidence, risk_level, and requires_confirmation.",
            "Only classify task meaning. Do not choose tools or implementation details.",
            "Task list:",
            *task_lines,
            "JSON schema:",
            str(schema),
        ]
    )


def _post_process_assignment(task: TaskFrame) -> TaskFrame:
    """Apply deterministic risk and confirmation rules after LLM assignment."""

    dry_run = bool(task.constraints.get("dry_run"))
    elevated_risk = bool(task.constraints.get("elevated_risk"))
    low_risk_verbs = {"read", "search", "analyze", "summarize", "render"}
    if task.semantic_verb == "delete" and not dry_run:
        task.requires_confirmation = True
    if task.semantic_verb in low_risk_verbs and not elevated_risk:
        task.risk_level = "low"
        task.requires_confirmation = False
    if task.semantic_verb in {"create", "update"} and task.risk_level in {"high", "critical"}:
        task.requires_confirmation = True
    if task.semantic_verb == "execute" and task.risk_level == "low":
        task.risk_level = "medium"
    if task.semantic_verb == "execute" and bool(task.constraints.get("read_only_capability")):
        task.risk_level = "low"
    return task


def assign_semantic_verbs(tasks: list[TaskFrame], llm_client) -> list[TaskFrame]:
    """Assign semantic verbs and semantic metadata to task frames through an LLM."""

    if not tasks:
        return []

    prompt = _build_assignment_prompt(tasks)
    assignment_result = structured_call(llm_client, prompt, TaskVerbAssignmentResult)
    assignments = {assignment.task_id: assignment for assignment in assignment_result.assignments}

    updated_tasks: list[TaskFrame] = []
    for task in tasks:
        assignment = assignments.get(task.id)
        if assignment is None:
            raise ValueError(f"Missing semantic verb assignment for task {task.id}.")
        updated_task = task.model_copy(
            update={
                "semantic_verb": assignment.semantic_verb,
                "object_type": assignment.object_type,
                "intent_confidence": assignment.intent_confidence,
                "risk_level": assignment.risk_level,
                "requires_confirmation": assignment.requires_confirmation,
            }
        )
        updated_tasks.append(_post_process_assignment(updated_task))
    return updated_tasks
