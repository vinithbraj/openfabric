from __future__ import annotations

from typing import Any

from aor_runtime.core.contracts import HighLevelPlan, PlannerConfig
from aor_runtime.core.utils import dumps_json, extract_json_object
from aor_runtime.llm.client import LLMClient


DEFAULT_DECOMPOSER_PROMPT = """You are the decomposition stage for a deterministic local agent runtime.

Your job is to break the user's goal into an ordered list of high-level tasks.
Output JSON only. The response must be a single JSON object that validates against the HighLevelPlan schema:
{
  "tasks": [
    "task 1",
    "task 2"
  ]
}

Rules:
- Each task must be meaningful, actionable, and ordered.
- Do not generate executable tool steps yet.
- Do not mention tools, code, or implementation details unless the goal explicitly requires them.
- Keep the task list as short as possible while still covering the full goal.
- Do not skip important prerequisites or verification work when they matter to the request.
- Do not output anything except the JSON object.
"""


def is_complex_goal(goal: str) -> bool:
    normalized_goal = str(goal or "").lower()
    return (
        len(normalized_goal.split()) > 12
        or " and " in normalized_goal
        or "then" in normalized_goal
        or "after" in normalized_goal
    )


class GoalDecomposer:
    def __init__(self, *, llm: LLMClient) -> None:
        self.llm = llm
        self.last_raw_output: str | None = None

    def decompose_goal(
        self,
        *,
        goal: str,
        planner: PlannerConfig,
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None = None,
    ) -> HighLevelPlan:
        system_prompt = self.llm.load_prompt(planner.decomposer_prompt, DEFAULT_DECOMPOSER_PROMPT)
        user_prompt = dumps_json(
            {
                "goal": goal,
                "input": input_payload,
                "failure_context": failure_context or {},
            },
            indent=2,
        )
        raw_output = self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=planner.model,
            temperature=planner.temperature,
        )
        self.last_raw_output = raw_output
        payload = extract_json_object(raw_output)
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object response from model")
        return HighLevelPlan.model_validate(payload)
