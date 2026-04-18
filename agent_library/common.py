from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel


class EventRequest(BaseModel):
    event: str
    payload: Dict


class EmittedEvent(BaseModel):
    event: str
    payload: Dict


class EventResponse(BaseModel):
    emits: List[EmittedEvent]


@dataclass(frozen=True)
class TaskPlanContext:
    step_task: str
    original_task: str
    target_agent: str
    previous_step_result: Dict[str, Any]
    prior_step_results: List[Dict[str, Any]]
    dependency_results: List[Dict[str, Any]]

    @property
    def structured_context(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"current_step": self.step_task}
        if self.original_task and self.original_task.strip() != self.step_task.strip():
            payload["original_task"] = self.original_task.strip()
        if self.previous_step_result:
            payload["previous_step_result"] = self.previous_step_result
        if self.prior_step_results:
            payload["prior_step_results"] = self.prior_step_results
        if self.dependency_results:
            payload["dependency_results"] = self.dependency_results
        return payload

    @property
    def classification_task(self) -> str:
        return self.step_task

    @property
    def execution_task(self) -> str:
        sections = [f"Current workflow step: {self.step_task.strip()}"]
        if self.original_task and self.original_task.strip() != self.step_task.strip():
            sections.extend(["Original user request:", self.original_task.strip()])
        context_payload = self.structured_context
        if context_payload:
            sections.extend(["Structured workflow context JSON:", _to_json(context_payload)])
        return "\n".join(sections)

    def targets(self, agent_name_or_family: str) -> bool:
        target = self.target_agent.strip()
        candidate = agent_name_or_family.strip()
        if not target or not candidate:
            return False
        return target == candidate or target.startswith(f"{candidate}_")


def task_plan_context(payload: Dict[str, Any]) -> TaskPlanContext:
    step_task = payload.get("task")
    original_task = payload.get("original_task")
    target_agent = payload.get("target_agent")
    previous_step_result = payload.get("previous_step_result")
    prior_step_results = payload.get("prior_step_results")
    dependency_results = payload.get("dependency_results")
    return TaskPlanContext(
        step_task=step_task.strip() if isinstance(step_task, str) else "",
        original_task=original_task.strip() if isinstance(original_task, str) else "",
        target_agent=target_agent.strip() if isinstance(target_agent, str) else "",
        previous_step_result=previous_step_result if isinstance(previous_step_result, dict) else {},
        prior_step_results=[item for item in prior_step_results if isinstance(item, dict)] if isinstance(prior_step_results, list) else [],
        dependency_results=[item for item in dependency_results if isinstance(item, dict)] if isinstance(dependency_results, list) else [],
    )


def _to_json(value: Any) -> str:
    return json_dumps(value)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, indent=2, default=str)
