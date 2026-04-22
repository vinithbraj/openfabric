from __future__ import annotations

from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import AgentAction, AgentRunResult, ToolCall, ToolResult
from aor_runtime.core.utils import dumps_json
from aor_runtime.dsl.models import AgentDefinition
from aor_runtime.llm.client import LLMClient
from aor_runtime.tools.base import ToolRegistry


DEFAULT_AGENT_PROMPT = """You are an execution agent inside a local multi-agent runtime.

You receive a task, shared run state, previous tool results, and a list of tools you may use.
Respond with a JSON object:
{
  "status": "continue|complete|blocked|failed",
  "summary": "short summary",
  "tool_calls": [{"tool":"tool.name","arguments":{}}],
  "output": {"content":"optional final answer","artifacts":[]},
  "score": 0.0,
  "reasoning": "short private summary"
}

Rules:
- Use tools when needed.
- Keep tool calls concrete and minimal.
- If you already have enough information, set status to complete.
- If you are blocked, explain why.
- Output must be valid JSON.
"""


class AgentExecutor:
    def __init__(self, *, llm: LLMClient, tool_registry: ToolRegistry, settings: Settings | None = None) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.settings = settings or get_settings()

    def run(
        self,
        *,
        agent_name: str,
        agent_definition: AgentDefinition,
        task: str,
        state: dict[str, Any],
        node_name: str,
    ) -> AgentRunResult:
        prompt = self.llm.load_prompt(agent_definition.prompt, DEFAULT_AGENT_PROMPT)
        allowed_tools = agent_definition.tools
        max_iterations = max(1, agent_definition.max_iterations or self.settings.max_agent_iterations)
        tool_results: list[ToolResult] = []
        raw_actions: list[dict[str, Any]] = []
        latest_action: AgentAction | None = None

        for iteration in range(1, max_iterations + 1):
            user_prompt = dumps_json(
                {
                    "node": node_name,
                    "agent": agent_name,
                    "task": task,
                    "input": state.get("input", {}),
                    "state": state,
                    "previous_tool_results": tool_results,
                    "available_tools": self.tool_registry.specs(allowed_tools),
                    "iteration": iteration,
                },
                indent=2,
            )
            action_payload = self.llm.complete_json(
                system_prompt=prompt,
                user_prompt=user_prompt,
                model=agent_definition.model,
                temperature=agent_definition.temperature,
            )
            latest_action = AgentAction.model_validate(action_payload)
            raw_actions.append(latest_action.model_dump())

            if latest_action.tool_calls:
                for tool_call in latest_action.tool_calls:
                    if not isinstance(tool_call, ToolCall):
                        tool_call = ToolCall.model_validate(tool_call)
                    result = self.tool_registry.invoke(tool_call.tool, tool_call.arguments)
                    tool_results.append(result)

            if latest_action.status == "complete" and not latest_action.tool_calls:
                return AgentRunResult(
                    agent_name=agent_name,
                    status="completed",
                    summary=latest_action.summary,
                    output=latest_action.output,
                    tool_results=tool_results,
                    iterations=iteration,
                    score=latest_action.score,
                    raw_actions=raw_actions,
                )

            if latest_action.status in {"blocked", "failed"}:
                return AgentRunResult(
                    agent_name=agent_name,
                    status="blocked" if latest_action.status == "blocked" else "failed",
                    summary=latest_action.summary,
                    output=latest_action.output,
                    tool_results=tool_results,
                    iterations=iteration,
                    score=latest_action.score,
                    error=latest_action.summary or latest_action.reasoning or "Agent reported failure.",
                    raw_actions=raw_actions,
                )

        final_output = latest_action.output if latest_action else {}
        return AgentRunResult(
            agent_name=agent_name,
            status="completed",
            summary=(latest_action.summary if latest_action else "Agent completed without explicit terminal status."),
            output=final_output | {"tool_results": [item.model_dump() for item in tool_results]},
            tool_results=tool_results,
            iterations=max_iterations,
            score=(latest_action.score if latest_action else None),
            raw_actions=raw_actions,
        )
