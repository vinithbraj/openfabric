from __future__ import annotations

from typing import Any

from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.core.utils import dumps_json
from aor_runtime.llm.client import LLMClient
from aor_runtime.tools.base import ToolRegistry


DEFAULT_PLANNER_PROMPT = """You are the planner for a deterministic local agent runtime.

Your job is to create a complete execution plan for the user's goal.
Return valid JSON only:
{
  "steps": [
    {
      "id": 1,
      "action": "tool.name",
      "args": {}
    }
  ]
}

Rules:
- Planning only. Do not execute anything.
- Every step must use a real tool from the provided tool list.
- No natural-language steps.
- Be explicit and sequential.
- Prefer filesystem tools over shell for filesystem work.
- Use fs.copy for copying files.
- Use fs.mkdir for creating directories.
- Use fs.write for exact content writes.
- Use fs.read or fs.exists for verification steps when needed.
- Use python.exec when loops are required, multiple files must be processed, conditional logic is needed, or multiple tool calls are required.
- Do not use python.exec for single-step tasks that a direct fs.* or shell.exec step can handle.
- python.exec code may call fs.exists, fs.copy, fs.read, fs.write, fs.mkdir, fs.list, and shell.exec.
- For python.exec, the code must assign a JSON-serializable value to a variable named result.
- For python.exec, keep the code in a single-line JSON string and use semicolons instead of raw newlines.
- Avoid shell.exec when a specific fs tool exists.

Examples:

Task:
copy source.txt to copy.txt
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.exists", "args": {"path": "source.txt"}},
    {"id": 2, "action": "fs.copy", "args": {"src": "source.txt", "dst": "copy.txt"}},
    {"id": 3, "action": "fs.exists", "args": {"path": "copy.txt"}},
    {"id": 4, "action": "fs.read", "args": {"path": "copy.txt"}}
  ]
}

Task:
create nested/deep and write result.txt with exact content "hello"
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.mkdir", "args": {"path": "nested/deep"}},
    {"id": 2, "action": "fs.write", "args": {"path": "nested/deep/result.txt", "content": "hello"}},
    {"id": 3, "action": "fs.read", "args": {"path": "nested/deep/result.txt"}}
  ]
}

Task:
how many txt files are in logs
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.exists", "args": {"path": "logs"}},
    {
      "id": 2,
      "action": "python.exec",
      "args": {
        "code": "files = fs.list('logs'); result = {'count': len([name for name in files if name.endswith('.txt')])}"
      }
    }
  ]
}

Task:
read line 2 from notes.txt
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.exists", "args": {"path": "notes.txt"}},
    {
      "id": 2,
      "action": "python.exec",
      "args": {
        "code": "lines = fs.read('notes.txt').splitlines(); result = {'value': lines[1]}"
      }
    }
  ]
}

Task:
copy all txt files from A to B
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.exists", "args": {"path": "A"}},
    {"id": 2, "action": "fs.mkdir", "args": {"path": "B"}},
    {
      "id": 3,
      "action": "python.exec",
      "args": {
        "code": "files = fs.list('A'); copied = []; [fs.copy(f'A/{name}', f'B/{name}') or copied.append(name) for name in files if name.endswith('.txt')]; result = {'operation': 'bulk_copy', 'src_dir': 'A', 'dst_dir': 'B', 'copied_files': copied}"
      }
    },
    {"id": 4, "action": "fs.list", "args": {"path": "B"}}
  ]
}
"""


class TaskPlanner:
    def __init__(self, *, llm: LLMClient, tools: ToolRegistry) -> None:
        self.llm = llm
        self.tools = tools

    def build_plan(
        self,
        *,
        goal: str,
        planner: PlannerConfig,
        allowed_tools: list[str],
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        system_prompt = self.llm.load_prompt(planner.prompt, DEFAULT_PLANNER_PROMPT)
        user_prompt = dumps_json(
            {
                "goal": goal,
                "input": input_payload,
                "allowed_tools": self.tools.specs(allowed_tools),
                "failure_context": failure_context or {},
            },
            indent=2,
        )
        payload = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=planner.model,
            temperature=planner.temperature,
        )
        plan = ExecutionPlan.model_validate(payload)
        for step in plan.steps:
            if step.action not in allowed_tools:
                raise ValueError(f"Planner selected disallowed tool {step.action!r}.")
            self.tools.validate_step(step.action, step.args)
        return plan
