from __future__ import annotations

import re
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.core.utils import dumps_json
from aor_runtime.llm.client import LLMClient
from aor_runtime.tools.base import ToolRegistry
from aor_runtime.tools.sql import get_schema, prune_schema, resolve_sql_databases


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
- Use sql.query for relational database questions when schema information is provided.
- Only use databases, tables, and columns from the provided schema. Never hallucinate schema.
- When multiple databases are shown in the schema, sql.query args must include an explicit database name.
- Never encode database selection inside the SQL text.
- Prefer sql.query over shell.exec or python.exec for direct database reads.
- Use python.exec for post-query local composition only when loops, filtering, or conditional logic are required after reading from sql.query.
- Use python.exec when loops are required, multiple files must be processed, conditional logic is needed, or multiple tool calls are required.
- Do not use python.exec for single-step tasks that a direct fs.* or shell.exec step can handle.
- python.exec code may call fs.exists, fs.copy, fs.read, fs.write, fs.mkdir, fs.list, shell.exec, and sql.query.
- In python.exec, call sql.query with explicit keyword arguments like sql.query(database='clinical_db', query='SELECT ...').
- For python.exec, the code must assign a JSON-serializable value to a variable named result.
- For python.exec, keep the code in a single-line JSON string and use semicolons instead of raw newlines.
- Avoid shell.exec when a specific fs tool exists.
- For folder or directory disk-usage questions, prefer du-based shell commands.
- For filesystem or disk-capacity questions, prefer df-based shell commands.

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

Task:
list all patient names from clinical_db
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "sql.query",
      "args": {
        "database": "clinical_db",
        "query": "SELECT name FROM patients ORDER BY name"
      }
    }
  ]
}

Task:
which folder is consuming the most space?
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "shell.exec",
      "args": {
        "command": "du -sh * | sort -hr"
      }
    }
  ]
}

Task:
which folder in my computer is consuming the most space?
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "shell.exec",
      "args": {
        "command": "du -xhd 1 / 2>/dev/null | sort -hr"
      }
    }
  ]
}

Task:
how full is my disk?
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "shell.exec",
      "args": {
        "command": "df -h /"
      }
    }
  ]
}
"""

DATABASE_NAME_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*_db\b")
STORAGE_TOKEN_RE = re.compile(r"[a-z0-9_]+")
DU_COMMAND_RE = re.compile(r"\bdu\b")
DF_COMMAND_RE = re.compile(r"\bdf\b")


class TaskPlanner:
    def __init__(self, *, llm: LLMClient, tools: ToolRegistry, settings: Settings | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.settings = settings or get_settings()

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
        planner_context: dict[str, Any] = {
            "goal": goal,
            "input": input_payload,
            "allowed_tools": self.tools.specs(allowed_tools),
            "failure_context": failure_context or {},
        }
        if "sql.query" in allowed_tools:
            try:
                planner_context["schema"] = prune_schema(get_schema(self.settings), goal, settings=self.settings).model_dump()
            except Exception as exc:  # noqa: BLE001
                planner_context["schema"] = {"databases": [], "error": str(exc)}
        user_prompt = dumps_json(planner_context, indent=2)
        payload = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=planner.model,
            temperature=planner.temperature,
        )
        plan = ExecutionPlan.model_validate(payload)
        self._apply_storage_shell_semantics(goal, plan)
        for step in plan.steps:
            if step.action not in allowed_tools:
                raise ValueError(f"Planner selected disallowed tool {step.action!r}.")
            self.tools.validate_step(step.action, step.args)
        self._validate_explicit_database_targets(goal, plan)
        return plan

    def _validate_explicit_database_targets(self, goal: str, plan: ExecutionPlan) -> None:
        configured_databases = resolve_sql_databases(self.settings)
        if not configured_databases:
            return

        goal_text = str(goal or "").lower()
        requested_names = {match.group(0).lower() for match in DATABASE_NAME_RE.finditer(goal_text)}
        for database_name in configured_databases:
            if database_name.lower() in goal_text or database_name.replace("_", " ").lower() in goal_text:
                requested_names.add(database_name.lower())

        if not requested_names:
            return

        for step in plan.steps:
            if step.action != "sql.query":
                continue
            database_name = step.args.get("database")
            if not isinstance(database_name, str) or not database_name.strip():
                raise ValueError("Planner must include an explicit database name for SQL steps when the goal names a database.")
            normalized = re.sub(r"[^a-z0-9_]+", "_", database_name.strip().lower()).strip("_")
            if normalized not in requested_names:
                requested = ", ".join(sorted(requested_names))
                raise ValueError(f"Planner changed the requested database target. Expected one of: {requested}.")

    def _apply_storage_shell_semantics(self, goal: str, plan: ExecutionPlan) -> None:
        intent = self._classify_storage_intent(goal)
        if intent is None:
            return

        preferred_command = self._preferred_storage_command(intent)
        if preferred_command is None:
            return

        for step in plan.steps:
            if step.action != "shell.exec":
                continue
            command = str(step.args.get("command", ""))
            if intent.startswith("folder_usage"):
                if not DU_COMMAND_RE.search(command):
                    step.args["command"] = preferred_command
                return
            if intent == "filesystem_capacity":
                if not DF_COMMAND_RE.search(command):
                    step.args["command"] = preferred_command
                return

    def _classify_storage_intent(self, goal: str) -> str | None:
        tokens = set(STORAGE_TOKEN_RE.findall(str(goal or "").lower()))
        if not tokens:
            return None

        folder_terms = {"folder", "folders", "directory", "directories"}
        filesystem_terms = {"disk", "disks", "filesystem", "filesystems", "partition", "partitions", "mount", "mounted", "drive", "drives"}
        usage_terms = {"space", "size", "usage", "used", "consuming", "largest", "biggest", "heaviest", "most"}
        system_scope_terms = {"computer", "system", "root", "whole", "entire"}

        mentions_usage = bool(tokens & usage_terms)
        mentions_folder = bool(tokens & folder_terms)
        mentions_filesystem = bool(tokens & filesystem_terms)
        mentions_system_scope = bool(tokens & system_scope_terms)

        if mentions_folder and mentions_usage:
            return "folder_usage_system" if mentions_system_scope else "folder_usage_workspace"
        if mentions_filesystem and mentions_usage:
            return "filesystem_capacity"
        return None

    def _preferred_storage_command(self, intent: str) -> str | None:
        if intent == "folder_usage_workspace":
            return "du -sh * | sort -hr"
        if intent == "folder_usage_system":
            return "du -xhd 1 / 2>/dev/null | sort -hr"
        if intent == "filesystem_capacity":
            return "df -h /"
        return None
