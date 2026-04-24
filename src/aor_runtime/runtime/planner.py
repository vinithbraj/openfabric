from __future__ import annotations

import re
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ExecutionPlan, HighLevelPlan, PlannerConfig
from aor_runtime.core.utils import dumps_json, extract_json_object
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.dataflow import normalize_execution_plan_dataflow
from aor_runtime.runtime.decomposer import GoalDecomposer, is_complex_goal
from aor_runtime.runtime.plan_canonicalizer import canonicalize_plan, coerce_plan_payload
from aor_runtime.runtime.policies import render_policy_text, select_policies, validate_plan
from aor_runtime.tools.base import ToolRegistry
from aor_runtime.tools.sql import get_schema, prune_schema, resolve_sql_databases


DEFAULT_PLANNER_PROMPT = """You are the planner for a deterministic local agent runtime.

Your job is to create a complete execution plan for the user's goal.
Output JSON only. The response must be a single JSON object that validates against the ExecutionPlan schema:
{
  "steps": [
    {
      "id": 1,
      "action": "tool.name",
      "args": {}
    }
  ]
}

You MUST:
- Produce an ExecutionPlan with strictly executable steps.
- Use only tools from the provided allowed tool list.
- Restrict tool selection to the available tool families in this runtime: fs.*, shell.exec, sql.query, and python.exec.
- Ensure every step can be executed exactly as written without further interpretation.
- Keep steps explicit, concrete, and sequential.
- Fully satisfy the user request.
- If a high_level_plan is provided in the planner context, refine it into executable steps and preserve the task order unless correctness requires a tighter merge.
- If explicit_tool_intent is provided in the planner context, you MUST use those requested tools and MUST NOT substitute a different tool family.
- Include all necessary prerequisite, execution, and verification steps.
- Include verification when the task changes state or requires exactness.
- Every step that produces data for later use must declare an output alias.
- Later steps that consume prior results must declare input aliases and reference previous outputs explicitly with structured refs like {"$ref": "alias"} or {"$ref": "alias", "path": "rows.0.name"}.
- Refinement must preserve outputs from previous steps, use outputs in subsequent steps, avoid placeholder or hardcoded values, and maintain logical dataflow across steps.
- Use sql.query for relational database questions when schema information is provided.
- Use SQL for aggregation, grouping, counting, filtering large datasets, and joins whenever the database can express the operation directly.
- Prefer pushing computation into SQL whenever possible.
- Only use databases, tables, and columns from the provided schema. Never hallucinate schema.
- If schema information includes a database dialect, generate SQL that is valid for that dialect.
- For PostgreSQL, do not use SQLite-only functions like strftime. Prefer PostgreSQL date functions such as CURRENT_DATE, AGE, DATE_PART, EXTRACT, or INTERVAL arithmetic.
- When multiple databases are shown in the schema, sql.query args must include an explicit database name.
- Never encode database selection inside the SQL text.
- Prefer sql.query over shell.exec or python.exec for direct database reads.
- Avoid pulling large datasets into python.exec for simple aggregation tasks that SQL can perform directly.
- Use fs.* for file operations.
- Use shell.exec for system-level commands.
- If the task explicitly names a node, include that node in shell.exec args.
- Use only node names from the provided logical node list when a node is specified.
- If a default node is provided in the planner context, you may omit node and shell.exec will run there.
- Never invent node names outside the provided logical node list.
- Use python.exec only when loops, conditional logic, or multi-step composition are required.
- Use python.exec once for simple composition, combine logic into a single block when possible, and use multiple python.exec steps only if necessary.
- Use python.exec for post-query local composition only when loops, filtering, or conditional logic are required after reading from sql.query.
- Use python.exec for formatting, visualization preparation, or post-processing only after SQL has already reduced the dataset when possible.
- In python.exec, you may call sql.query through the provided sql helper.
- In python.exec, upstream data is passed through args.inputs and available in the sandbox as the inputs dict.
- In python.exec, inputs[...] values are fully computed runtime results, not references, wrappers, or tool-response objects.
- In python.exec, inputs[...] is the value itself, not an object containing the value, and it does not contain implicit nested structure.
- In python.exec, upstream tool outputs passed through args.inputs resolve to these shapes: sql.query -> list of dict rows, fs.find -> list of file path strings, fs.read -> string content, shell.exec -> stdout string, python.exec -> arbitrary resolved value.
- In python.exec, use inputs[name] directly and never access nested wrapper fields like ["stdout"], ["rows"], or ["content"].
- In python.exec, do not add defensive wrapper-detection logic or shape-probing branches for inputs[...].
- In python.exec, do not wrap inputs[...] into new containers unless the actual computation requires it, and do not rename inputs[...] unless needed for readability or a real transformation.
- In python.exec, shell.exec output passed through inputs[...] is a string; use .splitlines() when you need a list of lines.
- In python.exec, always handle empty SQL result lists safely before indexing.
- In python.exec, do not assume SQL result fields unless they were explicitly selected by the SQL query.
- In python.exec, downstream steps consume the python.exec output value directly, so produce one clear output value rather than ambiguous nested containers.
- In python.exec, shell.exec(...) called directly inside the sandbox returns an object with stdout, stderr, and returncode fields. If you need command output text from a direct helper call, parse shell.exec(...).stdout.
- In python.exec, shell.exec(command, node='edge-1') runs on the requested logical node. If node is omitted, the default node is used when configured.
- In python.exec, call sql.query with explicit keyword arguments like sql.query(database='clinical_db', query='SELECT ...').
- In python.exec, assign the final JSON-serializable answer to a variable named result.
- If the user asks to return, list, show, or provide data, the final step must surface that data and not only write it to a file.
- Keep python.exec code in a single-line JSON string and use semicolons instead of raw newlines.
- Every args value must be valid JSON as written.
- For straightforward command-output extraction, filtering, or CSV/text formatting, prefer a single shell.exec step when shell can produce the final answer directly.
- Prefer filesystem tools over shell commands for filesystem tasks.
- Use fs.find for recursive file discovery and glob-style file matching such as *.txt under a directory.
- Use fs.size when the user asks for the size of a file or the total size of a set of files.
- Use fs.copy for copying files.
- Use fs.mkdir for creating directories.
- Use fs.write for exact file content.
- Use fs.read to verify exact contents when the task requires exact text verification.
- Use fs.exists to assert that a path exists before reads, copies, or removals, or after operations that create something.
- Use fs.not_exists to verify that a path is absent after deletion or cleanup.
- For delete-like shell commands, verify removed paths with fs.not_exists after the shell step.
- For folder or directory disk-usage questions, prefer du-based shell commands.
- For filesystem or disk-capacity questions, prefer df-based shell commands.

You MUST NOT:
- Output anything except the JSON ExecutionPlan object.
- Pretend to execute anything.
- Output high-level intent labels such as "copy file" instead of executable tool actions.
- Skip required prerequisite or verification steps.
- Rely on implicit assumptions or invent missing details.
- Generate non-executable natural-language descriptions.
- Emit natural-language pseudo-steps.
- Ignore a provided high_level_plan.
- Ignore explicit tool intent.
- Use shell.exec when a filesystem tool already covers the task.
- Use python.exec for a single-step task that a direct fs.* or shell.exec step can handle.
- Put expressions, string concatenation, comprehensions, or variable references in args outside a python.exec code string.
- Use shell -> python -> fs.write -> fs.read round-trips when a direct shell.exec step can produce the requested text output.
- For python.exec inputs, assume wrapper objects or nested response objects where resolved values are already provided.
- Do not reuse earlier fs.exists prechecks as deletion verification.
- Do not use fs.list when the user asked to find matching files recursively by pattern and fs.find is available.
- Do not use shell.exec or Python imports like os.path for file sizes when fs.size is available.
- Do not generate placeholder values like name1,name2,name3 when upstream data is available.

Tool selection policy:
- Use sql.query for filtering or querying structured data.
- Use fs.* for file and directory operations.
- Use shell.exec for system-level commands.
- Use python.exec only for loops, conditional logic, or multi-step composition.

Completeness rule:
- Every plan must fully satisfy the user request.
- Every plan must include all necessary steps.
- Every plan must include verification when applicable.

Policy guidance:
- You must follow the planning policies provided in the planner context.
- Policies define which tools to prefer, when to avoid certain tools, and how to minimize steps.
- Select the best tool for the task.
- Avoid unnecessary steps.
- Prefer domain-specific tools over generic ones.

Tool selection priority:
1. SQL:
   - Use for filtering, aggregation, grouping, counting, and joins.
   - Prefer over python.exec for data operations.
2. Filesystem:
   - Use for file operations.
   - Prefer over shell.exec.
3. Shell:
   - Use only when no direct tool exists.
   - Include node when the task is node-specific.
4. Python:
   - Use only for loops, complex composition, or multi-step logic.

Optimization rules:
- Use the minimal number of steps.
- Avoid switching between domains unless necessary.
- Combine operations when possible.
- Prefer a direct shell.exec step for simple command-output formatting tasks.

Examples:

Task:
create a file notes.txt with exact content "hello"
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.write", "args": {"path": "notes.txt", "content": "hello"}},
    {"id": 2, "action": "fs.read", "args": {"path": "notes.txt"}}
  ]
}

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
    {"id": 3, "action": "fs.exists", "args": {"path": "nested/deep/result.txt"}},
    {"id": 4, "action": "fs.read", "args": {"path": "nested/deep/result.txt"}}
  ]
}

Task:
how many txt files are in logs
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.find", "args": {"path": "logs", "pattern": "*.txt"}},
    {
      "id": 2,
      "action": "python.exec",
      "args": {
        "code": "files = fs.find('logs', '*.txt'); result = {'count': len(files)}"
      }
    }
  ]
}

Task:
find all *.txt files in this folder and provide list as csv
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.txt"}, "output": "txt_matches"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["txt_matches"],
      "output": "csv_result",
      "args": {
        "inputs": {"matches": {"$ref": "txt_matches", "path": "matches"}},
        "code": "result = {'csv': ','.join(inputs['matches'])}"
      }
    }
  ]
}

Task:
using shell, list all .py files under src/aor_runtime/runtime and return the list as a csv string
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "find src/aor_runtime/runtime -type f -name \"*.py\""}, "output": "py_files"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["py_files"],
      "output": "csv_result",
      "args": {
        "inputs": {"py_files": {"$ref": "py_files", "path": "stdout"}},
        "code": "result = {'csv': ','.join(inputs['py_files'].splitlines())}"
      }
    }
  ]
}

Task:
compute the total size of all the txt files in this folder
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.find", "args": {"path": ".", "pattern": "*.txt"}, "output": "txt_matches"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["txt_matches"],
      "output": "size_summary",
      "args": {
        "inputs": {"files": {"$ref": "txt_matches", "path": "matches"}},
        "code": "total_size = sum(fs.size(path) for path in inputs['files']); result = {'file_count': len(inputs['files']), 'total_size_bytes': total_size}"
      }
    }
  ]
}

Task:
count studies in clinical_db and return the count safely
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "sql.query",
      "output": "study_rows",
      "args": {
        "database": "clinical_db",
        "query": "SELECT COUNT(*) AS study_count FROM studies"
      }
    },
    {
      "id": 2,
      "action": "python.exec",
      "input": ["study_rows"],
      "output": "study_count",
      "args": {
        "inputs": {"rows": {"$ref": "study_rows", "path": "rows"}},
        "code": "rows = inputs['rows']; result = {'count': rows[0]['study_count'] if rows else 0}"
      }
    }
  ]
}

Task:
query the top 3 patients by score from clinical_db, format the names as csv, and save the result to outputs/top_patients.csv
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "sql.query",
      "output": "patient_rows",
      "args": {
        "database": "clinical_db",
        "query": "SELECT name FROM patients ORDER BY score DESC LIMIT 3"
      }
    },
    {
      "id": 2,
      "action": "python.exec",
      "input": ["patient_rows"],
      "output": "patient_csv",
      "args": {
        "inputs": {"rows": {"$ref": "patient_rows", "path": "rows"}},
        "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}"
      }
    },
    {"id": 3, "action": "fs.mkdir", "args": {"path": "outputs"}},
    {
      "id": 4,
      "action": "fs.write",
      "input": ["patient_csv"],
      "args": {
        "path": "outputs/top_patients.csv",
        "content": {"$ref": "patient_csv", "path": "csv"}
      }
    },
    {
      "id": 5,
      "action": "fs.read",
      "args": {
        "path": "outputs/top_patients.csv"
      }
    }
  ]
}

Task:
find all *.txt files under inputs, compute their total size, and write a JSON summary to reports/txt_summary.json
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.find", "args": {"path": "inputs", "pattern": "*.txt"}, "output": "txt_matches"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["txt_matches"],
      "output": "size_summary",
      "args": {
        "inputs": {"files": {"$ref": "txt_matches", "path": "matches"}},
        "code": "import json; total_size = sum(fs.size(f'inputs/{path}') for path in inputs['files']); result = {'summary_json': json.dumps({'file_count': len(inputs['files']), 'total_size_bytes': total_size})}"
      }
    },
    {"id": 3, "action": "fs.mkdir", "args": {"path": "reports"}},
    {
      "id": 4,
      "action": "fs.write",
      "input": ["size_summary"],
      "args": {
        "path": "reports/txt_summary.json",
        "content": {"$ref": "size_summary", "path": "summary_json"}
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
      "output": "patient_rows",
      "args": {
        "database": "clinical_db",
        "query": "SELECT name FROM patients ORDER BY name"
      }
    }
  ]
}

Task:
list all patients above 45 years of age in dicom
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "sql.query",
      "args": {
        "database": "dicom",
        "query": "SELECT patient_id, name, dob FROM patient WHERE dob <= CURRENT_DATE - INTERVAL '45 years' ORDER BY dob"
      }
    }
  ]
}

Task:
return the current directory entries as a csv string
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "shell.exec",
      "args": {
        "command": "ls -1 | paste -sd, -"
      }
    }
  ]
}

Task:
run uname -a on node edge-1
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "shell.exec",
      "args": {
        "node": "edge-1",
        "command": "uname -a"
      }
    }
  ]
}

Task:
delete notes.txt
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.exists", "args": {"path": "notes.txt"}},
    {"id": 2, "action": "shell.exec", "args": {"command": "rm notes.txt"}},
    {"id": 3, "action": "fs.not_exists", "args": {"path": "notes.txt"}}
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
PLANNER_RAW_OUTPUT_PREVIEW_CHARS = 600
TOOL_INTENT_PATTERNS = {
    "shell.exec": [r"\b(?:using|use|with)\s+shell(?:\.exec)?\b"],
    "python.exec": [r"\b(?:using|use|with)\s+python(?:\.exec)?\b"],
    "sql.query": [r"\b(?:using|use|with)\s+sql(?:\.query)?\b"],
    "fs.*": [r"\b(?:using|use|with)\s+(?:filesystem|fs)\b"],
}
FILESYSTEM_TOOL_INTENT_PATTERNS = {
    "fs.copy": [r"\b(?:using|use|with)\s+fs\.copy\b"],
    "fs.exists": [r"\b(?:using|use|with)\s+fs\.exists\b"],
    "fs.find": [r"\b(?:using|use|with)\s+fs\.find\b"],
    "fs.list": [r"\b(?:using|use|with)\s+fs\.list\b"],
    "fs.mkdir": [r"\b(?:using|use|with)\s+fs\.mkdir\b"],
    "fs.not_exists": [r"\b(?:using|use|with)\s+fs\.not_exists\b"],
    "fs.read": [r"\b(?:using|use|with)\s+fs\.read\b"],
    "fs.size": [r"\b(?:using|use|with)\s+fs\.size\b"],
    "fs.write": [r"\b(?:using|use|with)\s+fs\.write\b"],
}


def summarize_plan(plan: ExecutionPlan) -> str:
    actions = [step.action for step in plan.steps]
    return f"Plan with {len(actions)} steps: " + ", ".join(actions)


def summarize_planner_raw_output(raw_output: str | None, limit: int = PLANNER_RAW_OUTPUT_PREVIEW_CHARS) -> str | None:
    text = str(raw_output or "").strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_explicit_tool_intent(goal: str, allowed_tools: list[str]) -> list[str]:
    goal_text = str(goal or "").lower()
    requested: list[str] = []
    for tool_name, patterns in TOOL_INTENT_PATTERNS.items():
        if tool_name != "fs.*" and tool_name not in allowed_tools:
            continue
        if tool_name == "fs.*" and not any(name.startswith("fs.") for name in allowed_tools):
            continue
        if any(re.search(pattern, goal_text) for pattern in patterns):
            requested.append(tool_name)
    for tool_name, patterns in FILESYSTEM_TOOL_INTENT_PATTERNS.items():
        if tool_name not in allowed_tools:
            continue
        if any(re.search(pattern, goal_text) for pattern in patterns):
            requested.append(tool_name)
    return list(dict.fromkeys(requested))


class TaskPlanner:
    def __init__(self, *, llm: LLMClient, tools: ToolRegistry, settings: Settings | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.settings = settings or get_settings()
        self.decomposer = GoalDecomposer(llm=llm)
        self.last_policies_used: list[str] = []
        self.last_high_level_plan: list[str] | None = None
        self.last_planning_mode: str = "direct"
        self.last_llm_calls: int = 0
        self.last_error_stage: str | None = None
        self.last_raw_output: str | None = None
        self.last_error_type: str | None = None
        self.last_original_execution_plan: dict[str, Any] | None = None
        self.last_canonicalized_execution_plan: dict[str, Any] | None = None
        self.last_plan_repairs: list[str] = []
        self.last_plan_canonicalized: bool = False

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
        self._reset_tracking()
        schema_payload = self._schema_payload(goal, allowed_tools)
        policies = select_policies(goal, allowed_tools, schema_payload)
        explicit_tool_intent = extract_explicit_tool_intent(goal, allowed_tools)
        self.last_policies_used = [policy.name for policy in policies]
        self.last_planning_mode = "hierarchical" if is_complex_goal(goal) else "direct"

        try:
            if self.last_planning_mode == "hierarchical":
                high_level_plan = self._decompose_goal(
                    goal=goal,
                    planner=planner,
                    input_payload=input_payload,
                    failure_context=failure_context,
                )
                self.last_high_level_plan = list(high_level_plan.tasks)
                plan = self._generate_execution_plan(
                    system_prompt=system_prompt,
                    goal=goal,
                    planner=planner,
                    allowed_tools=allowed_tools,
                    input_payload=input_payload,
                    failure_context=failure_context,
                    schema_payload=schema_payload,
                    policies=policies,
                    explicit_tool_intent=explicit_tool_intent,
                    high_level_plan=high_level_plan,
                    stage="refine",
                )
            else:
                plan = self._generate_execution_plan(
                    system_prompt=system_prompt,
                    goal=goal,
                    planner=planner,
                    allowed_tools=allowed_tools,
                    input_payload=input_payload,
                    failure_context=failure_context,
                    schema_payload=schema_payload,
                    policies=policies,
                    explicit_tool_intent=explicit_tool_intent,
                    high_level_plan=None,
                    stage="direct",
                )
            finalized_plan = self._finalize_plan(goal, plan, allowed_tools, explicit_tool_intent)
            self.last_error_stage = None
            return finalized_plan
        except Exception as exc:
            self.last_error_type = type(exc).__name__
            raise

    def _reset_tracking(self) -> None:
        self.last_policies_used = []
        self.last_high_level_plan = None
        self.last_planning_mode = "direct"
        self.last_llm_calls = 0
        self.last_error_stage = None
        self.last_raw_output = None
        self.last_error_type = None
        self.last_original_execution_plan = None
        self.last_canonicalized_execution_plan = None
        self.last_plan_repairs = []
        self.last_plan_canonicalized = False

    def _schema_payload(self, goal: str, allowed_tools: list[str]) -> dict[str, Any] | None:
        if "sql.query" not in allowed_tools:
            return None
        try:
            return prune_schema(get_schema(self.settings), goal, settings=self.settings).model_dump()
        except Exception as exc:  # noqa: BLE001
            return {"databases": [], "error": str(exc)}

    def _decompose_goal(
        self,
        *,
        goal: str,
        planner: PlannerConfig,
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None,
    ) -> HighLevelPlan:
        self.last_error_stage = "decompose"
        self.last_llm_calls += 1
        try:
            high_level_plan = self.decomposer.decompose_goal(
                goal=goal,
                planner=planner,
                input_payload=input_payload,
                failure_context=failure_context,
            )
            self.last_raw_output = self.decomposer.last_raw_output
            self.last_error_stage = None
            return high_level_plan
        except Exception:
            self.last_raw_output = self.decomposer.last_raw_output
            raise

    def _generate_execution_plan(
        self,
        *,
        system_prompt: str,
        goal: str,
        planner: PlannerConfig,
        allowed_tools: list[str],
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None,
        schema_payload: dict[str, Any] | None,
        policies: list[Any],
        explicit_tool_intent: list[str],
        high_level_plan: HighLevelPlan | None,
        stage: str,
    ) -> ExecutionPlan:
        planner_context = self._build_planner_context(
            goal=goal,
            allowed_tools=allowed_tools,
            input_payload=input_payload,
            failure_context=failure_context,
            schema_payload=schema_payload,
            policies=policies,
            explicit_tool_intent=explicit_tool_intent,
            high_level_plan=high_level_plan,
        )
        user_prompt = dumps_json(planner_context, indent=2)
        self.last_error_stage = stage
        self.last_llm_calls += 1
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
        return ExecutionPlan.model_validate(coerce_plan_payload(payload))

    def _build_planner_context(
        self,
        *,
        goal: str,
        allowed_tools: list[str],
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None,
        schema_payload: dict[str, Any] | None,
        policies: list[Any],
        explicit_tool_intent: list[str],
        high_level_plan: HighLevelPlan | None,
    ) -> dict[str, Any]:
        planner_context: dict[str, Any] = {
            "goal": goal,
            "input": input_payload,
            "allowed_tools": self.tools.specs(allowed_tools),
            "failure_context": failure_context or {},
            "policies": render_policy_text(policies),
        }
        if explicit_tool_intent:
            planner_context["explicit_tool_intent"] = explicit_tool_intent
        if high_level_plan is not None:
            planner_context["high_level_plan"] = list(high_level_plan.tasks)

        if "shell.exec" in allowed_tools:
            planner_context["nodes"] = {"available": self.settings.available_nodes}
            default_node = self.settings.resolved_default_node()
            if default_node:
                planner_context["nodes"]["default"] = default_node

        if schema_payload is not None:
            planner_context["schema"] = schema_payload
        return planner_context

    def _finalize_plan(self, goal: str, plan: ExecutionPlan, allowed_tools: list[str], explicit_tool_intent: list[str]) -> ExecutionPlan:
        self._apply_storage_shell_semantics(goal, plan)
        normalize_execution_plan_dataflow(plan)
        self.last_original_execution_plan = plan.model_dump()
        canonicalized = canonicalize_plan(plan, goal, allowed_tools)
        plan = canonicalized.plan
        self.last_plan_repairs = list(canonicalized.repairs)
        self.last_plan_canonicalized = canonicalized.changed
        self.last_canonicalized_execution_plan = plan.model_dump() if canonicalized.changed else None
        normalize_execution_plan_dataflow(plan)
        for step in plan.steps:
            if step.action not in allowed_tools:
                raise ValueError(f"Planner selected disallowed tool {step.action!r}.")
            self.tools.validate_step(step.action, step.args)
        self._validate_explicit_database_targets(goal, plan)
        self._validate_shell_targets(plan)
        self._validate_explicit_tool_intent(plan, explicit_tool_intent)
        validate_plan(plan)
        return plan

    def _validate_explicit_tool_intent(self, plan: ExecutionPlan, explicit_tool_intent: list[str]) -> None:
        if not explicit_tool_intent:
            return
        actions = [step.action for step in plan.steps]
        for requested_tool in explicit_tool_intent:
            if requested_tool == "fs.*":
                if not any(action.startswith("fs.") for action in actions):
                    raise ValueError("Planner ignored the explicit filesystem tool request.")
                continue
            if requested_tool not in actions:
                raise ValueError(f"Planner ignored the explicit tool request for {requested_tool}.")

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

    def _validate_shell_targets(self, plan: ExecutionPlan) -> None:
        allowed_nodes = self.settings.available_nodes
        default_node = str(self.settings.resolved_default_node() or "").strip()
        for step in plan.steps:
            if step.action != "shell.exec":
                continue
            node = str(step.args.get("node", "")).strip()
            if node and node not in allowed_nodes:
                allowed = ", ".join(allowed_nodes) or "<none configured>"
                raise ValueError(f"Planner selected a disallowed node {node!r}. Available nodes: {allowed}.")
            if not node and not default_node:
                raise ValueError(
                    "Planner must include an explicit node name for shell.exec steps when no default node is configured."
                )

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
