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
from aor_runtime.runtime.policies import (
    PlanContractViolation,
    classify_plan_violations,
    render_policy_text,
    select_policies,
    validate_plan_contract,
)
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
- Preserve full file paths exactly as provided in the goal or returned by tools.
- Never strip directory prefixes, collapse to basenames, manually reconstruct alternate prefixes, or convert absolute paths into relative paths.
- When fs.find returns matches, keep those returned match strings unchanged unless a downstream fs.size, fs.read, or fs.copy operation needs a concrete path; in that case, join the returned fs.find.path root with the returned relative match.
- Use sql.query for relational database questions when schema information is provided.
- Use SQL for aggregation, grouping, counting, filtering, joins, and histograms whenever the database can express the operation directly.
- Prefer pushing computation into SQL whenever possible.
- Only use databases, tables, and columns from the provided schema. Never hallucinate schema.
- If schema information includes a database dialect, generate SQL that is valid for that dialect.
- For PostgreSQL, do not use SQLite-only functions like strftime. Prefer PostgreSQL date functions such as CURRENT_DATE, AGE, DATE_PART, EXTRACT, or INTERVAL arithmetic.
- When multiple databases are shown in the schema, sql.query args must include an explicit database name.
- Never encode database selection inside the SQL text.
- Prefer sql.query over shell.exec or python.exec for direct database reads.
- Avoid pulling large datasets into python.exec for simple aggregation tasks that SQL can perform directly.
- If the user requests modifying SQL data or schema, do not generate a modifying sql.query plan. Produce an explicit error outcome instead.
- Use fs.* for file operations.
- Use shell.exec for system-level commands.
- If the task explicitly names a node, include that node in shell.exec args.
- Use only node names from the provided logical node list when a node is specified.
- If a default node is provided in the planner context, you may omit node and shell.exec will run there.
- Never invent node names outside the provided logical node list.
- Use python.exec only for pure data transformation when loops, conditional logic, or multi-step composition are required to transform already-produced values.
- Do not use python.exec when sql.query, fs.*, or shell.exec can solve the task directly.
- Use python.exec once for simple transformation, combine logic into a single block when possible, and use multiple python.exec steps only if necessary.
- Use python.exec for formatting, combining, visualization preparation, or post-processing only after upstream tool steps have already produced the needed data.
- python.exec is a pure data transformation step: it must read from inputs[...], compute a transformed value, and assign the final value to result.
- python.exec must never call tools and must never perform side effects.
- In python.exec, code must be valid, minimal, and should only import json or re when an import is required.
- In python.exec, do not use os, subprocess, system calls, eval, exec, or direct fs.* / shell.exec(...) / sql.query(...) helper calls.
- In python.exec, all side-effecting work must appear as explicit non-Python tool steps in the plan.
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
- In python.exec, assign the final JSON-serializable answer to a variable named result.
- If the user asks to return, list, show, or provide data, the final step must surface that data and not only write it to a file.
- Return exactly what the user asked for: count requests return a number only, CSV requests return a CSV string only, and JSON requests return a JSON object only.
- Do not include extra text, file paths, acknowledgements, or wrapper objects in the final output unless the user explicitly asked for them.
- Output keys must match the user's requested keys exactly, including spelling and case. Do not rename keys like studies to study_count and do not change CT to ct.
- Keep python.exec code in a single-line JSON string and use semicolons instead of raw newlines.
- Every args value must be valid JSON as written.
- For straightforward command-output extraction, filtering, or CSV/text formatting, prefer a single shell.exec step when shell can produce the final answer directly.
- For fetch-and-extract tasks such as curl/fetch page + extract section/value, treat extraction as a shell-first pattern and prefer a single shell.exec step when shell can return the final extracted text directly.
- For web fetches in shell.exec, prefer curl -sL and prefer https:// for bare domains so redirects are followed and the fetched page is the canonical page.
- For simple HTML/text extraction, do not generate shell.exec -> python.exec or shell.exec -> shell.exec plans when one shell.exec pipeline can fetch once and return the requested extracted text directly.
- For simple HTML tag or section extraction such as <head>, <title>, or similar single-section extraction, use a single shell.exec pipeline and do not add python.exec unless shell cannot express the extraction cleanly.
- Do not generate multiple shell.exec steps that fetch the same URL twice for a single extraction task.
- For shell-based HTML/text extraction, prefer portable pipe-based commands and avoid process substitution like <(...).
- Use python.exec with re only when upstream shell or other tools already produced the full input text and the extraction is too awkward to express cleanly in shell.
- If python.exec uses re for extraction, result must be a plain string, list, dict, number, or boolean. Never assign a raw re.Match object to result; use match.group(0) if match else '' or similar.
- Prefer filesystem tools over shell commands for filesystem tasks.
- Use fs.find for recursive file discovery and glob-style file matching such as *.txt under a directory.
- For top-level or non-recursive file matching, prefer fs.list plus minimal filtering or formatting instead of fs.find.
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
- For text-content search across many files, prefer shell.exec with a portable find + grep command rather than python.exec file IO or fs.read loops.
- If the user specifies a file pattern such as *.txt or *.py, use find ROOT -type f -name "<pattern>" -exec grep -li -- "needle" {} + || true so the content search targets the requested files.
- If the user asks to search all files and does not specify a pattern, search all regular files with find ROOT -type f -exec grep -li -- "needle" {} + || true rather than inventing a restrictive extension filter.
- Use rg -l only as an optional faster variant when the shell environment is explicitly known to support rg; otherwise default to the portable find + grep form.

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
- Reimplement filesystem operations, SQL operations, shell operations, or any other side-effecting work in Python.
- Generate python.exec steps that call fs.*, shell.exec, or sql.query helpers.
- Do not reuse earlier fs.exists prechecks as deletion verification.
- Do not use fs.list when the user asked to find matching files recursively by pattern and fs.find is available.
- Do not use shell.exec or Python imports like os.path for file sizes when fs.size is available.
- Do not generate placeholder values like name1,name2,name3 when upstream data is available.
- Do not generate modifying SQL such as INSERT, UPDATE, DELETE, CREATE, ALTER, GRANT, or REVOKE.

Tool selection policy:
- Use sql.query for filtering or querying structured data.
- Use fs.* for file and directory operations.
- Use shell.exec for system-level commands.
- Use python.exec only for pure formatting, transformation, and value combination.

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
   - Use only for pure formatting, transformation, or value combination.

Optimization rules:
- Be correct first, minimal second, and efficient third.
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
    {"id": 1, "action": "fs.find", "args": {"path": "logs", "pattern": "*.txt"}, "output": "txt_matches"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["txt_matches"],
      "output": "txt_count",
      "args": {
        "inputs": {"matches": {"$ref": "txt_matches", "path": "matches"}},
        "code": "result = len(inputs['matches'])"
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
        "code": "result = ','.join(inputs['matches'])"
      }
    }
  ]
}

Task:
find all *.txt files in this folder with the word vinith in their contents
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "find . -type f -name \"*.txt\" -exec grep -li -- \"vinith\" {} + || true"}, "output": "matching_files"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["matching_files"],
      "output": "csv_result",
      "args": {
        "inputs": {"matching_files": {"$ref": "matching_files", "path": "stdout"}},
        "code": "result = ','.join(line for line in inputs['matching_files'].splitlines() if line)"
      }
    }
  ]
}

Task:
list all .py files with import in their contents
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "find . -type f -name \"*.py\" -exec grep -li -- \"import\" {} + || true"}, "output": "matching_files"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["matching_files"],
      "output": "csv_result",
      "args": {
        "inputs": {"matching_files": {"$ref": "matching_files", "path": "stdout"}},
        "code": "result = ','.join(line for line in inputs['matching_files'].splitlines() if line)"
      }
    }
  ]
}

Task:
list all files with vinith in their contents
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "find . -type f -exec grep -li -- \"vinith\" {} + || true"}, "output": "matching_files"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["matching_files"],
      "output": "csv_result",
      "args": {
        "inputs": {"matching_files": {"$ref": "matching_files", "path": "stdout"}},
        "code": "result = ','.join(line for line in inputs['matching_files'].splitlines() if line)"
      }
    }
  ]
}

Task:
if the shell environment is explicitly known to support rg, find all *.txt files in this folder with the word vinith in their contents
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "rg -l -i --glob \"*.txt\" \"vinith\" ."}, "output": "matching_files"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["matching_files"],
      "output": "csv_result",
      "args": {
        "inputs": {"matching_files": {"$ref": "matching_files", "path": "stdout"}},
        "code": "result = ','.join(line for line in inputs['matching_files'].splitlines() if line)"
      }
    }
  ]
}

Task:
curl google.com and extract <head>
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "curl -sL https://www.google.com | tr '\\n' ' ' | sed -n 's:.*\\(<head[^>]*>.*</head>\\).*:\\1:p'"}}
  ]
}

Task:
fetch a page and extract title text
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "curl -sL https://example.com | tr '\\n' ' ' | sed -n 's:.*<title[^>]*>\\([^<]*\\)</title>.*:\\1:p'"}}
  ]
}

Task:
curl example.com and extract <title>
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "curl -sL https://example.com | tr '\\n' ' ' | sed -n 's:.*\\(<title[^>]*>.*</title>\\).*:\\1:p'"}}
  ]
}

Task:
fetch a page, then extract a value with complex regex cleanup
Plan:
{
  "steps": [
    {"id": 1, "action": "shell.exec", "args": {"command": "curl -sL https://example.com"}, "output": "html_content"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["html_content"],
      "output": "description_text",
      "args": {
        "inputs": {"html": {"$ref": "html_content", "path": "stdout"}},
        "code": "import re; match = re.search(r'<meta[^>]+name=[\\\"\\']description[\\\"\\'][^>]+content=[\\\"\\']([^\\\"\\']+)[\\\"\\']', inputs['html'], re.IGNORECASE); result = match.group(1) if match else ''"
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
        "code": "result = ','.join(inputs['py_files'].splitlines())"
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
        "code": "rows = inputs['rows']; result = rows[0]['study_count'] if rows else 0"
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
        "code": "result = ','.join(row['name'] for row in inputs['rows'])"
      }
    },
    {"id": 3, "action": "fs.mkdir", "args": {"path": "outputs"}},
    {
      "id": 4,
      "action": "fs.write",
      "input": ["patient_csv"],
      "args": {
        "path": "outputs/top_patients.csv",
        "content": {"$ref": "patient_csv"}
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
count CT and MR series in dicom and return JSON with keys CT and MR
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "sql.query",
      "output": "modality_rows",
      "args": {
        "database": "dicom",
        "query": "SELECT SUM(CASE WHEN modality = 'CT' THEN 1 ELSE 0 END) AS \"CT\", SUM(CASE WHEN modality = 'MR' THEN 1 ELSE 0 END) AS \"MR\" FROM series"
      }
    },
    {
      "id": 2,
      "action": "python.exec",
      "input": ["modality_rows"],
      "output": "modality_counts",
      "args": {
        "inputs": {"rows": {"$ref": "modality_rows", "path": "rows"}},
        "code": "rows = inputs['rows']; result = rows[0] if rows else {'CT': 0, 'MR': 0}"
      }
    }
  ]
}

Task:
count studies and series in clinical_db, write a JSON summary to reports/summary.json, and return it
Plan:
{
  "steps": [
    {
      "id": 1,
      "action": "sql.query",
      "output": "study_rows",
      "args": {
        "database": "clinical_db",
        "query": "SELECT COUNT(*) AS studies FROM studies"
      }
    },
    {
      "id": 2,
      "action": "sql.query",
      "output": "series_rows",
      "args": {
        "database": "clinical_db",
        "query": "SELECT COUNT(*) AS series FROM series"
      }
    },
    {
      "id": 3,
      "action": "python.exec",
      "input": ["study_rows", "series_rows"],
      "output": "summary_json",
      "args": {
        "inputs": {
          "study_rows": {"$ref": "study_rows", "path": "rows"},
          "series_rows": {"$ref": "series_rows", "path": "rows"}
        },
        "code": "import json; study_rows = inputs['study_rows']; series_rows = inputs['series_rows']; result = json.dumps({'studies': study_rows[0]['studies'] if study_rows else 0, 'series': series_rows[0]['series'] if series_rows else 0}, sort_keys=True)"
      }
    },
    {"id": 4, "action": "fs.mkdir", "args": {"path": "reports"}},
    {
      "id": 5,
      "action": "fs.write",
      "input": ["summary_json"],
      "args": {
        "path": "reports/summary.json",
        "content": {"$ref": "summary_json"}
      }
    },
    {
      "id": 6,
      "action": "fs.read",
      "args": {
        "path": "reports/summary.json"
      }
    }
  ]
}

Task:
read line 2 from notes.txt
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.read", "args": {"path": "notes.txt"}, "output": "notes_text"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["notes_text"],
      "args": {
        "inputs": {"text": {"$ref": "notes_text", "path": "content"}},
        "code": "lines = inputs['text'].splitlines(); result = lines[1] if len(lines) > 1 else ''"
      }
    }
  ]
}

Task:
find all top-level *.txt files under reports and return them as csv
Plan:
{
  "steps": [
    {"id": 1, "action": "fs.list", "args": {"path": "reports"}, "output": "report_entries"},
    {
      "id": 2,
      "action": "python.exec",
      "input": ["report_entries"],
      "args": {
        "inputs": {"entries": {"$ref": "report_entries", "path": "entries"}},
        "code": "result = ','.join(name for name in inputs['entries'] if name.endswith('.txt'))"
      }
    }
  ]
}

Task:
Anti-patterns:
- Invalid: a python.exec step that calls fs.write(...), fs.read(...), fs.copy(...), fs.find(...), fs.list(...), or fs.size(...).
- Invalid: a python.exec step that calls shell.exec(...).
- Invalid: a python.exec step that calls sql.query(...).
- Invalid: any python.exec step that performs side effects instead of returning a transformed value through result.

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
        initial_violations = classify_plan_violations(plan, goal=goal)
        if initial_violations.hard:
            raise PlanContractViolation.from_violations(initial_violations)

        if initial_violations.soft:
            canonicalized = canonicalize_plan(plan, goal, allowed_tools)
            plan = canonicalized.plan
            self.last_plan_repairs = list(canonicalized.repairs)
            self.last_plan_canonicalized = canonicalized.changed
            self.last_canonicalized_execution_plan = plan.model_dump() if canonicalized.changed else None
            normalize_execution_plan_dataflow(plan)
        else:
            self.last_plan_repairs = []
            self.last_plan_canonicalized = False
            self.last_canonicalized_execution_plan = None

        for step in plan.steps:
            if step.action not in allowed_tools:
                raise ValueError(f"Planner selected disallowed tool {step.action!r}.")
            self.tools.validate_step(step.action, step.args)
        self._validate_explicit_database_targets(goal, plan)
        self._validate_shell_targets(plan)
        self._validate_explicit_tool_intent(plan, explicit_tool_intent)
        validate_plan_contract(plan, goal=goal)
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
