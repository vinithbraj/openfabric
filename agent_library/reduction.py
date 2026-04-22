import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

from agent_library.common import run_local_reducer_loop, serialize_for_stdin, shared_llm_api_settings

ALLOWED_REDUCTION_PREFIXES = (
    "python",
    "python3",
    "awk",
    "jq",
    "grep",
    "sed",
    "cut",
    "sort",
    "uniq",
    "wc",
    "tr",
    "head",
    "tail",
    "perl",
)

SLURM_LINE_COUNT_PRIMITIVES = {
    "slurm.jobs.queue_count",
    "slurm.jobs.history_count",
}
SLURM_PASS_THROUGH_PRIMITIVES = {
    "slurm.cluster.node_list",
    "slurm.cluster.partition_summary",
    "slurm.jobs.queue_list",
    "slurm.jobs.history_list",
    "slurm.jobs.details",
    "slurm.jobs.cancel",
    "slurm.jobs.hold",
    "slurm.jobs.release",
    "slurm.jobs.requeue",
    "slurm.jobs.resume",
    "slurm.jobs.suspend",
    "fallback_only",
}


@dataclass(frozen=True)
class ReductionExecutionResult:
    reduced_result: Any = None
    strategy: str = ""
    local_reduction_command: str = ""
    attempts: int = 0
    error: str = ""


def looks_like_safe_reducer_command(command: str) -> bool:
    compact = str(command or "").strip()
    if not compact:
        return False
    first_token = compact.split(None, 1)[0].strip().lower()
    return first_token.startswith(ALLOWED_REDUCTION_PREFIXES)


def generate_shell_reduction_command(
    task: str,
    original_cmd: str,
    sample_stdout: str,
    previous_command: str = "",
    previous_error: str = "",
) -> str:
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")
    if not api_key:
        return ""

    repair_context = ""
    if previous_error:
        repair_context = (
            "- The previous reducer failed. Generate a corrected command that avoids the prior issue.\n"
            f"- Previous command: {previous_command}\n"
            f"- Previous error: {previous_error}\n"
        )

    prompt = (
        "You are an expert shell data analyst. I have a large output from a shell command but I only want to "
        "send the relevant data back for analysis. Your job is to provide exactly ONE shell command "
        "(using awk, grep, tail, head, or jq) that will extract or calculate the necessary information "
        "from the FULL output locally.\n\n"
        f"User Intent: {task}\n"
        f"Original Command: {original_cmd}\n"
        "Sample Data (first few lines):\n"
        "```\n"
        f"{sample_stdout}\n"
        "```\n"
        "Instructions:\n"
        "- Return ONLY the shell command string, no markdown, no explanations.\n"
        "- The command will receive the full output via STDIN.\n"
        "- If the user wants a sum of a column, use awk.\n"
        "- If you cannot generate a reliable processing command, return 'NONE'.\n"
        f"{repair_context}"
        "Shell Command:"
    )

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a shell command generator. Return only the command."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        cmd = response.json()["choices"][0]["message"]["content"].strip().strip("`").strip()
        return cmd if cmd and cmd.upper() != "NONE" else ""
    except Exception:
        return ""


def should_reduce_sql_result(task: str, result: dict[str, Any]) -> bool:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    row_count = result.get("total_matching_rows", result.get("row_count"))
    row_total = row_count if isinstance(row_count, int) else len(rows)
    task_lc = str(task or "").lower()
    return (
        row_total > 20
        or len(json.dumps(rows[: min(len(rows), 25)], ensure_ascii=True, default=str)) > 5000
        or any(token in task_lc for token in ("list all", "full list", "all tables", "all rows", "every", "save it", "export"))
    )


def generate_sql_reduction_command(
    task: str,
    sql: str,
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    row_count: int,
    previous_command: str = "",
    previous_error: str = "",
) -> str:
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")
    if not api_key:
        return ""

    repair_context = ""
    if previous_error:
        repair_context = (
            "- The previous reducer failed. Generate a corrected command that avoids the prior issue.\n"
            f"- Previous command: {previous_command}\n"
            f"- Previous error: {previous_error}\n"
        )

    prompt = (
        "You are an expert local data reduction engineer.\n"
        "Generate exactly ONE shell command that reads the full SQL result JSON from STDIN and produces the reduced output needed for the user.\n\n"
        f"User request: {task}\n"
        f"SQL query: {sql}\n"
        f"Columns: {json.dumps(columns, ensure_ascii=True)}\n"
        f"Row count: {row_count}\n"
        "Sample rows:\n"
        "```json\n"
        f"{json.dumps(sample_rows, indent=2, ensure_ascii=True, default=str)}\n"
        "```\n"
        "Input JSON on STDIN has this shape:\n"
        '{"task":"...","sql":"...","columns":[...],"rows":[...],"row_count":123}\n'
        "Instructions:\n"
        "- Return ONLY the shell command string, no markdown, no explanations.\n"
        "- Prefer python3 -c for JSON processing; jq is also acceptable.\n"
        "- For inventory/list requests, print every requested item, one per line or as markdown bullets.\n"
        "- For count, aggregate, or filter requests, compute exactly from the JSON rows.\n"
        "- If you cannot generate a reliable reducer, return 'NONE'.\n"
        f"{repair_context}"
        "Reducer command:"
    )

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You generate one shell command only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        cmd = response.json()["choices"][0]["message"]["content"].strip().strip("`").strip()
        return cmd if cmd and cmd.upper() != "NONE" else ""
    except Exception:
        return ""


def summarize_sql_rows_locally(task: str, sql: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    column_names = [str(item).strip() for item in columns if isinstance(item, str) and str(item).strip()]
    if not column_names and isinstance(rows[0], dict):
        column_names = [str(key).strip() for key in rows[0].keys() if str(key).strip()]

    lines = [f"Returned {len(rows)} row(s)."]
    if column_names:
        lines.append("Columns: " + ", ".join(column_names[:12]))

    preview_chunks: list[str] = []
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        items = []
        for key, value in list(row.items())[:4]:
            compact = str(value).strip()
            if len(compact) > 60:
                compact = compact[:57].rstrip() + "..."
            items.append(f"{key}={compact}")
        if items:
            preview_chunks.append("; ".join(items))
    if preview_chunks:
        lines.append("Preview: " + " | ".join(preview_chunks))

    task_text = str(task or "").strip().lower()
    if any(token in task_text for token in ("count", "how many", "number of", "total")):
        lines.append(f"Local summary count: {len(rows)}")

    return "\n".join(lines)


def summarize_sql_rows(task: str, sql: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    return summarize_sql_rows_locally(task, sql, columns, rows)


def _progressive_text_samples(source_text: str) -> list[str]:
    text = str(source_text or "")
    if not text:
        return [""]
    limits = (5000, 12000, 25000)
    samples: list[str] = []
    for limit in limits:
        sample = text[:limit]
        if sample and sample not in samples:
            samples.append(sample)
        if len(text) <= limit:
            break
    if text not in samples and len(text) <= 25000:
        samples.append(text)
    return samples or [""]


def _progressive_row_samples(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return [[]]
    limits = (10, 25, 50)
    samples: list[list[dict[str, Any]]] = []
    for limit in limits:
        sample = rows[:limit]
        if sample and sample not in samples:
            samples.append(sample)
        if len(rows) <= limit:
            break
    if rows not in samples and len(rows) <= 50:
        samples.append(rows)
    return samples or [[]]


def looks_like_elapsed_summary_task(task: str) -> bool:
    text = str(task or "").strip().lower()
    if "sacct" not in text and not any(token in text for token in ("slurm", "job", "jobs", "partition")):
        return False
    duration_markers = (
        "how long",
        "took to complete",
        "take to complete",
        "total time",
        "total elapsed",
        "elapsed time",
        "duration",
        "average time",
        "avg time",
        "mean time",
        "longest",
        "shortest",
    )
    return any(marker in text for marker in duration_markers)


def looks_like_node_inventory_summary_task(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not any(token in text for token in ("slurm", "cluster", "sinfo", "scheduler")):
        return False
    has_nodes = any(token in text for token in ("node", "nodes", "nodelist", "compute node", "compute nodes"))
    asks_for_count = any(token in text for token in ("how many", "count", "number of", "total nodes", "total number"))
    asks_for_state = any(token in text for token in ("state", "states", "status", "statuses"))
    return has_nodes and asks_for_count and asks_for_state


def looks_like_job_id_list_task(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    mentions_ids = "job id" in text or re.search(r"\bjob ids\b", text) or re.search(r"\bids\b", text)
    mentions_jobs = any(token in text for token in ("job", "jobs", "slurm", "queue", "pending", "running"))
    return bool(mentions_ids and mentions_jobs)


def deterministic_line_count_reducer_command(label: str) -> str:
    script = f"""
import sys

lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
print("{label}: " + str(len(lines)))
""".strip()
    return f"python3 -c {shlex.quote(script)}"


def deterministic_state_breakdown_reducer_command(state_index: int) -> str:
    script = f"""
import sys
from collections import Counter

STATE_INDEX = {state_index}
counts = Counter()
for line in sys.stdin.read().splitlines():
    if not line.strip():
        continue
    cols = line.split("|")
    if len(cols) <= STATE_INDEX:
        continue
    state = cols[STATE_INDEX].strip().rstrip("*") or "UNKNOWN"
    counts[state] += 1

if not counts:
    print("No matching Slurm records were returned.")
    raise SystemExit(0)

lines_out = []
total = sum(counts.values())
lines_out.append(f"Total jobs: {{total}}")
for state, count in sorted(counts.items()):
    lines_out.append(f"State {{state}}: {{count}}")
print("\\n".join(lines_out))
""".strip()
    return f"python3 -c {shlex.quote(script)}"


def deterministic_elapsed_reducer_command(task: str) -> str:
    task_lc = str(task or "").lower()
    completed_only = any(token in task_lc for token in ("complete", "completed"))
    state_filter = "COMPLETED" if completed_only else ""
    script = f"""
import sys

STATE_FILTER = {state_filter!r}

def parse_elapsed(value: str):
    text = (value or "").strip()
    if not text:
        return None
    if text.lower() in {{"unknown", "n/a", "partition_limit", "infinite", "unlimited"}}:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        try:
            days = int(day_text)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    elif len(numbers) == 1:
        hours, minutes, seconds = 0, 0, numbers[0]
    else:
        return None
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds

def format_elapsed(total_seconds: int) -> str:
    total_seconds = int(total_seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{{days}}-{{hours:02d}}:{{minutes:02d}}:{{seconds:02d}}"
    return f"{{hours:02d}}:{{minutes:02d}}:{{seconds:02d}}"

text = sys.stdin.read()
lines = [line for line in text.splitlines() if line.strip()]
if not lines:
    print("No Slurm job records were returned.")
    raise SystemExit(0)

header = lines[0].split("|")
header_map = {{name.strip().lower(): index for index, name in enumerate(header)}}
elapsed_index = header_map.get("elapsed")
state_index = header_map.get("state")
job_id_index = header_map.get("jobid")
job_name_index = header_map.get("jobname")
if elapsed_index is None:
    print("Could not find an Elapsed column in the Slurm output.")
    raise SystemExit(0)

total_seconds = 0
count = 0
skipped = 0
longest = None
shortest = None
for line in lines[1:]:
    cols = line.split("|")
    if elapsed_index >= len(cols):
        skipped += 1
        continue
    state = cols[state_index].strip().upper() if state_index is not None and state_index < len(cols) else ""
    if STATE_FILTER and state != STATE_FILTER:
        continue
    seconds = parse_elapsed(cols[elapsed_index])
    if seconds is None:
        skipped += 1
        continue
    total_seconds += seconds
    count += 1
    job_id = cols[job_id_index].strip() if job_id_index is not None and job_id_index < len(cols) else ""
    job_name = cols[job_name_index].strip() if job_name_index is not None and job_name_index < len(cols) else ""
    record = (seconds, job_id, job_name)
    if longest is None or seconds > longest[0]:
        longest = record
    if shortest is None or seconds < shortest[0]:
        shortest = record

if count == 0:
    if STATE_FILTER:
        print(f"No matching Slurm jobs were found for state {{STATE_FILTER}}.")
    else:
        print("No matching Slurm jobs were found.")
    raise SystemExit(0)

average_seconds = total_seconds // count
lines_out = [
    f"Jobs considered: {{count}}",
    f"Total elapsed time: {{format_elapsed(total_seconds)}}",
    f"Average elapsed time: {{format_elapsed(average_seconds)}}",
]
if longest is not None:
    lines_out.append(f"Longest elapsed job: {{longest[1] or longest[2] or 'unknown'}} ({{format_elapsed(longest[0])}})")
if shortest is not None:
    lines_out.append(f"Shortest elapsed job: {{shortest[1] or shortest[2] or 'unknown'}} ({{format_elapsed(shortest[0])}})")
if skipped:
    lines_out.append(f"Skipped rows without parseable elapsed values: {{skipped}}")
print("\\n".join(lines_out))
""".strip()
    return f"python3 -c {shlex.quote(script)}"


def deterministic_node_inventory_reducer_command(_task: str) -> str:
    script = """
import sys
from collections import Counter

text = sys.stdin.read()
lines = [line.strip() for line in text.splitlines() if line.strip()]
if not lines:
    print("No Slurm node records were returned.")
    raise SystemExit(0)

state_counts = Counter()
unique_nodes = set()

for line in lines:
    if "|" not in line:
        continue
    node, state = line.split("|", 1)
    node = node.strip()
    state = state.strip().rstrip("*").lower()
    if not node:
        continue
    if node in unique_nodes:
        continue
    unique_nodes.add(node)
    state_counts[state or "unknown"] += 1

if not unique_nodes:
    print("No Slurm node records were returned.")
    raise SystemExit(0)

lines_out = [f"Total nodes: {len(unique_nodes)}"]
for state, count in sorted(state_counts.items()):
    lines_out.append(f"State {state}: {count}")
print("\\n".join(lines_out))
""".strip()
    return f"python3 -c {shlex.quote(script)}"


def deterministic_job_id_list_reducer_command() -> str:
    script = """
import sys

job_ids = []
seen = set()
for line in sys.stdin.read().splitlines():
    compact = line.strip()
    if not compact:
        continue
    job_id = compact.split("|", 1)[0].strip()
    if not job_id or job_id in seen:
        continue
    seen.add(job_id)
    job_ids.append(job_id)

if not job_ids:
    print("No matching job IDs were returned.")
    raise SystemExit(0)

print("\\n".join(job_ids))
""".strip()
    return f"python3 -c {shlex.quote(script)}"


def build_shell_reduction_request(task: str, original_command: str, stdout: str) -> dict[str, Any] | None:
    if not isinstance(stdout, str) or len(stdout) < 5000 or not str(task or "").strip():
        return None
    return {
        "kind": "shell.local_reducer",
        "task": task,
        "source_command": original_command,
        "sample": stdout[:5000],
        "input_format": "text",
    }


def build_sql_reduction_request(task: str, sql: str, result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict) or not isinstance(result.get("rows"), list) or not result.get("rows"):
        return None
    rows = result.get("rows", [])
    columns = result.get("columns", [])
    row_count = result.get("total_matching_rows", result.get("row_count", len(rows)))
    normalized_row_count = row_count if isinstance(row_count, int) else len(rows)
    if should_reduce_sql_result(task, result) or len(rows) > 5:
        return {
            "kind": "sql.local_reducer",
            "task": task,
            "source_sql": sql,
            "columns": columns,
            "row_count": normalized_row_count,
            "sample_rows": rows[:10],
            "input_format": "json",
        }
    return None


def build_slurm_reduction_request(
    task: str,
    command: str,
    stdout: str,
    stderr: str,
    *,
    primitive_id: str = "",
    natural_language_request: bool = False,
) -> dict[str, Any] | None:
    if not natural_language_request or not str(task or "").strip():
        return None

    normalized_primitive_id = str(primitive_id or "").strip()
    if normalized_primitive_id in SLURM_LINE_COUNT_PRIMITIVES:
        return {
            "kind": "slurm.line_count",
            "task": task,
            "source_command": command,
            "label": "Matching jobs",
            "input_format": "text",
        }
    if normalized_primitive_id == "slurm.jobs.queue_state_breakdown":
        return {
            "kind": "slurm.state_breakdown",
            "task": task,
            "source_command": command,
            "state_index": 2,
            "input_format": "text",
        }
    if looks_like_job_id_list_task(task):
        return {
            "kind": "slurm.job_id_list",
            "task": task,
            "source_command": command,
            "input_format": "text",
        }
    if normalized_primitive_id == "slurm.cluster.node_inventory_summary" or (
        not normalized_primitive_id and looks_like_node_inventory_summary_task(task)
    ):
        return {
            "kind": "slurm.node_inventory_summary",
            "task": task,
            "source_command": command,
            "input_format": "text",
        }
    if normalized_primitive_id == "slurm.jobs.elapsed_summary" or (
        not normalized_primitive_id and looks_like_elapsed_summary_task(task)
    ):
        return {
            "kind": "slurm.elapsed_summary",
            "task": task,
            "source_command": command,
            "input_format": "text",
        }
    if normalized_primitive_id in SLURM_PASS_THROUGH_PRIMITIVES:
        return {
            "kind": "pass_through",
            "task": task,
            "source_command": command,
            "source_field": "stdout",
            "input_format": "text",
        }
    if isinstance(stdout, str) and stdout.strip():
        return {
            "kind": "pass_through",
            "task": task,
            "source_command": command,
            "source_field": "stdout",
            "input_format": "text",
        }
    if isinstance(stderr, str) and stderr.strip():
        return {
            "kind": "pass_through",
            "task": task,
            "source_command": command,
            "source_field": "stderr",
            "input_format": "text",
        }
    return None


def execute_reduction_request(request: dict[str, Any], input_data: Any) -> ReductionExecutionResult:
    if not isinstance(request, dict):
        return ReductionExecutionResult(error="Missing reduction request.")

    kind = str(request.get("kind") or "").strip()
    if not kind:
        return ReductionExecutionResult(error="Missing reduction request kind.")

    if kind == "pass_through":
        if input_data in (None, "", [], {}):
            return ReductionExecutionResult(strategy="pass_through", error="No reducer input was available.")
        return ReductionExecutionResult(reduced_result=input_data, strategy="pass_through")

    if kind == "shell.local_reducer":
        sample = str(request.get("sample") or "")
        task = str(request.get("task") or "")
        source_command = str(request.get("source_command") or "")
        sample_candidates = _progressive_text_samples(
            serialize_for_stdin(input_data) or sample or ""
        )
        sample_state = {"index": 0}

        def _next_shell_reducer(previous_command: str, previous_error: str) -> str:
            if (previous_command or previous_error) and sample_state["index"] < len(sample_candidates) - 1:
                sample_state["index"] += 1
            return generate_shell_reduction_command(
                task,
                source_command,
                sample_candidates[sample_state["index"]],
                previous_command,
                previous_error,
            )

        reduction = run_local_reducer_loop(
            input_data,
            _next_shell_reducer,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="planned_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    if kind == "sql.local_reducer":
        task = str(request.get("task") or "")
        source_sql = str(request.get("source_sql") or "")
        columns = [str(item) for item in request.get("columns", [])] if isinstance(request.get("columns"), list) else []
        sample_rows = request.get("sample_rows", [])
        if not isinstance(sample_rows, list):
            sample_rows = []
        row_count = request.get("row_count")
        normalized_row_count = int(row_count) if isinstance(row_count, int) else len(sample_rows)
        full_rows = []
        if isinstance(input_data, dict):
            raw_rows = input_data.get("rows")
            if isinstance(raw_rows, list):
                full_rows = [row for row in raw_rows if isinstance(row, dict)]
        row_samples = _progressive_row_samples(full_rows or sample_rows)
        row_sample_state = {"index": 0}

        def _next_sql_reducer(previous_command: str, previous_error: str) -> str:
            if (previous_command or previous_error) and row_sample_state["index"] < len(row_samples) - 1:
                row_sample_state["index"] += 1
            return generate_sql_reduction_command(
                task,
                source_sql,
                columns,
                row_samples[row_sample_state["index"]],
                normalized_row_count,
                previous_command,
                previous_error,
            )

        reduction = run_local_reducer_loop(
            input_data,
            _next_sql_reducer,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="planned_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    if kind == "sql.summary":
        source_sql = str(request.get("source_sql") or "")
        columns = [str(item) for item in request.get("columns", [])] if isinstance(request.get("columns"), list) else []
        rows = input_data.get("rows") if isinstance(input_data, dict) else None
        if not isinstance(rows, list):
            return ReductionExecutionResult(strategy="summary", error="SQL summary request did not receive row data.")
        summary = summarize_sql_rows_locally(str(request.get("task") or ""), source_sql, columns, rows)
        return ReductionExecutionResult(
            reduced_result=summary or None,
            strategy="local_summary_compatibility",
            attempts=1 if summary else 0,
            error="" if summary else "SQL summary generation produced no output.",
        )

    if kind == "slurm.line_count":
        label = str(request.get("label") or "Matching jobs")
        reduction = run_local_reducer_loop(
            input_data,
            lambda previous_command, previous_error: deterministic_line_count_reducer_command(label),
            max_attempts=1,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="deterministic_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    if kind == "slurm.state_breakdown":
        state_index = request.get("state_index")
        normalized_state_index = int(state_index) if isinstance(state_index, int) else 2
        reduction = run_local_reducer_loop(
            input_data,
            lambda previous_command, previous_error: deterministic_state_breakdown_reducer_command(normalized_state_index),
            max_attempts=1,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="deterministic_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    if kind == "slurm.node_inventory_summary":
        reduction = run_local_reducer_loop(
            input_data,
            lambda previous_command, previous_error: deterministic_node_inventory_reducer_command(str(request.get("task") or "")),
            max_attempts=1,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="deterministic_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    if kind == "slurm.job_id_list":
        reduction = run_local_reducer_loop(
            input_data,
            lambda previous_command, previous_error: deterministic_job_id_list_reducer_command(),
            max_attempts=1,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="deterministic_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    if kind == "slurm.elapsed_summary":
        reduction = run_local_reducer_loop(
            input_data,
            lambda previous_command, previous_error: deterministic_elapsed_reducer_command(str(request.get("task") or "")),
            max_attempts=1,
            validate_output=lambda output: bool(output.strip()),
        )
        return ReductionExecutionResult(
            reduced_result=reduction.output or None,
            strategy="deterministic_local_reduction_command",
            local_reduction_command=reduction.command,
            attempts=reduction.attempts,
            error=reduction.error,
        )

    return ReductionExecutionResult(error=f"Unsupported reduction request kind: {kind}")
