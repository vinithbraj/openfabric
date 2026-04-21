import json
import os
import re
from typing import Any

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, shared_llm_api_settings, with_node_envelope
from runtime.console import log_debug

app = FastAPI()

AGENT_METADATA = {
    "description": "Builds final user-facing answers from tool results.",
    "capability_domains": ["response_synthesis", "final_answer"],
    "action_verbs": ["summarize", "format", "respond"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
    "methods": [
        {
            "name": "synthesize_file_result",
            "event": "file.content",
            "when": "Converts file content into final answer.",
        },
        {
            "name": "synthesize_shell_result",
            "event": "shell.result",
            "when": "Converts shell execution result into final answer.",
        },
        {
            "name": "synthesize_notify_result",
            "event": "notify.result",
            "when": "Converts notify result into final answer.",
        },
        {
            "name": "synthesize_sql_result",
            "event": "sql.result",
            "when": "Converts SQL schema/query results into final answer.",
        },
        {
            "name": "synthesize_slurm_result",
            "event": "slurm.result",
            "when": "Converts Slurm command results into final answer.",
        },
        {
            "name": "synthesize_task_result",
            "event": "task.result",
            "when": "Converts generic task result into final answer.",
        },
        {
            "name": "synthesize_workflow_result",
            "event": "workflow.result",
            "when": "Converts aggregated multi-step workflow results into final answer.",
        },
        {
            "name": "synthesize_clarification_required",
            "event": "clarification.required",
            "when": "Converts clarification requests into a user-facing follow-up question.",
        },
    ],
}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("SYNTH_DEBUG", message)


ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_terminal_output(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = ANSI_PATTERN.sub("", value)
    text = text.replace("\r", "\n")
    while "\b" in text:
        text = re.sub(r".\x08", "", text)
        text = text.replace("\b", "")
    text = CONTROL_PATTERN.sub("", text)
    return text.strip()


def _truncate_if_large(value: Any, limit: int = 5000) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n\n[... Truncated for synthesis. Actual size: {len(value)} chars. Raw data is visible in the technical details section above. ...]"


def _format_preformatted_block(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    if not text:
        return [">"]
    return [(f"> {line}" if line else ">") for line in text.splitlines()]


def _format_console_section(label: str, value: Any) -> list[str]:
    text = "" if value is None else str(value).rstrip()
    if not text:
        return []
    return [f"**{label}:**", "", "```text", text, "```"]


def _format_callout_section(label: str, value: Any) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    if "\n" not in text:
        return [f"**{label}:** `{text}`"]
    return _format_console_section(label, text)


def _markdown_section(title: str, body: Any, level: int = 3) -> str:
    text = ""
    if isinstance(body, list):
        text = "\n".join(str(item) for item in body if str(item).strip()).strip()
    elif body is not None:
        text = str(body).strip()
    if not title or not text:
        return text
    hashes = "#" * max(1, min(level, 6))
    return f"{hashes} {title}\n\n{text}"


def _join_markdown_sections(*sections: Any) -> str:
    parts = [str(section).strip() for section in sections if isinstance(section, str) and section.strip()]
    return "\n\n".join(parts).strip()


def _extract_file_paths_from_text(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    lines = [line.strip() for line in _clean_terminal_output(value).splitlines() if line.strip()]
    if not lines or len(lines) > 20:
        return []
    linked = []
    for line in lines:
        link = _markdown_link_for_file_path(line)
        if not link:
            return []
        linked.append(link)
    return linked


def _extract_labeled_paths_from_text(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    linked = []
    for line in _clean_terminal_output(value).splitlines():
        match = re.match(r"\s*([^:]{1,40}):\s+(.+?)\s*$", line)
        if not match:
            continue
        label = match.group(1).strip()
        path_text = match.group(2).strip()
        link = _markdown_link_for_file_path(path_text)
        if link:
            linked.append(f"- {label}: {link}")
    return linked


def _extract_path_values(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    lines = [line.strip() for line in _clean_terminal_output(value).splitlines() if line.strip()]
    if not lines or len(lines) > 20:
        return []
    paths: list[str] = []
    for line in lines:
        candidate = _candidate_file_path(line)
        if candidate:
            paths.append(candidate)
            continue
        match = re.match(r"\s*[^:]{1,40}:\s+(.+?)\s*$", line)
        if not match:
            return []
        candidate = _candidate_file_path(match.group(1).strip())
        if not candidate:
            return []
        paths.append(candidate)
    seen: set[str] = set()
    unique: list[str] = []
    for item in paths:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _command_output_details(command: str, stdout: str, stderr: str, returncode: int | None = None) -> str:
    clean_stdout = _clean_terminal_output(stdout) or "<empty>"
    clean_stderr = _clean_terminal_output(stderr)
    linked_paths = _extract_file_paths_from_text(stdout)
    labeled_paths = _extract_labeled_paths_from_text(stdout)

    lines = _format_console_section("Command", command.strip())
    if returncode is not None:
        status_text = "success" if returncode == 0 else f"error (exit {returncode})"
        lines.extend(["", f"**Status:** `{status_text}`"])

    if linked_paths or labeled_paths:
        lines.extend(["", "**Files**"])
        lines.extend([f"- {path}" for path in linked_paths])
        lines.extend(labeled_paths)

    lines.extend([""])
    lines.extend(_format_console_section("Output", clean_stdout))

    if clean_stderr:
        lines.extend([""])
        lines.extend(_format_console_section("Error output", clean_stderr))
    return "\n".join(lines)


def _format_agent_result(payload: dict[str, Any], event_name: str | None = None) -> str:
    """Generic formatter for standard agent result payloads."""
    # 1. Raw outputs
    stdout = payload.get("stdout") or payload.get("stdout_excerpt") or payload.get("content")
    stderr = payload.get("stderr") or payload.get("stderr_excerpt")
    if not stdout and isinstance(payload.get("result"), dict):
        stdout = payload["result"].get("stdout") or payload["result"].get("stdout_excerpt")
        stderr = payload["result"].get("stderr") or payload["result"].get("stderr_excerpt")

    # 2. Primary human-readable answer (priority)
    refined = payload.get("reduced_result") or payload.get("refined_answer")
    if not refined and isinstance(payload.get("result"), dict):
        refined = (
            payload["result"].get("reduced_result")
            or payload["result"].get("refined_answer")
        )
    if not refined:
        detail = payload.get("detail")
        if not stdout and isinstance(detail, str) and detail.strip():
            refined = detail
        elif not stdout and isinstance(payload.get("result"), dict):
            nested_detail = payload["result"].get("detail")
            if isinstance(nested_detail, str) and nested_detail.strip():
                refined = nested_detail

    # 3. Command/Operation
    op = payload.get("command") or payload.get("sql") or payload.get("task")
    if not op and isinstance(payload.get("result"), dict):
        op = payload["result"].get("command") or payload["result"].get("sql")

    # 4. Status
    rc = payload.get("returncode")
    if rc is None and isinstance(payload.get("result"), dict):
        rc = payload["result"].get("returncode")
    
    ok = payload.get("ok")
    if ok is None and isinstance(payload.get("result"), dict):
        ok = payload["result"].get("ok")
    
    if ok is False or (rc is not None and rc != 0):
        status = "failure"
    else:
        status = "success"

    # 5. Build the UI
    if refined and isinstance(refined, str) and refined.strip():
        # If we have a nice refined answer, keep it clean
        return refined.strip()

    linked_stdout_path = _markdown_link_for_file_path(stdout)
    if linked_stdout_path and not stderr and status == "success":
        return _markdown_section("Summary", f"File path: {linked_stdout_path}")

    # Fallback for raw tool results
    summary = _markdown_section(
        "Summary",
        f"Status: `{status}`" + (f"\n\nSource: `{event_name}`" if isinstance(event_name, str) and event_name.strip() else ""),
    )
    details = _markdown_section(
        "Details",
        _command_output_details(str(op or "unknown"), str(stdout or ""), str(stderr or ""), rc),
    )
    return _join_markdown_sections(summary, details)


def _format_shell_answer(command: str, returncode: int, stdout: str, stderr: str) -> str:
    return _format_agent_result({
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr
    }, "shell.result")


def _flatten_workflow_steps(steps: list[dict[str, Any]], prefix: str = "") -> list[dict[str, Any]]:
    flattened = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id", "step"))
        display_id = f"{prefix}.{step_id}" if prefix else step_id
        entry = dict(step)
        entry["display_id"] = display_id
        flattened.append(entry)
        nested = step.get("steps")
        if isinstance(nested, list):
            flattened.extend(_flatten_workflow_steps(nested, display_id))
    return flattened


def _markdown_table(columns: list[str], rows: list[dict[str, Any]], limit: int | None = None) -> str:
    if not columns:
        return ""
    display_rows = rows[:limit] if isinstance(limit, int) else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display_rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            linked_path = _markdown_link_for_file_path(text)
            if linked_path:
                values.append(linked_path)
            else:
                values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _path_markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _candidate_file_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\n" in text:
        return None
    if text.startswith(("http://", "https://", "app://", "file://")):
        return None
    candidate = text.strip("`")
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
        candidate = candidate[1:-1].strip()
    if not candidate or candidate.endswith(("/", "\\")):
        return None
    normalized = os.path.expanduser(candidate)
    has_separator = "/" in normalized or "\\" in normalized
    if not has_separator and not normalized.startswith("."):
        return None
    basename = os.path.basename(normalized.replace("\\", "/"))
    if not basename or basename in {".", ".."}:
        return None
    absolute_candidate = os.path.abspath(normalized)
    if not os.path.isfile(absolute_candidate):
        return None
    return candidate


def _markdown_link_for_file_path(value: Any) -> str | None:
    candidate = _candidate_file_path(value)
    if not candidate:
        return None
    absolute_target = os.path.abspath(os.path.expanduser(candidate))
    target = f"<{absolute_target}>" if " " in absolute_target else absolute_target
    return f"[{_path_markdown_label(candidate)}]({target})"


def _extract_sql_results_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for step in _flatten_workflow_steps(steps):
        payload = step.get("payload")
        if not isinstance(payload, dict) or step.get("event") != "sql.result":
            continue
        result = payload.get("result")
        if isinstance(result, dict):
            results.append(
                {
                    "step": step,
                    "sql": payload.get("sql") or result.get("sql"),
                    "queries": result.get("queries"),
                    "columns": result.get("columns", []),
                    "rows": result.get("rows", []),
                    "row_count": result.get("row_count"),
                    "returned_row_count": result.get("returned_row_count"),
                    "total_matching_rows": result.get("total_matching_rows"),
                    "truncated": result.get("truncated"),
                    "limit": result.get("limit"),
                }
            )
    return results


def _extract_slurm_results_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for step in _flatten_workflow_steps(steps):
        payload = step.get("payload")
        if not isinstance(payload, dict) or step.get("event") != "slurm.result":
            continue
        results.append(payload)
    return results


def _extract_general_results_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extracts any successful agent results that follow the standard pattern."""
    results = []
    for step in _flatten_workflow_steps(steps):
        payload = step.get("payload")
        event = step.get("event")
        if not isinstance(payload, dict) or not event:
            continue
        if event in {"slurm.result", "shell.result", "task.result", "file.content"}:
            results.append((event, payload))
    return results


def _step_clean_stdout(step: dict[str, Any]) -> str:
    stdout = _step_payload_field(step, "stdout") or _step_payload_field(step, "stdout_excerpt")
    if isinstance(stdout, dict):
        excerpt = stdout.get("excerpt")
        stdout = excerpt if isinstance(excerpt, str) else ""
    return _clean_terminal_output(stdout)


def _shell_fact_value(text: str) -> str | None:
    clean = _clean_terminal_output(text)
    if not clean:
        return None
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    value = lines[0]
    if value.startswith(("[", "{")):
        return None
    if value.lower() in {"true", "false"}:
        return "Yes" if value.lower() == "true" else "No"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return value
    if len(value) <= 80:
        return value
    return None


def _shell_fact_label(step: dict[str, Any]) -> str:
    task = _clean_shell_task_text(step.get("task") or "")
    task_lc = task.lower()
    if "docker" in task_lc and any(token in task_lc for token in ("installed", "available")):
        return "Docker installed"
    if "count" in task_lc and "container" in task_lc:
        return "Container count"
    if "count" in task_lc and "image" in task_lc:
        return "Image count"
    if "count" in task_lc and "python" in task_lc and "file" in task_lc:
        return "Python file count"
    if "current branch" in task_lc:
        return "Current branch"
    if "last commit message" in task_lc:
        return "Last commit message"
    if "working tree clean" in task_lc:
        return "Working tree clean"
    if "free space" in task_lc and "gb" in task_lc:
        return "Total usable free space (GB)"
    if "free space" in task_lc:
        return "Total usable free space"
    if "count" in task_lc:
        return "Count"
    return task[:1].upper() + task[1:] if task else "Result"


def _clean_shell_task_text(task: Any) -> str:
    text = re.sub(r"\s+", " ", str(task or "").strip(" ,;:."))
    if not text:
        return ""
    text = re.sub(r"^(?:and|then|also)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:and|then|also)\s*$", "", text, flags=re.IGNORECASE)
    return text.strip(" ,;:.")


def _shell_detail_heading(step: dict[str, Any]) -> str:
    task = _clean_shell_task_text(step.get("task") or "")
    task_lc = task.lower()
    if "nvidia-smi" in task_lc and any(
        token in task_lc for token in ("gpu", "gpus", "spec", "specs", "detail", "details", "cuda", "driver", "memory", "vram")
    ):
        return "GPU details from nvidia-smi"
    if "docker" in task_lc and "container" in task_lc and "count" in task_lc:
        return "Docker container count"
    if "docker" in task_lc and "image" in task_lc and "count" in task_lc:
        return "Docker image count"
    if task:
        return task[:1].upper() + task[1:]
    return "Details"


def _format_multi_shell_workflow_answer(steps: list[dict[str, Any]], task: str = "") -> str | None:
    facts: list[tuple[str, str]] = []
    for step in steps:
        if step.get("status") != "completed" or step.get("event") != "shell.result":
            continue
        value = _shell_fact_value(_step_clean_stdout(step))
        if value is None:
            continue
        facts.append((_shell_fact_label(step), value))
    if not facts:
        return None
    summary_lines = [f"- **{label}:** `{value}`" for label, value in facts]
    return _markdown_section("Summary", summary_lines)


def _format_size_path_table_answer(text: Any, intro: str = "") -> str | None:
    clean = _clean_terminal_output(text)
    if not clean:
        return None
    rows: list[dict[str, Any]] = []
    for line in clean.splitlines():
        compact = line.strip()
        if not compact:
            continue
        match = re.match(r"^(\d+)\s+(.+)$", compact)
        if not match:
            return None
        rows.append({"Size (bytes)": match.group(1), "Path": match.group(2).strip()})
    if not rows:
        return None
    lines: list[str] = []
    if isinstance(intro, str) and intro.strip():
        lines.extend([f"#### {intro.strip()}", ""])
    lines.append(_markdown_table(["Size (bytes)", "Path"], rows, limit=50))
    return "\n".join(lines).strip()


def _format_labeled_stdout_answer(text: Any, intro: str = "") -> str | None:
    clean = _clean_terminal_output(text)
    if not clean:
        return None
    pairs: list[tuple[str, str]] = []
    for line in clean.splitlines():
        compact = line.strip()
        if not compact:
            continue
        if ":" not in compact:
            return None
        label, value = compact.split(":", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            return None
        pairs.append((label, value))
    if not pairs:
        return None
    lines: list[str] = []
    if isinstance(intro, str) and intro.strip():
        lines.extend([f"#### {intro.strip()}", ""])
    lines.extend(f"- **{label}:** `{value}`" for label, value in pairs)
    return "\n".join(lines).strip()


def _format_shell_detail_answer(step: dict[str, Any]) -> str | None:
    stdout = _step_clean_stdout(step)
    if not stdout:
        return None
    intro = _shell_detail_heading(step)
    rendered = _format_labeled_stdout_answer(stdout, intro)
    if rendered:
        return rendered
    rendered = _format_fixed_width_table_answer(stdout, intro)
    if rendered:
        return rendered
    rendered = _format_size_path_table_answer(stdout, intro)
    if rendered:
        return rendered
    linked_paths = _extract_file_paths_from_text(stdout)
    if linked_paths:
        lines = [f"#### {intro}"] if intro else []
        if lines:
            lines.append("")
        lines.extend(f"- {path}" for path in linked_paths)
        return "\n".join(lines).strip()
    return None


def _shell_step_prefers_detail_render(step: dict[str, Any]) -> bool:
    task = str(step.get("task") or "").strip().lower()
    stdout = _step_clean_stdout(step)
    if "\n" not in stdout:
        return False
    return any(token in task for token in ("gpu", "gpus", "spec", "specs", "detail", "details", "driver", "cuda", "memory", "vram"))


def _format_compound_shell_workflow_answer(steps: list[dict[str, Any]], task: str = "") -> str | None:
    facts: list[tuple[str, str]] = []
    detail_sections: list[str] = []
    for step in steps:
        if step.get("status") != "completed" or step.get("event") != "shell.result":
            continue
        if _shell_step_prefers_detail_render(step):
            rendered = _format_shell_detail_answer(step)
            if not rendered:
                reduced = _step_payload_field(step, "reduced_result") or _step_payload_field(step, "refined_answer")
                if isinstance(reduced, str) and reduced.strip():
                    rendered = _format_labeled_stdout_answer(reduced, _shell_detail_heading(step))
                    if not rendered:
                        intro = _shell_detail_heading(step)
                        body = ["```text", reduced.strip(), "```"]
                        rendered = _join_markdown_sections(
                            _markdown_section(intro, "\n".join(body), level=4) if intro else "\n".join(body)
                        )
            if rendered:
                detail_sections.append(rendered)
                continue
        reduced = _step_payload_field(step, "reduced_result") or _step_payload_field(step, "refined_answer")
        value = _shell_fact_value(reduced) if isinstance(reduced, str) else None
        if value is None:
            value = _shell_fact_value(_step_clean_stdout(step))
        if value is not None:
            facts.append((_shell_fact_label(step), value))
            continue
        rendered = _format_shell_detail_answer(step)
        if rendered:
            detail_sections.append(rendered)
    if not facts and not detail_sections:
        return None
    summary_section = ""
    details_section = ""
    if facts:
        summary_section = _markdown_section(
            "Summary",
            [f"- **{label}:** `{value}`" for label, value in facts],
        )
    if detail_sections:
        details_section = _markdown_section("Details", "\n\n".join(detail_sections))
    return _join_markdown_sections(summary_section, details_section)


def _workflow_requests_internal_steps(payload: dict[str, Any]) -> bool:
    presentation = payload.get("presentation")
    return isinstance(presentation, dict) and bool(presentation.get("include_internal_steps"))


def _workflow_presentation_format(payload: dict[str, Any]) -> str:
    presentation = payload.get("presentation")
    if not isinstance(presentation, dict):
        return ""
    value = presentation.get("format")
    return value.strip().lower() if isinstance(value, str) else ""


def _step_payload_field(step: dict[str, Any], field: str) -> Any:
    if field in step:
        return step.get(field)
    payload = step.get("payload")
    if isinstance(payload, dict) and field in payload:
        return payload.get(field)
    evidence = step.get("evidence")
    if isinstance(evidence, dict):
        evidence_payload = evidence.get("payload")
        if isinstance(evidence_payload, dict) and field in evidence_payload:
            return evidence_payload.get(field)
    return None


def _extract_fixed_width_columns(header_line: str) -> list[tuple[str, int]]:
    columns = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\s{2,}|$)", header_line):
        label = match.group(0).strip()
        if label:
            columns.append((label, match.start()))
    return columns


def _parse_fixed_width_table(text: Any) -> tuple[list[str], list[dict[str, Any]]] | None:
    clean = _clean_terminal_output(text)
    if not clean:
        return None
    lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header_line = lines[0]
    column_specs = _extract_fixed_width_columns(header_line)
    if len(column_specs) < 2:
        return None
    columns = [label for label, _ in column_specs]
    starts = [start for _, start in column_specs]
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        row: dict[str, Any] = {}
        non_empty_cells = 0
        for index, (label, start) in enumerate(column_specs):
            end = starts[index + 1] if index + 1 < len(starts) else None
            cell = line[start:end].strip() if end is not None else line[start:].strip()
            row[label] = cell
            if cell:
                non_empty_cells += 1
        if non_empty_cells:
            rows.append(row)
    if not rows:
        return None
    return columns, rows


def _format_fixed_width_table_answer(text: Any, intro: str = "") -> str | None:
    parsed = _parse_fixed_width_table(text)
    if not parsed:
        return None
    columns, rows = parsed
    lines = []
    if isinstance(intro, str) and intro.strip():
        lines.extend([f"#### {intro.strip()}", ""])
    lines.append(_markdown_table(columns, rows, limit=50))
    if len(rows) > 50:
        lines.extend(["", "Showing first 50 rows."])
    return "\n".join(lines).strip()


def _format_sql_result_answer(result: dict[str, Any], task: str = "") -> str:
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    row_count = result.get("total_matching_rows", result.get("row_count"))
    returned_row_count = result.get("returned_row_count")
    if not isinstance(returned_row_count, int):
        returned_row_count = len(rows) if isinstance(rows, list) else 0
    truncated = result.get("truncated")
    if not isinstance(truncated, bool):
        truncated = isinstance(row_count, int) and row_count > returned_row_count
    sql = result.get("sql", "")
    queries = result.get("queries")
    limit = result.get("limit")

    summary_lines = []
    if isinstance(row_count, int):
        summary_lines.append(f"Found {row_count} matching result row(s).")
    elif task:
        summary_lines.append(f"Returned {returned_row_count} result row(s).")
    elif returned_row_count:
        summary_lines.append(f"Returned {returned_row_count} result row(s).")

    detail_lines = []
    if isinstance(columns, list) and isinstance(rows, list) and columns:
        table_limit = 50
        detail_lines.append(_markdown_table(columns, rows, table_limit))
        if len(rows) > table_limit:
            if isinstance(row_count, int):
                detail_lines.extend(["", f"Showing first {table_limit} of {returned_row_count} returned row(s)."])
            else:
                detail_lines.extend(["", f"Showing first {table_limit} rows."])
        elif truncated:
            if isinstance(row_count, int):
                detail_lines.extend(["", f"Showing {returned_row_count} returned row(s) out of {row_count} matching row(s)."])
            else:
                detail_lines.extend(["", f"Showing up to {limit if isinstance(limit, int) else returned_row_count} rows."])
    else:
        # Check if this is actually a schema object masquerading as a plain result
        schema_check = result if isinstance(result, dict) else {}
        if "dialect" in schema_check and "tables" in schema_check:
            detail_lines.append(_format_schema_answer(schema_check))
        else:
            detail_lines.append("```json\n" + json.dumps(result, indent=2, ensure_ascii=True) + "\n```")

    sql_sections: list[str] = []
    if isinstance(queries, list) and queries:
        sql_lines = []
        for index, query in enumerate(queries, start=1):
            if not isinstance(query, dict):
                continue
            query_sql = query.get("sql")
            if not isinstance(query_sql, str) or not query_sql.strip():
                continue
            label = query.get("label") or f"Query {index}"
            sql_lines.extend([f"{index}. {label}", "", "```sql", query_sql.strip(), "```", ""])
        if sql_lines:
            sql_sections.append(_markdown_section("SQL Used", "\n".join(sql_lines).strip()))
    elif isinstance(sql, str) and sql.strip():
        sql_sections.append(_markdown_section("SQL Used", "\n".join(["```sql", sql.strip(), "```"])))

    return _join_markdown_sections(
        _markdown_section("Summary", summary_lines) if summary_lines else "",
        _markdown_section("Details", "\n".join(detail_lines)) if detail_lines else "",
        *sql_sections,
    )


def _extract_scalar_sql_value(result: dict[str, Any]) -> Any:
    if not isinstance(result, dict):
        return None
    rows = result.get("rows")
    columns = result.get("columns")
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    if not isinstance(columns, list) or len(columns) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict):
        return None
    return row.get(columns[0])


def _format_multi_sql_workflow_answer(results: list[dict[str, Any]], task: str = "") -> str | None:
    if len(results) < 2:
        return None
    scalar_lines = []
    table_result = None
    for item in results:
        scalar_value = _extract_scalar_sql_value(item)
        if scalar_value is not None:
            step = item.get("step") if isinstance(item.get("step"), dict) else {}
            label = str(step.get("task") or task or "Count").strip()
            scalar_lines.append(f"{label}: `{scalar_value}`")
        elif table_result is None and isinstance(item.get("rows"), list) and item.get("rows"):
            table_result = item
    if table_result is None:
        return None
    table_answer = _format_sql_result_answer(table_result, "")
    summary_section = _markdown_section("Summary", [f"- {line}" for line in scalar_lines]) if scalar_lines else ""
    return _join_markdown_sections(summary_section, table_answer) or None


def _artifact_paths_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for step in _flatten_workflow_steps(steps):
        if step.get("status") != "completed":
            continue
        event = step.get("event")
        if event not in {"shell.result", "task.result"}:
            continue
        value = step.get("result")
        if isinstance(value, str):
            candidate = value.strip()
        elif isinstance(value, dict):
            excerpt = value.get("excerpt")
            candidate = excerpt.strip() if isinstance(excerpt, str) else ""
        else:
            candidate = ""
        paths.extend(_extract_path_values(candidate))
        stdout = _step_payload_field(step, "stdout") or _step_payload_field(step, "stdout_excerpt")
        if isinstance(stdout, str):
            paths.extend(_extract_path_values(stdout))
    seen: set[str] = set()
    unique = []
    for item in paths:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique

def _schema_focus_terms(task: str = "") -> set[str]:
    lowered = str(task or "").lower()
    terms = set()
    if re.search(r"\bschemas\b", lowered) or "schema names" in lowered:
        terms.add("schemas")
    if "table" in lowered:
        terms.add("tables")
    if "column" in lowered:
        terms.add("columns")
    if "relationship" in lowered or "foreign key" in lowered or "relation" in lowered:
        terms.add("relationships")
    if "schema" in lowered:
        terms.add("schema")
    return terms


def _schema_focus_result(schema: dict[str, Any], task: str = "") -> dict[str, Any] | None:
    tables = schema.get("tables")
    if not isinstance(tables, list):
        return None
    terms = _schema_focus_terms(task)
    if "schemas" in terms and "tables" not in terms and "columns" not in terms and "relationships" not in terms:
        names = sorted({str(table.get("schema") or "").strip() for table in tables if str(table.get("schema") or "").strip()})
        return {
            "columns": ["schema"],
            "rows": [{"schema": name} for name in names],
            "row_count": len(names),
            "limit": len(names),
        }
    if "tables" in terms and "schemas" not in terms and "columns" not in terms and "relationships" not in terms:
        rows = [
            {
                "schema": table.get("schema"),
                "table": table.get("name"),
                "type": table.get("type", "table"),
            }
            for table in tables
        ]
        return {
            "columns": ["schema", "table", "type"],
            "rows": rows,
            "row_count": len(rows),
            "limit": len(rows),
        }
    if "columns" in terms and "schemas" not in terms and "tables" not in terms and "relationships" not in terms:
        rows = []
        for table in tables:
            for column in table.get("columns", []):
                rows.append(
                    {
                        "schema": table.get("schema"),
                        "table": table.get("name"),
                        "column": column.get("name"),
                        "type": column.get("type"),
                        "nullable": column.get("nullable"),
                    }
                )
        return {
            "columns": ["schema", "table", "column", "type", "nullable"],
            "rows": rows,
            "row_count": len(rows),
            "limit": len(rows),
        }
    if "relationships" in terms and "schemas" not in terms and "tables" not in terms and "columns" not in terms:
        rows = []
        for table in tables:
            for foreign_key in table.get("foreign_keys", []):
                ref_schema = str(foreign_key.get("references_schema") or "").strip()
                ref_table = str(foreign_key.get("references_table") or "").strip()
                ref_column = str(foreign_key.get("references_column") or "").strip()
                qualified_ref = ".".join(part for part in (ref_schema, ref_table) if part)
                if ref_column:
                    qualified_ref = f"{qualified_ref}.{ref_column}" if qualified_ref else ref_column
                rows.append(
                    {
                        "schema": table.get("schema"),
                        "table": table.get("name"),
                        "column": foreign_key.get("column"),
                        "references": qualified_ref,
                    }
                )
        return {
            "columns": ["schema", "table", "column", "references"],
            "rows": rows,
            "row_count": len(rows),
            "limit": len(rows),
        }
    return None


def _format_schema_answer(schema: dict[str, Any], task: str = "") -> str:
    """Produce a compact Markdown summary of a SQL schema object."""
    focused_result = _schema_focus_result(schema, task)
    if isinstance(focused_result, dict):
        return _format_sql_result_answer(focused_result, task)

    dialect = schema.get("dialect", "unknown")
    tables = schema.get("tables", [])
    if not isinstance(tables, list):
        return json.dumps(schema, indent=2, ensure_ascii=True)

    schema_names = sorted({str(table.get("schema") or "").strip() for table in tables if str(table.get("schema") or "").strip()})
    summary_rows = []
    for table in tables:
        foreign_keys = table.get("foreign_keys", [])
        columns = table.get("columns", [])
        summary_rows.append(
            {
                "schema": table.get("schema"),
                "table": table.get("name"),
                "columns": len(columns) if isinstance(columns, list) else 0,
                "relationships": len(foreign_keys) if isinstance(foreign_keys, list) else 0,
            }
        )

    lines = [f"**Database schema** (`{dialect}`)"]
    if schema_names:
        lines.extend(["", f"Schemas: {', '.join(f'`{name}`' for name in schema_names)}"])
    if summary_rows:
        lines.extend(["", _markdown_table(["schema", "table", "columns", "relationships"], summary_rows, limit=50)])
        if len(summary_rows) > 50:
            lines.extend(["", "Showing first 50 tables."])
    return "\n".join(lines).strip()


def _format_schema_payload_answer(payload: dict[str, Any], task: str = "") -> str | None:
    result = payload.get("result")
    if isinstance(result, dict):
        return _format_sql_result_answer(result, task)
    schema = payload.get("schema")
    if isinstance(schema, dict):
        return _format_schema_answer(schema, task)
    return None


def _format_slurm_result_answer(payload: dict[str, Any]) -> str:
    return _format_agent_result(payload, "slurm.result")


def _format_workflow_answer(payload: dict[str, Any]) -> str:
    task = payload.get("task", "")
    status = payload.get("status", "unknown")
    task_shape = str(payload.get("task_shape") or "").strip().lower()
    include_internal_steps = _workflow_requests_internal_steps(payload)
    presentation_format = _workflow_presentation_format(payload)
    steps = payload.get("steps", [])
    flat_steps = _flatten_workflow_steps(steps if isinstance(steps, list) else [])
    artifact_paths = _artifact_paths_from_steps(steps if isinstance(steps, list) else [])
    sql_results = _extract_sql_results_from_steps(steps if isinstance(steps, list) else [])
    primary_answer = None
    if status != "completed":
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
    if task_shape == "save_artifact" and status == "completed":
        if sql_results:
            scalar_value = _extract_scalar_sql_value(sql_results[-1]) if len(sql_results) == 1 else None
            if scalar_value is not None:
                primary_answer = f"Result: `{scalar_value}`"
            else:
                primary_answer = _format_multi_sql_workflow_answer(sql_results, task) or _format_sql_result_answer(sql_results[-1], task)
        if artifact_paths:
            artifact_links = [(_markdown_link_for_file_path(path) or path) for path in artifact_paths]
            artifact_text = _markdown_section(
                "Files",
                artifact_links[0] if len(artifact_links) == 1 else "\n".join(f"- {link}" for link in artifact_links),
            )
            if isinstance(primary_answer, str) and primary_answer.strip():
                primary_answer = _join_markdown_sections(primary_answer, artifact_text)
            else:
                primary_answer = artifact_text
    else:
        if status == "completed" and sql_results:
            primary_answer = _format_multi_sql_workflow_answer(sql_results, task) or _format_sql_result_answer(sql_results[-1], task)
        shell_steps = [step for step in flat_steps if step.get("status") == "completed" and step.get("event") == "shell.result"]
        if primary_answer is None and shell_steps:
            primary_answer = _format_compound_shell_workflow_answer(shell_steps, task)
        for step in reversed(flat_steps):
            if primary_answer:
                break
            reduced = _step_payload_field(step, "reduced_result") or _step_payload_field(step, "refined_answer")
            if isinstance(reduced, str) and reduced.strip():
                primary_answer = reduced.strip()
                break
    if (
        task_shape == "save_artifact"
        and artifact_paths
        and isinstance(primary_answer, str)
        and primary_answer.strip()
    ):
        artifact_links = [(_markdown_link_for_file_path(path) or path) for path in artifact_paths]
        artifact_text = (
            _markdown_section("Files", artifact_links[0])
            if len(artifact_links) == 1
            else _markdown_section("Files", "\n".join(f"- {link}" for link in artifact_links))
        )
        if artifact_text not in primary_answer:
            primary_answer = _join_markdown_sections(primary_answer, artifact_text)

    if primary_answer is None and status == "completed" and task_shape == "schema_summary":
        for step in reversed(flat_steps):
            if step.get("event") != "sql.result":
                continue
            step_payload = step.get("payload")
            if not isinstance(step_payload, dict):
                continue
            rendered = _format_schema_payload_answer(step_payload, task)
            if rendered:
                primary_answer = rendered
                break

    slurm_results = _extract_slurm_results_from_steps(steps if isinstance(steps, list) else [])
    if primary_answer is None and status == "completed" and slurm_results:
        primary_answer = _format_slurm_result_answer(slurm_results[-1])

    if primary_answer is None and status == "completed" and presentation_format == "markdown_table":
        for step in reversed(flat_steps):
            if step.get("event") not in {"shell.result", "slurm.result"}:
                continue
            stdout = _step_payload_field(step, "stdout") or _step_payload_field(step, "stdout_excerpt")
            rendered = _format_fixed_width_table_answer(stdout, task)
            if rendered:
                primary_answer = rendered
                break

    if primary_answer is None and status == "completed":
        primary_answer = _format_multi_shell_workflow_answer(flat_steps, task)

    general_results = _extract_general_results_from_steps(steps if isinstance(steps, list) else [])
    if primary_answer is None and status == "completed" and general_results:
        event, payload = general_results[-1]
        primary_answer = _format_agent_result(payload, event)

    if primary_answer and not include_internal_steps:
        return primary_answer

    lines = [
        primary_answer.strip() if isinstance(primary_answer, str) and primary_answer.strip() else f"Workflow {status}: {task}".strip(),
    ]
    if not include_internal_steps:
        lines.extend(
            [
                "",
                "| Step | Agent | Status | Result |",
                "| --- | --- | --- | --- |",
            ]
        )
        for step in flat_steps:
            result = step.get("result")
            if isinstance(result, str):
                result_text = result.strip().replace("\n", "<br>")
            elif result is None:
                result_text = step.get("error", "")
            else:
                result_text = json.dumps(result, ensure_ascii=True)
            if len(result_text) > 500:
                result_text = f"{result_text[:500]}..."
            lines.append(
                "| {step_id} | {agent} | {status} | {result} |".format(
                    step_id=step.get("display_id", step.get("id", "")),
                    agent=step.get("target_agent", "workflow"),
                    status=step.get("status", ""),
                    result=result_text or "<empty>",
                )
            )

    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        lines.extend(["", f"Error: {error.strip()}"])

    stage_details = []
    for step in flat_steps:
        event = step.get("event")
        if event not in {"shell.result", "slurm.result", "file.content"}:
            continue
        command = _step_payload_field(step, "command")
        content = _step_payload_field(step, "content")
        stdout = _step_payload_field(step, "stdout") or _step_payload_field(step, "stdout_excerpt")
        stderr = _step_payload_field(step, "stderr") or _step_payload_field(step, "stderr_excerpt")
        if event == "file.content":
            stdout = content or stdout
        returncode = _step_payload_field(step, "returncode")
        detail = step.get("detail")
        stage_lines = [f"**{step.get('display_id', step.get('id', 'step'))}** {step.get('task', '').strip()}".strip()]
        if isinstance(event, str) and event.strip():
            stage_lines.append(f"Event: `{event}`")
        if isinstance(command, str) and command.strip():
            stage_lines.extend(["", "**Command**", ""])
            stage_lines.extend(_format_preformatted_block(command.strip()))
        if returncode is not None:
            stage_lines.append(f"Return code: `{returncode}`")
        if isinstance(detail, str) and detail.strip():
            stage_lines.append(f"Detail: {detail.strip()}")
        linked_paths = _extract_file_paths_from_text(stdout)
        labeled_paths = _extract_labeled_paths_from_text(stdout)
        if linked_paths or labeled_paths:
            stage_lines.append("Files:")
            stage_lines.extend([f"- {path}" for path in linked_paths])
            stage_lines.extend(labeled_paths)
        if isinstance(stdout, str) and stdout.strip():
            stage_lines.extend(["", "**Output**", ""])
            stage_lines.extend(_format_preformatted_block(_clean_terminal_output(stdout) or stdout.strip()))
        if isinstance(stderr, str) and stderr.strip():
            stage_lines.extend(["", "**Error output**", ""])
            stage_lines.extend(_format_preformatted_block(_clean_terminal_output(stderr) or stderr.strip()))
        if len(stage_lines) > 1:
            stage_details.append("\n".join(stage_lines))

    if include_internal_steps and stage_details:
        lines.extend(["", "**Stage Outputs**", "", *stage_details])

    shell_details = []
    for step in flat_steps:
        if step.get("event") != "shell.result":
            continue
        command = _step_payload_field(step, "command")
        if not isinstance(command, str) or not command.strip():
            continue
        detail = _command_output_details(
            command,
            str(_step_payload_field(step, "stdout") or _step_payload_field(step, "stdout_excerpt") or ""),
            str(_step_payload_field(step, "stderr") or _step_payload_field(step, "stderr_excerpt") or ""),
            _step_payload_field(step, "returncode"),
        )
        shell_details.append(
            "\n".join(
                [
                    f"**{step.get('display_id', step.get('id', 'step'))}**",
                    "",
                    detail,
                ]
            )
        )
    if shell_details:
        lines.extend(["", "**Command details**", "", *shell_details])

    return "\n".join(lines)


def _fallback_answer(req: EventRequest) -> str:
    if req.event == "research.result":
        content = req.payload["content"]
        return f"Research summary: {content}"

    if req.event == "task.result":
        detail = req.payload["detail"]
        return f"Task execution summary: {detail}"

    if req.event == "file.content":
        path = req.payload["path"]
        content = req.payload["content"]
        linked_path = _markdown_link_for_file_path(path) or path
        return f"File {linked_path} content preview:\n{content}"

    if req.event == "shell.result":
        command = req.payload["command"]
        stdout = req.payload["stdout"]
        stderr = req.payload["stderr"]
        returncode = req.payload["returncode"]
        return _format_shell_answer(command, returncode, stdout, stderr)

    if req.event == "sql.result":
        result = req.payload.get("result")
        if isinstance(result, dict):
            return _format_sql_result_answer(result)
        reduced = req.payload.get("reduced_result") or req.payload.get("refined_answer")
        if isinstance(reduced, str) and reduced.strip():
            return reduced.strip()
        schema = req.payload.get("schema")
        if isinstance(schema, dict):
            return _format_schema_answer(schema, str(req.payload.get("detail") or ""))
        return f"SQL result:\n{json.dumps(result, indent=2, ensure_ascii=True)}"

    if req.event == "slurm.result":
        return _format_slurm_result_answer(req.payload)

    if req.event == "workflow.result":
        return _format_workflow_answer(req.payload)

    if req.event == "clarification.required":
        question = req.payload.get("question")
        detail = req.payload.get("detail")
        missing_information = req.payload.get("missing_information")
        lines = []
        if isinstance(detail, str) and detail.strip():
            lines.append(detail.strip())
        if isinstance(question, str) and question.strip():
            lines.extend(["", question.strip()] if lines else [question.strip()])
        if isinstance(missing_information, list):
            items = [item.strip() for item in missing_information if isinstance(item, str) and item.strip()]
            if items:
                lines.extend(["", "Needed to continue:"])
                lines.extend([f"- {item}" for item in items])
        return "\n".join(lines).strip()

    if req.event == "notify.result":
        detail = req.payload["detail"]
        return f"Notification: {detail}"

    return ""


def _should_use_grounded_workflow_fallback(payload: dict[str, Any]) -> bool:
    steps = payload.get("steps", [])
    flat_steps = _flatten_workflow_steps(steps if isinstance(steps, list) else [])
    for step in flat_steps:
        if step.get("status") != "completed":
            continue
        event = step.get("event")
        step_payload = step.get("payload")
        if not isinstance(step_payload, dict):
            continue
        if event in {"shell.result", "slurm.result", "file.content"}:
            reduced = step_payload.get("reduced_result") or step_payload.get("refined_answer")
            if isinstance(reduced, str) and reduced.strip():
                continue
            stdout = (
                step_payload.get("stdout")
                or step_payload.get("stdout_excerpt")
                or step_payload.get("content")
            )
            if isinstance(stdout, str) and stdout.strip():
                return True
    return False


def _should_use_grounded_sql_fallback(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("result"), dict) or isinstance(payload.get("schema"), dict):
        return True
    return False


def _workflow_has_grounded_sql_output(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("task_shape") or "").strip().lower() == "schema_summary":
        return True
    return bool(_extract_sql_results_from_steps(payload.get("steps", []) if isinstance(payload.get("steps"), list) else []))


def _build_source_payload(req: EventRequest) -> dict[str, Any]:
    # Gatekeeper: Truncate large raw data before sending to LLM for final polish
    if req.event == "shell.result":
        return {
            "event": req.event,
            "command": req.payload.get("command"),
            "reduced_result": req.payload.get("reduced_result") or req.payload.get("refined_answer"),
            "local_reduction_command": req.payload.get("local_reduction_command"),
            "stdout": _truncate_if_large(req.payload.get("stdout")),
            "stderr": _truncate_if_large(req.payload.get("stderr"), limit=2000),
            "returncode": req.payload.get("returncode"),
        }
    if req.event == "workflow.result":
        sql_queries = []
        compact_sql_results = []
        for step in req.payload.get("steps", []):
            if not isinstance(step, dict):
                continue
            evidence = step.get("evidence") if isinstance(step.get("evidence"), dict) else {}
            payload = step.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("sql"), str):
                sql_queries.append(payload["sql"])
            if isinstance(payload, dict) and step.get("event") == "sql.result":
                result = payload.get("result")
                if isinstance(result, dict):
                    # Truncate rows in workflow summary
                    rows = result.get("rows", [])
                    compact_rows = rows[:5] if len(rows) > 5 else rows
                    compact_sql_results.append(
                        {
                            "sql": payload.get("sql") or result.get("sql"),
                            "queries": result.get("queries"),
                            "columns": result.get("columns"),
                            "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
                            "rows": compact_rows,
                            "row_count": result.get("row_count"),
                            "returned_row_count": result.get("returned_row_count"),
                            "total_matching_rows": result.get("total_matching_rows"),
                            "truncated": result.get("truncated"),
                            "limit": result.get("limit"),
                            "note": "Rows truncated for synthesis" if len(rows) > 5 else None
                        }
                    )
            elif isinstance(evidence, dict):
                evidence_payload = evidence.get("payload")
                if isinstance(evidence_payload, dict) and isinstance(evidence_payload.get("sql"), str):
                    sql_queries.append(evidence_payload["sql"])

        # Generic step outcome extraction (Phase 11 Refined)
        step_outcomes = []
        for step in req.payload.get("steps", []):
            if not isinstance(step, dict) or step.get("status") != "completed":
                continue
            evidence = step.get("evidence") if isinstance(step.get("evidence"), dict) else {}
            payload = step.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            
            # Extract the best human-readable summary from this step
            best_summary = (
                evidence.get("summary_text") or
                (evidence.get("payload", {}) if isinstance(evidence.get("payload"), dict) else {}).get("reduced_result") or
                (evidence.get("payload", {}) if isinstance(evidence.get("payload"), dict) else {}).get("detail") or
                payload.get("reduced_result") or
                payload.get("refined_answer") or 
                payload.get("detail") or 
                (payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}).get("reduced_result") or
                (payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}).get("refined_answer") or
                (payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}).get("detail")
            )

            if (not best_summary or not isinstance(best_summary, str) or not best_summary.strip()) and step.get("event") == "shell.result":
                stdout = payload.get("stdout") or payload.get("stdout_excerpt") or (evidence.get("payload", {}) if isinstance(evidence.get("payload"), dict) else {}).get("stdout_excerpt")
                if isinstance(stdout, str) and stdout.strip():
                    trimmed = stdout.strip()
                    if "\n" not in trimmed and len(trimmed) <= 400:
                        best_summary = trimmed
            
            if not best_summary or not isinstance(best_summary, str) or not best_summary.strip():
                # Fallback: Capture the semantic signal of the return code
                rc = payload.get("returncode")
                if rc == 0:
                    best_summary = f"Command succeeded (exit 0)."
                elif rc is not None:
                    best_summary = f"Command failed with exit code {rc}."
                else:
                    best_summary = "Step completed."
            
            if best_summary and isinstance(best_summary, str) and best_summary.strip():
                step_outcomes.append({
                    "step_id": step.get("id"),
                    "task": step.get("task"),
                    "outcome": best_summary.strip(),
                    "local_reduction_command": (
                        payload.get("local_reduction_command")
                        or payload.get("result", {}).get("local_reduction_command")
                        or (evidence.get("payload", {}) if isinstance(evidence.get("payload"), dict) else {}).get("local_reduction_command")
                    )
                })

        # Construction of a unified summary to prevent last-step bias
        combined_detail = req.payload.get("detail") or ""
        if step_outcomes:
            outcome_text = "\n".join([f"- Step '{so['task']}': {so['outcome']}" for so in step_outcomes])
            combined_detail = f"The workflow completed several steps with the following findings:\n{outcome_text}"

        return {
            "event": req.event,
            "task": req.payload.get("task"),
            "status": req.payload.get("status"),
            "presentation": req.payload.get("presentation"),
            "detail": combined_detail,
            "sql_queries": sql_queries,
            "sql_results": compact_sql_results,
            "step_outcomes": step_outcomes,
            "steps": [] if step_outcomes else req.payload.get("steps"), # Hide noisy raw steps if we have clean outcomes
            "result": None, # CRITICAL: Suppress last-step result to prevent LLM from ignoring step_outcomes
            "error": req.payload.get("error"),
            "task_shape": req.payload.get("task_shape"),
        }
    if req.event == "sql.result":
        result = req.payload.get("result")
        if isinstance(result, dict):
            rows = result.get("rows", [])
            compact_result = dict(result)
            if len(rows) > 10:
                compact_result["rows"] = rows[:10]
                compact_result["note"] = "Rows truncated for synthesis to respect data privacy and efficiency."
        else:
            compact_result = result

        schema = req.payload.get("schema")
        # Compact the schema: send only a table-name->column-count map to avoid
        # token overflow. The synthesizer LLM doesn't need the full DDL.
        compact_schema = None
        if isinstance(schema, dict) and "tables" in schema:
            compact_schema = {
                "dialect": schema.get("dialect", "unknown"),
                "tables": {
                    tbl: {"column_count": len(info.get("columns", []))} if isinstance(info, dict) else {}
                    for tbl, info in schema.get("tables", {}).items()
                },
            }

        return {
            "event": req.event,
            "detail": req.payload.get("detail"),
            "sql": req.payload.get("sql"),
            "reduced_result": req.payload.get("reduced_result") or req.payload.get("refined_answer"),
            "local_reduction_command": req.payload.get("local_reduction_command"),
            "schema": compact_schema,
            "result": compact_result,
        }
    if req.event == "slurm.result":
        return {
            "event": req.event,
            "detail": req.payload.get("detail"),
            "command": req.payload.get("command"),
            "reduced_result": req.payload.get("reduced_result") or req.payload.get("refined_answer"),
            "local_reduction_command": req.payload.get("local_reduction_command"),
            "stdout": _truncate_if_large(req.payload.get("stdout")),
            "stderr": _truncate_if_large(req.payload.get("stderr"), limit=2000),
            "returncode": req.payload.get("returncode"),
            "result": req.payload.get("result"),
        }
    if req.event == "clarification.required":
        return {
            "event": req.event,
            "task": req.payload.get("task"),
            "step_id": req.payload.get("step_id"),
            "detail": req.payload.get("detail"),
            "question": req.payload.get("question"),
            "missing_information": req.payload.get("missing_information"),
            "available_context": req.payload.get("available_context"),
        }
    return {
        "event": req.event,
        "payload": req.payload,
    }


def _build_prompt(req: EventRequest) -> str:
    source = _build_source_payload(req)
    presentation = source.get("presentation") if isinstance(source, dict) else None
    if not isinstance(presentation, dict):
        presentation = {}
    presentation_format = presentation.get("format", "markdown")
    presentation_task = presentation.get("task", "Answer the user request directly.")
    include_context = presentation.get("include_context", True)
    include_internal_steps = presentation.get("include_internal_steps", False)
    include_sql = bool(source.get("sql") or source.get("sql_queries")) if isinstance(source, dict) else False
    return (
        "You are writing the final assistant message in Open WebUI.\n"
        "Open WebUI renders Markdown tables, lists, headings, inline code, and fenced code blocks.\n"
        "Your job is to produce the final user-facing answer, not a debug report.\n"
        f"Presentation task: {presentation_task}\n"
        f"Requested format: {presentation_format}\n"
        f"Include helpful context: {include_context}\n"
        f"Include internal workflow steps/commands: {include_internal_steps}\n"
        f"Include executed SQL query: {include_sql}\n"
        "Requirements:\n"
        "- Answer the user's original request directly.\n"
        "- Look for a 'reduced_result', 'refined_answer', or 'detail' field in the source JSON; this is the high-quality reduced summary from the agent. Use it as the primary source for your response.\n"
        "- If a 'local_reduction_command' is present for a step, it means the agent performed a 'Compute Locally' step to reduce a large dataset. You may mention this briefly in your summary if it helps the user understand how the calculation was performed (e.g., 'Calculated via local awk script').\n"
        "- Output only the final answer. Do not explain your formatting choices.\n"
        "- Return concise Markdown unless the requested format is JSON or plain text.\n"
        "- Prefer Markdown tables for tabular data, inventories, comparisons, status reports, and anything requested as a table.\n"
        "- For SQL query results, always include the actual executed SQL query in a compact 'SQL used' section after the result table.\n"
        "- For SQL schema introspection, summarize tables, columns, and relationships clearly.\n"
        "- Put the useful answer first; add a one-sentence context line only when include_context is true.\n"
        "- Do not use boilerplate headings like Outcome Summary, Important Data, Key Value, Conclusion, or Shell Output.\n"
        "- Do not mention source event JSON, GitHub-flavored Markdown, Open WebUI, formatting, workflow, steps, agents, planner, synthesizer, commands, stdout, stderr, JSON, or logs unless include_internal_steps is true or needed to explain an error.\n"
        "- Never end with commentary such as 'This is a concise answer' or 'This is formatted as Markdown'.\n"
        "- If include_internal_steps is false, use command outputs only as source data and hide implementation details.\n"
        "- If include_internal_steps is true, include a compact 'How it was done' section after the answer.\n"
        "- For multi-step workflows, review the 'step_outcomes' array. It contains key findings from each individual step. Aggregrate these into one comprehensive response that fully addresses the user's original request.\n"
        "- Never output HTML details, summary, or collapsible blocks.\n"
        "- If command, stdout, stderr, logs, or raw execution details are needed, use one plain fenced Markdown block and no HTML tags.\n"
        "- If the workflow failed or partially failed, include the failed step task and the exact error message from that step.\n"
        "- If there are completed steps before a failure, show any useful partial result first, then mention what failed.\n"
        "- If a failed step references unavailable previous results, say which dependency was missing.\n"
        "- If requested format is json, output only valid JSON and no Markdown.\n"
        "- **CRITICAL**: For 'Existence Checks' (e.g., using `grep -q`, `ls`, `find`, or `test`), the **Return Code (Exit Code)** is your primary signal. An exit code of 0 usually confirms the presence or success of the object/pattern, even if the output is empty. Interpret these correctly (e.g., if `grep -q` succeeded, the answer is 'Yes').\n"
        "- Do not invent facts not present in the source event.\n"
        "- Keep the answer compact, practical, and attractive in a chat UI.\n"
        "Source event JSON:\n"
        f"{json.dumps(source, indent=2)}"
    )


def _llm_synthesize(req: EventRequest) -> str | None:
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")

    prompt = _build_prompt(req)
    _debug_log("Constructed synthesizer prompt:")
    _debug_log(prompt)

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You create polished final answers for users."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        return None
    answer = content.strip()
    _debug_log("Raw synthesizer response:")
    _debug_log(answer)
    return answer or None


def _synthesize(req: EventRequest) -> str:
    if req.event == "task.result":
        detail = req.payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    if req.event == "clarification.required":
        question = req.payload.get("question")
        if isinstance(question, str) and question.strip():
            return _fallback_answer(req)
    if req.event == "sql.result" and _should_use_grounded_sql_fallback(req.payload):
        return _fallback_answer(req)
    if req.event == "workflow.result" and (
        str(req.payload.get("status") or "").strip().lower() != "completed"
        or
        _should_use_grounded_workflow_fallback(req.payload)
        or _workflow_has_grounded_sql_output(req.payload)
        or str(req.payload.get("task_shape") or "").strip().lower() == "save_artifact"
        or _workflow_requests_internal_steps(req.payload)
    ):
        return _fallback_answer(req)
    try:
        answer = _llm_synthesize(req)
        if answer:
            return answer
    except Exception as exc:
        _debug_log(f"Synthesizer LLM failed: {type(exc).__name__}: {exc}")
    return _fallback_answer(req)


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("synthesizer", "synthesizer")
def handle_event(req: EventRequest):
    if req.event not in {"research.result", "task.result", "file.content", "shell.result", "sql.result", "slurm.result", "notify.result", "workflow.result", "clarification.required"}:
        return {"emits": []}
    answer = _synthesize(req)
    return {"emits": [{"event": "answer.final", "payload": {"answer": answer}}]}
