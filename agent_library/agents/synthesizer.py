import json
import os
from typing import Any

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse
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
            "name": "synthesize_task_result",
            "event": "task.result",
            "when": "Converts generic task result into final answer.",
        },
        {
            "name": "synthesize_workflow_result",
            "event": "workflow.result",
            "when": "Converts aggregated multi-step workflow results into final answer.",
        },
    ],
}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("SYNTH_DEBUG", message)


def _format_shell_answer(command: str, returncode: int, stdout: str, stderr: str) -> str:
    clean_stdout = (stdout or "").strip()
    clean_stderr = (stderr or "").strip()
    status = "success" if returncode == 0 else "failure"
    lines = [
        "Shell execution result",
        f"- command: {command}",
        f"- exit_code: {returncode} ({status})",
        "- stdout:",
        clean_stdout or "<empty>",
        "- stderr:",
        clean_stderr or "<empty>",
    ]
    return "\n".join(lines)


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
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
                    "limit": result.get("limit"),
                }
            )
    return results


def _format_sql_result_answer(result: dict[str, Any], task: str = "") -> str:
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    row_count = result.get("row_count")
    sql = result.get("sql", "")
    queries = result.get("queries")
    limit = result.get("limit")

    lines = []
    if task:
        lines.append(f"Found {row_count if row_count is not None else len(rows)} result row(s).")
        lines.append("")

    if isinstance(columns, list) and isinstance(rows, list) and columns:
        table_limit = 50
        lines.append(_markdown_table(columns, rows, table_limit))
        if len(rows) > table_limit:
            lines.append("")
            lines.append(f"Showing first {table_limit} rows.")
        elif isinstance(limit, int) and row_count == limit:
            lines.append("")
            lines.append(f"Showing up to {limit} rows.")
    else:
        lines.append(json.dumps(result, indent=2, ensure_ascii=True))

    if isinstance(queries, list) and queries:
        lines.extend(["", "**SQL used**"])
        for index, query in enumerate(queries, start=1):
            if not isinstance(query, dict):
                continue
            query_sql = query.get("sql")
            if not isinstance(query_sql, str) or not query_sql.strip():
                continue
            label = query.get("label") or f"Query {index}"
            lines.extend(["", f"{index}. {label}", "", "```sql", query_sql.strip(), "```"])
    elif isinstance(sql, str) and sql.strip():
        lines.extend(["", "**SQL used**", "", "```sql", sql.strip(), "```"])
    return "\n".join(lines).strip()


def _format_workflow_answer(payload: dict[str, Any]) -> str:
    task = payload.get("task", "")
    status = payload.get("status", "unknown")
    steps = payload.get("steps", [])
    flat_steps = _flatten_workflow_steps(steps if isinstance(steps, list) else [])
    sql_results = _extract_sql_results_from_steps(steps if isinstance(steps, list) else [])
    if status == "completed" and sql_results:
        return _format_sql_result_answer(sql_results[-1], task)

    lines = [
        f"Workflow {status}: {task}".strip(),
        "",
        "| Step | Agent | Status | Result |",
        "| --- | --- | --- | --- |",
    ]
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
        return f"File '{path}' content preview:\n{content}"

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
        return f"SQL result:\n{json.dumps(result, indent=2, ensure_ascii=True)}"

    if req.event == "workflow.result":
        return _format_workflow_answer(req.payload)

    if req.event == "notify.result":
        detail = req.payload["detail"]
        return f"Notification: {detail}"

    return ""


def _build_source_payload(req: EventRequest) -> dict[str, Any]:
    if req.event == "shell.result":
        return {
            "event": req.event,
            "command": req.payload.get("command"),
            "stdout": req.payload.get("stdout"),
            "stderr": req.payload.get("stderr"),
            "returncode": req.payload.get("returncode"),
        }
    if req.event == "workflow.result":
        sql_queries = []
        compact_sql_results = []
        for step in req.payload.get("steps", []):
            if not isinstance(step, dict):
                continue
            payload = step.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("sql"), str):
                sql_queries.append(payload["sql"])
            if isinstance(payload, dict) and step.get("event") == "sql.result":
                result = payload.get("result")
                if isinstance(result, dict):
                    compact_sql_results.append(
                        {
                            "sql": payload.get("sql") or result.get("sql"),
                            "queries": result.get("queries"),
                            "columns": result.get("columns"),
                            "rows": result.get("rows"),
                            "row_count": result.get("row_count"),
                            "limit": result.get("limit"),
                        }
                    )
        return {
            "event": req.event,
            "task": req.payload.get("task"),
            "status": req.payload.get("status"),
            "presentation": req.payload.get("presentation"),
            "sql_queries": sql_queries,
            "sql_results": compact_sql_results,
            "steps": [] if compact_sql_results else req.payload.get("steps"),
            "result": req.payload.get("result"),
            "error": req.payload.get("error"),
        }
    if req.event == "sql.result":
        return {
            "event": req.event,
            "detail": req.payload.get("detail"),
            "sql": req.payload.get("sql"),
            "schema": req.payload.get("schema"),
            "result": req.payload.get("result"),
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
        "- If the workflow failed or partially failed, include the failed step task and the exact error message from that step.\n"
        "- If there are completed steps before a failure, show any useful partial result first, then mention what failed.\n"
        "- If a failed step references unavailable previous results, say which dependency was missing.\n"
        "- If requested format is json, output only valid JSON and no Markdown.\n"
        "- Do not invent facts not present in the source event.\n"
        "- Keep the answer compact, practical, and attractive in a chat UI.\n"
        "Source event JSON:\n"
        f"{json.dumps(source, indent=2)}"
    )


def _llm_synthesize(req: EventRequest) -> str | None:
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_SYNTH_MODEL") or os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

    if api_key.startswith("sk-") and api_key.lower() != "dummy":
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4.1"
    elif not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

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
    try:
        answer = _llm_synthesize(req)
        if answer:
            return answer
    except Exception as exc:
        _debug_log(f"Synthesizer LLM failed: {type(exc).__name__}: {exc}")
    return _fallback_answer(req)


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event not in {"research.result", "task.result", "file.content", "shell.result", "sql.result", "notify.result", "workflow.result"}:
        return {"emits": []}
    answer = _synthesize(req)
    return {"emits": [{"event": "answer.final", "payload": {"answer": answer}}]}
