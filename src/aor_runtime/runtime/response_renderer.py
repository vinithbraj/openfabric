from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.facts import build_sanitized_facts
from aor_runtime.runtime.insights import InsightContext, InsightResult, generate_insights, summarize_insights_with_llm
from aor_runtime.runtime.intelligent_output import render_intelligent_output
from aor_runtime.runtime.markdown import add_section_breaks
from aor_runtime.runtime.markdown import code_block as md_code_block
from aor_runtime.runtime.markdown import section as md_section
from aor_runtime.runtime.markdown import table as md_table
from aor_runtime.runtime.presentation import (
    PresentationContext,
    build_sanitized_presentation_facts,
    present_result,
    strip_internal_telemetry,
    summarize_presented_facts_with_llm,
)
from aor_runtime.tools.slurm import SACCT_FORMAT, SINFO_NODE_FORMAT, SINFO_PARTITION_FORMAT, SQUEUE_FORMAT


ResponseRenderMode = Literal["user", "debug", "raw"]
ExecutionStatus = Literal["completed", "failed", "skipped"]
DISPLAY_CODE_WRAP_WIDTH = 100

SQL_DISPLAY_CLAUSES = [
    "LEFT OUTER JOIN",
    "RIGHT OUTER JOIN",
    "FULL OUTER JOIN",
    "INNER JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "CROSS JOIN",
    "GROUP BY",
    "ORDER BY",
    "UNION ALL",
    "SELECT",
    "FROM",
    "JOIN",
    "WHERE",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "UNION",
]


@dataclass
class ExecutedAction:
    tool: str
    label: str
    command: str | None = None
    sql: str | None = None
    database: str | None = None
    args_summary: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    status: ExecutionStatus = "completed"


@dataclass
class ResponseRenderContext:
    mode: ResponseRenderMode = "user"
    show_executed_commands: bool = True
    show_validation: bool = False
    show_planning_steps: bool = False
    show_tool_events: bool = False
    show_debug_metadata: bool = False
    enable_llm_summary: bool = False
    enable_insight_layer: bool = True
    enable_llm_insights: bool = False
    insight_max_facts: int = 50
    insight_max_input_chars: int = 4000
    insight_max_output_chars: int = 1500
    max_rows: int = 20
    max_command_length: int = 4000
    output_mode: str = "text"
    source_action: str | None = None
    goal: str = ""
    llm_settings: Any | None = None
    intelligent_output_mode: str = "off"
    intelligent_output_max_fields: int = 8


@dataclass
class RenderedResponse:
    markdown: str
    title: str | None = None
    sections: list[dict[str, Any]] = field(default_factory=list)
    hidden_events: list[str] = field(default_factory=list)
    debug: dict[str, Any] | None = None


def render_agent_response(
    final_result: Any,
    execution_events: list[Any] | None = None,
    plan: Any | None = None,
    metadata: dict[str, Any] | None = None,
    context: ResponseRenderContext | None = None,
) -> RenderedResponse:
    ctx = context or ResponseRenderContext()
    meta = dict(metadata or {})
    actions = extract_executed_actions(plan=plan, execution_events=execution_events, metadata=meta)
    presentation_result = _result_for_presentation(final_result, actions)
    if ctx.mode == "raw":
        raw = final_result if isinstance(final_result, str) else json.dumps(final_result, ensure_ascii=False, sort_keys=True, default=str)
        return RenderedResponse(markdown=str(raw).strip(), title="Raw Output", sections=[{"title": "Raw Output"}])

    source_action = _presentation_source_action(ctx.source_action, actions)
    source_args = _presentation_source_args(source_action, actions)
    presentation_context = PresentationContext(
        mode=ctx.mode,
        enable_llm_summary=ctx.enable_llm_summary,
        max_rows=ctx.max_rows,
        max_items=ctx.max_rows,
        include_telemetry=ctx.show_debug_metadata,
        include_raw=ctx.mode == "raw",
        source_action=source_action,
        source_args=source_args,
        output_mode=ctx.output_mode,
        goal=ctx.goal,
    )
    presented = present_result(presentation_result, presentation_context)
    intelligent = render_intelligent_output(
        presentation_result,
        actions,
        presentation_context,
        ctx.llm_settings,
        mode=ctx.intelligent_output_mode,
        max_fields=ctx.intelligent_output_max_fields,
    )

    sections: list[dict[str, Any]] = []
    lines: list[str] = []
    insight_result: InsightResult | None = None
    facts: dict[str, Any] = {}

    if ctx.enable_insight_layer:
        facts = build_sanitized_facts(presentation_result, actions, context=_InsightFactContext(ctx, presentation_context))
        insight_context = InsightContext(
            enable_llm=ctx.enable_llm_insights,
            max_facts=ctx.insight_max_facts,
            max_input_chars=ctx.insight_max_input_chars,
            max_output_chars=ctx.insight_max_output_chars,
            mode=ctx.mode,
        )
        insight_result = generate_insights(facts, insight_context)
        llm_summary = (
            summarize_insights_with_llm(facts, insight_result, insight_context, ctx.llm_settings)
            if ctx.enable_llm_insights and ctx.llm_settings is not None
            else None
        )
        summary_text = llm_summary or insight_result.summary
        if summary_text:
            lines.extend(md_section("Summary", [_truncate(summary_text.strip(), ctx.insight_max_output_chars)]))
            lines.append("")
            sections.append({"title": "Summary"})
        if llm_summary:
            insight_result.llm_used = True
        if insight_result.insights:
            lines.extend(md_section("Key Findings"))
            lines.append("")
            for insight in insight_result.insights:
                lines.append(f"- **{insight.title}:** {insight.message}")
            lines.append("")
            sections.append({"title": "Key Findings"})
    else:
        facts = build_sanitized_presentation_facts(presentation_result, actions, presentation_context)
    if not ctx.enable_insight_layer and ctx.enable_llm_summary and ctx.llm_settings is not None:
        summary = summarize_presented_facts_with_llm(facts, presentation_context, ctx.llm_settings)
        if summary:
            lines.extend(md_section("Summary", [_truncate(summary.strip(), ctx.max_command_length)]))
            lines.append("")
            sections.append({"title": "Summary"})

    if ctx.intelligent_output_mode == "replace" and intelligent is not None:
        result_markdown = intelligent.markdown
    else:
        result_markdown = _result_section_markdown(presented.markdown)
    if result_markdown:
        lines.append(result_markdown)
        sections.append({"title": "Result"})
    if ctx.intelligent_output_mode == "compare" and intelligent is not None:
        lines.extend(["", intelligent.markdown])
        sections.append({"title": "Intelligent Output"})

    if ctx.show_executed_commands:
        action_markdown = _executed_actions_markdown(actions, ctx)
        if action_markdown:
            lines.extend(["", action_markdown])
            sections.append({"title": "Execution Details"})

    if ctx.mode == "debug" or ctx.show_debug_metadata:
        debug_payload = _compact_debug_metadata(meta, actions)
        if insight_result is not None:
            debug_payload["insights"] = insight_result.to_dict()
        if intelligent is not None:
            debug_payload["intelligent_output"] = {
                "selected_fields": intelligent.selected_fields,
                "llm_used": intelligent.llm_used,
            }
        if debug_payload:
            lines.extend(["", *md_section("Debug Metadata")])
            lines.extend(md_code_block("json", json.dumps(debug_payload, ensure_ascii=False, sort_keys=True, default=str)))
            sections.append({"title": "Debug Metadata"})

    markdown = add_section_breaks("\n".join(line for line in lines if line is not None).strip())
    return RenderedResponse(
        markdown=markdown,
        title=_infer_title(markdown),
        sections=sections,
        hidden_events=_hidden_event_names(execution_events or [], ctx),
        debug=_compact_debug_metadata(meta, actions) if ctx.mode == "debug" else None,
    )


def extract_executed_actions(
    plan: Any | None,
    execution_events: list[Any] | None,
    metadata: dict[str, Any] | None,
) -> list[ExecutedAction]:
    actions: list[ExecutedAction] = []
    for item in execution_events or []:
        action = _action_from_step_log(item)
        if action is not None:
            actions.append(action)
            continue
        action = _action_from_event(item)
        if action is not None:
            actions.append(action)
    if actions:
        return _dedupe_actions(actions)

    for step in _iter_plan_steps(plan):
        action = _action_from_step(step, result={}, success=True)
        if action is not None:
            actions.append(action)
    return _dedupe_actions(actions)


def _action_from_step_log(item: Any) -> ExecutedAction | None:
    step = getattr(item, "step", None)
    if step is not None:
        return _action_from_step(step, result=dict(getattr(item, "result", {}) or {}), success=bool(getattr(item, "success", False)))
    if isinstance(item, dict) and "step" in item:
        return _action_from_step(item.get("step") or {}, result=dict(item.get("result") or {}), success=bool(item.get("success", False)))
    return None


def _action_from_event(item: Any) -> ExecutedAction | None:
    if not isinstance(item, dict):
        return None
    event_type = str(item.get("event_type") or "")
    payload = dict(item.get("payload") or {})
    if event_type == "executor.step.started":
        step = dict(payload.get("step") or {})
        action = str(step.get("action") or "")
        if not action:
            return None
        command = str(payload.get("command") or "").strip() or None
        built = _action_from_step(step, result={}, success=True)
        if built is not None and command:
            built.command = command
        return built
    if event_type == "executor.step.completed":
        return _action_from_step_log(payload)
    return None


def _action_from_step(step: Any, *, result: dict[str, Any], success: bool) -> ExecutedAction | None:
    if hasattr(step, "action"):
        tool = str(step.action)
        args = dict(getattr(step, "args", {}) or {})
    else:
        step_dict = dict(step or {})
        tool = str(step_dict.get("action") or "")
        args = dict(step_dict.get("args") or {})
    if not tool:
        return None
    status: ExecutionStatus = "completed" if success else "failed"

    if tool == "runtime.return":
        return ExecutedAction(tool=tool, label="Return", status=status)
    if tool == "sql.query":
        sql = str(result.get("sql_final") or result.get("sql_normalized") or args.get("query") or "").strip()
        return ExecutedAction(
            tool=tool,
            label="SQL query",
            sql=sql or None,
            database=str(result.get("database") or args.get("database") or "").strip() or None,
            result=dict(result or {}),
            status=status,
        )
    if tool.startswith("slurm."):
        return ExecutedAction(
            tool=tool,
            label=_tool_label(tool),
            command=_slurm_command(tool, args),
            args_summary=_safe_args(args),
            result=dict(result or {}),
            status=status,
        )
    if tool.startswith("fs."):
        return ExecutedAction(tool=tool, label=_tool_label(tool), args_summary=_safe_args(args or result), status=status)
    if tool == "shell.exec":
        args_summary: dict[str, Any] = {"node": args.get("node")} if args.get("node") else {}
        if result.get("risk"):
            args_summary["risk"] = result.get("risk")
        return ExecutedAction(
            tool=tool,
            label="Shell command",
            command=str(args.get("command") or "").strip() or None,
            args_summary=args_summary,
            result=dict(result or {}),
            status=status,
        )
    return ExecutedAction(tool=tool, label=_tool_label(tool), args_summary=_safe_args(args), status=status)


def _executed_actions_markdown(actions: list[ExecutedAction], context: ResponseRenderContext) -> str:
    visible = [action for action in actions if context.mode != "user" or action.tool != "runtime.return"]
    if not visible:
        return ""

    sql_actions = [action for action in visible if action.sql]
    slurm_actions = [action for action in visible if action.tool.startswith("slurm.") and action.command]
    fs_actions = [action for action in visible if action.tool.startswith("fs.")]
    shell_actions = [action for action in visible if action.tool == "shell.exec" and action.command]
    other_actions = [
        action
        for action in visible
        if action not in sql_actions and action not in slurm_actions and action not in fs_actions and action not in shell_actions and action.tool != "runtime.return"
    ]

    lines: list[str] = []
    if sql_actions:
        action = sql_actions[-1]
        lines.extend(md_section("Query Used"))
        lines.extend(md_code_block("sql", _format_sql_for_display(action.sql or "", context.max_command_length)))
        lines.extend(_execution_table([("Tool", action.tool), ("Database", action.database or ""), ("Status", _title(action.status))]))
    if slurm_actions:
        lines.extend(["", *md_section("Commands Used")])
        commands = _unique([action.command or "" for action in slurm_actions])
        rendered_commands = [_format_command_for_display(command, context.max_command_length) for command in commands[:5]]
        if len(commands) > 5:
            rendered_commands.append(f"{len(commands) - 5} additional commands hidden; use debug/raw mode for full list.")
        lines.extend(md_code_block("bash", "\n\n".join(rendered_commands)))
        lines.extend(_execution_table([("Backend", "SLURM gateway"), ("Tools", ", ".join(_unique([action.tool for action in slurm_actions]))), ("Status", _overall_status(slurm_actions))]))
    if fs_actions:
        lines.extend(["", *md_section("Operation")])
        lines.append("")
        rows = [("Tool", ", ".join(_unique([action.tool for action in fs_actions])))]
        merged: dict[str, Any] = {}
        for action in fs_actions:
            merged.update(action.args_summary or {})
        for key in ("path", "pattern", "recursive", "aggregate", "size_unit", "path_style"):
            if key in merged and merged[key] is not None:
                rows.append((_title(key), merged[key]))
        rows.append(("Status", _overall_status(fs_actions)))
        lines.extend(_execution_table(rows))
    if shell_actions:
        lines.extend(["", *md_section("Command Used")])
        lines.extend(
            md_code_block(
                "bash",
                "\n\n".join(
                    _format_command_for_display(command, context.max_command_length)
                    for command in _unique([action.command or "" for action in shell_actions])
                ),
            )
        )
        risks = _unique([str((action.result or {}).get("risk") or (action.args_summary or {}).get("risk") or "") for action in shell_actions])
        reasons = _unique([str((action.result or {}).get("policy_reason") or "") for action in shell_actions])
        rows: list[tuple[str, Any]] = [("Tool", "shell.exec")]
        if any(risks):
            rows.append(("Risk", ", ".join(risk for risk in risks if risk)))
        if any(reasons) and context.mode != "user":
            rows.append(("Policy", "; ".join(reason for reason in reasons if reason)))
        rows.append(("Status", _overall_status(shell_actions)))
        lines.extend(_execution_table(rows))
    if context.mode == "debug" and other_actions:
        lines.extend(["", *md_section("Additional Tools")])
        lines.append("")
        lines.extend(_execution_table([("Tools", ", ".join(_unique([action.tool for action in other_actions]))), ("Status", _overall_status(other_actions))]))
    return "\n".join(lines).strip()


def _result_section_markdown(markdown: str) -> str:
    body = str(markdown or "").strip()
    if not body:
        return ""
    if body.startswith("## SQL Results"):
        return body.replace("## SQL Results", "## Result", 1)
    if body.startswith("##"):
        return body
    if body.startswith("# SLURM"):
        return "##" + body[1:]
    if body.startswith("# SQL Results"):
        return body.replace("# SQL Results", "## Result", 1)
    if body.startswith("#"):
        return "##" + body[1:]
    return "\n".join(md_section("Result", [body]))


def _execution_table(rows: list[tuple[str, Any]]) -> list[str]:
    return md_table(["Field", "Value"], [[_code_cell(key), _code_cell(value)] for key, value in rows if value is not None and value != ""])


class _InsightFactContext:
    def __init__(self, render_context: ResponseRenderContext, presentation_context: PresentationContext) -> None:
        self.max_facts = render_context.insight_max_facts
        self.max_input_chars = render_context.insight_max_input_chars
        self.source_action = presentation_context.source_action
        self.source_args = presentation_context.source_args
        self.output_mode = presentation_context.output_mode
        self.include_row_samples = presentation_context.include_row_samples
        self.include_paths = presentation_context.include_paths


def _slurm_command(tool: str, args: dict[str, Any]) -> str | None:
    if tool == "slurm.queue":
        return _join_argv(["squeue", "-h", "-o", SQUEUE_FORMAT])
    if tool == "slurm.nodes":
        return _join_argv(["sinfo", "-h", "-N", "-o", SINFO_NODE_FORMAT])
    if tool == "slurm.partitions":
        return _join_argv(["sinfo", "-h", "-o", SINFO_PARTITION_FORMAT])
    if tool == "slurm.job_detail" and args.get("job_id"):
        return _join_argv(["scontrol", "show", "job", str(args.get("job_id"))])
    if tool == "slurm.node_detail" and args.get("node"):
        return _join_argv(["scontrol", "show", "node", str(args.get("node"))])
    if tool in {"slurm.accounting", "slurm.accounting_aggregate"}:
        return _join_argv(_accounting_argv(args))
    if tool == "slurm.slurmdbd_health":
        return _join_argv(["sacctmgr", "show", "cluster", "-P"])
    if tool == "slurm.metrics":
        return _metric_command_summary(str(args.get("metric_group") or "cluster_summary"), args)
    return None


def _metric_command_summary(metric_group: str, args: dict[str, Any]) -> str:
    commands: list[str]
    if metric_group == "cluster_summary":
        commands = [
            _join_argv(["squeue", "-h", "-o", SQUEUE_FORMAT]),
            _join_argv(["sinfo", "-h", "-N", "-o", SINFO_NODE_FORMAT]),
        ]
    elif metric_group in {"queue_summary", "scheduler_health"}:
        commands = [_join_argv(["squeue", "-h", "-o", SQUEUE_FORMAT])]
    elif metric_group in {"node_summary", "problematic_nodes", "gpu_summary"}:
        commands = [_join_argv(["sinfo", "-h", "-N", "-o", SINFO_NODE_FORMAT])]
    elif metric_group == "partition_summary":
        commands = [_join_argv(["sinfo", "-h", "-o", SINFO_PARTITION_FORMAT])]
    else:
        commands = [_join_argv(_accounting_argv(args))]
    return "\n".join(commands)


def _accounting_argv(args: dict[str, Any]) -> list[str]:
    command = ["sacct", "-X", "-P", f"--format={SACCT_FORMAT}"]
    for key, flag in (("user", "--user"), ("state", "--state"), ("partition", "--partition"), ("start", "--starttime"), ("end", "--endtime")):
        value = str(args.get(key) or "").strip()
        if value:
            command.append(f"{flag}={value}")
    return command


def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in dict(args or {}).items():
        if str(key) in {"command", "query", "output_contract", "value"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
    return safe


def _compact_debug_metadata(metadata: dict[str, Any], actions: list[ExecutedAction]) -> dict[str, Any]:
    compact = strip_internal_telemetry(metadata)
    if actions:
        compact["executed_tools"] = [action.tool for action in actions if action.tool != "runtime.return"]
    return compact


def _hidden_event_names(events: list[Any], context: ResponseRenderContext) -> list[str]:
    if context.mode != "user":
        return []
    hidden: list[str] = []
    for event in events:
        if isinstance(event, dict) and event.get("event_type"):
            hidden.append(str(event["event_type"]))
    return hidden


def _result_for_presentation(final_result: Any, actions: list[ExecutedAction]) -> Any:
    slurm_actions = [action for action in actions if action.tool.startswith("slurm.") and isinstance(action.result, dict)]
    if not slurm_actions:
        return final_result
    if isinstance(final_result, dict) and "results" in final_result:
        return final_result
    if _looks_like_multi_metric_slurm(final_result):
        return final_result
    last = slurm_actions[-1]
    if isinstance(final_result, list) or isinstance(final_result, (int, float, str)):
        return last.result or final_result
    if isinstance(final_result, dict) and not any(key in final_result for key in ("jobs", "nodes", "partitions", "metric_group", "payload")):
        return last.result or final_result
    return final_result


def _looks_like_multi_metric_slurm(value: Any) -> bool:
    if not isinstance(value, dict) or "result_kind" in value:
        return False
    children = [
        item
        for item in value.values()
        if isinstance(item, dict)
        and (item.get("result_kind") == "accounting_aggregate" or {"average_elapsed_seconds", "min_elapsed_seconds", "max_elapsed_seconds"} & set(item))
    ]
    return len(children) >= 2 and len(children) == len(value)


def _iter_plan_steps(plan: Any | None) -> list[Any]:
    if plan is None:
        return []
    if hasattr(plan, "steps"):
        return list(plan.steps)
    if isinstance(plan, dict):
        return list(plan.get("steps") or [])
    return []


def _last_visible_action(actions: list[ExecutedAction]) -> str | None:
    for action in reversed(actions):
        if action.tool != "runtime.return":
            return action.tool
    return None


def _presentation_source_action(requested: str | None, actions: list[ExecutedAction]) -> str | None:
    requested_value = str(requested or "").strip()
    if requested_value and requested_value not in {"runtime.return", "text.format"}:
        return requested_value
    for action in reversed(actions):
        if action.tool not in {"runtime.return", "text.format"}:
            return action.tool
    return requested_value or None


def _presentation_source_args(source_action: str | None, actions: list[ExecutedAction]) -> dict[str, Any]:
    if source_action:
        for action in reversed(actions):
            if action.tool == source_action:
                return dict(action.args_summary or {})
    for action in reversed(actions):
        if action.tool != "runtime.return":
            return dict(action.args_summary or {})
    return {}


def _infer_title(markdown: str) -> str | None:
    match = re.match(r"^#+\s+(.+)$", markdown.strip())
    return match.group(1).strip() if match else None


def _overall_status(actions: list[ExecutedAction]) -> str:
    return "Failed" if any(action.status == "failed" for action in actions) else "Completed"


def _tool_label(tool: str) -> str:
    return tool.replace(".", " ").replace("_", " ").title()


def _title(value: Any) -> str:
    return str(value).replace("_", " ").title()


def _code_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("`", "'").strip()
    return f"`{text or '-'}`"


def _join_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _format_sql_for_display(sql: str, max_chars: int, *, width: int = DISPLAY_CODE_WRAP_WIDTH) -> str:
    text = _normalize_whitespace_outside_quotes(sql)
    if not text:
        return ""
    clause_lines = _break_sql_clauses(text)
    wrapped: list[str] = []
    for line in clause_lines:
        wrapped.extend(_wrap_sql_line(line, width=width))
    return _truncate("\n".join(wrapped), max_chars)


def _format_command_for_display(command: str, max_chars: int, *, width: int = DISPLAY_CODE_WRAP_WIDTH) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    wrapped_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_shell_line(line.strip(), width=width))
    return _truncate("\n".join(wrapped_lines), max_chars)


def _normalize_whitespace_outside_quotes(value: str) -> str:
    text = str(value or "")
    result: list[str] = []
    quote: str | None = None
    escaped = False
    pending_space = False
    for char in text:
        if escaped:
            if pending_space and quote is None and result:
                result.append(" ")
                pending_space = False
            result.append(char)
            escaped = False
            continue
        if char == "\\":
            if pending_space and quote is None and result:
                result.append(" ")
                pending_space = False
            result.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            if pending_space and quote is None and result:
                result.append(" ")
                pending_space = False
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            result.append(char)
            continue
        if quote is None and char.isspace():
            pending_space = bool(result)
            continue
        if pending_space and quote is None and result:
            result.append(" ")
            pending_space = False
        result.append(char)
    return "".join(result).strip()


def _break_sql_clauses(sql: str) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(sql):
        char = sql[index]
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            current.append(char)
            index += 1
            continue
        if quote is None:
            clause = _match_sql_clause(sql, index)
            if clause:
                current_text = "".join(current).strip()
                if current_text:
                    lines.append(current_text)
                    current = []
                current.append(sql[index : index + len(clause)])
                index += len(clause)
                continue
        current.append(char)
        index += 1
    tail = "".join(current).strip()
    if tail:
        lines.append(tail)
    return lines or [sql]


def _match_sql_clause(sql: str, index: int) -> str | None:
    for clause in SQL_DISPLAY_CLAUSES:
        end = index + len(clause)
        if sql[index:end].upper() != clause:
            continue
        previous = sql[index - 1] if index > 0 else ""
        following = sql[end] if end < len(sql) else ""
        if previous and (previous.isalnum() or previous == "_"):
            continue
        if following and (following.isalnum() or following == "_"):
            continue
        if index == 0:
            return None
        return clause
    return None


def _wrap_sql_line(line: str, *, width: int) -> list[str]:
    text = line.strip()
    if len(text) <= width:
        return [text]
    comma_parts = _split_sql_commas(text)
    if len(comma_parts) > 1:
        return _wrap_segments(comma_parts, width=width, continuation_indent="  ")
    return _wrap_text_line(text, width=width, continuation_indent="  ")


def _split_sql_commas(line: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in line:
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            current.append(char)
            continue
        if quote is None and char == ",":
            parts.append("".join(current).strip() + ",")
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _wrap_segments(segments: list[str], *, width: int, continuation_indent: str = "") -> list[str]:
    lines: list[str] = []
    current = ""
    for index, segment in enumerate(segments):
        segment_text = segment.strip()
        candidate = f"{current} {segment_text}".strip() if current else segment_text
        if current and len(candidate) > width:
            lines.append(current)
            current = f"{continuation_indent}{segment_text}" if index > 0 else segment_text
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _wrap_text_line(line: str, *, width: int, continuation_indent: str = "") -> list[str]:
    tokens = line.split()
    if not tokens:
        return [line]
    return _wrap_segments(tokens, width=width, continuation_indent=continuation_indent)


def _wrap_shell_line(line: str, *, width: int) -> list[str]:
    tokens = _shell_tokens_preserving_quotes(line)
    if len(line) <= width or len(tokens) <= 1:
        return [line]
    raw_lines = _wrap_segments(tokens, width=width, continuation_indent="  ")
    if len(raw_lines) <= 1:
        return raw_lines
    return [f"{part} \\" if index < len(raw_lines) - 1 else part for index, part in enumerate(raw_lines)]


def _shell_tokens_preserving_quotes(command: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in str(command or ""):
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            current.append(char)
            continue
        if quote is None and char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_actions(actions: list[ExecutedAction]) -> list[ExecutedAction]:
    seen: dict[tuple[str, str, str, str], int] = {}
    result: list[ExecutedAction] = []
    for action in actions:
        key = (action.tool, action.command or "", action.sql or "", action.status)
        if key in seen:
            existing = result[seen[key]]
            if not existing.result and action.result:
                result[seen[key]] = action
            continue
        seen[key] = len(result)
        result.append(action)
    return result


def _truncate(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 20)]}\n... truncated ..."
