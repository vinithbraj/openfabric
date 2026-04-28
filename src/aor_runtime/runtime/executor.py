from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlanStep, StepLog
from aor_runtime.runtime.dataflow import resolve_execution_step
from aor_runtime.runtime.policies import infer_output_mode
from aor_runtime.runtime.presentation import strip_internal_telemetry
from aor_runtime.runtime.response_renderer import ResponseRenderContext, RenderedResponse, render_agent_response
from aor_runtime.tools.base import ToolExecutionError, ToolRegistry


COUNT_TEXT_RE = re.compile(r"^\s*(-?\d+)(?:\s+\S.*)?$")


class PlanExecutor:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def describe_step(self, step: PlanStep) -> dict[str, Any]:
        preview: dict[str, Any] = {}
        node = str(step.args.get("node") or step.args.get("gateway_node") or "").strip()
        if node:
            preview["node"] = node
        command = str(step.args.get("command") or "").strip()
        if command:
            preview["command"] = command
            return preview

        tool = self.tools.get(step.action)
        preview_method = getattr(tool, "preview_command", None)
        if callable(preview_method):
            validated_args = tool.args_model.model_validate(step.args)
            described = str(preview_method(validated_args) or "").strip()
            if described:
                preview["command"] = described
        return preview

    def execute_step(
        self,
        step: PlanStep,
        *,
        step_outputs: dict[str, Any] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> StepLog:
        started = datetime.now(timezone.utc).isoformat()
        try:
            resolved_step = resolve_execution_step(step, step_outputs or {})
            if event_sink is not None and self._supports_streaming(resolved_step.action):
                output = self._invoke_streaming_tool(resolved_step, event_sink)
            else:
                output = self.tools.invoke(resolved_step.action, resolved_step.args)
            finished = datetime.now(timezone.utc).isoformat()
            if resolved_step.action == "python.exec" and not bool(output.get("success", False)):
                raise ToolExecutionError(str(output.get("error") or "python.exec failed."))
            if resolved_step.action == "fs.exists" and not bool(output.get("exists")):
                raise ToolExecutionError(f"Path does not exist: {resolved_step.args.get('path', '')}")
            if resolved_step.action == "fs.not_exists" and bool(output.get("exists")):
                raise ToolExecutionError(f"Path still exists: {resolved_step.args.get('path', '')}")
            return StepLog(
                step=resolved_step,
                result=output,
                success=True,
                started_at=started,
                finished_at=finished,
            )
        except Exception as exc:  # noqa: BLE001
            finished = datetime.now(timezone.utc).isoformat()
            return StepLog(
                step=step,
                result={},
                success=False,
                error=str(exc),
                started_at=started,
                finished_at=finished,
            )

    def _supports_streaming(self, action: str) -> bool:
        tool = self.tools.get(action)
        return callable(getattr(tool, "stream", None))

    def _invoke_streaming_tool(self, step: PlanStep, event_sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        tool = self.tools.get(step.action)
        validated_args = tool.args_model.model_validate(step.args)
        stream_method = getattr(tool, "stream", None)
        if stream_method is None:
            return tool.invoke(step.args)

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        returncode: int | None = None
        resolved_node = str(step.args.get("node") or step.args.get("gateway_node") or "")
        resolved_command = str(step.args.get("command", ""))
        stream_metadata: dict[str, Any] = {}

        for chunk in stream_method(validated_args):
            chunk_type = str(chunk.get("type") or "")
            if chunk_type in {"stdout", "stderr"}:
                text = str(chunk.get("text") or "")
                if not text:
                    continue
                if chunk_type == "stdout":
                    stdout_chunks.append(text)
                else:
                    stderr_chunks.append(text)
                event_sink(
                    {
                        "step_id": step.id,
                        "action": step.action,
                        "channel": chunk_type,
                        "text": text,
                        "node": str(chunk.get("node") or resolved_node),
                        "command": str(chunk.get("command") or resolved_command),
                    }
                )
                continue
            if chunk_type == "completed":
                returncode = int(chunk.get("exit_code") or 0)
                resolved_node = str(chunk.get("node") or resolved_node)
                resolved_command = str(chunk.get("command") or resolved_command)
                stream_metadata = dict(chunk)
                continue
            if chunk_type == "error":
                message = str(chunk.get("text") or f"Streaming {step.action} execution failed.")
                raise ToolExecutionError(message)

        build_stream_result = getattr(tool, "build_stream_result", None)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        finalized_returncode = int(returncode or 0)
        if callable(build_stream_result):
            return build_stream_result(
                validated_args,
                stdout=stdout,
                stderr=stderr,
                returncode=finalized_returncode,
                metadata=stream_metadata,
            )

        result = {
            "command": resolved_command,
            "node": resolved_node,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": finalized_returncode,
        }
        if result["returncode"] != 0:
            raise ToolExecutionError(result["stderr"].strip() or f"Command exited with {result['returncode']}")
        return result

    def execute(self, plan: ExecutionPlan) -> tuple[list[StepLog], dict[str, Any] | None]:
        history: list[StepLog] = []
        failure: dict[str, Any] | None = None
        step_outputs: dict[str, Any] = {}
        for step in plan.steps:
            log = self.execute_step(step, step_outputs=step_outputs)
            history.append(log)
            if log.success and log.step.output:
                step_outputs[log.step.output] = log.result
            if not log.success:
                failure = {
                    "reason": "tool_execution_failed",
                    "step": step.model_dump(),
                    "error": log.error or "step failed",
                    "history": [item.model_dump() for item in history],
                }
                break
        return history, failure


def summarize_final_output(
    goal: str,
    history: list[StepLog],
    settings: Settings | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts: list[str] = []
    if history:
        for item in history:
            result = item.result
            for key in ("path", "src", "dst"):
                value = result.get(key)
                if isinstance(value, str):
                    artifacts.append(value)
        artifacts = list(dict.fromkeys(artifacts))

    if not history:
        return {"content": "", "artifacts": artifacts, "metadata": {"goal": goal}}

    last = history[-1]
    action = last.step.action
    result = last.result
    output_mode = infer_output_mode(goal)

    if settings is not None and settings.response_render_mode != "raw":
        rendered = _render_final_result(goal, history, settings=settings, output_mode=output_mode, metadata=metadata or {})
        if rendered is not None:
            output_metadata = {"goal": goal, "response_render_mode": settings.response_render_mode}
            if rendered.title:
                output_metadata["title"] = rendered.title
            if rendered.hidden_events:
                output_metadata["hidden_events"] = rendered.hidden_events
            return {"content": rendered.markdown.strip(), "artifacts": artifacts, "metadata": output_metadata}

    if action == "fs.read":
        content = _shape_text_like_content(str(result.get("content", "")), output_mode)
    elif action == "fs.list":
        entries = result.get("entries", [])
        content = _shape_sequence_content(list(entries), output_mode, key="entries")
    elif action == "fs.find":
        matches = result.get("matches", [])
        content = _shape_sequence_content(list(matches), output_mode, key="matches")
    elif action == "fs.search_content":
        matches = result.get("matches", [])
        content = _shape_sequence_content(list(matches), output_mode, key="matches")
    elif action == "fs.aggregate":
        if output_mode == "count":
            content = str(result.get("file_count", "0"))
        else:
            content = str(result.get("summary_text", "")).strip()
    elif action == "fs.size":
        content = str(result.get("size_bytes", ""))
    elif action == "fs.exists":
        content = "true" if result.get("exists") else "false"
    elif action == "fs.not_exists":
        content = "true" if not result.get("exists") else "false"
    elif action == "python.exec":
        content = _shape_python_result(result, output_mode)
    elif action == "runtime.return":
        rendered = result.get("output")
        if isinstance(rendered, str):
            content = rendered
        else:
            value = result.get("value")
            content = _shape_text_like_content(str(value or ""), output_mode)
    elif action == "sql.query":
        content = _shape_sql_result(result, output_mode)
    elif action == "shell.exec":
        content = _shape_shell_output(str(result.get("stdout", "")).strip(), output_mode)
    else:
        lines: list[str] = []
        for item in history:
            step = item.step
            if step.action == "fs.write":
                lines.append(f"- wrote `{step.args.get('path', '')}`")
            elif step.action == "fs.copy":
                lines.append(f"- copied `{step.args.get('src', '')}` -> `{step.args.get('dst', '')}`")
            elif step.action == "fs.mkdir":
                lines.append(f"- created directory `{step.args.get('path', '')}`")
        content = "\n".join(lines)

    return {"content": content.strip(), "artifacts": artifacts, "metadata": {"goal": goal}}


def _render_final_result(
    goal: str,
    history: list[StepLog],
    *,
    settings: Settings,
    output_mode: str,
    metadata: dict[str, Any],
):
    if output_mode == "csv":
        return None
    last = history[-1]
    action = last.step.action
    previous_action = history[-2].step.action if len(history) >= 2 else None
    context = ResponseRenderContext(
        mode=settings.response_render_mode,  # type: ignore[arg-type]
        show_executed_commands=settings.show_executed_commands,
        show_validation=settings.show_validation_events,
        show_planning_steps=settings.show_planner_events,
        show_tool_events=settings.show_tool_events,
        show_debug_metadata=settings.show_debug_metadata or settings.include_internal_telemetry,
        enable_llm_summary=settings.enable_presentation_llm_summary,
        enable_insight_layer=settings.enable_insight_layer,
        enable_llm_insights=settings.enable_llm_insights,
        insight_max_facts=settings.insight_max_facts,
        insight_max_input_chars=settings.insight_max_input_chars,
        insight_max_output_chars=settings.insight_max_output_chars,
        max_rows=settings.sql_row_limit if settings.sql_row_limit < 50 else 20,
        max_command_length=settings.presentation_llm_max_input_chars,
        source_action=previous_action if action == "runtime.return" else action,
        output_mode=output_mode,
        goal=goal,
        llm_settings=settings,
    )
    if action == "runtime.return":
        value = last.result.get("value")
        if not _presentation_source_supported(previous_action) and not _presentation_value_supported(value):
            return None
        if output_mode == "json":
            clean = strip_internal_telemetry(value) if settings.response_render_mode == "user" else value
            return RenderedResponse(markdown=json.dumps(clean, ensure_ascii=False, sort_keys=True), title="JSON")
        return render_agent_response(value, execution_events=history, metadata={**metadata, "goal": goal}, context=context)
    elif action in {"sql.query", "shell.exec", "fs.aggregate", "fs.search_content", "fs.find", "fs.list", "fs.size"} or action.startswith("slurm."):
        if output_mode == "json":
            clean = strip_internal_telemetry(last.result) if settings.response_render_mode == "user" else last.result
            return RenderedResponse(markdown=json.dumps(clean, ensure_ascii=False, sort_keys=True), title="JSON")
        final_result: Any = str(last.result.get("stdout") or "").strip() if action == "shell.exec" else last.result
        return render_agent_response(final_result, execution_events=history, metadata={**metadata, "goal": goal}, context=context)
    else:
        return None


def _presentation_source_supported(action: str | None) -> bool:
    if not action:
        return False
    return action in {"sql.query", "shell.exec"} or action.startswith("slurm.") or action.startswith("fs.")


def _presentation_value_supported(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "results" in value and isinstance(value.get("results"), dict):
        return True
    if "rows" in value and "database" in value:
        return True
    if any(key in value for key in ("file_count", "total_size_bytes", "matches", "entries")):
        return True
    if any(key in value for key in ("jobs", "nodes", "partitions", "metric_group")):
        return True
    return False


def _shape_text_like_content(content: str, mode: str) -> str:
    stripped = str(content or "").strip()
    if mode == "count":
        count_value = _extract_count_value(stripped)
        if count_value is not None:
            return str(count_value)
    return stripped


def _shape_sequence_content(values: list[Any], mode: str, *, key: str) -> str:
    items = [str(value) for value in values]
    if mode == "count":
        return str(len(items))
    if mode == "csv":
        return ",".join(items)
    if mode == "json":
        return _dump_json({key: items})
    return "\n".join(items)


def _shape_python_result(result: dict[str, Any], mode: str) -> str:
    value = result.get("result")
    fallback = str(result.get("output") or "").strip()
    if mode == "count":
        count_value = _extract_count_value(value)
        if count_value is None:
            count_value = _extract_count_value(fallback)
        if count_value is not None:
            return str(count_value)
    if mode == "csv":
        csv_value = _extract_textual_value(value, preferred_keys=("csv", "value", "text", "content"))
        if csv_value is not None:
            return csv_value
        return fallback
    if mode == "json":
        json_value = _extract_json_value(value)
        if json_value is not None:
            return json_value
        if _looks_like_json_object_or_array(fallback):
            return fallback
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is not None:
        return _dump_json(value)
    return fallback


def _shape_sql_result(result: dict[str, Any], mode: str) -> str:
    rows = list(result.get("rows", []) or [])
    database = str(result.get("database", ""))
    row_count = int(result.get("row_count", 0))
    if mode == "count":
        scalar = _extract_single_row_scalar(rows)
        if scalar is not None:
            return str(scalar)
        return str(row_count)
    if mode == "csv":
        return _rows_to_csv(rows)
    if mode == "json":
        return _dump_json({"database": database, "row_count": row_count, "rows": rows})
    return _dump_json({"database": database, "row_count": row_count, "rows": rows})


def _shape_shell_output(stdout: str, mode: str) -> str:
    if mode == "count":
        count_value = _extract_count_value(stdout)
        if count_value is not None:
            return str(count_value)
    return stdout.strip()


def _extract_textual_value(value: Any, *, preferred_keys: tuple[str, ...]) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in preferred_keys:
            nested = value.get(key)
            if isinstance(nested, str):
                return nested.strip()
        if len(value) == 1:
            only_value = next(iter(value.values()))
            if isinstance(only_value, str):
                return only_value.strip()
    return None


def _extract_json_value(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("json", "value"):
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                return _dump_json(nested)
        return _dump_json(value)
    if isinstance(value, list):
        return _dump_json(value)
    if isinstance(value, str) and _looks_like_json_object_or_array(value):
        return value.strip()
    return None


def _extract_single_row_scalar(rows: list[Any]) -> Any | None:
    if len(rows) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict) or len(row) != 1:
        return None
    return next(iter(row.values()))


def _rows_to_csv(rows: list[Any]) -> str:
    if not rows:
        return ""
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return ",".join(str(item) for item in rows)
    headers = list(normalized_rows[0].keys())
    if len(headers) == 1:
        header = headers[0]
        return ",".join(str(row.get(header, "")) for row in normalized_rows)
    lines = [",".join(headers)]
    for row in normalized_rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    return "\n".join(lines)


def _extract_count_value(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = COUNT_TEXT_RE.match(value.strip())
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("count", "value", "result"):
            if key in value:
                nested = _extract_count_value(value.get(key))
                if nested is not None:
                    return nested
        if len(value) == 1:
            return _extract_count_value(next(iter(value.values())))
    if isinstance(value, list) and len(value) == 1:
        return _extract_count_value(value[0])
    return None


def _looks_like_json_object_or_array(value: str) -> bool:
    stripped = str(value or "").strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
