from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aor_runtime.core.contracts import ExecutionPlan, PlanStep, StepLog
from aor_runtime.tools.base import ToolExecutionError, ToolRegistry


class PlanExecutor:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def execute(self, plan: ExecutionPlan) -> tuple[list[StepLog], dict[str, Any] | None]:
        history: list[StepLog] = []
        failure: dict[str, Any] | None = None

        for step in plan.steps:
            started = datetime.now(timezone.utc).isoformat()
            try:
                output = self.tools.invoke(step.action, step.args)
                finished = datetime.now(timezone.utc).isoformat()
                if step.action == "python.exec" and not bool(output.get("success", False)):
                    raise ToolExecutionError(str(output.get("error") or "python.exec failed."))
                if step.action == "fs.exists" and not bool(output.get("exists")):
                    raise ToolExecutionError(f"Path does not exist: {step.args.get('path', '')}")
                history.append(
                    StepLog(
                        step=step,
                        result=output,
                        success=True,
                        started_at=started,
                        finished_at=finished,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                finished = datetime.now(timezone.utc).isoformat()
                history.append(
                    StepLog(
                        step=step,
                        result={},
                        success=False,
                        error=str(exc),
                        started_at=started,
                        finished_at=finished,
                    )
                )
                failure = {
                    "reason": "tool_execution_failed",
                    "step": step.model_dump(),
                    "error": str(exc),
                    "history": [item.model_dump() for item in history],
                }
                break

        return history, failure


def summarize_final_output(goal: str, history: list[StepLog]) -> dict[str, Any]:
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

    if action == "fs.read":
        content = str(result.get("content", ""))
    elif action == "fs.list":
        entries = result.get("entries", [])
        content = "\n".join(str(entry) for entry in entries)
    elif action == "fs.exists":
        content = "true" if result.get("exists") else "false"
    elif action == "python.exec":
        content = str(result.get("output") or "").strip()
    elif action == "shell.exec":
        content = str(result.get("stdout", "")).strip()
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
