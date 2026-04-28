from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import Field

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.runtime.shell_safety import classify_shell_command
from aor_runtime.tools.gateway import execute_gateway_command, resolve_execution_node, stream_gateway_command
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


def run_shell(settings: Settings, command: str, node: str = "") -> dict[str, Any]:
    trimmed = command.strip()
    if not trimmed:
        raise ToolExecutionError("Empty command.")

    policy = classify_shell_command(
        trimmed,
        mode=settings.shell_mode,
        allow_mutation_with_approval=settings.shell_allow_mutation_with_approval or settings.allow_destructive_shell,
    )
    if not policy.allowed:
        if policy.requires_approval:
            raise ToolExecutionError(f"Shell command requires approval: {policy.reason}")
        raise ToolExecutionError(f"Shell command blocked by policy: {policy.reason}")

    resolved_node = resolve_execution_node(settings, node)
    completed = _execute_with_shell_timeout(settings, node=resolved_node, command=trimmed)
    stdout = _cap_output(completed.stdout, settings.shell_max_output_chars)
    stderr = _cap_output(completed.stderr, settings.shell_max_output_chars)
    result = {
        "command": trimmed,
        "node": resolved_node,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": completed.exit_code,
        "risk": policy.risk,
        "requires_approval": policy.requires_approval,
        "policy_reason": policy.reason,
        "detected_operations": policy.detected_operations,
        "truncated": len(completed.stdout) > len(stdout) or len(completed.stderr) > len(stderr),
    }
    if completed.exit_code != 0:
        raise ToolExecutionError(stderr.strip() or f"Command exited with {completed.exit_code}")
    return result


def stream_shell(settings: Settings, command: str, node: str = "") -> Iterator[dict[str, Any]]:
    trimmed = command.strip()
    if not trimmed:
        raise ToolExecutionError("Empty command.")

    policy = classify_shell_command(
        trimmed,
        mode=settings.shell_mode,
        allow_mutation_with_approval=settings.shell_allow_mutation_with_approval or settings.allow_destructive_shell,
    )
    if not policy.allowed:
        if policy.requires_approval:
            raise ToolExecutionError(f"Shell command requires approval: {policy.reason}")
        raise ToolExecutionError(f"Shell command blocked by policy: {policy.reason}")

    resolved_node = resolve_execution_node(settings, node)
    for chunk in _stream_with_shell_timeout(settings, node=resolved_node, command=trimmed):
        payload = chunk.model_dump()
        payload["node"] = resolved_node
        payload["command"] = trimmed
        payload["risk"] = policy.risk
        payload["requires_approval"] = policy.requires_approval
        payload["policy_reason"] = policy.reason
        yield payload


class ShellExecTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        command: str
        node: str = ""

    class ToolResult(ToolResultModel):
        command: str
        node: str
        stdout: str
        stderr: str
        returncode: int
        risk: str = ""
        requires_approval: bool = False
        policy_reason: str = ""
        detected_operations: list[str] = Field(default_factory=list)
        truncated: bool = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="shell.exec",
            description="Execute a shell command on a logical node.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "node": {"type": "string"},
                },
                "required": ["command"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(run_shell(self.settings, arguments.command, arguments.node))

    def stream(self, arguments: ToolArgs) -> Iterator[dict[str, Any]]:
        return stream_shell(self.settings, arguments.command, arguments.node)

    def preview_command(self, arguments: ToolArgs) -> str:
        return str(arguments.command).strip()


def _cap_output(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 24)]}\n... truncated by policy ..."


def _execute_with_shell_timeout(settings: Settings, *, node: str, command: str):
    try:
        return execute_gateway_command(settings, node=node, command=command, timeout=settings.shell_command_timeout_seconds)
    except TypeError as exc:
        if "timeout" not in str(exc):
            raise
        return execute_gateway_command(settings, node=node, command=command)


def _stream_with_shell_timeout(settings: Settings, *, node: str, command: str):
    try:
        yield from stream_gateway_command(settings, node=node, command=command, timeout=settings.shell_command_timeout_seconds)
    except TypeError as exc:
        if "timeout" not in str(exc):
            raise
        yield from stream_gateway_command(settings, node=node, command=command)
