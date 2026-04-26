from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.gateway import execute_gateway_command, resolve_execution_node, stream_gateway_command
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


HIGH_RISK_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",
]


def run_shell(settings: Settings, command: str, node: str = "") -> dict[str, Any]:
    trimmed = command.strip()
    if not trimmed:
        raise ToolExecutionError("Empty command.")

    if not settings.allow_destructive_shell:
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, trimmed):
                raise ToolExecutionError("Blocked high-risk shell command by policy.")

    resolved_node = resolve_execution_node(settings, node)
    completed = execute_gateway_command(settings, node=resolved_node, command=trimmed)
    result = {
        "command": trimmed,
        "node": resolved_node,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.exit_code,
    }
    if completed.exit_code != 0:
        raise ToolExecutionError(completed.stderr.strip() or f"Command exited with {completed.exit_code}")
    return result


def stream_shell(settings: Settings, command: str, node: str = "") -> Iterator[dict[str, Any]]:
    trimmed = command.strip()
    if not trimmed:
        raise ToolExecutionError("Empty command.")

    if not settings.allow_destructive_shell:
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, trimmed):
                raise ToolExecutionError("Blocked high-risk shell command by policy.")

    resolved_node = resolve_execution_node(settings, node)
    for chunk in stream_gateway_command(settings, node=resolved_node, command=trimmed):
        payload = chunk.model_dump()
        payload["node"] = resolved_node
        payload["command"] = trimmed
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
