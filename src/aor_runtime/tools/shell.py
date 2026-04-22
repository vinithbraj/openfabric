from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


HIGH_RISK_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",
]


def run_shell(settings: Settings, command: str, cwd: str = "", timeout: int = 60) -> dict[str, Any]:
    trimmed = command.strip()
    if not trimmed:
        raise ToolExecutionError("Empty command.")

    if not settings.allow_destructive_shell:
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, trimmed):
                raise ToolExecutionError("Blocked high-risk shell command by policy.")

    if not cwd:
        resolved_cwd = settings.workspace_root
    else:
        candidate = Path(cwd).expanduser()
        resolved_cwd = candidate.resolve() if candidate.is_absolute() else (settings.workspace_root / candidate).resolve()

    completed = subprocess.run(
        ["bash", "-lc", trimmed],
        cwd=resolved_cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    result = {
        "command": trimmed,
        "cwd": str(resolved_cwd),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise ToolExecutionError(completed.stderr.strip() or f"Command exited with {completed.returncode}")
    return result


class ShellExecTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        command: str
        cwd: str = ""
        timeout: int = 60

    class ToolResult(ToolResultModel):
        command: str
        cwd: str
        stdout: str
        stderr: str
        returncode: int

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="shell.exec",
            description="Execute a shell command on the local machine.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            run_shell(self.settings, arguments.command, arguments.cwd, arguments.timeout)
        )
