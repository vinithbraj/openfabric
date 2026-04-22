from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolResult
from aor_runtime.tools.base import BaseTool, ToolSpec


HIGH_RISK_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",
]


class ShellExecTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
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

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments.get("command", "")).strip()
        cwd = str(arguments.get("cwd", "")).strip()
        timeout = int(arguments.get("timeout", 60))
        if not command:
            return ToolResult(tool=self.spec.name, ok=False, error="Empty command.")

        if not self.settings.allow_destructive_shell:
            for pattern in HIGH_RISK_PATTERNS:
                if re.search(pattern, command):
                    return ToolResult(tool=self.spec.name, ok=False, error="Blocked high-risk shell command by policy.")

        if not cwd:
            resolved_cwd = self.settings.workspace_root
        else:
            candidate = Path(cwd).expanduser()
            resolved_cwd = candidate.resolve() if candidate.is_absolute() else (self.settings.workspace_root / candidate).resolve()
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=resolved_cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return ToolResult(
                tool=self.spec.name,
                ok=completed.returncode == 0,
                output={
                    "command": command,
                    "cwd": str(resolved_cwd),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                error=None if completed.returncode == 0 else completed.stderr.strip() or f"Command exited with {completed.returncode}",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc))
