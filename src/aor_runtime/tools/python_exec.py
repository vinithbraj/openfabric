from __future__ import annotations

import subprocess
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolResult
from aor_runtime.tools.base import BaseTool, ToolSpec


class PythonRunTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="python.run",
            description="Run a short Python snippet locally.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["code"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        code = str(arguments.get("code", ""))
        timeout = int(arguments.get("timeout", 30))
        try:
            completed = subprocess.run(
                [".venv/bin/python", "-c", code],
                cwd=self.settings.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return ToolResult(
                tool=self.spec.name,
                ok=completed.returncode == 0,
                output={"stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode},
                error=None if completed.returncode == 0 else completed.stderr.strip() or f"Python exited with {completed.returncode}",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc))
