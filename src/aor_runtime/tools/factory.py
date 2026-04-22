from __future__ import annotations

from aor_runtime.config import Settings, get_settings
from aor_runtime.tools.base import ToolRegistry
from aor_runtime.tools.filesystem import FileReadTool, FileWriteTool, ListDirectoryTool
from aor_runtime.tools.python_exec import PythonRunTool
from aor_runtime.tools.shell import ShellExecTool


def build_tool_registry(settings: Settings | None = None) -> ToolRegistry:
    configured = settings or get_settings()
    return ToolRegistry(
        [
            ShellExecTool(configured),
            FileReadTool(configured),
            FileWriteTool(configured),
            ListDirectoryTool(configured),
            PythonRunTool(configured),
        ]
    )
