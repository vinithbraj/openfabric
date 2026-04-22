from __future__ import annotations

from pathlib import Path
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolResult
from aor_runtime.tools.base import BaseTool, ToolSpec


class FileReadTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.read_text",
            description="Read a text file from the local workspace.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        path = (self.settings.workspace_root / str(arguments.get("path", ""))).expanduser().resolve()
        try:
            return ToolResult(tool=self.spec.name, output={"path": str(path), "content": path.read_text()})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc))


class FileWriteTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.write_text",
            description="Write text content to a file inside the local workspace.",
            arguments_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        path = (self.settings.workspace_root / str(arguments.get("path", ""))).expanduser().resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments.get("content", "")))
            return ToolResult(tool=self.spec.name, output={"path": str(path)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc))


class ListDirectoryTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.list_dir",
            description="List directory entries in the local workspace.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        path = (self.settings.workspace_root / str(arguments.get("path", "."))).expanduser().resolve()
        try:
            entries = [
                {"name": item.name, "path": str(item), "is_dir": item.is_dir()}
                for item in sorted(path.iterdir(), key=lambda value: value.name.lower())
            ]
            return ToolResult(tool=self.spec.name, output={"path": str(path), "entries": entries})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=self.spec.name, ok=False, error=str(exc))
