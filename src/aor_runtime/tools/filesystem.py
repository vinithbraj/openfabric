from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolExecutionError


def resolve_path(settings: Settings, raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (settings.workspace_root / candidate).resolve()


def fs_exists(settings: Settings, path: str) -> dict[str, Any]:
    resolved = resolve_path(settings, path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
        "is_dir": resolved.is_dir(),
    }


def fs_read(settings: Settings, path: str) -> dict[str, Any]:
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise ToolExecutionError(f"Path is not a file: {resolved}")
    return {"path": str(resolved), "content": resolved.read_text()}


def fs_write(settings: Settings, path: str, content: str) -> dict[str, Any]:
    resolved = resolve_path(settings, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return {"path": str(resolved), "bytes_written": len(content.encode())}


def fs_copy(settings: Settings, src: str, dst: str) -> dict[str, Any]:
    source = resolve_path(settings, src)
    target = resolve_path(settings, dst)
    if not source.exists():
        raise ToolExecutionError(f"Source does not exist: {source}")
    if not source.is_file():
        raise ToolExecutionError(f"Source is not a file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"src": str(source), "dst": str(target)}


def fs_mkdir(settings: Settings, path: str) -> dict[str, Any]:
    resolved = resolve_path(settings, path)
    resolved.mkdir(parents=True, exist_ok=True)
    return {"path": str(resolved)}


def fs_list(settings: Settings, path: str) -> dict[str, Any]:
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {resolved}")
    entries = sorted(item.name for item in resolved.iterdir())
    return {"path": str(resolved), "entries": entries}


class FileExistsTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.exists",
            description="Check whether a file or directory exists.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return fs_exists(self.settings, str(arguments["path"]))


class FileCopyTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.copy",
            description="Copy a file from src to dst.",
            arguments_schema={
                "type": "object",
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                "required": ["src", "dst"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return fs_copy(self.settings, str(arguments["src"]), str(arguments["dst"]))


class FileReadTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.read",
            description="Read a text file from the local workspace.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return fs_read(self.settings, str(arguments["path"]))


class FileWriteTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.write",
            description="Write exact text content to a file.",
            arguments_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return fs_write(self.settings, str(arguments["path"]), str(arguments["content"]))


class MakeDirectoryTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.mkdir",
            description="Create a directory and any missing parents.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return fs_mkdir(self.settings, str(arguments["path"]))


class ListDirectoryTool(BaseTool):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.spec = ToolSpec(
            name="fs.list",
            description="List directory entries.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return fs_list(self.settings, str(arguments["path"]))
