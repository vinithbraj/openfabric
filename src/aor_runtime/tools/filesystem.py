from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


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


def fs_not_exists(settings: Settings, path: str) -> dict[str, Any]:
    return fs_exists(settings, path)


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
    class ToolArgs(ToolArgsModel):
        path: str

    class ToolResult(ToolResultModel):
        path: str
        exists: bool
        is_file: bool
        is_dir: bool

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.exists",
            description="Check whether a file or directory exists.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_exists(self.settings, arguments.path))


class FileNotExistsTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        path: str

    class ToolResult(ToolResultModel):
        path: str
        exists: bool
        is_file: bool
        is_dir: bool

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.not_exists",
            description="Check that a file or directory does not exist.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_not_exists(self.settings, arguments.path))


class FileCopyTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        src: str
        dst: str

    class ToolResult(ToolResultModel):
        src: str
        dst: str

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.copy",
            description="Copy a file from src to dst.",
            arguments_schema={
                "type": "object",
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                "required": ["src", "dst"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_copy(self.settings, arguments.src, arguments.dst))


class FileReadTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        path: str

    class ToolResult(ToolResultModel):
        path: str
        content: str

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.read",
            description="Read a text file from the local workspace.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_read(self.settings, arguments.path))


class FileWriteTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        path: str
        content: str

    class ToolResult(ToolResultModel):
        path: str
        bytes_written: int

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.write",
            description="Write exact text content to a file.",
            arguments_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_write(self.settings, arguments.path, arguments.content))


class MakeDirectoryTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        path: str

    class ToolResult(ToolResultModel):
        path: str

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.mkdir",
            description="Create a directory and any missing parents.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_mkdir(self.settings, arguments.path))


class ListDirectoryTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        path: str

    class ToolResult(ToolResultModel):
        path: str
        entries: list[str]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.list",
            description="List directory entries.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(fs_list(self.settings, arguments.path))
