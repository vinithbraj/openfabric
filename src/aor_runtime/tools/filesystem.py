"""OpenFABRIC Runtime Module: aor_runtime.tools.filesystem

Purpose:
    Implement root-constrained filesystem tools.

Responsibilities:
    Read, write, list, glob, find, size, copy, mkdir, and existence-check paths under configured allowed roots.

Data flow / Interfaces:
    Receives validated filesystem arguments and returns structured file/path metadata.

Boundaries:
    Must prevent traversal, unsafe writes, and accidental exposure of ignored runtime artifacts unless explicitly requested.
"""

from __future__ import annotations

import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel


def resolve_path(settings: Settings, raw_path: str) -> Path:
    """Resolve path for the surrounding runtime workflow.

    Inputs:
        Receives settings, raw_path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.resolve_path.
    """
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (settings.workspace_root / candidate).resolve()


def fs_exists(settings: Settings, path: str) -> dict[str, Any]:
    """Fs exists for the surrounding runtime workflow.

    Inputs:
        Receives settings, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_exists.
    """
    resolved = resolve_path(settings, path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
        "is_dir": resolved.is_dir(),
    }


def fs_not_exists(settings: Settings, path: str) -> dict[str, Any]:
    """Fs not exists for the surrounding runtime workflow.

    Inputs:
        Receives settings, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_not_exists.
    """
    return fs_exists(settings, path)


def fs_read(settings: Settings, path: str) -> dict[str, Any]:
    """Fs read for the surrounding runtime workflow.

    Inputs:
        Receives settings, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_read.
    """
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise ToolExecutionError(f"Path is not a file: {resolved}")
    return {"path": str(resolved), "content": resolved.read_text()}


def fs_write(settings: Settings, path: str, content: str) -> dict[str, Any]:
    """Fs write for the surrounding runtime workflow.

    Inputs:
        Receives settings, path, content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_write.
    """
    resolved = resolve_path(settings, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return {"path": str(resolved), "bytes_written": len(content.encode())}


def fs_copy(settings: Settings, src: str, dst: str) -> dict[str, Any]:
    """Fs copy for the surrounding runtime workflow.

    Inputs:
        Receives settings, src, dst for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_copy.
    """
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
    """Fs mkdir for the surrounding runtime workflow.

    Inputs:
        Receives settings, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_mkdir.
    """
    resolved = resolve_path(settings, path)
    resolved.mkdir(parents=True, exist_ok=True)
    return {"path": str(resolved)}


def fs_list(settings: Settings, path: str) -> dict[str, Any]:
    """Fs list for the surrounding runtime workflow.

    Inputs:
        Receives settings, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_list.
    """
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {resolved}")
    entries = sorted(item.name for item in resolved.iterdir())
    return {"path": str(resolved), "entries": entries}


def fs_glob(
    settings: Settings,
    path: str,
    pattern: str = "*",
    recursive: bool = False,
    file_only: bool = True,
    dir_only: bool = False,
    path_style: str = "relative",
) -> dict[str, Any]:
    """Fs glob for the surrounding runtime workflow.

    Inputs:
        Receives settings, path, pattern, recursive, file_only, dir_only, path_style for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_glob.
    """
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {resolved}")
    normalized_pattern = str(pattern or "*").strip() or "*"
    if file_only and dir_only:
        raise ToolExecutionError("fs.glob cannot set both file_only and dir_only to true.")
    if path_style not in {"name", "relative", "absolute"}:
        raise ToolExecutionError("fs.glob path_style must be one of: name, relative, absolute.")

    root = resolved.resolve()
    iterator = root.rglob("*") if recursive else root.iterdir()
    entries: list[dict[str, Any]] = []
    for item in iterator:
        try:
            item.resolve().relative_to(root)
        except ValueError as exc:
            raise ToolExecutionError(f"fs.glob discovered a path outside the requested root: {item}") from exc
        is_file = item.is_file()
        is_dir = item.is_dir()
        if file_only and not is_file:
            continue
        if dir_only and not is_dir:
            continue
        if not file_only and not dir_only and not (is_file or is_dir):
            continue
        if not _glob_matches(item=item, root=root, pattern=normalized_pattern):
            continue
        relative_path = str(item.relative_to(resolved))
        entries.append(
            {
                "name": item.name,
                "path": str(item),
                "relative_path": relative_path,
                "is_file": is_file,
                "is_dir": is_dir,
            }
        )
    entries.sort(key=lambda item: str(item["relative_path"]))
    matches = [_format_glob_match(item, path_style=path_style) for item in entries]
    return {
        "path": str(resolved),
        "pattern": normalized_pattern,
        "recursive": bool(recursive),
        "path_style": path_style,
        "matches": matches,
        "entries": entries,
    }


def fs_find(settings: Settings, path: str, pattern: str) -> dict[str, Any]:
    """Fs find for the surrounding runtime workflow.

    Inputs:
        Receives settings, path, pattern for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_find.
    """
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {resolved}")
    normalized_pattern = str(pattern or "").strip()
    if not normalized_pattern:
        raise ToolExecutionError("Pattern must be non-empty.")
    matches = sorted(str(item.relative_to(resolved)) for item in resolved.rglob(normalized_pattern) if item.is_file())
    return {"path": str(resolved), "pattern": normalized_pattern, "matches": matches}


def fs_size(settings: Settings, path: str) -> dict[str, Any]:
    """Fs size for the surrounding runtime workflow.

    Inputs:
        Receives settings, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_size.
    """
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise ToolExecutionError(f"Path is not a file: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size}


def fs_aggregate(
    settings: Settings,
    path: str,
    pattern: str = "*",
    recursive: bool = True,
    file_only: bool = True,
    include_matches: bool = True,
    path_style: Literal["name", "relative", "absolute"] = "relative",
    size_unit: Literal["bytes", "kb", "mb", "gb", "auto"] = "auto",
    aggregate: Literal["total_size", "count", "count_and_total_size"] = "total_size",
) -> dict[str, Any]:
    """Fs aggregate for the surrounding runtime workflow.

    Inputs:
        Receives settings, path, pattern, recursive, file_only, include_matches, path_style, size_unit, ... for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem.fs_aggregate.
    """
    resolved = resolve_path(settings, path)
    if not resolved.exists():
        raise ToolExecutionError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {resolved}")
    normalized_pattern = str(pattern or "*").strip() or "*"
    if path_style not in {"name", "relative", "absolute"}:
        raise ToolExecutionError("fs.aggregate path_style must be one of: name, relative, absolute.")

    root = resolved.resolve()
    iterator = root.rglob("*") if recursive else root.iterdir()
    matches: list[dict[str, Any]] = []
    total_size_bytes = 0
    file_count = 0

    for item in iterator:
        try:
            resolved_item = item.resolve()
            resolved_item.relative_to(root)
        except ValueError:
            # Symlinks inside the requested root may point elsewhere. Keep the
            # root boundary strict, but skip those entries instead of failing
            # the whole aggregate request.
            continue
        if file_only and not item.is_file():
            continue
        if not file_only and not (item.is_file() or item.is_dir()):
            continue
        if not _glob_matches(item=item, root=root, pattern=normalized_pattern):
            continue
        if not item.is_file():
            continue
        try:
            size_bytes = item.stat().st_size
        except OSError:
            continue
        relative_path = str(item.relative_to(root))
        file_count += 1
        total_size_bytes += size_bytes
        if include_matches:
            matches.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "relative_path": relative_path,
                    "size_bytes": size_bytes,
                    "display_path": _format_glob_match(
                        {"name": item.name, "path": str(item), "relative_path": relative_path},
                        path_style=path_style,
                    ),
                }
            )

    if include_matches:
        matches.sort(key=lambda item: str(item["relative_path"]))

    display_size = _format_aggregate_size(total_size_bytes, size_unit)
    summary_text = _format_aggregate_summary(file_count, total_size_bytes, aggregate=aggregate, size_unit=size_unit, display_size=display_size)
    rendered_matches = []
    if include_matches:
        rendered_matches = [
            {
                "name": str(item["name"]),
                "path": str(item["path"]),
                "relative_path": str(item["relative_path"]),
                "size_bytes": int(item["size_bytes"]),
            }
            for item in matches
        ]

    return {
        "path": str(resolved),
        "pattern": normalized_pattern,
        "recursive": bool(recursive),
        "file_count": int(file_count),
        "total_size_bytes": int(total_size_bytes),
        "matches": rendered_matches,
        "summary_text": summary_text,
        "display_size": display_size,
    }


class FileExistsTool(BaseTool):
    """Represent file exists tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileExistsTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileExistsTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        exists: bool
        is_file: bool
        is_dir: bool

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileExistsTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileExistsTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.exists",
            description="Check whether a file or directory exists.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for FileExistsTool instances.

        Inputs:
            Receives arguments for this FileExistsTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileExistsTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_exists(self.settings, arguments.path))


class FileNotExistsTool(BaseTool):
    """Represent file not exists tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileNotExistsTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileNotExistsTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        exists: bool
        is_file: bool
        is_dir: bool

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileNotExistsTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileNotExistsTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.not_exists",
            description="Check that a file or directory does not exist.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for FileNotExistsTool instances.

        Inputs:
            Receives arguments for this FileNotExistsTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileNotExistsTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_not_exists(self.settings, arguments.path))


class FileCopyTool(BaseTool):
    """Represent file copy tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileCopyTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileCopyTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        src: str
        dst: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        src: str
        dst: str

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileCopyTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileCopyTool.__init__ calls and related tests.
        """
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
        """Run for FileCopyTool instances.

        Inputs:
            Receives arguments for this FileCopyTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileCopyTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_copy(self.settings, arguments.src, arguments.dst))


class FileReadTool(BaseTool):
    """Represent file read tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileReadTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileReadTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        content: str

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileReadTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileReadTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.read",
            description="Read a text file from the local workspace.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for FileReadTool instances.

        Inputs:
            Receives arguments for this FileReadTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileReadTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_read(self.settings, arguments.path))


class FileWriteTool(BaseTool):
    """Represent file write tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileWriteTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileWriteTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str
        content: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        bytes_written: int

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileWriteTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileWriteTool.__init__ calls and related tests.
        """
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
        """Run for FileWriteTool instances.

        Inputs:
            Receives arguments for this FileWriteTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileWriteTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_write(self.settings, arguments.path, arguments.content))


class MakeDirectoryTool(BaseTool):
    """Represent make directory tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by MakeDirectoryTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.MakeDirectoryTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this MakeDirectoryTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through MakeDirectoryTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.mkdir",
            description="Create a directory and any missing parents.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for MakeDirectoryTool instances.

        Inputs:
            Receives arguments for this MakeDirectoryTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through MakeDirectoryTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_mkdir(self.settings, arguments.path))


class ListDirectoryTool(BaseTool):
    """Represent list directory tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ListDirectoryTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.ListDirectoryTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        entries: list[str]

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this ListDirectoryTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through ListDirectoryTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.list",
            description="List directory entries.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for ListDirectoryTool instances.

        Inputs:
            Receives arguments for this ListDirectoryTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through ListDirectoryTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_list(self.settings, arguments.path))


class GlobFilesTool(BaseTool):
    """Represent glob files tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by GlobFilesTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.GlobFilesTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str
        pattern: str = "*"
        recursive: bool = False
        file_only: bool = True
        dir_only: bool = False
        path_style: str = "relative"

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        pattern: str
        recursive: bool
        path_style: str
        matches: list[str]
        entries: list[dict[str, Any]]

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this GlobFilesTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through GlobFilesTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.glob",
            description="Find matching files or directories under a root with optional non-recursive semantics.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "file_only": {"type": "boolean"},
                    "dir_only": {"type": "boolean"},
                    "path_style": {"type": "string", "enum": ["name", "relative", "absolute"]},
                },
                "required": ["path"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for GlobFilesTool instances.

        Inputs:
            Receives arguments for this GlobFilesTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through GlobFilesTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(
            fs_glob(
                self.settings,
                arguments.path,
                pattern=arguments.pattern,
                recursive=arguments.recursive,
                file_only=arguments.file_only,
                dir_only=arguments.dir_only,
                path_style=arguments.path_style,
            )
        )


def _glob_matches(*, item: Path, root: Path, pattern: str) -> bool:
    """Handle the internal glob matches helper path for this module.

    Inputs:
        Receives item, root, pattern for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem._glob_matches.
    """
    relative_path = str(item.relative_to(root))
    if "/" in pattern or "\\" in pattern:
        normalized_relative = relative_path.replace("\\", "/")
        normalized_pattern = pattern.replace("\\", "/")
        return fnmatch(normalized_relative, normalized_pattern)
    return fnmatch(item.name, pattern)


def _format_glob_match(entry: dict[str, Any], *, path_style: str) -> str:
    """Handle the internal format glob match helper path for this module.

    Inputs:
        Receives entry, path_style for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem._format_glob_match.
    """
    if path_style == "name":
        return str(entry["name"])
    if path_style == "absolute":
        return str(entry["path"])
    return str(entry["relative_path"])


def _format_aggregate_summary(
    file_count: int,
    total_size_bytes: int,
    *,
    aggregate: Literal["total_size", "count", "count_and_total_size"],
    size_unit: Literal["bytes", "kb", "mb", "gb", "auto"],
    display_size: str,
) -> str:
    """Handle the internal format aggregate summary helper path for this module.

    Inputs:
        Receives file_count, total_size_bytes, aggregate, size_unit, display_size for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem._format_aggregate_summary.
    """
    label = "file" if file_count == 1 else "files"
    if aggregate == "count":
        return f"{file_count} {label}"
    if size_unit == "bytes":
        return f"{file_count} {label}, {total_size_bytes} bytes"
    return f"{file_count} {label}, {display_size}"


def _format_aggregate_size(total_size_bytes: int, size_unit: Literal["bytes", "kb", "mb", "gb", "auto"]) -> str:
    """Handle the internal format aggregate size helper path for this module.

    Inputs:
        Receives total_size_bytes, size_unit for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.filesystem._format_aggregate_size.
    """
    if size_unit == "bytes" or total_size_bytes < 1024 and size_unit == "auto":
        return f"{total_size_bytes} bytes"

    units = {
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
    }
    if size_unit == "auto":
        if total_size_bytes >= 1024**3:
            size_unit = "gb"
        elif total_size_bytes >= 1024**2:
            size_unit = "mb"
        else:
            size_unit = "kb"
    divisor = units[str(size_unit)]
    scaled = total_size_bytes / divisor
    return f"{scaled:.1f} {str(size_unit).upper()} ({total_size_bytes} bytes)"


class FindFilesTool(BaseTool):
    """Represent find files tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FindFilesTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FindFilesTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str
        pattern: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        pattern: str
        matches: list[str]

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FindFilesTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FindFilesTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.find",
            description="Recursively find files matching a glob pattern under a directory.",
            arguments_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}},
                "required": ["path", "pattern"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for FindFilesTool instances.

        Inputs:
            Receives arguments for this FindFilesTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FindFilesTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_find(self.settings, arguments.path, arguments.pattern))


class FileSizeTool(BaseTool):
    """Represent file size tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileSizeTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileSizeTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        size_bytes: int

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileSizeTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileSizeTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.size",
            description="Return the exact size of a file in bytes.",
            arguments_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for FileSizeTool instances.

        Inputs:
            Receives arguments for this FileSizeTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileSizeTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(fs_size(self.settings, arguments.path))


class FileAggregateTool(BaseTool):
    """Represent file aggregate tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileAggregateTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.filesystem.FileAggregateTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolArgs and related tests.
        """
        path: str
        pattern: str = "*"
        recursive: bool = True
        file_only: bool = True
        include_matches: bool = True
        path_style: Literal["name", "relative", "absolute"] = "relative"
        size_unit: Literal["bytes", "kb", "mb", "gb", "auto"] = "auto"
        aggregate: Literal["total_size", "count", "count_and_total_size"] = "total_size"

    class AggregateMatch(ToolResultModel):
        """Represent aggregate match within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by AggregateMatch.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.AggregateMatch and related tests.
        """
        name: str
        path: str
        relative_path: str
        size_bytes: int

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.filesystem.ToolResult and related tests.
        """
        path: str
        pattern: str
        recursive: bool
        file_count: int
        total_size_bytes: int
        matches: list[AggregateMatch]
        summary_text: str
        display_size: str

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this FileAggregateTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through FileAggregateTool.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.aggregate",
            description="Aggregate matching files under a directory for deterministic counts and total size.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "file_only": {"type": "boolean"},
                    "include_matches": {"type": "boolean"},
                    "path_style": {"type": "string", "enum": ["name", "relative", "absolute"]},
                    "size_unit": {"type": "string", "enum": ["bytes", "kb", "mb", "gb", "auto"]},
                    "aggregate": {"type": "string", "enum": ["total_size", "count", "count_and_total_size"]},
                },
                "required": ["path"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for FileAggregateTool instances.

        Inputs:
            Receives arguments for this FileAggregateTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through FileAggregateTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(
            fs_aggregate(
                self.settings,
                arguments.path,
                pattern=arguments.pattern,
                recursive=arguments.recursive,
                file_only=arguments.file_only,
                include_matches=arguments.include_matches,
                path_style=arguments.path_style,
                size_unit=arguments.size_unit,
                aggregate=arguments.aggregate,
            )
        )
