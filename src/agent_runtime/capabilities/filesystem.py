"""Filesystem capabilities constrained to the workspace root."""

from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ExecutionResult


def _workspace_root(context: dict[str, Any]) -> Path:
    """Resolve the workspace root from execution context or the current directory."""

    execution_context = dict(context.get("execution_context") or {})
    raw_root = execution_context.get("workspace_root") or execution_context.get("cwd") or "."
    return Path(str(raw_root)).resolve()


def _normalize_workspace_path(raw_path: str, workspace_root: Path) -> tuple[Path, str]:
    """Normalize one path and ensure it stays inside the workspace root."""

    candidate = Path(str(raw_path or ".").strip() or ".")
    resolved = (
        (workspace_root / candidate).resolve(strict=False)
        if not candidate.is_absolute()
        else candidate.resolve(strict=False)
    )
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValidationError(f"path escapes workspace root: {raw_path}") from exc
    return resolved, relative.as_posix() or "."


def _is_secret_path(path: Path) -> bool:
    """Return whether a path points to an obvious secret filename."""

    name = path.name.lower()
    return name in {".env", "id_rsa", "id_ed25519", "credentials.json"} or name.startswith("secrets.")


def _entry_type(path: Path) -> str:
    """Return a stable entry type label for filesystem output."""

    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _entry_record(path: Path, workspace_root: Path) -> dict[str, Any]:
    """Convert a filesystem entry into a deterministic record."""

    stat = path.stat()
    relative = path.resolve(strict=False).relative_to(workspace_root).as_posix() or "."
    return {
        "name": path.name,
        "path": relative,
        "type": _entry_type(path),
        "size": int(stat.st_size),
        "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


class ListDirectoryCapability(BaseCapability):
    """List workspace-bounded filesystem entries."""

    manifest = CapabilityManifest(
        capability_id="filesystem.list_directory",
        domain="filesystem",
        operation_id="list_directory",
        name="List Directory",
        description="List entries in a directory path.",
        semantic_verbs=["read", "search"],
        object_types=["directory", "filesystem"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=["path"],
        optional_arguments=["recursive", "include_hidden", "limit"],
        output_schema={"entries": {"type": "array"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": "."}}],
        safety_notes=["Read-only directory metadata access."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Return directory entries while staying inside the workspace root."""

        payload = {"path": ".", "recursive": False, "include_hidden": False, "limit": 1000}
        payload.update(dict(arguments or {}))
        validated = self.validate_arguments(payload)

        workspace_root = _workspace_root(context)
        directory, relative_path = _normalize_workspace_path(str(validated["path"]), workspace_root)
        if not directory.exists():
            raise ValidationError(f"directory does not exist: {relative_path}")
        if not directory.is_dir():
            raise ValidationError(f"path is not a directory: {relative_path}")

        recursive = bool(validated.get("recursive", False))
        include_hidden = bool(validated.get("include_hidden", False))
        limit = max(1, int(validated.get("limit", 1000)))
        entries: list[dict[str, Any]] = []

        iterator = directory.rglob("*") if recursive else directory.iterdir()
        for entry in sorted(iterator, key=lambda item: item.as_posix()):
            if not include_hidden and any(part.startswith(".") for part in entry.relative_to(directory).parts):
                continue
            entries.append(_entry_record(entry, workspace_root))
            if len(entries) >= limit:
                break

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "path": relative_path,
                "entries": entries,
                "truncated": len(entries) >= limit,
            },
            metadata={"entry_count": len(entries)},
        )


class ReadFileCapability(BaseCapability):
    """Read a single workspace-bounded file with truncation safeguards."""

    manifest = CapabilityManifest(
        capability_id="filesystem.read_file",
        domain="filesystem",
        operation_id="read_file",
        name="Read File",
        description="Read a file from disk.",
        semantic_verbs=["read"],
        object_types=["file", "filesystem"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=["path"],
        optional_arguments=["max_bytes"],
        output_schema={"content_preview": {"type": "string"}, "truncated": {"type": "boolean"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": "README.md"}}],
        safety_notes=["Read-only file access."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Read a text preview from a workspace file and block obvious secrets."""

        payload = {"max_bytes": 100000}
        payload.update(dict(arguments or {}))
        validated = self.validate_arguments(payload)

        workspace_root = _workspace_root(context)
        file_path, relative_path = _normalize_workspace_path(str(validated["path"]), workspace_root)
        if _is_secret_path(file_path):
            raise ValidationError(f"access to secret file is blocked: {relative_path}")
        if not file_path.exists():
            raise ValidationError(f"file does not exist: {relative_path}")
        if not file_path.is_file():
            raise ValidationError(f"path is not a file: {relative_path}")

        max_bytes = max(1, int(validated.get("max_bytes", 100000)))
        raw_bytes = file_path.read_bytes()
        preview_bytes = raw_bytes[:max_bytes]
        truncated = len(raw_bytes) > max_bytes

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "path": relative_path,
                "content_preview": preview_bytes.decode("utf-8", errors="replace"),
                "truncated": truncated,
            },
            metadata={"bytes_read": len(preview_bytes), "total_bytes": len(raw_bytes)},
        )


class SearchFilesCapability(BaseCapability):
    """Search for files by glob-style pattern within the workspace."""

    manifest = CapabilityManifest(
        capability_id="filesystem.search_files",
        domain="filesystem",
        operation_id="search_files",
        name="Search Files",
        description="Search files by name or pattern.",
        semantic_verbs=["search"],
        object_types=["file", "filesystem"],
        argument_schema={"pattern": {"type": "string"}},
        required_arguments=["path", "pattern"],
        optional_arguments=["recursive", "limit"],
        output_schema={"matches": {"type": "array"}},
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"pattern": "*.py", "path": "src"}}],
        safety_notes=["Read-only filesystem search."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Return workspace-relative file matches for a glob pattern."""

        payload = {"path": ".", "recursive": True, "limit": 1000}
        payload.update(dict(arguments or {}))
        validated = self.validate_arguments(payload)

        workspace_root = _workspace_root(context)
        search_root, relative_root = _normalize_workspace_path(str(validated["path"]), workspace_root)
        if not search_root.exists():
            raise ValidationError(f"search path does not exist: {relative_root}")
        if not search_root.is_dir():
            raise ValidationError(f"search path is not a directory: {relative_root}")

        recursive = bool(validated.get("recursive", True))
        limit = max(1, int(validated.get("limit", 1000)))
        pattern = str(validated["pattern"])
        iterator = search_root.rglob("*") if recursive else search_root.iterdir()
        matches: list[str] = []

        for entry in sorted(iterator, key=lambda item: item.as_posix()):
            if not entry.is_file():
                continue
            if fnmatch(entry.name, pattern):
                matches.append(entry.resolve(strict=False).relative_to(workspace_root).as_posix())
                if len(matches) >= limit:
                    break

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "path": relative_root,
                "pattern": pattern,
                "matches": matches,
                "truncated": len(matches) >= limit,
            },
            metadata={"match_count": len(matches)},
        )
