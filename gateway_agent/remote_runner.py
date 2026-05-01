from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


class RemoteToolError(Exception):
    """Raised when a gateway-owned remote tool request is invalid or fails safely."""


OPERATION_SPECS: dict[str, dict[str, set[str]]] = {
    "filesystem.list_directory": {
        "required": {"path"},
        "optional": {"recursive", "include_hidden", "limit"},
    },
    "filesystem.read_file": {
        "required": {"path"},
        "optional": {"max_bytes"},
    },
    "filesystem.search_files": {
        "required": {"path", "pattern"},
        "optional": {"recursive", "limit"},
    },
    "shell.inspect_system": {
        "required": set(),
        "optional": {"scope"},
    },
}


def _workspace_root(workspace_root: Path | None = None) -> Path:
    """Resolve the remote workspace root for filesystem operations."""

    return (workspace_root or Path.cwd()).resolve()


def _normalize_workspace_path(raw_path: str, workspace_root: Path) -> tuple[Path, str]:
    """Normalize one remote path and ensure it stays inside the remote workspace root."""

    candidate = Path(str(raw_path or ".").strip() or ".")
    resolved = (
        (workspace_root / candidate).resolve(strict=False)
        if not candidate.is_absolute()
        else candidate.resolve(strict=False)
    )
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise RemoteToolError(f"path escapes workspace root: {raw_path}") from exc
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


def _list_directory(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """List workspace-bounded entries from the remote host."""

    path = str(arguments.get("path") or ".")
    recursive = bool(arguments.get("recursive", False))
    include_hidden = bool(arguments.get("include_hidden", False))
    limit = max(1, int(arguments.get("limit", 1000)))

    directory, relative_path = _normalize_workspace_path(path, workspace_root)
    if not directory.exists():
        raise RemoteToolError(f"directory does not exist: {relative_path}")
    if not directory.is_dir():
        raise RemoteToolError(f"path is not a directory: {relative_path}")

    entries: list[dict[str, Any]] = []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for entry in sorted(iterator, key=lambda item: item.as_posix()):
        if not include_hidden and any(part.startswith(".") for part in entry.relative_to(directory).parts):
            continue
        entries.append(_entry_record(entry, workspace_root))
        if len(entries) >= limit:
            break

    return {
        "data_preview": {
            "path": relative_path,
            "entries": entries,
            "truncated": len(entries) >= limit,
        },
        "metadata": {"entry_count": len(entries)},
    }


def _read_file(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Read a workspace-bounded file preview from the remote host."""

    path = str(arguments.get("path") or "").strip()
    max_bytes = max(1, int(arguments.get("max_bytes", 100000)))

    file_path, relative_path = _normalize_workspace_path(path, workspace_root)
    if _is_secret_path(file_path):
        raise RemoteToolError(f"access to secret file is blocked: {relative_path}")
    if not file_path.exists():
        raise RemoteToolError(f"file does not exist: {relative_path}")
    if not file_path.is_file():
        raise RemoteToolError(f"path is not a file: {relative_path}")

    raw_bytes = file_path.read_bytes()
    preview_bytes = raw_bytes[:max_bytes]
    truncated = len(raw_bytes) > max_bytes
    return {
        "data_preview": {
            "path": relative_path,
            "content_preview": preview_bytes.decode("utf-8", errors="replace"),
            "truncated": truncated,
        },
        "metadata": {"bytes_read": len(preview_bytes), "total_bytes": len(raw_bytes)},
    }


def _search_files(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Search for files by glob pattern on the remote host."""

    path = str(arguments.get("path") or ".")
    pattern = str(arguments.get("pattern") or "").strip()
    recursive = bool(arguments.get("recursive", True))
    limit = max(1, int(arguments.get("limit", 1000)))
    if not pattern:
        raise RemoteToolError("pattern is required.")

    search_root, relative_root = _normalize_workspace_path(path, workspace_root)
    if not search_root.exists():
        raise RemoteToolError(f"search path does not exist: {relative_root}")
    if not search_root.is_dir():
        raise RemoteToolError(f"search path is not a directory: {relative_root}")

    iterator = search_root.rglob("*") if recursive else search_root.iterdir()
    matches: list[str] = []
    for entry in sorted(iterator, key=lambda item: item.as_posix()):
        if not entry.is_file():
            continue
        if fnmatch(entry.name, pattern):
            matches.append(entry.resolve(strict=False).relative_to(workspace_root).as_posix())
            if len(matches) >= limit:
                break

    return {
        "data_preview": {
            "path": relative_root,
            "pattern": pattern,
            "matches": matches,
            "truncated": len(matches) >= limit,
        },
        "metadata": {"match_count": len(matches)},
    }


SHELL_SCOPE_COMMANDS: dict[str, list[str]] = {
    "hostname": ["hostname"],
    "user": ["whoami"],
    "uptime": ["uptime"],
    "disk": ["df", "-h"],
    "memory": ["free", "-h"],
    "cpu": ["nproc"],
    "ports": ["ss", "-ltn"],
    "processes": ["ps", "-eo", "pid,comm,%cpu,%mem", "--sort=-%cpu"],
}


def _inspect_system(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Run an allowlisted read-only system inspection command on the remote host."""

    _ = workspace_root
    scope = str(arguments.get("scope") or "uptime").strip().lower()
    command = SHELL_SCOPE_COMMANDS.get(scope)
    if command is None:
        allowed = ", ".join(sorted(SHELL_SCOPE_COMMANDS))
        raise RemoteToolError(f"unsupported shell.inspect_system scope: {scope}. Allowed: {allowed}")

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RemoteToolError(f"system inspection command is unavailable for scope {scope}: {exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"system inspection failed for scope {scope}"
        raise RemoteToolError(message)

    facts = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "data_preview": {"scope": scope, "facts": facts},
        "metadata": {"scope": scope, "fact_count": len(facts)},
    }


def run_remote_operation(
    operation: str,
    arguments: dict[str, Any],
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Execute one typed remote operation and return a JSON-safe result envelope."""

    root = _workspace_root(workspace_root)
    operation = str(operation).strip()
    operation_map = {
        "filesystem.list_directory": _list_directory,
        "filesystem.read_file": _read_file,
        "filesystem.search_files": _search_files,
        "shell.inspect_system": _inspect_system,
    }
    handler = operation_map.get(operation)
    if handler is None:
        raise RemoteToolError(f"unsupported remote operation: {operation}")

    spec = OPERATION_SPECS[operation]
    payload_arguments = dict(arguments or {})
    allowed = set(spec["required"]) | set(spec["optional"])
    unexpected = sorted(set(payload_arguments) - allowed)
    missing = sorted(name for name in spec["required"] if name not in payload_arguments)
    if unexpected:
        raise RemoteToolError(
            f"unexpected arguments for {operation}: {', '.join(unexpected)}"
        )
    if missing:
        raise RemoteToolError(f"missing required arguments for {operation}: {', '.join(missing)}")

    payload = handler(payload_arguments, root)
    return {
        "status": "success",
        "data_preview": payload.get("data_preview"),
        "metadata": payload.get("metadata", {}),
    }


def _decode_payload(raw_payload: str) -> dict[str, Any]:
    """Decode one base64 JSON payload from the runtime."""

    try:
        decoded = base64.b64decode(raw_payload.encode("ascii"), validate=True).decode("utf-8")
    except Exception as exc:
        raise RemoteToolError("payload is not valid base64 JSON") from exc
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise RemoteToolError("payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RemoteToolError("payload must decode to a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by the generic gateway exec transport."""

    parser = argparse.ArgumentParser(description="OpenFABRIC gateway remote tool runner")
    parser.add_argument("--operation", required=True)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)

    try:
        payload = _decode_payload(args.payload)
        result = run_remote_operation(args.operation, payload)
    except RemoteToolError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
