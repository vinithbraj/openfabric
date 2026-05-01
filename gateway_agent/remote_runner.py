from __future__ import annotations

import argparse
import base64
import json
import os
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
    "shell.which": {
        "required": {"program"},
        "optional": set(),
    },
    "shell.pwd": {
        "required": set(),
        "optional": set(),
    },
    "shell.list_processes": {
        "required": set(),
        "optional": {"pattern", "limit"},
    },
    "shell.check_port": {
        "required": {"port"},
        "optional": set(),
    },
    "shell.git_status": {
        "required": set(),
        "optional": {"path"},
    },
    "shell.run_tests_readonly": {
        "required": set(),
        "optional": {"target", "max_failures"},
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


def _contains_shell_metacharacters(value: str) -> bool:
    """Return whether one user string contains shell metacharacters we reject outright."""

    text = str(value or "")
    return any(token in text for token in (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r", "\x00"))


def _reject_shell_text(value: str, field_name: str) -> str:
    """Normalize one shell-facing argument and reject command-like text."""

    normalized = str(value or "").strip()
    if not normalized:
        raise RemoteToolError(f"{field_name} must be a non-empty string.")
    if _contains_shell_metacharacters(normalized):
        raise RemoteToolError(f"{field_name} contains rejected shell command text.")
    return normalized


def _run_readonly_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run one internally-constructed shell command without invoking a shell."""

    try:
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        joined = " ".join(command)
        raise RemoteToolError(f"remote shell command is unavailable: {joined}: {exc}") from exc


def _which(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Resolve one program path through `which`."""

    _ = workspace_root
    program = _reject_shell_text(arguments.get("program"), "program")
    if "/" in program:
        raise RemoteToolError("program must be a bare executable name, not a path.")

    completed = _run_readonly_command(["which", program])
    return {
        "data_preview": {
            "program": program,
            "found": completed.returncode == 0,
            "path": completed.stdout.strip() or None,
        },
        "metadata": {"program": program, "exit_code": completed.returncode},
    }


def _pwd(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Return the remote working directory through `pwd`."""

    _ = arguments
    completed = _run_readonly_command(["pwd"], cwd=workspace_root)
    if completed.returncode != 0:
        raise RemoteToolError(completed.stderr.strip() or "pwd failed")
    return {
        "data_preview": {"cwd": completed.stdout.strip()},
        "metadata": {"exit_code": completed.returncode},
    }


def _list_processes(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """List running processes through a fixed `ps` command."""

    _ = workspace_root
    pattern = arguments.get("pattern")
    normalized_pattern = None
    if pattern is not None:
        normalized_pattern = _reject_shell_text(pattern, "pattern").lower()
    limit = max(1, int(arguments.get("limit", 50)))

    completed = _run_readonly_command(["ps", "-eo", "pid=,%cpu=,%mem=,comm=", "--sort=-%cpu"])
    if completed.returncode != 0:
        raise RemoteToolError(completed.stderr.strip() or "process listing failed")

    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        pid_text, cpu_text, memory_text, command_name = parts
        record = {
            "pid": int(pid_text),
            "command": command_name,
            "cpu_percent": float(cpu_text),
            "memory_percent": float(memory_text),
        }
        if normalized_pattern and normalized_pattern not in command_name.lower():
            continue
        processes.append(record)
        if len(processes) >= limit:
            break

    return {
        "data_preview": {
            "pattern": normalized_pattern,
            "processes": processes,
            "truncated": len(processes) >= limit,
        },
        "metadata": {"process_count": len(processes)},
    }


def _check_port(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Inspect listeners on one TCP port through `lsof`."""

    _ = workspace_root
    try:
        port = int(arguments.get("port"))
    except (TypeError, ValueError) as exc:
        raise RemoteToolError("port must be an integer.") from exc
    if port <= 0 or port > 65535:
        raise RemoteToolError("port must be between 1 and 65535.")

    listeners: list[dict[str, Any]] = []
    completed = _run_readonly_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"])
    if completed.returncode not in {0, 1}:
        raise RemoteToolError(completed.stderr.strip() or f"port inspection failed for {port}")

    for line in completed.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        listeners.append(
            {
                "command": parts[0],
                "pid": int(parts[1]),
                "user": parts[2],
                "name": parts[-1],
            }
        )

    if not listeners:
        ss_completed = _run_readonly_command(["ss", "-ltnp"])
        if ss_completed.returncode == 0:
            port_suffix = f":{port}"
            for line in ss_completed.stdout.splitlines()[1:]:
                if port_suffix not in line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                local_address = parts[3]
                process_info = parts[5] if len(parts) > 5 else ""
                pid = None
                command = None
                if "pid=" in process_info:
                    try:
                        pid = int(process_info.split("pid=", 1)[1].split(",", 1)[0].rstrip(")"))
                    except ValueError:
                        pid = None
                if 'users:(("' in process_info:
                    command = process_info.split('users:(("', 1)[1].split('"', 1)[0]
                listeners.append(
                    {
                        "command": command or "unknown",
                        "pid": pid,
                        "user": None,
                        "name": local_address,
                    }
                )

    return {
        "data_preview": {"port": port, "listeners": listeners},
        "metadata": {"listener_count": len(listeners)},
    }


def _git_status(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Return concise git status through a fixed git command."""

    path = str(arguments.get("path") or ".")
    repo_path, relative_path = _normalize_workspace_path(path, workspace_root)
    completed = _run_readonly_command(["git", "-C", str(repo_path), "status", "--short", "--branch"])
    if completed.returncode != 0:
        raise RemoteToolError(completed.stderr.strip() or f"git status failed for {relative_path}")

    lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    branch = None
    if lines and lines[0].startswith("## "):
        branch = lines[0][3:]

    return {
        "data_preview": {
            "path": relative_path,
            "branch": branch,
            "is_clean": len(lines) <= 1,
            "status_lines": lines,
        },
        "metadata": {"status_line_count": len(lines)},
    }


def _run_tests_readonly(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Run one constrained pytest command without accepting arbitrary shell flags."""

    target = _reject_shell_text(arguments.get("target") or ".", "target")
    max_failures = max(1, int(arguments.get("max_failures", 1)))
    target_path, relative_target = _normalize_workspace_path(target, workspace_root)

    if target_path.is_file() or target_path.is_dir():
        target_argument = relative_target
    elif "::" in target:
        base_path = target.split("::", 1)[0]
        if not base_path:
            raise RemoteToolError("pytest node id target must include a file path prefix.")
        base_target_path, base_relative = _normalize_workspace_path(base_path, workspace_root)
        if not base_target_path.exists():
            raise RemoteToolError(f"test target does not exist: {base_relative}")
        suffix = target[len(base_path) :]
        target_argument = f"{base_relative}{suffix}"
    else:
        raise RemoteToolError(f"test target does not exist: {target}")

    python_candidates = [
        workspace_root / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    python_command = next((candidate for candidate in python_candidates if candidate.exists()), None)
    if python_command is None:
        raise RemoteToolError("no Python interpreter is available for constrained test execution.")

    command = [
        os.fspath(python_command),
        "-m",
        "pytest",
        "-q",
        "--maxfail",
        str(max_failures),
        target_argument,
    ]
    completed = _run_readonly_command(command, cwd=workspace_root)
    if completed.returncode not in {0, 1, 5}:
        raise RemoteToolError(completed.stderr.strip() or f"pytest failed unexpectedly for {target_argument}")

    return {
        "data_preview": {
            "target": target_argument,
            "exit_code": completed.returncode,
            "stdout_lines": [line.rstrip() for line in completed.stdout.splitlines() if line.strip()],
            "stderr_lines": [line.rstrip() for line in completed.stderr.splitlines() if line.strip()],
        },
        "metadata": {"exit_code": completed.returncode, "target": target_argument},
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
        "shell.which": _which,
        "shell.pwd": _pwd,
        "shell.list_processes": _list_processes,
        "shell.check_port": _check_port,
        "shell.git_status": _git_status,
        "shell.run_tests_readonly": _run_tests_readonly,
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
