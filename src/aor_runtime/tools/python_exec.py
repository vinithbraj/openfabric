from __future__ import annotations

import ast
import io
import importlib
import json
import multiprocessing as mp
import re
from contextlib import redirect_stdout
from typing import Any

from pydantic import Field

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel
from aor_runtime.tools.filesystem import fs_copy, fs_exists, fs_list, fs_mkdir, fs_read, fs_write
from aor_runtime.tools.shell import run_shell


SAFE_BUILTINS = {
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

FORBIDDEN_NODES = (
    ast.Global,
    ast.Nonlocal,
    ast.Try,
    ast.With,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
)

FORBIDDEN_NAMES = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "input",
    "help",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}

NETWORK_COMMAND_RE = re.compile(r"\b(curl|wget|ssh|scp|sftp|ftp|ping|nc|ncat|telnet)\b", re.IGNORECASE)

SAFE_IMPORT_MODULES = {
    "collections",
    "functools",
    "itertools",
    "json",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
}


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _PythonFsFacade:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def exists(self, path: str) -> bool:
        return bool(fs_exists(self.settings, path)["exists"])

    def copy(self, src: str, dst: str) -> None:
        fs_copy(self.settings, src, dst)

    def read(self, path: str) -> str:
        return str(fs_read(self.settings, path)["content"])

    def write(self, path: str, content: str) -> None:
        fs_write(self.settings, path, content)

    def mkdir(self, path: str) -> None:
        fs_mkdir(self.settings, path)

    def list(self, path: str) -> list[str]:
        return list(fs_list(self.settings, path)["entries"])


class _PythonShellFacade:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def exec(self, command: str, cwd: str = "", timeout: int = 60) -> _AttrDict:
        if NETWORK_COMMAND_RE.search(command):
            raise ToolExecutionError("Network-oriented shell commands are not allowed inside python.exec.")
        bounded_timeout = max(1, min(int(timeout), 5))
        return _AttrDict(run_shell(self.settings, command, cwd=cwd, timeout=bounded_timeout))


def _validate_code(code: str) -> ast.AST:
    parsed = ast.parse(code, mode="exec")
    for node in ast.walk(parsed):
        if isinstance(node, FORBIDDEN_NODES):
            raise ToolExecutionError(f"Forbidden Python construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ToolExecutionError(f"Forbidden Python name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ToolExecutionError("Dunder attribute access is not allowed in python.exec.")
    return parsed


def _safe_import(name: str, globals: dict[str, Any] | None = None, locals: dict[str, Any] | None = None, fromlist: tuple[str, ...] = (), level: int = 0):
    if level != 0:
        raise ToolExecutionError("Relative imports are not allowed in python.exec.")
    root = name.split(".", 1)[0]
    if root not in SAFE_IMPORT_MODULES:
        raise ToolExecutionError(f"Import of module {root!r} is not allowed in python.exec.")
    return importlib.import_module(name)


def _python_exec_worker(queue: mp.Queue, code: str, settings_data: dict[str, Any]) -> None:
    settings = Settings.model_validate(settings_data)
    parsed = _validate_code(code)
    fs = _PythonFsFacade(settings)
    shell = _PythonShellFacade(settings)
    namespace: dict[str, Any] = {"fs": fs, "shell": shell, "result": None}
    stdout_buffer = io.StringIO()
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    try:
        with redirect_stdout(stdout_buffer):
            exec(compile(parsed, "<python.exec>", "exec"), {"__builtins__": safe_builtins}, namespace)
        queue.put(
            {
                "ok": True,
                "stdout": stdout_buffer.getvalue(),
                "result": namespace.get("result"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc), "stdout": stdout_buffer.getvalue()})


class PythonExecTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        code: str
        timeout: int = Field(default=5, ge=1, le=5)

    class ToolResult(ToolResultModel):
        success: bool
        output: str | None = None
        error: str | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="python.exec",
            description="Execute minimal sandboxed Python that can call fs and shell helpers.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["code"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        code = arguments.code
        timeout = arguments.timeout
        _validate_code(code)

        queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=_python_exec_worker,
            args=(queue, code, self.settings.model_dump(mode="json")),
        )
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return self.ToolResult(success=False, output=None, error="python.exec timed out.")

        if queue.empty():
            return self.ToolResult(success=False, output=None, error="python.exec did not return a result.")

        payload = queue.get()
        if not payload.get("ok"):
            return self.ToolResult(
                success=False,
                output=str(payload.get("stdout") or "") or None,
                error=str(payload.get("error") or "python.exec failed."),
            )

        result = payload.get("result")
        try:
            json.dumps(result, default=str)
        except TypeError as exc:  # noqa: PERF203
            return self.ToolResult(success=False, output=str(payload.get("stdout") or "") or None, error=f"python.exec result must be JSON serializable: {exc}")

        stdout = str(payload.get("stdout") or "").strip()
        output: str | None
        if result is None:
            output = stdout or None
        elif isinstance(result, str):
            output = result
        elif isinstance(result, (int, float, bool)):
            output = str(result)
        else:
            output = json.dumps(result, ensure_ascii=False, sort_keys=True)
        if stdout and output and stdout != output:
            output = f"{stdout}\n{output}"

        return self.ToolResult(success=True, output=output, error=None)
