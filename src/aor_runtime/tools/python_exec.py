from __future__ import annotations

import ast
import io
import importlib
import json
import multiprocessing as mp
import re
import shlex
from contextlib import redirect_stdout
from typing import Any

from pydantic import Field

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel
from aor_runtime.tools.filesystem import fs_copy, fs_exists, fs_find, fs_list, fs_mkdir, fs_read, fs_write
from aor_runtime.tools.shell import run_shell
from aor_runtime.tools.sql import sql_query


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
SHELL_SUB_RE = re.compile(r"\$\(([^()]+)\)")

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
    "subprocess",
}


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _RowResult(_AttrDict):
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _SqlResult(_AttrDict):
    def __len__(self) -> int:
        row_count = self.get("row_count")
        if isinstance(row_count, int):
            return row_count
        return super().__len__()

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            rows = self.get("rows", [])
            if not isinstance(rows, list):
                raise KeyError(key)
            row = rows[key]
            if isinstance(row, dict):
                return _RowResult(row)
            return row
        return super().__getitem__(key)

    def __iter__(self):
        rows = self.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield _RowResult(row)
                else:
                    yield row
            return
        yield from super().__iter__()


class _CompletedProcess(_AttrDict):
    pass


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

    def find(self, path: str, pattern: str) -> list[str]:
        return list(fs_find(self.settings, path, pattern)["matches"])


class _PythonShellFacade:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def exec(self, command: str, node: str = "", timeout: int = 60, cwd: str = "") -> _AttrDict:
        if NETWORK_COMMAND_RE.search(command):
            raise ToolExecutionError("Network-oriented shell commands are not allowed inside python.exec.")
        if str(cwd or "").strip():
            raise ToolExecutionError("cwd is not supported for gateway-backed shell execution inside python.exec.")
        bounded_timeout = max(1, min(int(timeout), 5))
        del bounded_timeout  # The gateway transport timeout comes from runtime settings.
        return _AttrDict(run_shell(self.settings, command, node=node))


class _PythonSqlFacade:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def query(self, query: str, database: str | None = None) -> _AttrDict:
        return _SqlResult(sql_query(self.settings, query=query, database=database))


class _SubprocessModule:
    PIPE = -1
    STDOUT = -2

    def __init__(self, shell: _PythonShellFacade) -> None:
        self._shell = shell

    def _normalize_command(self, command: Any, shell: bool) -> str:
        if isinstance(command, str):
            return command
        if isinstance(command, (list, tuple)):
            parts = [str(part) for part in command]
            if shell:
                return " ".join(parts)
            return shlex.join(parts)
        raise ToolExecutionError("subprocess command must be a string or a list of strings.")

    def run(
        self,
        command: Any,
        *,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
        shell: bool = False,
        timeout: int | float | None = None,
        cwd: str = "",
        stdout: Any = None,
        stderr: Any = None,
        **_: Any,
    ) -> _CompletedProcess:
        normalized_command = self._normalize_command(command, shell=shell)
        bounded_timeout = 5 if timeout is None else max(1, min(int(timeout), 5))
        del bounded_timeout
        result = self._shell.exec(normalized_command, cwd=cwd)
        stdout_text = str(result.get("stdout", ""))
        stderr_text = str(result.get("stderr", ""))
        payload: dict[str, Any] = {
            "args": command,
            "returncode": int(result.get("returncode", 0)),
            "stdout": stdout_text if (capture_output or text or stdout == self.PIPE) else None,
            "stderr": stderr_text if (capture_output or text or stderr == self.PIPE) else None,
        }
        completed = _CompletedProcess(payload)
        if check and completed.returncode != 0:
            raise ToolExecutionError(completed.stderr or f"Command exited with {completed.returncode}")
        return completed

    def check_output(
        self,
        command: Any,
        *,
        text: bool = False,
        shell: bool = False,
        timeout: int | float | None = None,
        cwd: str = "",
        **kwargs: Any,
    ) -> str | bytes:
        completed = self.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            shell=shell,
            timeout=timeout,
            cwd=cwd,
            **kwargs,
        )
        output = str(completed.stdout or "")
        if text:
            return output
        return output.encode("utf-8")


def _shell_substitution(shell: _PythonShellFacade, command: str) -> str:
    return str(shell.exec(command, timeout=5).get("stdout", "")).strip()


def _rewrite_shell_substitutions(code: str) -> str:
    rewritten = code
    for _ in range(4):
        updated = SHELL_SUB_RE.sub(lambda match: f"_shell_substitution({json.dumps(match.group(1).strip())})", rewritten)
        if updated == rewritten:
            break
        rewritten = updated
    return rewritten


def _validate_code(code: str) -> tuple[ast.AST, str]:
    candidate = str(code or "")
    try:
        parsed = ast.parse(candidate, mode="exec")
    except SyntaxError:
        rewritten = _rewrite_shell_substitutions(candidate)
        try:
            parsed = ast.parse(rewritten, mode="exec")
        except SyntaxError as exc:
            raise ToolExecutionError(str(exc)) from exc
        candidate = rewritten
    for node in ast.walk(parsed):
        if isinstance(node, FORBIDDEN_NODES):
            raise ToolExecutionError(f"Forbidden Python construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ToolExecutionError(f"Forbidden Python name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ToolExecutionError("Dunder attribute access is not allowed in python.exec.")
    return parsed, candidate


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    if level != 0:
        raise ToolExecutionError("Relative imports are not allowed in python.exec.")
    root = name.split(".", 1)[0]
    if root not in SAFE_IMPORT_MODULES:
        raise ToolExecutionError(f"Import of module {root!r} is not allowed in python.exec.")
    if root == "subprocess":
        if globals is None or "__codex_subprocess__" not in globals:
            raise ToolExecutionError("subprocess shim is unavailable in python.exec.")
        return globals["__codex_subprocess__"]
    return importlib.import_module(name)


def _python_exec_worker(queue: mp.Queue, code: str, settings_data: dict[str, Any]) -> None:
    settings = Settings.model_validate(settings_data)
    parsed, _ = _validate_code(code)
    fs = _PythonFsFacade(settings)
    shell = _PythonShellFacade(settings)
    sql = _PythonSqlFacade(settings)
    namespace: dict[str, Any] = {
        "fs": fs,
        "shell": shell,
        "sql": sql,
        "result": None,
        "__codex_subprocess__": _SubprocessModule(shell),
        "_shell_substitution": lambda command: _shell_substitution(shell, str(command)),
    }
    stdout_buffer = io.StringIO()
    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    execution_globals: dict[str, Any] = {"__builtins__": safe_builtins, **namespace}
    try:
        with redirect_stdout(stdout_buffer):
            exec(compile(parsed, "<python.exec>", "exec"), execution_globals, execution_globals)
        queue.put(
            {
                "ok": True,
                "stdout": stdout_buffer.getvalue(),
                "result": execution_globals.get("result"),
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
            description="Execute minimal sandboxed Python that can call fs, shell, and sql helpers.",
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
        _, normalized_code = _validate_code(code)

        queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=_python_exec_worker,
            args=(queue, normalized_code, self.settings.model_dump(mode="json")),
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
        if output:
            output = self._expand_shell_substitutions(output)

        return self.ToolResult(success=True, output=output, error=None)

    def _expand_shell_substitutions(self, value: str) -> str:
        expanded = str(value)
        shell = _PythonShellFacade(self.settings)
        for _ in range(4):
            match = SHELL_SUB_RE.search(expanded)
            if not match:
                break
            command = match.group(1).strip()
            replacement = _shell_substitution(shell, command)
            expanded = expanded[: match.start()] + replacement + expanded[match.end() :]
        return expanded
