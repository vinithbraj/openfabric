"""OpenFABRIC Runtime Module: aor_runtime.tools.python_exec

Purpose:
    Execute bounded Python snippets for explicitly allowed internal workflows.

Responsibilities:
    Expose typed tool arguments/results for filesystem, SQL, shell, SLURM, text formatting, Python, and runtime return operations.

Data flow / Interfaces:
    Receives validated tool arguments from the executor and returns structured result models for downstream contracts and presenters.

Boundaries:
    Does not decide user intent; every tool must preserve safety, allowed-root, read-only, timeout, and result-shape boundaries.
"""

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
from aor_runtime.runtime.lifecycle import CancellationError, ToolInvocationContext, run_process_with_queue
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel
from aor_runtime.tools.filesystem import fs_copy, fs_exists, fs_find, fs_list, fs_mkdir, fs_read, fs_size, fs_write
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
    """Represent attr dict within the OpenFABRIC runtime. It extends dict.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _AttrDict.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._AttrDict and related tests.
    """
    def __getattr__(self, name: str) -> Any:
        """Handle the internal getattr helper path for this module.

        Inputs:
            Receives name for this _AttrDict method; type hints and validators define accepted shapes.

        Returns:
            Returns the standard Python protocol value for __getattr__ when one is required.

        Used by:
            Used by registered tool execution through _AttrDict.__getattr__ calls and related tests.
        """
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _RowResult(_AttrDict):
    """Represent row result within the OpenFABRIC runtime. It extends _AttrDict.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _RowResult.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._RowResult and related tests.
    """
    def __getitem__(self, key: Any) -> Any:
        """Handle the internal getitem helper path for this module.

        Inputs:
            Receives key for this _RowResult method; type hints and validators define accepted shapes.

        Returns:
            Returns the standard Python protocol value for __getitem__ when one is required.

        Used by:
            Used by registered tool execution through _RowResult.__getitem__ calls and related tests.
        """
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _SqlResult(_AttrDict):
    """Represent sql result within the OpenFABRIC runtime. It extends _AttrDict.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _SqlResult.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._SqlResult and related tests.
    """
    def __len__(self) -> int:
        """Handle the internal len helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the standard Python protocol value for __len__ when one is required.

        Used by:
            Used by registered tool execution through _SqlResult.__len__ calls and related tests.
        """
        row_count = self.get("row_count")
        if isinstance(row_count, int):
            return row_count
        return super().__len__()

    def __getitem__(self, key: Any) -> Any:
        """Handle the internal getitem helper path for this module.

        Inputs:
            Receives key for this _SqlResult method; type hints and validators define accepted shapes.

        Returns:
            Returns the standard Python protocol value for __getitem__ when one is required.

        Used by:
            Used by registered tool execution through _SqlResult.__getitem__ calls and related tests.
        """
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
        """Handle the internal iter helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the standard Python protocol value for __iter__ when one is required.

        Used by:
            Used by registered tool execution through _SqlResult.__iter__ calls and related tests.
        """
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
    """Represent completed process within the OpenFABRIC runtime. It extends _AttrDict.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _CompletedProcess.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._CompletedProcess and related tests.
    """
    pass


class _PythonFsFacade:
    """Represent python fs facade within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _PythonFsFacade.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._PythonFsFacade and related tests.
    """
    def __init__(self, settings: Settings) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through _PythonFsFacade.__init__ calls and related tests.
        """
        self.settings = settings

    def exists(self, path: str) -> bool:
        """Exists for _PythonFsFacade instances.

        Inputs:
            Receives path for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonFsFacade.exists calls and related tests.
        """
        return bool(fs_exists(self.settings, path)["exists"])

    def copy(self, src: str, dst: str) -> None:
        """Copy for _PythonFsFacade instances.

        Inputs:
            Receives src, dst for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by registered tool execution through _PythonFsFacade.copy calls and related tests.
        """
        fs_copy(self.settings, src, dst)

    def read(self, path: str) -> str:
        """Read for _PythonFsFacade instances.

        Inputs:
            Receives path for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonFsFacade.read calls and related tests.
        """
        return str(fs_read(self.settings, path)["content"])

    def write(self, path: str, content: str) -> None:
        """Write for _PythonFsFacade instances.

        Inputs:
            Receives path, content for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by registered tool execution through _PythonFsFacade.write calls and related tests.
        """
        fs_write(self.settings, path, content)

    def mkdir(self, path: str) -> None:
        """Mkdir for _PythonFsFacade instances.

        Inputs:
            Receives path for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by registered tool execution through _PythonFsFacade.mkdir calls and related tests.
        """
        fs_mkdir(self.settings, path)

    def list(self, path: str) -> list[str]:
        """List for _PythonFsFacade instances.

        Inputs:
            Receives path for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonFsFacade.list calls and related tests.
        """
        return list(fs_list(self.settings, path)["entries"])

    def find(self, path: str, pattern: str) -> list[str]:
        """Find for _PythonFsFacade instances.

        Inputs:
            Receives path, pattern for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonFsFacade.find calls and related tests.
        """
        return list(fs_find(self.settings, path, pattern)["matches"])

    def size(self, path: str) -> int:
        """Size for _PythonFsFacade instances.

        Inputs:
            Receives path for this _PythonFsFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonFsFacade.size calls and related tests.
        """
        return int(fs_size(self.settings, path)["size_bytes"])


class _PythonShellFacade:
    """Represent python shell facade within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _PythonShellFacade.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._PythonShellFacade and related tests.
    """
    def __init__(self, settings: Settings) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this _PythonShellFacade method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through _PythonShellFacade.__init__ calls and related tests.
        """
        self.settings = settings

    def exec(self, command: str, node: str = "", timeout: int = 60, cwd: str = "") -> _AttrDict:
        """Exec for _PythonShellFacade instances.

        Inputs:
            Receives command, node, timeout, cwd for this _PythonShellFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonShellFacade.exec calls and related tests.
        """
        if NETWORK_COMMAND_RE.search(command):
            raise ToolExecutionError("Network-oriented shell commands are not allowed inside python.exec.")
        if str(cwd or "").strip():
            raise ToolExecutionError("cwd is not supported for gateway-backed shell execution inside python.exec.")
        bounded_timeout = max(1, min(int(timeout), 5))
        del bounded_timeout  # The gateway transport timeout comes from runtime settings.
        return _AttrDict(run_shell(self.settings, command, node=node))


class _PythonSqlFacade:
    """Represent python sql facade within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _PythonSqlFacade.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._PythonSqlFacade and related tests.
    """
    def __init__(self, settings: Settings) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this _PythonSqlFacade method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through _PythonSqlFacade.__init__ calls and related tests.
        """
        self.settings = settings

    def query(self, query: str, database: str | None = None) -> _AttrDict:
        """Query for _PythonSqlFacade instances.

        Inputs:
            Receives query, database for this _PythonSqlFacade method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _PythonSqlFacade.query calls and related tests.
        """
        return _SqlResult(sql_query(self.settings, query=query, database=database))


class _SubprocessModule:
    """Represent subprocess module within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _SubprocessModule.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec._SubprocessModule and related tests.
    """
    PIPE = -1
    STDOUT = -2

    def __init__(self, shell: _PythonShellFacade) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives shell for this _SubprocessModule method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through _SubprocessModule.__init__ calls and related tests.
        """
        self._shell = shell

    def _normalize_command(self, command: Any, shell: bool) -> str:
        """Handle the internal normalize command helper path for this module.

        Inputs:
            Receives command, shell for this _SubprocessModule method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _SubprocessModule._normalize_command calls and related tests.
        """
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
        """Run for _SubprocessModule instances.

        Inputs:
            Receives command, capture_output, text, check, shell, timeout, cwd, stdout, ... for this _SubprocessModule method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _SubprocessModule.run calls and related tests.
        """
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
        """Check output for _SubprocessModule instances.

        Inputs:
            Receives command, text, shell, timeout, cwd for this _SubprocessModule method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through _SubprocessModule.check_output calls and related tests.
        """
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
    """Handle the internal shell substitution helper path for this module.

    Inputs:
        Receives shell, command for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.python_exec._shell_substitution.
    """
    return str(shell.exec(command, timeout=5).get("stdout", "")).strip()


def _rewrite_shell_substitutions(code: str) -> str:
    """Handle the internal rewrite shell substitutions helper path for this module.

    Inputs:
        Receives code for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.python_exec._rewrite_shell_substitutions.
    """
    rewritten = code
    for _ in range(4):
        updated = SHELL_SUB_RE.sub(lambda match: f"_shell_substitution({json.dumps(match.group(1).strip())})", rewritten)
        if updated == rewritten:
            break
        rewritten = updated
    return rewritten


def _validate_code(code: str) -> tuple[ast.AST, str]:
    """Handle the internal validate code helper path for this module.

    Inputs:
        Receives code for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.python_exec._validate_code.
    """
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
    """Handle the internal safe import helper path for this module.

    Inputs:
        Receives name, globals, locals, fromlist, level for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.python_exec._safe_import.
    """
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
    """Handle the internal python exec worker helper path for this module.

    Inputs:
        Receives queue, code, settings_data for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.python_exec._python_exec_worker.
    """
    settings_payload = dict(settings_data)
    inputs = dict(settings_payload.pop("__python_inputs__", {}) or {})
    settings = Settings.model_validate(settings_payload)
    parsed, _ = _validate_code(code)
    fs = _PythonFsFacade(settings)
    shell = _PythonShellFacade(settings)
    sql = _PythonSqlFacade(settings)
    namespace: dict[str, Any] = {
        "fs": fs,
        "shell": shell,
        "sql": sql,
        "inputs": inputs,
        "result": None,
        "_json_dumps_safe": _json_dumps_safe,
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


def _json_dumps_safe(value: Any) -> str:
    """Handle the internal json dumps safe helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.python_exec._json_dumps_safe.
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


class PythonExecTool(BaseTool):
    """Represent python exec tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PythonExecTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.python_exec.PythonExecTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.python_exec.ToolArgs and related tests.
        """
        code: str
        inputs: dict[str, Any] = Field(default_factory=dict)
        timeout: int = Field(default=5, ge=1, le=5)

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.python_exec.ToolResult and related tests.
        """
        success: bool
        output: str | None = None
        result: Any | None = None
        error: str | None = None

    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this PythonExecTool method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through PythonExecTool.__init__ calls and related tests.
        """
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
                    "inputs": {"type": "object"},
                    "timeout": {"type": "integer"},
                },
                "required": ["code"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        """Run for PythonExecTool instances.

        Inputs:
            Receives arguments for this PythonExecTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through PythonExecTool.run calls and related tests.
        """
        return self._run(arguments, context=None)

    def run_with_context(self, arguments: ToolArgs, context: ToolInvocationContext) -> ToolResult:
        """Run with context for PythonExecTool instances.

        Inputs:
            Receives arguments, context for this PythonExecTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through PythonExecTool.run_with_context calls and related tests.
        """
        return self._run(arguments, context=context)

    def _run(self, arguments: ToolArgs, *, context: ToolInvocationContext | None) -> ToolResult:
        """Handle the internal run helper path for this module.

        Inputs:
            Receives arguments, context for this PythonExecTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through PythonExecTool._run calls and related tests.
        """
        code = arguments.code
        inputs = dict(arguments.inputs or {})
        timeout = arguments.timeout
        _, normalized_code = _validate_code(code)

        try:
            payload = run_process_with_queue(
                target=_python_exec_worker,
                args=(normalized_code, {**self.settings.model_dump(mode="json"), "__python_inputs__": inputs}),
                timeout_seconds=timeout,
                timeout_message="python.exec timed out.",
                context=context,
                process_name="aor-python-exec",
            )
        except TimeoutError:
            return self.ToolResult(success=False, output=None, result=None, error="python.exec timed out.")
        except RuntimeError as exc:
            if isinstance(exc, CancellationError):
                raise
            return self.ToolResult(success=False, output=None, result=None, error="python.exec did not return a result.")

        if not payload.get("ok"):
            return self.ToolResult(
                success=False,
                output=str(payload.get("stdout") or "") or None,
                result=None,
                error=str(payload.get("error") or "python.exec failed."),
            )

        result = payload.get("result")
        try:
            serialized_result = json.dumps(result, default=str, ensure_ascii=False, sort_keys=True)
        except TypeError as exc:  # noqa: PERF203
            return self.ToolResult(
                success=False,
                output=str(payload.get("stdout") or "") or None,
                result=None,
                error=f"python.exec result must be JSON serializable: {exc}",
            )

        stdout = str(payload.get("stdout") or "").strip()
        output: str | None
        if result is None:
            output = stdout or None
        elif isinstance(result, str):
            output = result
        elif isinstance(result, (int, float, bool)):
            output = str(result)
        else:
            output = serialized_result
        if stdout and output and stdout != output:
            output = f"{stdout}\n{output}"
        if output:
            output = self._expand_shell_substitutions(output)

        return self.ToolResult(success=True, output=output, result=result, error=None)

    def _expand_shell_substitutions(self, value: str) -> str:
        """Handle the internal expand shell substitutions helper path for this module.

        Inputs:
            Receives value for this PythonExecTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through PythonExecTool._expand_shell_substitutions calls and related tests.
        """
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
