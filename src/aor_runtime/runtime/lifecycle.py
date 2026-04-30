"""OpenFABRIC Runtime Module: aor_runtime.runtime.lifecycle

Purpose:
    Manage request lifecycle, cancellation, and child worker ownership.

Responsibilities:
    Track active runs, cancellation tokens, managed processes, and bounded shutdown behavior.

Data flow / Interfaces:
    Used by API, engine, SQL/Python workers, and long-running tool execution paths.

Boundaries:
    Prevents leaked child processes, inherited sockets, blocked joins, and shutdown hangs.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class CancellationError(RuntimeError):
    """Represent cancellation error within the OpenFABRIC runtime. It extends RuntimeError.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CancellationError.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.lifecycle.CancellationError and related tests.
    """


class CancellationToken:
    """Represent cancellation token within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CancellationToken.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.lifecycle.CancellationToken and related tests.
    """
    def __init__(self) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through CancellationToken.__init__ calls and related tests.
        """
        self._event = threading.Event()
        self._reason = "cancelled"

    @property
    def cancelled(self) -> bool:
        """Cancelled for CancellationToken instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through CancellationToken.cancelled calls and related tests.
        """
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Reason for CancellationToken instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through CancellationToken.reason calls and related tests.
        """
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancel for CancellationToken instances.

        Inputs:
            Receives reason for this CancellationToken method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through CancellationToken.cancel calls and related tests.
        """
        normalized = str(reason or "cancelled").strip() or "cancelled"
        self._reason = normalized
        self._event.set()

    def throw_if_cancelled(self) -> None:
        """Throw if cancelled for CancellationToken instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through CancellationToken.throw_if_cancelled calls and related tests.
        """
        if self.cancelled:
            raise CancellationError(self.reason)


@dataclass
class ManagedProcess:
    """Represent managed process within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ManagedProcess.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.lifecycle.ManagedProcess and related tests.
    """
    process: mp.Process
    name: str
    started_at: float = field(default_factory=time.monotonic)

    @property
    def pid(self) -> int | None:
        """Pid for ManagedProcess instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through ManagedProcess.pid calls and related tests.
        """
        return self.process.pid


@dataclass
class ToolInvocationContext:
    """Represent tool invocation context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolInvocationContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.lifecycle.ToolInvocationContext and related tests.
    """
    cancellation: CancellationToken | None = None
    process_registry: "ActiveRunRegistry | None" = None
    worker_join_timeout_seconds: float = 2.0
    tool_process_kill_grace_seconds: float = 1.0

    def throw_if_cancelled(self) -> None:
        """Throw if cancelled for ToolInvocationContext instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ToolInvocationContext.throw_if_cancelled calls and related tests.
        """
        if self.cancellation is not None:
            self.cancellation.throw_if_cancelled()


class RunHandle:
    """Represent run handle within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RunHandle.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.lifecycle.RunHandle and related tests.
    """
    def __init__(self, run_id: str, registry: "ActiveRunRegistry", token: CancellationToken | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives run_id, registry, token for this RunHandle method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through RunHandle.__init__ calls and related tests.
        """
        self.run_id = run_id
        self.registry = registry
        self.token = token or CancellationToken()
        self.outcome: dict[str, Any] = {}
        self.thread: threading.Thread | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancel for RunHandle instances.

        Inputs:
            Receives reason for this RunHandle method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through RunHandle.cancel calls and related tests.
        """
        self.token.cancel(reason)

    def join(self, timeout: float | None = None) -> None:
        """Join for RunHandle instances.

        Inputs:
            Receives timeout for this RunHandle method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through RunHandle.join calls and related tests.
        """
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        """Is alive for RunHandle instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through RunHandle.is_alive calls and related tests.
        """
        return bool(self.thread and self.thread.is_alive())


class ActiveRunRegistry:
    """Represent active run registry within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ActiveRunRegistry.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.lifecycle.ActiveRunRegistry and related tests.
    """
    def __init__(self, *, shutdown_grace_seconds: float = 5.0, process_kill_grace_seconds: float = 1.0) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives shutdown_grace_seconds, process_kill_grace_seconds for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.__init__ calls and related tests.
        """
        self.shutdown_grace_seconds = float(shutdown_grace_seconds)
        self.process_kill_grace_seconds = float(process_kill_grace_seconds)
        self._lock = threading.RLock()
        self._runs: dict[str, RunHandle] = {}
        self._processes: dict[int, ManagedProcess] = {}

    def register_run(self, run_id: str | None = None) -> RunHandle:
        """Register run for ActiveRunRegistry instances.

        Inputs:
            Receives run_id for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.register_run calls and related tests.
        """
        normalized_run_id = str(run_id or uuid.uuid4().hex)
        handle = RunHandle(normalized_run_id, self)
        with self._lock:
            self._runs[normalized_run_id] = handle
        return handle

    def start_background(self, run_id: str | None, target: Callable[[CancellationToken], Any]) -> RunHandle:
        """Start background for ActiveRunRegistry instances.

        Inputs:
            Receives run_id, target for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.start_background calls and related tests.
        """
        handle = self.register_run(run_id)

        def runner() -> None:
            """Runner for the surrounding runtime workflow.

            Inputs:
                Uses module or instance state; no caller-supplied data parameters are required.

            Returns:
                Returns None; side effects are limited to the local runtime operation described above.

            Used by:
                Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.lifecycle.runner.
            """
            try:
                handle.outcome["result"] = target(handle.token)
            except Exception as exc:  # noqa: BLE001
                handle.outcome["error"] = exc
            finally:
                self.unregister_run(handle.run_id)

        handle.thread = threading.Thread(target=runner, name=f"aor-run-{handle.run_id}", daemon=True)
        handle.thread.start()
        return handle

    def unregister_run(self, run_id: str) -> None:
        """Unregister run for ActiveRunRegistry instances.

        Inputs:
            Receives run_id for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.unregister_run calls and related tests.
        """
        with self._lock:
            self._runs.pop(str(run_id), None)

    def register_process(self, process: mp.Process, *, name: str) -> ManagedProcess:
        """Register process for ActiveRunRegistry instances.

        Inputs:
            Receives process, name for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.register_process calls and related tests.
        """
        managed = ManagedProcess(process=process, name=name)
        if process.pid is not None:
            with self._lock:
                self._processes[int(process.pid)] = managed
        return managed

    def unregister_process(self, process: mp.Process) -> None:
        """Unregister process for ActiveRunRegistry instances.

        Inputs:
            Receives process for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.unregister_process calls and related tests.
        """
        if process.pid is not None:
            with self._lock:
                self._processes.pop(int(process.pid), None)

    def active_run_count(self) -> int:
        """Active run count for ActiveRunRegistry instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.active_run_count calls and related tests.
        """
        with self._lock:
            return len(self._runs)

    def active_process_count(self) -> int:
        """Active process count for ActiveRunRegistry instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.active_process_count calls and related tests.
        """
        with self._lock:
            return len(self._processes)

    def cancel_all(self, reason: str = "server shutdown", *, wait_seconds: float | None = None) -> None:
        """Cancel all for ActiveRunRegistry instances.

        Inputs:
            Receives reason, wait_seconds for this ActiveRunRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActiveRunRegistry.cancel_all calls and related tests.
        """
        with self._lock:
            handles = list(self._runs.values())
            processes = [managed.process for managed in self._processes.values()]
        for handle in handles:
            handle.cancel(reason)
        deadline = time.monotonic() + float(wait_seconds if wait_seconds is not None else self.shutdown_grace_seconds)
        for handle in handles:
            remaining = max(0.0, deadline - time.monotonic())
            handle.join(timeout=remaining)
        for process in processes:
            terminate_process(process, kill_grace_seconds=self.process_kill_grace_seconds)


def terminate_process(process: mp.Process, *, kill_grace_seconds: float = 1.0) -> None:
    """Terminate process for the surrounding runtime workflow.

    Inputs:
        Receives process, kill_grace_seconds for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.lifecycle.terminate_process.
    """
    if process.is_alive():
        process.terminate()
        process.join(timeout=max(0.0, float(kill_grace_seconds)))
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=max(0.0, float(kill_grace_seconds)))
    if process.is_alive():
        process.join(timeout=0)


def run_process_with_queue(
    *,
    target: Callable[..., None],
    args: tuple[Any, ...],
    timeout_seconds: float,
    timeout_message: str,
    context: ToolInvocationContext | None = None,
    process_name: str = "tool-worker",
) -> Any:
    """Run process with queue for the surrounding runtime workflow.

    Inputs:
        Receives target, args, timeout_seconds, timeout_message, context, process_name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.lifecycle.run_process_with_queue.
    """
    runtime_context = context or ToolInvocationContext()
    spawn_context = mp.get_context("spawn")
    result_queue: mp.Queue = spawn_context.Queue()
    process = spawn_context.Process(target=target, args=(result_queue, *args), name=process_name)
    registry = runtime_context.process_registry
    try:
        runtime_context.throw_if_cancelled()
        process.start()
        if registry is not None:
            registry.register_process(process, name=process_name)
        payload = _wait_for_queue_payload(
            process=process,
            result_queue=result_queue,
            timeout_seconds=timeout_seconds,
            timeout_message=timeout_message,
            context=runtime_context,
        )
        process.join(timeout=max(0.0, float(runtime_context.worker_join_timeout_seconds)))
        if process.is_alive():
            terminate_process(process, kill_grace_seconds=runtime_context.tool_process_kill_grace_seconds)
        return payload
    finally:
        if registry is not None:
            registry.unregister_process(process)
        try:
            result_queue.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            result_queue.join_thread()
        except Exception:  # noqa: BLE001
            pass
        if process.is_alive():
            terminate_process(process, kill_grace_seconds=runtime_context.tool_process_kill_grace_seconds)


def _wait_for_queue_payload(
    *,
    process: mp.Process,
    result_queue: mp.Queue,
    timeout_seconds: float,
    timeout_message: str,
    context: ToolInvocationContext,
) -> Any:
    """Handle the internal wait for queue payload helper path for this module.

    Inputs:
        Receives process, result_queue, timeout_seconds, timeout_message, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.lifecycle._wait_for_queue_payload.
    """
    deadline = time.monotonic() + max(0.001, float(timeout_seconds))
    saw_exit = False
    while True:
        context.throw_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            terminate_process(process, kill_grace_seconds=context.tool_process_kill_grace_seconds)
            raise TimeoutError(timeout_message)
        try:
            return result_queue.get(timeout=min(0.05, max(0.001, remaining)))
        except queue_module.Empty:
            if not process.is_alive():
                if saw_exit:
                    break
                saw_exit = True
                continue
    raise RuntimeError("Worker process did not return a result.")
