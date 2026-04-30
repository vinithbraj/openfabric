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
    """Raised when a runtime request has been cancelled."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "cancelled"

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        normalized = str(reason or "cancelled").strip() or "cancelled"
        self._reason = normalized
        self._event.set()

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancellationError(self.reason)


@dataclass
class ManagedProcess:
    process: mp.Process
    name: str
    started_at: float = field(default_factory=time.monotonic)

    @property
    def pid(self) -> int | None:
        return self.process.pid


@dataclass
class ToolInvocationContext:
    cancellation: CancellationToken | None = None
    process_registry: "ActiveRunRegistry | None" = None
    worker_join_timeout_seconds: float = 2.0
    tool_process_kill_grace_seconds: float = 1.0

    def throw_if_cancelled(self) -> None:
        if self.cancellation is not None:
            self.cancellation.throw_if_cancelled()


class RunHandle:
    def __init__(self, run_id: str, registry: "ActiveRunRegistry", token: CancellationToken | None = None) -> None:
        self.run_id = run_id
        self.registry = registry
        self.token = token or CancellationToken()
        self.outcome: dict[str, Any] = {}
        self.thread: threading.Thread | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self.token.cancel(reason)

    def join(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())


class ActiveRunRegistry:
    def __init__(self, *, shutdown_grace_seconds: float = 5.0, process_kill_grace_seconds: float = 1.0) -> None:
        self.shutdown_grace_seconds = float(shutdown_grace_seconds)
        self.process_kill_grace_seconds = float(process_kill_grace_seconds)
        self._lock = threading.RLock()
        self._runs: dict[str, RunHandle] = {}
        self._processes: dict[int, ManagedProcess] = {}

    def register_run(self, run_id: str | None = None) -> RunHandle:
        normalized_run_id = str(run_id or uuid.uuid4().hex)
        handle = RunHandle(normalized_run_id, self)
        with self._lock:
            self._runs[normalized_run_id] = handle
        return handle

    def start_background(self, run_id: str | None, target: Callable[[CancellationToken], Any]) -> RunHandle:
        handle = self.register_run(run_id)

        def runner() -> None:
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
        with self._lock:
            self._runs.pop(str(run_id), None)

    def register_process(self, process: mp.Process, *, name: str) -> ManagedProcess:
        managed = ManagedProcess(process=process, name=name)
        if process.pid is not None:
            with self._lock:
                self._processes[int(process.pid)] = managed
        return managed

    def unregister_process(self, process: mp.Process) -> None:
        if process.pid is not None:
            with self._lock:
                self._processes.pop(int(process.pid), None)

    def active_run_count(self) -> int:
        with self._lock:
            return len(self._runs)

    def active_process_count(self) -> int:
        with self._lock:
            return len(self._processes)

    def cancel_all(self, reason: str = "server shutdown", *, wait_seconds: float | None = None) -> None:
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
