from __future__ import annotations

import time

import pytest

from aor_runtime.runtime.lifecycle import (
    ActiveRunRegistry,
    CancellationError,
    CancellationToken,
    ToolInvocationContext,
    run_process_with_queue,
)


def _large_payload_worker(queue, count: int) -> None:
    queue.put({"ok": True, "rows": [{"id": index, "value": "x" * 40} for index in range(count)]})


def _sleeping_worker(queue, seconds: float) -> None:
    time.sleep(seconds)
    queue.put({"ok": True})


def test_managed_process_drains_large_queue_payload_before_join() -> None:
    registry = ActiveRunRegistry(shutdown_grace_seconds=0.5, process_kill_grace_seconds=0.1)
    context = ToolInvocationContext(
        cancellation=CancellationToken(),
        process_registry=registry,
        worker_join_timeout_seconds=0.5,
        tool_process_kill_grace_seconds=0.1,
    )

    payload = run_process_with_queue(
        target=_large_payload_worker,
        args=(2500,),
        timeout_seconds=10,
        timeout_message="timed out",
        context=context,
        process_name="test-large-payload",
    )

    assert payload["ok"] is True
    assert len(payload["rows"]) == 2500
    assert registry.active_process_count() == 0


def test_managed_process_timeout_terminates_and_unregisters_child() -> None:
    registry = ActiveRunRegistry(shutdown_grace_seconds=0.5, process_kill_grace_seconds=0.1)
    context = ToolInvocationContext(
        cancellation=CancellationToken(),
        process_registry=registry,
        worker_join_timeout_seconds=0.1,
        tool_process_kill_grace_seconds=0.1,
    )

    with pytest.raises(TimeoutError):
        run_process_with_queue(
            target=_sleeping_worker,
            args=(5,),
            timeout_seconds=0.2,
            timeout_message="timed out",
            context=context,
            process_name="test-timeout",
        )

    assert registry.active_process_count() == 0


def test_managed_process_respects_pre_cancelled_token_without_child_leak() -> None:
    registry = ActiveRunRegistry(shutdown_grace_seconds=0.5, process_kill_grace_seconds=0.1)
    token = CancellationToken()
    token.cancel("client disconnected")
    context = ToolInvocationContext(
        cancellation=token,
        process_registry=registry,
        worker_join_timeout_seconds=0.1,
        tool_process_kill_grace_seconds=0.1,
    )

    with pytest.raises(CancellationError):
        run_process_with_queue(
            target=_sleeping_worker,
            args=(5,),
            timeout_seconds=1,
            timeout_message="timed out",
            context=context,
            process_name="test-cancelled",
        )

    assert registry.active_process_count() == 0


def test_active_run_registry_cancel_all_cancels_background_handles() -> None:
    registry = ActiveRunRegistry(shutdown_grace_seconds=0.2, process_kill_grace_seconds=0.1)

    def wait_for_cancel(token: CancellationToken) -> str:
        while not token.cancelled:
            time.sleep(0.01)
        return token.reason

    handle = registry.start_background("run-1", wait_for_cancel)
    assert registry.active_run_count() == 1

    registry.cancel_all("server shutdown", wait_seconds=1)

    assert handle.outcome["result"] == "server shutdown"
    assert registry.active_run_count() == 0
