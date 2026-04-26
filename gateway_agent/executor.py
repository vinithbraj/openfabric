from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Iterator

from gateway_agent.config import Settings
from gateway_agent.models import ExecResponse, ExecStreamEvent


TIMEOUT_EXIT_CODE = 124


def _coerce_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute_command(settings: Settings, command: str) -> ExecResponse:
    completed = None
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=settings.exec_timeout_seconds,
            cwd=str(settings.workdir) if settings.workdir is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _coerce_stream(exc.stderr).rstrip("\n")
        timeout_message = f"Command timed out after {settings.exec_timeout_seconds:g} seconds."
        if stderr:
            stderr = f"{stderr}\n{timeout_message}\n"
        else:
            stderr = f"{timeout_message}\n"
        return ExecResponse(
            stdout=_coerce_stream(exc.stdout),
            stderr=stderr,
            exit_code=TIMEOUT_EXIT_CODE,
        )

    return ExecResponse(
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
    )


def _stream_reader(stream, channel: str, output_queue: "queue.Queue[tuple[str, str | None]]") -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put((channel, _coerce_stream(line)))
    finally:
        try:
            stream.close()
        finally:
            output_queue.put((f"{channel}_eof", None))


def stream_command(settings: Settings, command: str) -> Iterator[ExecStreamEvent]:
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(settings.workdir) if settings.workdir is not None else None,
    )
    output_queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    stdout_done = False
    stderr_done = False
    deadline = time.monotonic() + settings.exec_timeout_seconds

    stdout_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stdout, "stdout", output_queue),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_reader,
        args=(process.stderr, "stderr", output_queue),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    while not (stdout_done and stderr_done and process.poll() is not None and output_queue.empty()):
        remaining = deadline - time.monotonic()
        if remaining <= 0 and process.poll() is None:
            timed_out = True
            process.kill()
            yield ExecStreamEvent(
                type="stderr",
                text=f"Command timed out after {settings.exec_timeout_seconds:g} seconds.\n",
            )
            break
        try:
            channel, text = output_queue.get(timeout=min(0.1, max(remaining, 0.0) or 0.1))
        except queue.Empty:
            continue
        if channel == "stdout_eof":
            stdout_done = True
            continue
        if channel == "stderr_eof":
            stderr_done = True
            continue
        yield ExecStreamEvent(type=channel, text=text or "")

    process.wait()
    stdout_thread.join(timeout=0.2)
    stderr_thread.join(timeout=0.2)

    while not output_queue.empty():
        channel, text = output_queue.get_nowait()
        if channel == "stdout_eof":
            stdout_done = True
            continue
        if channel == "stderr_eof":
            stderr_done = True
            continue
        yield ExecStreamEvent(type=channel, text=text or "")

    yield ExecStreamEvent(type="completed", exit_code=TIMEOUT_EXIT_CODE if timed_out else int(process.returncode or 0))
