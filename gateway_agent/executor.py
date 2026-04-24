from __future__ import annotations

import subprocess

from gateway_agent.config import Settings
from gateway_agent.models import ExecResponse


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
