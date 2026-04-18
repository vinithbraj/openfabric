import os
import shutil
import subprocess
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class ExecuteRequest(BaseModel):
    command: str
    args: list[str] = []
    timeout_seconds: float | None = None


DEFAULT_ALLOWED_COMMANDS = {
    "sinfo",
    "squeue",
    "sacct",
    "scontrol",
    "sreport",
    "sshare",
    "sacctmgr",
    "sdiag",
    "sprio",
    "sstat",
    "scancel",
}


def _allowed_commands() -> set[str]:
    raw = os.getenv("SLURM_GATEWAY_ALLOWED_COMMANDS", "")
    if not raw.strip():
        return set(DEFAULT_ALLOWED_COMMANDS)
    return {item.strip() for item in raw.split(",") if item.strip()}


def _default_timeout() -> float:
    raw = os.getenv("SLURM_GATEWAY_COMMAND_TIMEOUT_SECONDS", "60")
    try:
        return max(1.0, min(float(raw), 600.0))
    except ValueError:
        return 60.0


def _resolve_command(command: str) -> str:
    if command not in _allowed_commands():
        raise HTTPException(status_code=403, detail=f"Command '{command}' is not allowed.")
    resolved = shutil.which(command)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Command '{command}' is not installed on this node.")
    return resolved


@app.get("/health")
def health() -> dict[str, Any]:
    availability = {command: bool(shutil.which(command)) for command in sorted(_allowed_commands())}
    return {"ok": True, "allowed_commands": sorted(_allowed_commands()), "available_commands": availability}


@app.post("/execute")
def execute(req: ExecuteRequest) -> dict[str, Any]:
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="command is required")
    binary = _resolve_command(command)
    timeout_seconds = req.timeout_seconds if isinstance(req.timeout_seconds, (int, float)) else _default_timeout()
    argv = [binary, *[str(item) for item in req.args]]
    started = time.perf_counter()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=408, detail=f"Command timed out after {timeout_seconds} seconds.") from exc
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "args": [str(item) for item in req.args],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "duration_ms": duration_ms,
    }
