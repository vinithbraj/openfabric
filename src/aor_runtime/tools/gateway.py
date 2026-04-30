"""OpenFABRIC Runtime Module: aor_runtime.tools.gateway

Purpose:
    Implement gateway HTTP transport for shell and SLURM execution.

Responsibilities:
    Expose typed tool arguments/results for filesystem, SQL, shell, SLURM, text formatting, Python, and runtime return operations.

Data flow / Interfaces:
    Receives validated tool arguments from the executor and returns structured result models for downstream contracts and presenters.

Boundaries:
    Does not decide user intent; every tool must preserve safety, allowed-root, read-only, timeout, and result-shape boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests
from pydantic import BaseModel, ValidationError

from aor_runtime.config import Settings
from aor_runtime.tools.base import ToolExecutionError


class GatewayExecResult(BaseModel):
    """Represent gateway exec result within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by GatewayExecResult.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.gateway.GatewayExecResult and related tests.
    """
    stdout: str
    stderr: str
    exit_code: int


class GatewayExecStreamChunk(BaseModel):
    """Represent gateway exec stream chunk within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by GatewayExecStreamChunk.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.gateway.GatewayExecStreamChunk and related tests.
    """
    type: str
    text: str = ""
    exit_code: int | None = None


def resolve_execution_node(settings: Settings, node: str = "") -> str:
    """Resolve execution node for the surrounding runtime workflow.

    Inputs:
        Receives settings, node for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.gateway.resolve_execution_node.
    """
    try:
        return settings.resolve_node(node)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_gateway_command(settings: Settings, *, node: str, command: str, timeout: float | None = None) -> GatewayExecResult:
    """Execute gateway command for the surrounding runtime workflow.

    Inputs:
        Receives settings, node, command, timeout for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.gateway.execute_gateway_command.
    """
    normalized_command = str(command).strip()
    if not normalized_command:
        raise ToolExecutionError("Command is required.")

    try:
        gateway_url = settings.resolve_gateway_url(node)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc

    try:
        response = requests.post(
            gateway_url,
            json={"node": node, "command": normalized_command},
            timeout=timeout if timeout is not None else settings.gateway_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return GatewayExecResult.model_validate(payload)
    except requests.RequestException as exc:
        raise ToolExecutionError(f"Gateway request failed: {exc}") from exc
    except ValidationError as exc:
        raise ToolExecutionError(f"Gateway response validation failed: {exc}") from exc
    except ValueError as exc:
        raise ToolExecutionError("Gateway response is not valid JSON.") from exc


def stream_gateway_command(settings: Settings, *, node: str, command: str, timeout: float | None = None) -> Iterator[GatewayExecStreamChunk]:
    """Stream gateway command for the surrounding runtime workflow.

    Inputs:
        Receives settings, node, command, timeout for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.gateway.stream_gateway_command.
    """
    normalized_command = str(command or "").strip()
    if not normalized_command:
        raise ToolExecutionError("Command is required.")

    try:
        gateway_url = settings.resolve_gateway_url(node)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc

    stream_url = f"{gateway_url.rstrip('/')}/stream"

    try:
        with requests.post(
            stream_url,
            json={"node": node, "command": normalized_command},
            timeout=timeout if timeout is not None else settings.gateway_timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            yield from _parse_sse_stream(response)
    except requests.RequestException as exc:
        raise ToolExecutionError(f"Gateway request failed: {exc}") from exc
    except ValidationError as exc:
        raise ToolExecutionError(f"Gateway response validation failed: {exc}") from exc
    except ValueError as exc:
        raise ToolExecutionError("Gateway response is not valid JSON.") from exc


def _parse_sse_stream(response: requests.Response) -> Iterator[GatewayExecStreamChunk]:
    """Handle the internal parse sse stream helper path for this module.

    Inputs:
        Receives response for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.gateway._parse_sse_stream.
    """
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
        if line == "":
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                if isinstance(payload, dict) and "type" not in payload:
                    payload["type"] = event_name
                yield GatewayExecStreamChunk.model_validate(payload)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.partition(":")[2].lstrip())

    if data_lines:
        payload = json.loads("\n".join(data_lines))
        if isinstance(payload, dict) and "type" not in payload:
            payload["type"] = event_name
        yield GatewayExecStreamChunk.model_validate(payload)
