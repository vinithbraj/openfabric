from __future__ import annotations

import json
from collections.abc import Iterator

import requests
from pydantic import BaseModel, ValidationError

from aor_runtime.config import Settings
from aor_runtime.tools.base import ToolExecutionError


class GatewayExecResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


class GatewayExecStreamChunk(BaseModel):
    type: str
    text: str = ""
    exit_code: int | None = None


def resolve_execution_node(settings: Settings, node: str = "") -> str:
    try:
        return settings.resolve_node(node)
    except ValueError as exc:
        raise ToolExecutionError(str(exc)) from exc


def execute_gateway_command(settings: Settings, *, node: str, command: str) -> GatewayExecResult:
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
            timeout=settings.gateway_timeout_seconds,
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


def stream_gateway_command(settings: Settings, *, node: str, command: str) -> Iterator[GatewayExecStreamChunk]:
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
            timeout=settings.gateway_timeout_seconds,
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
