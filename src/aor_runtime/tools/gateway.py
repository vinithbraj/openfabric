from __future__ import annotations

import requests
from pydantic import BaseModel, ValidationError

from aor_runtime.config import Settings
from aor_runtime.tools.base import ToolExecutionError


class GatewayExecResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


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
