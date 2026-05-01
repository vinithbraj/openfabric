"""HTTP client for gateway-backed capability execution."""

from __future__ import annotations

import base64
import json
import shlex
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.core.config import RuntimeConfig
from agent_runtime.core.types import ActionNode, ExecutionResult
from agent_runtime.execution.errors import GatewayConfigurationError, GatewayExecutionError


class _GatewayExecResponse(BaseModel):
    """Compact HTTP response returned by the generic gateway exec endpoint."""

    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int


class _GatewayToolResponse(BaseModel):
    """Structured success envelope emitted by the remote tool runner."""

    model_config = ConfigDict(extra="forbid")

    status: str = "success"
    data_preview: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayClient:
    """Execute gateway-backed tools through the generic gateway HTTP API."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def resolve_node(self, execution_context: dict[str, Any]) -> str:
        """Resolve the target gateway node from request context or runtime defaults."""

        for key in ("gateway_node", "node", "target_node"):
            value = str(execution_context.get(key) or "").strip()
            if value:
                return value
        value = str(self.config.gateway_default_node or "").strip()
        if value:
            return value
        raise GatewayConfigurationError("Gateway default node is not configured.")

    def resolve_url(self, node: str) -> str:
        """Resolve the gateway base URL for one node."""

        gateway_url = str(self.config.gateway_endpoints.get(node, "") or self.config.gateway_url or "").strip()
        if gateway_url:
            return gateway_url.rstrip("/")
        raise GatewayConfigurationError(f"Gateway URL is not configured for node: {node}.")

    def build_command(self, capability: BaseCapability, arguments: dict[str, Any]) -> str:
        """Build the deterministic remote-runner command executed by the gateway."""

        operation = str(
            capability.manifest.backend_operation or capability.manifest.capability_id
        ).strip()
        if not operation:
            raise GatewayConfigurationError(
                f"Gateway-backed capability is missing backend operation: {capability.manifest.capability_id}."
            )

        payload = base64.b64encode(
            json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).decode("ascii")
        pieces = [
            "python3",
            "-m",
            "gateway_agent.remote_runner",
            "--operation",
            operation,
            "--payload",
            payload,
        ]
        return " ".join(shlex.quote(piece) for piece in pieces)

    def _post_exec(self, url: str, node: str, command: str) -> _GatewayExecResponse:
        """Send one command to the gateway exec endpoint and validate the HTTP payload."""

        payload = json.dumps({"node": node, "command": command}).encode("utf-8")
        request = urllib_request.Request(
            f"{url}/exec",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.config.gateway_timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise GatewayExecutionError(
                f"Gateway request failed with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:
            raise GatewayExecutionError(f"Gateway request failed: {exc.reason}") from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise GatewayExecutionError("Gateway returned invalid JSON for exec response.") from exc
        return _GatewayExecResponse.model_validate(body)

    def invoke(
        self,
        *,
        node: ActionNode,
        capability: BaseCapability,
        arguments: dict[str, Any],
        execution_context: dict[str, Any],
    ) -> ExecutionResult:
        """Invoke one gateway-backed capability and return a normalized raw result."""

        arguments = capability.validate_arguments(dict(arguments or {}))
        target_node = self.resolve_node(execution_context)
        gateway_url = self.resolve_url(target_node)
        command = self.build_command(capability, arguments)
        exec_response = self._post_exec(gateway_url, target_node, command)
        if exec_response.exit_code != 0:
            message = exec_response.stderr.strip() or exec_response.stdout.strip() or "Gateway execution failed."
            raise GatewayExecutionError(message)

        try:
            envelope = _GatewayToolResponse.model_validate_json(exec_response.stdout)
        except Exception as exc:
            raise GatewayExecutionError("Gateway tool runner returned invalid JSON.") from exc

        return ExecutionResult(
            node_id=node.id,
            status="success" if envelope.status == "success" else "error",
            data_preview=envelope.data_preview,
            error=None if envelope.status == "success" else "Gateway tool runner reported an error.",
            metadata={
                "gateway_node": target_node,
                "gateway_url": gateway_url,
                "backend_operation": capability.manifest.backend_operation or capability.manifest.capability_id,
                **dict(envelope.metadata),
            },
        )
