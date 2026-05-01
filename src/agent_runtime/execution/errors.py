"""Execution-specific errors."""

from agent_runtime.core.errors import AgentRuntimeError


class ExecutionError(AgentRuntimeError):
    """Raised when DAG execution fails."""


class GatewayConfigurationError(ExecutionError):
    """Raised when gateway-backed execution lacks required configuration."""


class GatewayExecutionError(ExecutionError):
    """Raised when a gateway request fails or returns an invalid response."""
