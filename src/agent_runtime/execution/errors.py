"""Execution-specific errors."""

from agent_runtime.core.errors import AgentRuntimeError


class ExecutionError(AgentRuntimeError):
    """Raised when DAG execution fails."""
