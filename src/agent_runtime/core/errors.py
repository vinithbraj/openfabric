"""Shared exception hierarchy for the agent runtime."""

from __future__ import annotations


class AgentRuntimeError(Exception):
    """Base class for expected runtime failures."""


class ValidationError(AgentRuntimeError):
    """Raised when a typed contract fails validation."""


class CapabilityNotFoundError(AgentRuntimeError):
    """Raised when a DAG references an unknown capability."""


class SafetyError(AgentRuntimeError):
    """Raised when a request violates a deterministic safety policy."""
