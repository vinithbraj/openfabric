"""Base capability interfaces."""

from __future__ import annotations

from typing import Any

from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ExecutionResult


class BaseCapability:
    """Base class for manifest-driven capabilities."""

    manifest: CapabilityManifest

    @property
    def spec(self) -> CapabilityManifest:
        """Compatibility alias for older scaffold code."""

        return self.manifest

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate required and allowed arguments against the manifest."""

        payload = dict(arguments or {})
        missing = [name for name in self.manifest.required_arguments if name not in payload]
        if missing:
            raise ValidationError(
                f"missing required arguments for {self.manifest.capability_id}: {', '.join(missing)}"
            )
        allowed = set(self.manifest.required_arguments) | set(self.manifest.optional_arguments)
        unexpected = [name for name in payload if allowed and name not in allowed]
        if unexpected:
            raise ValidationError(
                f"unexpected arguments for {self.manifest.capability_id}: {', '.join(sorted(unexpected))}"
            )
        return payload

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Execute a capability and return a placeholder normalized result."""

        validated = self.validate_arguments(arguments)
        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={
                "capability_id": self.manifest.capability_id,
                "operation_id": self.manifest.operation_id,
                "arguments": validated,
            },
            metadata={"context": dict(context or {})},
        )
