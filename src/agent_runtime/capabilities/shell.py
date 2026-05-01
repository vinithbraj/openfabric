"""Shell/system mock capability."""

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest


class ShellInspectCapability(BaseCapability):
    """Mock capability for read-only shell/system inspection."""

    manifest = CapabilityManifest(
        capability_id="shell.inspect_system",
        domain="shell",
        operation_id="inspect_system",
        name="Inspect System",
        description="Inspect read-only system information through approved shell access.",
        semantic_verbs=["read", "analyze"],
        object_types=["system", "process", "port"],
        argument_schema={},
        required_arguments=[],
        optional_arguments=["scope"],
        output_schema={"facts": {"type": "array"}},
        risk_level="medium",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"scope": "processes"}}],
        safety_notes=["Read-only system inspection only."],
    )
