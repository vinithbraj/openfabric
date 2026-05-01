"""Safe read-only system inspection capability manifests."""

from __future__ import annotations

from agent_runtime.capabilities.base import GatewayBackedCapability
from agent_runtime.capabilities.schemas import CapabilityManifest


class SystemMemoryStatusCapability(GatewayBackedCapability):
    """Inspect memory and swap status through a constrained gateway operation."""

    manifest = CapabilityManifest(
        capability_id="system.memory_status",
        domain="system",
        operation_id="memory_status",
        name="System Memory Status",
        description="Read safe memory and swap usage information from the remote system.",
        semantic_verbs=["read", "analyze"],
        object_types=["system.memory", "memory", "ram", "swap", "system.resources"],
        argument_schema={"human_readable": {"type": "boolean"}},
        required_arguments=[],
        optional_arguments=["human_readable"],
        output_schema={
            "memory": {"type": "object"},
            "swap": {"type": "object"},
            "rows": {"type": "array"},
        },
        execution_backend="gateway",
        backend_operation="system.memory_status",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"human_readable": True}}],
        safety_notes=[
            "Reads safe memory summary data only.",
            "Does not expose arbitrary shell or environment access.",
        ],
    )


class SystemDiskUsageCapability(GatewayBackedCapability):
    """Inspect disk usage through a constrained gateway operation."""

    manifest = CapabilityManifest(
        capability_id="system.disk_usage",
        domain="system",
        operation_id="disk_usage",
        name="System Disk Usage",
        description="Read safe disk usage information for a workspace-bounded path on the remote system.",
        semantic_verbs=["read", "analyze"],
        object_types=["system.disk", "disk", "storage", "filesystem.storage"],
        argument_schema={
            "path": {"type": "string"},
            "human_readable": {"type": "boolean"},
        },
        required_arguments=[],
        optional_arguments=["path", "human_readable"],
        output_schema={
            "path": {"type": "string"},
            "total": {"type": "integer"},
            "used": {"type": "integer"},
            "free": {"type": "integer"},
            "rows": {"type": "array"},
        },
        execution_backend="gateway",
        backend_operation="system.disk_usage",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": ".", "human_readable": True}}],
        safety_notes=[
            "Uses deterministic disk usage inspection only.",
            "Workspace-bounded path input; no arbitrary shell.",
        ],
    )


class SystemCpuLoadCapability(GatewayBackedCapability):
    """Inspect CPU load through a constrained gateway operation."""

    manifest = CapabilityManifest(
        capability_id="system.cpu_load",
        domain="system",
        operation_id="cpu_load",
        name="System CPU Load",
        description="Read safe CPU load and core count information from the remote system.",
        semantic_verbs=["read", "analyze"],
        object_types=["system.cpu", "cpu", "load", "load_average"],
        argument_schema={},
        required_arguments=[],
        optional_arguments=[],
        output_schema={
            "load_1m": {"type": ["number", "null"]},
            "load_5m": {"type": ["number", "null"]},
            "load_15m": {"type": ["number", "null"]},
            "cpu_count": {"type": ["integer", "null"]},
            "rows": {"type": "array"},
        },
        execution_backend="gateway",
        backend_operation="system.cpu_load",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {}}],
        safety_notes=[
            "Reads CPU load with Python or OS APIs only.",
            "No arbitrary shell or process mutation.",
        ],
    )


class SystemUptimeCapability(GatewayBackedCapability):
    """Inspect system uptime through a constrained gateway operation."""

    manifest = CapabilityManifest(
        capability_id="system.uptime",
        domain="system",
        operation_id="uptime",
        name="System Uptime",
        description="Read safe uptime information from the remote system.",
        semantic_verbs=["read", "analyze"],
        object_types=["system.uptime", "uptime", "load"],
        argument_schema={},
        required_arguments=[],
        optional_arguments=[],
        output_schema={
            "seconds": {"type": ["number", "null"]},
            "human": {"type": "string"},
        },
        execution_backend="gateway",
        backend_operation="system.uptime",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {}}],
        safety_notes=["Reads uptime metadata only; no shell command injection surface."],
    )


class SystemEnvironmentSummaryCapability(GatewayBackedCapability):
    """Inspect a safe environment summary through a constrained gateway operation."""

    manifest = CapabilityManifest(
        capability_id="system.environment_summary",
        domain="system",
        operation_id="environment_summary",
        name="System Environment Summary",
        description="Return a safe summary of the remote OS, Python version, working directory, CPU count, and memory summary when available.",
        semantic_verbs=["read", "summarize"],
        object_types=["system.environment", "environment", "runtime.environment", "os"],
        argument_schema={},
        required_arguments=[],
        optional_arguments=[],
        output_schema={
            "os": {"type": "string"},
            "python_version": {"type": "string"},
            "working_directory": {"type": "string"},
            "cpu_count": {"type": ["integer", "null"]},
            "memory": {"type": ["object", "null"]},
            "rows": {"type": "array"},
        },
        execution_backend="gateway",
        backend_operation="system.environment_summary",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {}}],
        safety_notes=[
            "Does not expose environment variables by default.",
            "Returns only safe summary metadata.",
        ],
    )


__all__ = [
    "SystemCpuLoadCapability",
    "SystemDiskUsageCapability",
    "SystemEnvironmentSummaryCapability",
    "SystemMemoryStatusCapability",
    "SystemUptimeCapability",
]
