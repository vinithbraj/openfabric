"""Gateway-backed filesystem capability manifests."""

from __future__ import annotations

from agent_runtime.capabilities.base import GatewayBackedCapability
from agent_runtime.capabilities.schemas import CapabilityManifest


class ListDirectoryCapability(GatewayBackedCapability):
    """List workspace-bounded filesystem entries through the gateway."""

    manifest = CapabilityManifest(
        capability_id="filesystem.list_directory",
        domain="filesystem",
        operation_id="list_directory",
        name="List Directory",
        description="List entries in a directory path.",
        semantic_verbs=["read", "search"],
        object_types=["directory", "filesystem"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=["path"],
        optional_arguments=["recursive", "include_hidden", "limit"],
        output_schema={"entries": {"type": "array"}},
        execution_backend="gateway",
        backend_operation="filesystem.list_directory",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": "."}}],
        safety_notes=["Read-only directory metadata access through the gateway."],
    )


class ReadFileCapability(GatewayBackedCapability):
    """Read a single workspace-bounded file through the gateway."""

    manifest = CapabilityManifest(
        capability_id="filesystem.read_file",
        domain="filesystem",
        operation_id="read_file",
        name="Read File",
        description="Read a file from disk.",
        semantic_verbs=["read"],
        object_types=["file", "filesystem"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=["path"],
        optional_arguments=["max_bytes"],
        output_schema={"content_preview": {"type": "string"}, "truncated": {"type": "boolean"}},
        execution_backend="gateway",
        backend_operation="filesystem.read_file",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": "README.md"}}],
        safety_notes=["Read-only file access through the gateway."],
    )


class SearchFilesCapability(GatewayBackedCapability):
    """Search for files by glob-style pattern through the gateway."""

    manifest = CapabilityManifest(
        capability_id="filesystem.search_files",
        domain="filesystem",
        operation_id="search_files",
        name="Search Files",
        description="Search files by name or pattern.",
        semantic_verbs=["search"],
        object_types=["file", "filesystem"],
        argument_schema={"pattern": {"type": "string"}},
        required_arguments=["path", "pattern"],
        optional_arguments=["recursive", "limit"],
        output_schema={"matches": {"type": "array"}},
        execution_backend="gateway",
        backend_operation="filesystem.search_files",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"pattern": "*.py", "path": "src"}}],
        safety_notes=["Read-only filesystem search through the gateway."],
    )
