"""Gateway-backed filesystem capability manifests."""

from __future__ import annotations

from typing import Any

from agent_runtime.capabilities.base import GatewayBackedCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError


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
        output_object_types=["filesystem.file", "filesystem.path"],
        output_fields=["path", "content_preview", "truncated"],
        output_affordances=["returns.relative_path", "returns.file_preview"],
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


class WriteFileCapability(GatewayBackedCapability):
    """Write UTF-8 text content to one workspace-bounded file through the gateway."""

    manifest = CapabilityManifest(
        capability_id="filesystem.write_file",
        domain="filesystem",
        operation_id="write_file",
        name="Write File",
        description="Write text, JSON, or Markdown content to one file inside the workspace.",
        semantic_verbs=["create", "update", "render"],
        object_types=["filesystem.file", "report", "document", "markdown", "json"],
        argument_schema={
            "path": {"type": "string"},
            "format": {"type": "string"},
            "content": {"type": "string"},
            "input_ref": {"type": "string"},
            "overwrite": {"type": "boolean"},
        },
        required_arguments=["path", "format"],
        optional_arguments=["content", "input_ref", "overwrite"],
        output_schema={
            "message": {"type": "string"},
            "path": {"type": "string"},
            "absolute_path": {"type": "string"},
            "format": {"type": "string"},
            "bytes_written": {"type": "integer"},
            "created": {"type": "boolean"},
            "overwritten": {"type": "boolean"},
        },
        output_object_types=["filesystem.file", "filesystem.path"],
        output_fields=[
            "message",
            "path",
            "absolute_path",
            "format",
            "bytes_written",
            "created",
            "overwritten",
        ],
        output_affordances=[
            "returns.absolute_path",
            "returns.relative_path",
            "returns.write_confirmation",
            "returns.file_metadata",
        ],
        execution_backend="gateway",
        backend_operation="filesystem.write_file",
        risk_level="low",
        read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        examples=[
            {"arguments": {"path": "report.txt", "format": "text", "content": "hello"}},
            {
                "arguments": {
                    "path": "report.md",
                    "format": "markdown",
                    "input_ref": "node::task_1",
                }
            },
        ],
        safety_notes=[
            "Writes are limited to the configured workspace root.",
            "Explicit confirmation is always required.",
            "Existing files are not overwritten unless overwrite=true is provided.",
        ],
    )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate write-file arguments including cross-field constraints."""

        payload = super().validate_arguments(arguments)
        file_format = str(payload.get("format") or "").strip().lower()
        if file_format not in {"text", "json", "markdown"}:
            raise ValidationError(
                "filesystem.write_file format must be one of: text, json, markdown"
            )
        has_content = "content" in payload and payload.get("content") is not None
        has_input_ref = "input_ref" in payload and payload.get("input_ref") is not None
        if has_content == has_input_ref:
            raise ValidationError(
                "filesystem.write_file requires exactly one of content or input_ref"
            )
        if has_content and not isinstance(payload.get("content"), str):
            raise ValidationError("filesystem.write_file content must be a string")
        if "path" in payload and (not isinstance(payload["path"], str) or not payload["path"].strip()):
            raise ValidationError("filesystem.write_file path must be a non-empty string")
        if "overwrite" in payload:
            payload["overwrite"] = bool(payload["overwrite"])
        payload["format"] = file_format
        return payload
