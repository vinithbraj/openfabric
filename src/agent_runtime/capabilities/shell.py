"""Gateway-backed constrained shell capability manifests."""

from __future__ import annotations

from agent_runtime.capabilities.base import GatewayBackedCapability
from agent_runtime.capabilities.schemas import CapabilityManifest


class ShellWhichCapability(GatewayBackedCapability):
    """Locate one executable through a constrained gateway command."""

    manifest = CapabilityManifest(
        capability_id="shell.which",
        domain="shell",
        operation_id="which",
        name="Locate Executable",
        description="Find the installed path for one program using a constrained shell lookup.",
        semantic_verbs=["read", "search"],
        object_types=["program", "executable", "binary"],
        argument_schema={"program": {"type": "string"}},
        required_arguments=["program"],
        optional_arguments=[],
        output_schema={
            "program": {"type": "string"},
            "found": {"type": "boolean"},
            "path": {"type": ["string", "null"]},
        },
        execution_backend="gateway",
        backend_operation="shell.which",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"program": "git"}}],
        safety_notes=[
            "Runs a fixed executable lookup only.",
            "Does not accept arbitrary shell command text.",
        ],
    )


class ShellPwdCapability(GatewayBackedCapability):
    """Return the remote working directory through the gateway."""

    manifest = CapabilityManifest(
        capability_id="shell.pwd",
        domain="shell",
        operation_id="pwd",
        name="Print Working Directory",
        description="Return the remote working directory through a constrained shell command.",
        semantic_verbs=["read"],
        object_types=["directory", "workspace", "path"],
        argument_schema={},
        required_arguments=[],
        optional_arguments=[],
        output_schema={"cwd": {"type": "string"}},
        execution_backend="gateway",
        backend_operation="shell.pwd",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {}}],
        safety_notes=["Read-only shell command routed through the gateway only."],
    )


class ShellListProcessesCapability(GatewayBackedCapability):
    """List running processes through a constrained gateway command."""

    manifest = CapabilityManifest(
        capability_id="shell.list_processes",
        domain="shell",
        operation_id="list_processes",
        name="List Processes",
        description="List running processes with optional name filtering through a constrained shell command.",
        semantic_verbs=["read", "search", "analyze"],
        object_types=["process", "system"],
        argument_schema={
            "pattern": {"type": "string"},
            "limit": {"type": "integer"},
        },
        required_arguments=[],
        optional_arguments=["pattern", "limit"],
        output_schema={
            "processes": {"type": "array"},
            "truncated": {"type": "boolean"},
        },
        execution_backend="gateway",
        backend_operation="shell.list_processes",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"pattern": "python", "limit": 10}}],
        safety_notes=[
            "Runs a fixed process listing command only.",
            "Filtering is applied to structured output, not shell text.",
        ],
    )


class ShellCheckPortCapability(GatewayBackedCapability):
    """Inspect which process is bound to one port through the gateway."""

    manifest = CapabilityManifest(
        capability_id="shell.check_port",
        domain="shell",
        operation_id="check_port",
        name="Check Port",
        description="Inspect which process is listening on a port through a constrained shell command.",
        semantic_verbs=["read", "search", "analyze"],
        object_types=["port", "process", "system"],
        argument_schema={"port": {"type": "integer"}},
        required_arguments=["port"],
        optional_arguments=[],
        output_schema={"port": {"type": "integer"}, "listeners": {"type": "array"}},
        execution_backend="gateway",
        backend_operation="shell.check_port",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"port": 8310}}],
        safety_notes=["Runs a fixed port inspection command only."],
    )


class ShellGitStatusCapability(GatewayBackedCapability):
    """Read git status through a constrained gateway command."""

    manifest = CapabilityManifest(
        capability_id="shell.git_status",
        domain="shell",
        operation_id="git_status",
        name="Git Status",
        description="Read git status for one workspace path through a constrained shell command.",
        semantic_verbs=["read", "analyze"],
        object_types=["git", "repository", "workspace"],
        argument_schema={"path": {"type": "string"}},
        required_arguments=[],
        optional_arguments=["path"],
        output_schema={
            "path": {"type": "string"},
            "branch": {"type": ["string", "null"]},
            "is_clean": {"type": "boolean"},
            "status_lines": {"type": "array"},
        },
        execution_backend="gateway",
        backend_operation="shell.git_status",
        risk_level="low",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"path": "."}}],
        safety_notes=["Runs a fixed git status command only."],
    )


class ShellRunTestsReadonlyCapability(GatewayBackedCapability):
    """Run a constrained read-only test command through the gateway."""

    manifest = CapabilityManifest(
        capability_id="shell.run_tests_readonly",
        domain="shell",
        operation_id="run_tests_readonly",
        name="Run Tests Read Only",
        description="Run a constrained pytest invocation without accepting arbitrary shell flags.",
        semantic_verbs=["execute", "analyze"],
        object_types=["tests", "workspace", "repository"],
        argument_schema={
            "target": {"type": "string"},
            "max_failures": {"type": "integer"},
        },
        required_arguments=[],
        optional_arguments=["target", "max_failures"],
        output_schema={
            "target": {"type": "string"},
            "exit_code": {"type": "integer"},
            "stdout_lines": {"type": "array"},
            "stderr_lines": {"type": "array"},
        },
        execution_backend="gateway",
        backend_operation="shell.run_tests_readonly",
        risk_level="medium",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[{"arguments": {"target": "tests/test_execution_safety.py", "max_failures": 1}}],
        safety_notes=[
            "Runs a fixed pytest command only.",
            "Does not accept arbitrary shell flags or command text.",
            "Test code itself may still have side effects; keep shell execution policy disabled unless explicitly enabled.",
        ],
    )


__all__ = [
    "ShellCheckPortCapability",
    "ShellGitStatusCapability",
    "ShellListProcessesCapability",
    "ShellPwdCapability",
    "ShellRunTestsReadonlyCapability",
    "ShellWhichCapability",
]
