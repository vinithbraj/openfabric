"""Capability contracts, mock capabilities, and registry."""

from agent_runtime.capabilities.base import BaseCapability, GatewayBackedCapability
from agent_runtime.capabilities.filesystem import (
    ListDirectoryCapability,
    ReadFileCapability,
    SearchFilesCapability,
)
from agent_runtime.capabilities.markdown import MarkdownRenderCapability
from agent_runtime.capabilities.python_data import TransformTableCapability
from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.capabilities.shell import (
    ShellCheckPortCapability,
    ShellGitStatusCapability,
    ShellListProcessesCapability,
    ShellPwdCapability,
    ShellRunTestsReadonlyCapability,
    ShellWhichCapability,
)
from agent_runtime.capabilities.sql import ReadQueryCapability


def build_default_registry() -> CapabilityRegistry:
    """Build the default safe capability registry for the agent runtime."""

    registry = CapabilityRegistry()
    registry.register(ListDirectoryCapability())
    registry.register(ReadFileCapability())
    registry.register(SearchFilesCapability())
    registry.register(ReadQueryCapability())
    registry.register(TransformTableCapability())
    registry.register(MarkdownRenderCapability())
    registry.register(ShellWhichCapability())
    registry.register(ShellPwdCapability())
    registry.register(ShellListProcessesCapability())
    registry.register(ShellCheckPortCapability())
    registry.register(ShellGitStatusCapability())
    registry.register(ShellRunTestsReadonlyCapability())
    return registry

__all__ = [
    "BaseCapability",
    "build_default_registry",
    "CapabilityManifest",
    "CapabilityRegistry",
    "GatewayBackedCapability",
    "ListDirectoryCapability",
    "MarkdownRenderCapability",
    "ReadFileCapability",
    "ReadQueryCapability",
    "SearchFilesCapability",
    "ShellCheckPortCapability",
    "ShellGitStatusCapability",
    "ShellListProcessesCapability",
    "ShellPwdCapability",
    "ShellRunTestsReadonlyCapability",
    "ShellWhichCapability",
    "TransformTableCapability",
]
