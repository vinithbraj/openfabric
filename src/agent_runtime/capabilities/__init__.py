"""Capability contracts, mock capabilities, and registry."""

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.filesystem import (
    ListDirectoryCapability,
    ReadFileCapability,
    SearchFilesCapability,
)
from agent_runtime.capabilities.markdown import MarkdownRenderCapability
from agent_runtime.capabilities.python_data import TransformTableCapability
from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.capabilities.shell import ShellInspectCapability
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
    registry.register(ShellInspectCapability())
    return registry

__all__ = [
    "BaseCapability",
    "build_default_registry",
    "CapabilityManifest",
    "CapabilityRegistry",
    "ListDirectoryCapability",
    "MarkdownRenderCapability",
    "ReadFileCapability",
    "ReadQueryCapability",
    "SearchFilesCapability",
    "ShellInspectCapability",
    "TransformTableCapability",
]
