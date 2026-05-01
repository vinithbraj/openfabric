"""Capability contracts, mock capabilities, and registry."""

from agent_runtime.capabilities.base import BaseCapability, GatewayBackedCapability
from agent_runtime.capabilities.data import (
    DataAggregateCapability,
    DataHeadCapability,
    DataProjectCapability,
)
from agent_runtime.capabilities.filesystem import (
    ListDirectoryCapability,
    ReadFileCapability,
    SearchFilesCapability,
    WriteFileCapability,
)
from agent_runtime.capabilities.markdown import MarkdownRenderCapability
from agent_runtime.capabilities.python_data import TransformTableCapability
from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.runtime import (
    RuntimeDescribeCapabilitiesCapability,
    RuntimeDescribePipelineCapability,
    RuntimeExplainLastFailureCapability,
    RuntimeShowLastPlanCapability,
)
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
from agent_runtime.capabilities.system import (
    SystemCpuLoadCapability,
    SystemDiskUsageCapability,
    SystemEnvironmentSummaryCapability,
    SystemMemoryStatusCapability,
    SystemUptimeCapability,
)


def build_default_registry() -> CapabilityRegistry:
    """Build the default safe capability registry for the agent runtime."""

    registry = CapabilityRegistry()
    registry.register(RuntimeDescribeCapabilitiesCapability(registry))
    registry.register(RuntimeDescribePipelineCapability())
    registry.register(RuntimeShowLastPlanCapability())
    registry.register(RuntimeExplainLastFailureCapability())
    registry.register(DataAggregateCapability())
    registry.register(DataProjectCapability())
    registry.register(DataHeadCapability())
    registry.register(ListDirectoryCapability())
    registry.register(ReadFileCapability())
    registry.register(SearchFilesCapability())
    registry.register(WriteFileCapability())
    registry.register(ReadQueryCapability())
    registry.register(TransformTableCapability())
    registry.register(MarkdownRenderCapability())
    registry.register(ShellWhichCapability())
    registry.register(ShellPwdCapability())
    registry.register(ShellListProcessesCapability())
    registry.register(ShellCheckPortCapability())
    registry.register(ShellGitStatusCapability())
    registry.register(ShellRunTestsReadonlyCapability())
    registry.register(SystemMemoryStatusCapability())
    registry.register(SystemDiskUsageCapability())
    registry.register(SystemCpuLoadCapability())
    registry.register(SystemUptimeCapability())
    registry.register(SystemEnvironmentSummaryCapability())
    return registry

__all__ = [
    "BaseCapability",
    "build_default_registry",
    "CapabilityManifest",
    "CapabilityRegistry",
    "DataAggregateCapability",
    "DataHeadCapability",
    "DataProjectCapability",
    "GatewayBackedCapability",
    "ListDirectoryCapability",
    "MarkdownRenderCapability",
    "ReadFileCapability",
    "ReadQueryCapability",
    "RuntimeDescribeCapabilitiesCapability",
    "RuntimeDescribePipelineCapability",
    "RuntimeExplainLastFailureCapability",
    "RuntimeShowLastPlanCapability",
    "SearchFilesCapability",
    "ShellCheckPortCapability",
    "ShellGitStatusCapability",
    "ShellListProcessesCapability",
    "ShellPwdCapability",
    "ShellRunTestsReadonlyCapability",
    "ShellWhichCapability",
    "SystemCpuLoadCapability",
    "SystemDiskUsageCapability",
    "SystemEnvironmentSummaryCapability",
    "SystemMemoryStatusCapability",
    "SystemUptimeCapability",
    "TransformTableCapability",
    "WriteFileCapability",
]
