"""OpenFABRIC Runtime Module: aor_runtime.tools.factory

Purpose:
    Build the registered tool set from runtime settings.

Responsibilities:
    Expose typed tool arguments/results for filesystem, SQL, shell, SLURM, text formatting, Python, and runtime return operations.

Data flow / Interfaces:
    Receives validated tool arguments from the executor and returns structured result models for downstream contracts and presenters.

Boundaries:
    Does not decide user intent; every tool must preserve safety, allowed-root, read-only, timeout, and result-shape boundaries.
"""

from __future__ import annotations

from aor_runtime.config import Settings, get_settings
from aor_runtime.tools.base import ToolRegistry
from aor_runtime.tools.filesystem import (
    FileAggregateTool,
    FileCopyTool,
    FileExistsTool,
    FileSizeTool,
    FindFilesTool,
    FileNotExistsTool,
    GlobFilesTool,
    FileReadTool,
    FileWriteTool,
    ListDirectoryTool,
    MakeDirectoryTool,
)
from aor_runtime.tools.python_exec import PythonExecTool
from aor_runtime.tools.runtime_return import RuntimeReturnTool
from aor_runtime.tools.search_content import SearchContentTool
from aor_runtime.tools.shell import ShellExecTool
from aor_runtime.tools.slurm import (
    SlurmAccountingAggregateTool,
    SlurmAccountingTool,
    SlurmDBDHealthTool,
    SlurmJobDetailTool,
    SlurmMetricsTool,
    SlurmNodeDetailTool,
    SlurmNodesTool,
    SlurmPartitionsTool,
    SlurmQueueTool,
)
from aor_runtime.tools.sql import SQLQueryTool, SQLSchemaTool, SQLValidateTool
from aor_runtime.tools.text_format import TextFormatTool


def build_tool_registry(settings: Settings | None = None) -> ToolRegistry:
    """Build tool registry for the surrounding runtime workflow.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.factory.build_tool_registry.
    """
    configured = settings or get_settings()
    return ToolRegistry(
        [
            FileExistsTool(configured),
            FileNotExistsTool(configured),
            FileCopyTool(configured),
            ShellExecTool(configured),
            FileReadTool(configured),
            FileWriteTool(configured),
            MakeDirectoryTool(configured),
            ListDirectoryTool(configured),
            GlobFilesTool(configured),
            FindFilesTool(configured),
            SearchContentTool(configured),
            FileSizeTool(configured),
            FileAggregateTool(configured),
            SlurmQueueTool(configured),
            SlurmJobDetailTool(configured),
            SlurmNodesTool(configured),
            SlurmNodeDetailTool(configured),
            SlurmPartitionsTool(configured),
            SlurmAccountingTool(configured),
            SlurmAccountingAggregateTool(configured),
            SlurmMetricsTool(configured),
            SlurmDBDHealthTool(configured),
            PythonExecTool(configured),
            SQLQueryTool(configured),
            SQLSchemaTool(configured),
            SQLValidateTool(configured),
            TextFormatTool(),
            RuntimeReturnTool(),
        ]
    )
