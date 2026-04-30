"""OpenFABRIC Runtime Module: aor_runtime.tools.base

Purpose:
    Define the base tool interfaces and typed argument/result contracts.

Responsibilities:
    Expose typed tool arguments/results for filesystem, SQL, shell, SLURM, text formatting, Python, and runtime return operations.

Data flow / Interfaces:
    Receives validated tool arguments from the executor and returns structured result models for downstream contracts and presenters.

Boundaries:
    Does not decide user intent; every tool must preserve safety, allowed-root, read-only, timeout, and result-shape boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from aor_runtime.core.contracts import ToolSpec
from aor_runtime.runtime.dataflow import collect_step_references
from aor_runtime.runtime.lifecycle import ToolInvocationContext


class ToolExecutionError(RuntimeError):
    """Represent tool execution error within the OpenFABRIC runtime. It extends RuntimeError.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolExecutionError.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.base.ToolExecutionError and related tests.
    """
    pass


class ToolArgsModel(BaseModel):
    """Represent tool args model within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolArgsModel.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.base.ToolArgsModel and related tests.
    """
    model_config = ConfigDict(extra="forbid")


class ToolResultModel(BaseModel):
    """Represent tool result model within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolResultModel.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.base.ToolResultModel and related tests.
    """
    model_config = ConfigDict(extra="forbid")


class BaseTool(ABC):
    """Represent base tool within the OpenFABRIC runtime. It extends ABC.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by BaseTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.base.BaseTool and related tests.
    """
    spec: ToolSpec
    args_model: type[ToolArgsModel]
    result_model: type[ToolResultModel]

    @abstractmethod
    def run(self, arguments: ToolArgsModel) -> ToolResultModel | dict[str, Any]:
        """Run for BaseTool instances.

        Inputs:
            Receives arguments for this BaseTool method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by registered tool execution through BaseTool.run calls and related tests.
        """
        raise NotImplementedError

    def run_with_context(
        self,
        arguments: ToolArgsModel,
        context: ToolInvocationContext,
    ) -> ToolResultModel | dict[str, Any]:
        """Run with context for BaseTool instances.

        Inputs:
            Receives arguments, context for this BaseTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through BaseTool.run_with_context calls and related tests.
        """
        context.throw_if_cancelled()
        return self.run(arguments)

    def invoke(self, arguments: dict[str, Any], *, context: ToolInvocationContext | None = None) -> dict[str, Any]:
        """Invoke for BaseTool instances.

        Inputs:
            Receives arguments, context for this BaseTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through BaseTool.invoke calls and related tests.
        """
        validated_args = self.args_model.model_validate(_strip_internal_args(arguments))
        if context is not None:
            raw_result = self.run_with_context(validated_args, context)
        else:
            raw_result = self.run(validated_args)
        if isinstance(raw_result, self.result_model):
            return raw_result.model_dump()
        validated_result = self.result_model.model_validate(raw_result)
        return validated_result.model_dump()


class ToolRegistry:
    """Represent tool registry within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolRegistry.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.base.ToolRegistry and related tests.
    """
    def __init__(self, tools: list[BaseTool]) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives tools for this ToolRegistry method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through ToolRegistry.__init__ calls and related tests.
        """
        self._tools = {tool.spec.name: tool for tool in tools}

    def contains(self, name: str) -> bool:
        """Contains for ToolRegistry instances.

        Inputs:
            Receives name for this ToolRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through ToolRegistry.contains calls and related tests.
        """
        return str(name or "") in self._tools

    def get(self, name: str) -> BaseTool:
        """Get for ToolRegistry instances.

        Inputs:
            Receives name for this ToolRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through ToolRegistry.get calls and related tests.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        """Names for ToolRegistry instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through ToolRegistry.names calls and related tests.
        """
        return sorted(self._tools)

    def specs(self, names: list[str]) -> list[dict[str, Any]]:
        """Specs for ToolRegistry instances.

        Inputs:
            Receives names for this ToolRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through ToolRegistry.specs calls and related tests.
        """
        return [self.get(name).spec.model_dump() for name in names]

    def validate_step(self, action: str, args: dict[str, Any]) -> None:
        """Validate step for ToolRegistry instances.

        Inputs:
            Receives action, args for this ToolRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by registered tool execution through ToolRegistry.validate_step calls and related tests.
        """
        tool = self.get(action)
        if collect_step_references(args):
            return
        tool.args_model.model_validate(_strip_internal_args(args))

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolInvocationContext | None = None,
    ) -> dict[str, Any]:
        """Invoke for ToolRegistry instances.

        Inputs:
            Receives name, arguments, context for this ToolRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through ToolRegistry.invoke calls and related tests.
        """
        tool = self.get(name)
        return tool.invoke(arguments, context=context)


def _strip_internal_args(value: Any) -> Any:
    """Handle the internal strip internal args helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.base._strip_internal_args.
    """
    if isinstance(value, dict):
        return {key: _strip_internal_args(nested) for key, nested in value.items() if not str(key).startswith("__")}
    if isinstance(value, list):
        return [_strip_internal_args(item) for item in value]
    return value
