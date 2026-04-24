from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from aor_runtime.core.contracts import ToolSpec
from aor_runtime.runtime.dataflow import collect_step_references


class ToolExecutionError(RuntimeError):
    pass


class ToolArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseTool(ABC):
    spec: ToolSpec
    args_model: type[ToolArgsModel]
    result_model: type[ToolResultModel]

    @abstractmethod
    def run(self, arguments: ToolArgsModel) -> ToolResultModel | dict[str, Any]:
        raise NotImplementedError

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated_args = self.args_model.model_validate(_strip_internal_args(arguments))
        raw_result = self.run(validated_args)
        if isinstance(raw_result, self.result_model):
            return raw_result.model_dump()
        validated_result = self.result_model.model_validate(raw_result)
        return validated_result.model_dump()


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def specs(self, names: list[str]) -> list[dict[str, Any]]:
        return [self.get(name).spec.model_dump() for name in names]

    def validate_step(self, action: str, args: dict[str, Any]) -> None:
        tool = self.get(action)
        if collect_step_references(args):
            return
        tool.args_model.model_validate(_strip_internal_args(args))

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        return tool.invoke(arguments)


def _strip_internal_args(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_internal_args(nested) for key, nested in value.items() if not str(key).startswith("__")}
    if isinstance(value, list):
        return [_strip_internal_args(item) for item in value]
    return value
