from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from aor_runtime.core.contracts import ToolResult


class ToolSpec(BaseModel):
    name: str
    description: str
    arguments_schema: dict[str, Any]


class BaseTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def specs(self, names: list[str]) -> list[dict[str, Any]]:
        return [self.get(name).spec.model_dump() for name in names]

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.get(name).invoke(arguments)
