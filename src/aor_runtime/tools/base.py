from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aor_runtime.core.contracts import ToolSpec


class ToolExecutionError(RuntimeError):
    pass


class BaseTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
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

    def validate_step(self, action: str, args: dict[str, Any]) -> None:
        spec = self.get(action).spec
        schema = spec.arguments_schema or {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = [key for key in required if key not in args]
        if missing:
            raise ValueError(f"Step {action!r} missing required args: {', '.join(missing)}")
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"Step {action!r} arg {key!r} must be a string.")
            if expected == "integer" and not isinstance(value, int):
                raise ValueError(f"Step {action!r} arg {key!r} must be an integer.")
            if expected == "object" and not isinstance(value, dict):
                raise ValueError(f"Step {action!r} arg {key!r} must be an object.")

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.validate_step(name, arguments)
        return self.get(name).invoke(arguments)
