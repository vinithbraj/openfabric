from __future__ import annotations

from pydantic import BaseModel, Field

from aor_runtime.core.contracts import PlannerConfig, RuntimePolicy


DEFAULT_TOOLS = [
    "fs.exists",
    "fs.not_exists",
    "fs.copy",
    "fs.read",
    "fs.write",
    "fs.mkdir",
    "fs.list",
    "shell.exec",
    "python.exec",
    "sql.query",
]


class RuntimeSpec(BaseModel):
    version: int = 1
    name: str = "agent_runtime"
    description: str = ""
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    runtime: RuntimePolicy = Field(default_factory=RuntimePolicy)
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))


class CompiledRuntimeSpec(BaseModel):
    name: str
    description: str = ""
    planner: PlannerConfig
    runtime: RuntimePolicy
    tools: list[str] = Field(default_factory=list)
    graph: list[str] = Field(default_factory=lambda: ["planner", "executor", "validator", "finalize"])
