from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from aor_runtime.core.contracts import PlannerConfig, RuntimePolicy


DEFAULT_TOOLS = [
    "fs.exists",
    "fs.not_exists",
    "fs.copy",
    "fs.read",
    "fs.write",
    "fs.mkdir",
    "fs.list",
    "fs.glob",
    "fs.find",
    "fs.search_content",
    "fs.size",
    "slurm.queue",
    "slurm.job_detail",
    "slurm.nodes",
    "slurm.node_detail",
    "slurm.partitions",
    "slurm.accounting",
    "slurm.metrics",
    "slurm.slurmdbd_health",
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
    nodes: "RuntimeNodesConfig" = Field(default_factory=lambda: RuntimeNodesConfig())


class CompiledRuntimeSpec(BaseModel):
    name: str
    description: str = ""
    planner: PlannerConfig
    runtime: RuntimePolicy
    tools: list[str] = Field(default_factory=list)
    nodes: "RuntimeNodesConfig" = Field(default_factory=lambda: RuntimeNodesConfig())
    graph: list[str] = Field(default_factory=lambda: ["planner", "executor", "validator", "finalize"])


class RuntimeNodeEndpoint(BaseModel):
    name: str
    url: str


class RuntimeNodesConfig(BaseModel):
    default: str | None = None
    endpoints: list[RuntimeNodeEndpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nodes(self) -> "RuntimeNodesConfig":
        seen: set[str] = set()
        normalized_default = str(self.default or "").strip()
        self.default = normalized_default or None
        for endpoint in self.endpoints:
            endpoint.name = str(endpoint.name or "").strip()
            endpoint.url = str(endpoint.url or "").strip()
            if not endpoint.name:
                raise ValueError("Node names must be non-empty.")
            if not endpoint.url:
                raise ValueError(f"Node URL must be non-empty for node {endpoint.name!r}.")
            if endpoint.name in seen:
                raise ValueError(f"Duplicate node name {endpoint.name!r}.")
            seen.add(endpoint.name)
        if self.default and self.endpoints and self.default not in seen:
            available = ", ".join(sorted(seen))
            raise ValueError(f"Default node must match one of the configured nodes. Available nodes: {available}.")
        return self
