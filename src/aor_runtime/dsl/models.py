"""OpenFABRIC Runtime Module: aor_runtime.dsl.models

Purpose:
    Define Pydantic models for runtime DSL specifications.

Responsibilities:
    Parse assistant specs into typed configuration objects before compilation and execution.

Data flow / Interfaces:
    Receives YAML/runtime spec data and returns validated DSL models for the compiler.

Boundaries:
    Keeps configuration parsing separate from request-time LLM planning and tool execution.
"""

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
    "slurm.accounting_aggregate",
    "slurm.metrics",
    "slurm.slurmdbd_health",
    "shell.exec",
    "python.exec",
    "sql.query",
    "sql.schema",
]


class RuntimeSpec(BaseModel):
    """Represent runtime spec within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeSpec.

    Data flow / Interfaces:
        Instances are created and consumed by runtime spec loading code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.dsl.models.RuntimeSpec and related tests.
    """
    version: int = 1
    name: str = "agent_runtime"
    description: str = ""
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    runtime: RuntimePolicy = Field(default_factory=RuntimePolicy)
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    nodes: "RuntimeNodesConfig" = Field(default_factory=lambda: RuntimeNodesConfig())


class CompiledRuntimeSpec(BaseModel):
    """Represent compiled runtime spec within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CompiledRuntimeSpec.

    Data flow / Interfaces:
        Instances are created and consumed by runtime spec loading code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.dsl.models.CompiledRuntimeSpec and related tests.
    """
    name: str
    description: str = ""
    planner: PlannerConfig
    runtime: RuntimePolicy
    tools: list[str] = Field(default_factory=list)
    nodes: "RuntimeNodesConfig" = Field(default_factory=lambda: RuntimeNodesConfig())
    graph: list[str] = Field(default_factory=lambda: ["planner", "executor", "validator", "finalize"])


class RuntimeNodeEndpoint(BaseModel):
    """Represent runtime node endpoint within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeNodeEndpoint.

    Data flow / Interfaces:
        Instances are created and consumed by runtime spec loading code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.dsl.models.RuntimeNodeEndpoint and related tests.
    """
    name: str
    url: str


class RuntimeNodesConfig(BaseModel):
    """Represent runtime nodes config within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeNodesConfig.

    Data flow / Interfaces:
        Instances are created and consumed by runtime spec loading code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.dsl.models.RuntimeNodesConfig and related tests.
    """
    default: str | None = None
    endpoints: list[RuntimeNodeEndpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nodes(self) -> "RuntimeNodesConfig":
        """Validate validate nodes invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by runtime spec loading through RuntimeNodesConfig.validate_nodes calls and related tests.
        """
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
