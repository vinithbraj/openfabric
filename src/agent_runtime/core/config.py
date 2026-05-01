"""Configuration schema for the foundational runtime."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    """Process-level settings shared by all three pipelines."""

    model_config = ConfigDict(extra="forbid")

    max_decomposition_depth: int = Field(default=10, gt=0)
    max_dag_nodes: int = Field(default=64, gt=0)
    allow_mutating_capabilities: bool = False
    default_output_format: str = "markdown"
    workspace_root: str = "."
    allow_shell_execution: bool = False
    allow_network_operations: bool = False
    gateway_default_node: str = "localhost"
    gateway_url: str | None = None
    gateway_endpoints: dict[str, str] = Field(default_factory=dict)
    gateway_timeout_seconds: float = Field(default=30.0, gt=0.0)
    confirmation_granted: bool = False
    max_files_listed: int = Field(default=1000, gt=0)
    max_rows_returned: int = Field(default=100, gt=0)
    max_output_preview_bytes: int = Field(default=4096, gt=0)
    low_risk_mutation_allowlist: list[str] = Field(default_factory=list)
    stop_on_error: bool = True
