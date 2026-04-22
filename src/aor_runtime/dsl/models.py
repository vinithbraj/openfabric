from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AgentDefinition(BaseModel):
    model: str | None = None
    tools: list[str] = Field(default_factory=list)
    prompt: str | None = None
    input_schema: str | None = None
    output_schema: str | None = None
    max_iterations: int = 3
    temperature: float = 0.1
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConditionalTransition(BaseModel):
    if_: str = Field(alias="if")
    then: str
    else_: str = Field(alias="else")


class GraphNodeDefinition(BaseModel):
    type: Literal["router", "agent", "end"] = "agent"
    agent: str | None = None
    next: str | list[str] | None = None
    condition: ConditionalTransition | None = None
    prompt: str | None = None
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> "GraphNodeDefinition":
        if self.type == "agent" and not self.agent:
            raise ValueError("Agent nodes require an 'agent' reference.")
        if self.type == "router":
            if not isinstance(self.next, list) or not self.next:
                raise ValueError("Router nodes require a non-empty list in 'next'.")
        if self.type == "end" and self.next:
            raise ValueError("End nodes cannot define next transitions.")
        return self


class GraphDefinition(BaseModel):
    start: str
    nodes: dict[str, GraphNodeDefinition]


class RuntimeSpec(BaseModel):
    version: int = 1
    name: str = "agent_runtime"
    description: str = ""
    agents: dict[str, AgentDefinition] = Field(default_factory=dict)
    graph: GraphDefinition


class CompiledNode(BaseModel):
    name: str
    kind: Literal["router", "agent", "end"]
    agent_name: str | None = None
    next_nodes: list[str] = Field(default_factory=list)
    condition: ConditionalTransition | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompiledGraphSpec(BaseModel):
    name: str
    description: str = ""
    start_node: str
    nodes: dict[str, CompiledNode]
    agents: dict[str, AgentDefinition]
