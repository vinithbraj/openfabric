from __future__ import annotations

from pydantic import BaseModel


class ExecRequest(BaseModel):
    node: str
    command: str


class ExecResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


class HealthResponse(BaseModel):
    status: str
    node: str


class CapabilityInfo(BaseModel):
    name: str
    description: str


class CapabilitiesResponse(BaseModel):
    node: str
    version: str
    capabilities: list[CapabilityInfo]
