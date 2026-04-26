from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ArtifactKind(str, Enum):
    TEXT = "text"
    COUNT = "count"
    FILE_MATCHES = "file_matches"
    ROWS = "rows"
    CSV = "csv"
    JSON = "json"
    FILE_CONTENT = "file_content"
    SHELL_STDOUT = "shell_stdout"
    UNKNOWN = "unknown"


class ArtifactRef(BaseModel):
    name: str
    kind: ArtifactKind
    path: str | None = None


class TaskNode(BaseModel):
    id: str
    intent: Any
    output: ArtifactRef | None = None


class TaskGraph(BaseModel):
    nodes: list[TaskNode] = Field(default_factory=list)
    final_output: ArtifactRef | None = None
