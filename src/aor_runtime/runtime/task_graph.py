"""OpenFABRIC Runtime Module: aor_runtime.runtime.task_graph

Purpose:
    Represent task graph structures used by compiled plans and diagnostics.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ArtifactKind(str, Enum):
    """Represent artifact kind within the OpenFABRIC runtime. It extends str, Enum.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ArtifactKind.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.task_graph.ArtifactKind and related tests.
    """
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
    """Represent artifact ref within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ArtifactRef.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.task_graph.ArtifactRef and related tests.
    """
    name: str
    kind: ArtifactKind
    path: str | None = None


class TaskNode(BaseModel):
    """Represent task node within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TaskNode.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.task_graph.TaskNode and related tests.
    """
    id: str
    intent: Any
    output: ArtifactRef | None = None


class TaskGraph(BaseModel):
    """Represent task graph within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TaskGraph.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.task_graph.TaskGraph and related tests.
    """
    nodes: list[TaskNode] = Field(default_factory=list)
    final_output: ArtifactRef | None = None
