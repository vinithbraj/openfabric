"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compilers.sql

Purpose:
    Provide the SQL semantic compiler seam.

Responsibilities:
    Compile SQL semantic frames through the central semantic compiler.

Data flow / Interfaces:
    Receives canonical SQL frames, settings, and allowed tool names; returns semantic compilation results.

Boundaries:
    Produces only read-only, schema-grounded SQL actions and relies on SQL validators before execution.
"""

from __future__ import annotations

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic.core import SemanticCompilationResult, SemanticFrame, SemanticFrameCompiler


def compile_sql_frame(frame: SemanticFrame, *, settings: Settings, allowed_tools: list[str]) -> SemanticCompilationResult | None:
    """Compile a SQL semantic frame using the shared compiler dispatcher.

    Inputs:
        Receives a canonical semantic frame, settings, and allowed tool names.

    Returns:
        A semantic compilation result when the frame is a supported SQL request.

    Used by:
        Domain compiler tests and future semantic compiler dispatch refactors.
    """
    if frame.domain != "sql":
        return None
    return SemanticFrameCompiler(settings=settings, allowed_tools=allowed_tools).compile(frame)


__all__ = ["compile_sql_frame"]
