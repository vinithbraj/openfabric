"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compilers.filesystem

Purpose:
    Provide the filesystem semantic compiler seam.

Responsibilities:
    Compile filesystem semantic frames through the central semantic compiler.

Data flow / Interfaces:
    Receives canonical filesystem frames, settings, and allowed tool names; returns semantic compilation results.

Boundaries:
    Does not read or write files; it only creates validated execution-plan actions for filesystem tools.
"""

from __future__ import annotations

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic.core import SemanticCompilationResult, SemanticFrame, SemanticFrameCompiler


def compile_filesystem_frame(frame: SemanticFrame, *, settings: Settings, allowed_tools: list[str]) -> SemanticCompilationResult | None:
    """Compile a filesystem semantic frame using the shared compiler dispatcher.

    Inputs:
        Receives a canonical semantic frame, settings, and allowed tool names.

    Returns:
        A semantic compilation result when the frame is a supported filesystem request.

    Used by:
        Domain compiler tests and future semantic compiler dispatch refactors.
    """
    if frame.domain != "filesystem":
        return None
    return SemanticFrameCompiler(settings=settings, allowed_tools=allowed_tools).compile(frame)


__all__ = ["compile_filesystem_frame"]
