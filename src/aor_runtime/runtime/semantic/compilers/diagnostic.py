"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compilers.diagnostic

Purpose:
    Provide the diagnostic semantic compiler seam.

Responsibilities:
    Compile bounded diagnostic semantic frames through the central semantic compiler.

Data flow / Interfaces:
    Receives canonical diagnostic frames, settings, and allowed tool names; returns semantic compilation results.

Boundaries:
    Keeps broad diagnostics bounded and delegates execution to existing diagnostic orchestration tools.
"""

from __future__ import annotations

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic.core import SemanticCompilationResult, SemanticFrame, SemanticFrameCompiler


def compile_diagnostic_frame(frame: SemanticFrame, *, settings: Settings, allowed_tools: list[str]) -> SemanticCompilationResult | None:
    """Compile a diagnostic semantic frame using the shared compiler dispatcher.

    Inputs:
        Receives a canonical semantic frame, settings, and allowed tool names.

    Returns:
        A semantic compilation result when the frame is a supported diagnostic request.

    Used by:
        Domain compiler tests and future semantic compiler dispatch refactors.
    """
    if frame.domain != "diagnostic":
        return None
    return SemanticFrameCompiler(settings=settings, allowed_tools=allowed_tools).compile(frame)


__all__ = ["compile_diagnostic_frame"]
