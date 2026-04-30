"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compilers.slurm

Purpose:
    Provide the SLURM semantic compiler seam.

Responsibilities:
    Compile SLURM semantic frames through the central semantic compiler while preserving SLURM policy ownership.

Data flow / Interfaces:
    Receives canonical SLURM frames, runtime settings, and allowed tool names; returns semantic compilation results.

Boundaries:
    Does not construct shell commands directly; it produces validated execution-plan actions for the SLURM tool layer.
"""

from __future__ import annotations

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic.core import SemanticCompilationResult, SemanticFrame, SemanticFrameCompiler


def compile_slurm_frame(frame: SemanticFrame, *, settings: Settings, allowed_tools: list[str]) -> SemanticCompilationResult | None:
    """Compile a SLURM semantic frame using the shared compiler dispatcher.

    Inputs:
        Receives a canonical semantic frame, settings, and allowed tool names.

    Returns:
        A semantic compilation result when the frame is a supported SLURM request.

    Used by:
        Domain compiler tests and future semantic compiler dispatch refactors.
    """
    if frame.domain != "slurm":
        return None
    return SemanticFrameCompiler(settings=settings, allowed_tools=allowed_tools).compile(frame)


__all__ = ["compile_slurm_frame"]
