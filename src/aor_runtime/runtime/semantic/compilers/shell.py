"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compilers.shell

Purpose:
    Provide the shell/system semantic compiler seam.

Responsibilities:
    Compile read-only shell/system semantic frames through the central semantic compiler.

Data flow / Interfaces:
    Receives canonical shell frames, settings, and allowed tool names; returns semantic compilation results.

Boundaries:
    Does not accept arbitrary shell commands from semantic extraction; only declared safe inspection strategies compile.
"""

from __future__ import annotations

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic.core import SemanticCompilationResult, SemanticFrame, SemanticFrameCompiler


def compile_shell_frame(frame: SemanticFrame, *, settings: Settings, allowed_tools: list[str]) -> SemanticCompilationResult | None:
    """Compile a shell/system semantic frame using the shared compiler dispatcher.

    Inputs:
        Receives a canonical semantic frame, settings, and allowed tool names.

    Returns:
        A semantic compilation result when the frame is a supported shell/system request.

    Used by:
        Domain compiler tests and future semantic compiler dispatch refactors.
    """
    if frame.domain != "shell":
        return None
    return SemanticFrameCompiler(settings=settings, allowed_tools=allowed_tools).compile(frame)


__all__ = ["compile_shell_frame"]
