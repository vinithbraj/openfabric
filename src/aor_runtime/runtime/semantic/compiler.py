"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compiler

Purpose:
    Provide the central semantic-frame compiler and planner imports.

Responsibilities:
    Expose the compiler dispatcher and planner integration used before LLM action planning fallback.

Data flow / Interfaces:
    Consumes canonical semantic frames, settings, allowed tools, and optional LLM clients; returns execution plans with semantic metadata.

Boundaries:
    Compilers may choose safe tool strategies but must preserve runtime validation, coverage checks, and tool contracts.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import SemanticFrameCompiler, SemanticFramePlanner

__all__ = ["SemanticFrameCompiler", "SemanticFramePlanner"]
