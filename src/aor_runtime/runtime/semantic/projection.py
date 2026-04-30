"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.projection

Purpose:
    Provide semantic result projection imports.

Responsibilities:
    Expose post-execution projection that keeps requested target groups and recomputes summaries.

Data flow / Interfaces:
    Consumes executed action names, action args, and structured tool results; returns projected structured results.

Boundaries:
    Projection may narrow safe structured results but must not introduce raw rows, executable payloads, or unvalidated values.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import project_semantic_result

__all__ = ["project_semantic_result"]
