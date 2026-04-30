"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.canonicalizer

Purpose:
    Provide semantic-frame canonicalization imports.

Responsibilities:
    Expose canonicalization of targets, filters, time windows, output contracts, and recursion limits.

Data flow / Interfaces:
    Consumes raw semantic frames and original user goals; returns canonical frames ready for compilation and coverage checks.

Boundaries:
    Canonicalization may normalize meaning but must not choose unsafe executable payloads or run tools.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import SemanticFrameCanonicalizer, semantic_frame_mode

__all__ = ["SemanticFrameCanonicalizer", "semantic_frame_mode"]
