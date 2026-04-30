"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.coverage

Purpose:
    Provide semantic coverage validation imports.

Responsibilities:
    Expose validation that compiled tool plans satisfy semantic frame requirements.

Data flow / Interfaces:
    Consumes semantic frames and execution plans; returns coverage results before execution.

Boundaries:
    Coverage validation rejects safe-but-wrong plans and never repairs by weakening requested meaning.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import SemanticCoverageValidator, validate_semantic_coverage

__all__ = ["SemanticCoverageValidator", "validate_semantic_coverage"]
