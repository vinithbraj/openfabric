"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.capabilities

Purpose:
    Provide semantic capability metadata and strategy-selection imports.

Responsibilities:
    Expose the registry of domain semantic capabilities and the deterministic strategy selector.

Data flow / Interfaces:
    Used by semantic compilers and extraction prompts to describe supported domains, filters, metrics, dimensions, and execution strategies.

Boundaries:
    Stores capability facts only; tool execution and domain-specific meaning policies remain elsewhere.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import SEMANTIC_CAPABILITIES, SemanticStrategySelector

__all__ = ["SEMANTIC_CAPABILITIES", "SemanticStrategySelector"]
