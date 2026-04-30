"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.models

Purpose:
    Provide the shared semantic-frame data model import surface.

Responsibilities:
    Re-export typed frame, filter, target, metric, output, strategy, capability, and result models.

Data flow / Interfaces:
    Used by semantic policies, compilers, tests, and compatibility imports that need semantic types without importing the facade.

Boundaries:
    Contains model imports only; domain meaning rules belong in semantic policies and compilers.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import (
    SemanticCapabilitySpec,
    SemanticCompilationResult,
    SemanticCompound,
    SemanticCoverageResult,
    SemanticDomain,
    SemanticFilter,
    SemanticFrame,
    SemanticFrameExtractionResult,
    SemanticMetric,
    SemanticOutputContract,
    SemanticOutputKind,
    SemanticStrategy,
    SemanticStrategyDecision,
    SemanticTargetSet,
    SemanticTimeWindow,
    SlurmAccountingStatePolicy,
)

__all__ = [
    "SemanticCapabilitySpec",
    "SemanticCompilationResult",
    "SemanticCompound",
    "SemanticCoverageResult",
    "SemanticDomain",
    "SemanticFilter",
    "SemanticFrame",
    "SemanticFrameExtractionResult",
    "SemanticMetric",
    "SemanticOutputContract",
    "SemanticOutputKind",
    "SemanticStrategy",
    "SemanticStrategyDecision",
    "SemanticTargetSet",
    "SemanticTimeWindow",
    "SlurmAccountingStatePolicy",
]
