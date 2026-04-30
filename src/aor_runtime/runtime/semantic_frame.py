"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic_frame

Purpose:
    Preserve the public semantic-frame import surface while implementation lives in the semantic package.

Responsibilities:
    Re-export the semantic-frame models, planners, compilers, extractors, validators, and projection helpers.

Data flow / Interfaces:
    Existing planner, executor, tests, and scripts import this module; it forwards those imports to runtime.semantic.

Boundaries:
    This file is a compatibility facade and must not accumulate domain-specific semantic policy or compiler logic.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import (
    LLMSemanticFrameExtractor,
    SEMANTIC_CAPABILITIES,
    SemanticCapabilitySpec,
    SemanticCompilationResult,
    SemanticCompound,
    SemanticCoverageResult,
    SemanticCoverageValidator,
    SemanticDomain,
    SemanticFilter,
    SemanticFrame,
    SemanticFrameCanonicalizer,
    SemanticFrameCompiler,
    SemanticFrameExtractionResult,
    SemanticFramePlanner,
    SemanticMetric,
    SemanticOutputContract,
    SemanticOutputKind,
    SemanticStrategy,
    SemanticStrategyDecision,
    SemanticStrategySelector,
    SemanticTargetSet,
    SemanticTimeWindow,
    SlurmAccountingStatePolicy,
    build_semantic_frame_prompt,
    deterministic_semantic_frame,
    normalize_metric_name,
    project_semantic_result,
    semantic_frame_mode,
    validate_semantic_coverage,
)

__all__ = [
    "LLMSemanticFrameExtractor",
    "SEMANTIC_CAPABILITIES",
    "SemanticCapabilitySpec",
    "SemanticCompilationResult",
    "SemanticCompound",
    "SemanticCoverageResult",
    "SemanticCoverageValidator",
    "SemanticDomain",
    "SemanticFilter",
    "SemanticFrame",
    "SemanticFrameCanonicalizer",
    "SemanticFrameCompiler",
    "SemanticFrameExtractionResult",
    "SemanticFramePlanner",
    "SemanticMetric",
    "SemanticOutputContract",
    "SemanticOutputKind",
    "SemanticStrategy",
    "SemanticStrategyDecision",
    "SemanticStrategySelector",
    "SemanticTargetSet",
    "SemanticTimeWindow",
    "SlurmAccountingStatePolicy",
    "build_semantic_frame_prompt",
    "deterministic_semantic_frame",
    "normalize_metric_name",
    "project_semantic_result",
    "semantic_frame_mode",
    "validate_semantic_coverage",
]
