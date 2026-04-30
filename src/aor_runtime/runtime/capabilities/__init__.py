"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.__init__

Purpose:
    Provide compatibility capability-pack helpers and fixtures for domain-specific tests and utilities.

Responsibilities:
    Classify or compile typed intents when called directly by tests or compatibility surfaces.

Data flow / Interfaces:
    Consumes compile contexts, allowed tools, and typed intents; returns execution-plan fragments or eval metadata.

Boundaries:
    These modules are not the active top-level natural-language planner; user prompts route through LLMActionPlanner.
"""

from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.capabilities.eval import (
    CapabilityEvalCase,
    CapabilityEvalPack,
    CapabilityEvalResult,
    ensure_unique_case_ids,
    load_capability_eval_pack,
    load_capability_eval_packs,
)
from aor_runtime.runtime.capabilities.registry import CapabilityRegistry, build_default_capability_registry

__all__ = [
    "CapabilityPack",
    "CapabilityRegistry",
    "CapabilityEvalCase",
    "CapabilityEvalPack",
    "CapabilityEvalResult",
    "ClassificationContext",
    "CompileContext",
    "CompiledIntentPlan",
    "ensure_unique_case_ids",
    "build_default_capability_registry",
    "load_capability_eval_pack",
    "load_capability_eval_packs",
]
