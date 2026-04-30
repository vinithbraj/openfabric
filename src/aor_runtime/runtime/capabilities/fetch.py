"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.fetch

Purpose:
    Provide compatibility capability-pack helpers and fixtures for domain-specific tests and utilities.

Responsibilities:
    Classify or compile typed intents when called directly by tests or compatibility surfaces.

Data flow / Interfaces:
    Consumes compile contexts, allowed tools, and typed intents; returns execution-plan fragments or eval metadata.

Boundaries:
    These modules are not the active top-level natural-language planner; user prompts route through LLMActionPlanner.
"""

from __future__ import annotations

from typing import Any

from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.intent_classifier import classify_single_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import FetchExtractIntent, IntentResult


class FetchCapabilityPack(CapabilityPack):
    """Represent fetch capability pack within the OpenFABRIC runtime. It extends CapabilityPack.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FetchCapabilityPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.fetch.FetchCapabilityPack and related tests.
    """
    name = "fetch"
    intent_types = (FetchExtractIntent,)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        """Classify for FetchCapabilityPack instances.

        Inputs:
            Receives goal, context for this FetchCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through FetchCapabilityPack.classify calls and related tests.
        """
        result = classify_single_intent(goal, schema_payload=context.schema_payload)
        if result.matched and isinstance(result.intent, self.intent_types):
            return result
        return IntentResult(matched=False, reason=f"{self.name}_no_match")

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        """Compile for FetchCapabilityPack instances.

        Inputs:
            Receives intent, context for this FetchCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through FetchCapabilityPack.compile calls and related tests.
        """
        if not isinstance(intent, self.intent_types):
            return None
        return CompiledIntentPlan(
            plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
            metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
        )
