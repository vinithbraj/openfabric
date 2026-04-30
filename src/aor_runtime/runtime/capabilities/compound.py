"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.compound

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
from aor_runtime.runtime.intent_classifier import classify_compound_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import CompoundIntent, IntentResult


class CompoundCapabilityPack(CapabilityPack):
    """Represent compound capability pack within the OpenFABRIC runtime. It extends CapabilityPack.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CompoundCapabilityPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.compound.CompoundCapabilityPack and related tests.
    """
    name = "compound"
    intent_types = (CompoundIntent,)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        """Classify for CompoundCapabilityPack instances.

        Inputs:
            Receives goal, context for this CompoundCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CompoundCapabilityPack.classify calls and related tests.
        """
        return classify_compound_intent(goal, schema_payload=context.schema_payload)

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        """Compile for CompoundCapabilityPack instances.

        Inputs:
            Receives intent, context for this CompoundCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CompoundCapabilityPack.compile calls and related tests.
        """
        if not isinstance(intent, self.intent_types):
            return None
        return CompiledIntentPlan(
            plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
            metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
        )
