"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.text_transform

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
from aor_runtime.runtime.intents import IntentResult, TransformChainIntent, TransformIntent


class TextTransformCapabilityPack(CapabilityPack):
    """Represent text transform capability pack within the OpenFABRIC runtime. It extends CapabilityPack.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TextTransformCapabilityPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.text_transform.TextTransformCapabilityPack and related tests.
    """
    name = "text_transform"
    intent_types = (TransformIntent, TransformChainIntent)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        """Classify for TextTransformCapabilityPack instances.

        Inputs:
            Receives goal, context for this TextTransformCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TextTransformCapabilityPack.classify calls and related tests.
        """
        return IntentResult(matched=False, reason=f"{self.name}_top_level_disabled")

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        """Compile for TextTransformCapabilityPack instances.

        Inputs:
            Receives intent, context for this TextTransformCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TextTransformCapabilityPack.compile calls and related tests.
        """
        return None
