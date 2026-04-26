from __future__ import annotations

from typing import Any

from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.intent_classifier import classify_single_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import FetchExtractIntent, IntentResult


class FetchCapabilityPack(CapabilityPack):
    name = "fetch"
    intent_types = (FetchExtractIntent,)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        result = classify_single_intent(goal, schema_payload=context.schema_payload)
        if result.matched and isinstance(result.intent, self.intent_types):
            return result
        return IntentResult(matched=False, reason=f"{self.name}_no_match")

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        if not isinstance(intent, self.intent_types):
            return None
        return CompiledIntentPlan(
            plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
            metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
        )
