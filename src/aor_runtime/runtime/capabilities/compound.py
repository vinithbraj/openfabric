from __future__ import annotations

from typing import Any

from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.intent_classifier import classify_compound_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import CompoundIntent, IntentResult


class CompoundCapabilityPack(CapabilityPack):
    name = "compound"
    intent_types = (CompoundIntent,)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        return classify_compound_intent(goal, schema_payload=context.schema_payload)

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        if not isinstance(intent, self.intent_types):
            return None
        return CompiledIntentPlan(
            plan=compile_intent_to_plan(intent, context.allowed_tools, context.settings),
            metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
        )
