from __future__ import annotations

from typing import Any

from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.intents import IntentResult, TransformChainIntent, TransformIntent


class TextTransformCapabilityPack(CapabilityPack):
    name = "text_transform"
    intent_types = (TransformIntent, TransformChainIntent)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        return IntentResult(matched=False, reason=f"{self.name}_top_level_disabled")

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        return None
