from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.intents import IntentResult


@dataclass(frozen=True)
class ClassificationContext:
    schema_payload: dict[str, Any] | None
    allowed_tools: list[str]
    settings: Settings
    input_payload: dict[str, Any] | None = None
    failure_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompileContext:
    allowed_tools: list[str]
    settings: Settings


@dataclass(frozen=True)
class CompiledIntentPlan:
    plan: ExecutionPlan
    planning_mode: str = "deterministic_intent"
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilityPack(ABC):
    name: str = ""
    intent_types: tuple[type, ...] = ()
    examples: tuple[str, ...] = ()
    eval_prompts: tuple[str, ...] = ()
    supports_llm_intent_extraction: bool = False

    @abstractmethod
    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        raise NotImplementedError

    @abstractmethod
    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        raise NotImplementedError

    def is_llm_intent_domain(self, goal: str, context: ClassificationContext) -> bool:
        return False

    def try_llm_extract(self, goal: str, context: ClassificationContext, extractor: Any) -> IntentResult:
        return IntentResult(matched=False, reason=f"{self.name}_llm_intent_not_supported")
