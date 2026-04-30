"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.base

Purpose:
    Define the base tool interfaces and typed argument/result contracts.

Responsibilities:
    Classify or compile typed intents when called directly by tests or compatibility surfaces.

Data flow / Interfaces:
    Consumes compile contexts, allowed tools, and typed intents; returns execution-plan fragments or eval metadata.

Boundaries:
    These modules are not the active top-level natural-language planner; user prompts route through LLMActionPlanner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.intents import IntentResult


@dataclass(frozen=True)
class ClassificationContext:
    """Represent classification context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ClassificationContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.base.ClassificationContext and related tests.
    """
    schema_payload: dict[str, Any] | None
    allowed_tools: list[str]
    settings: Settings
    input_payload: dict[str, Any] | None = None
    failure_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompileContext:
    """Represent compile context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CompileContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.base.CompileContext and related tests.
    """
    allowed_tools: list[str]
    settings: Settings


@dataclass(frozen=True)
class CompiledIntentPlan:
    """Represent compiled intent plan within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CompiledIntentPlan.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.base.CompiledIntentPlan and related tests.
    """
    plan: ExecutionPlan
    planning_mode: str = "deterministic_intent"
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilityPack(ABC):
    """Represent capability pack within the OpenFABRIC runtime. It extends ABC.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CapabilityPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.base.CapabilityPack and related tests.
    """
    name: str = ""
    intent_types: tuple[type, ...] = ()
    examples: tuple[str, ...] = ()
    eval_prompts: tuple[str, ...] = ()
    supports_llm_intent_extraction: bool = False

    @abstractmethod
    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        """Classify for CapabilityPack instances.

        Inputs:
            Receives goal, context for this CapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityPack.classify calls and related tests.
        """
        raise NotImplementedError

    @abstractmethod
    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        """Compile for CapabilityPack instances.

        Inputs:
            Receives intent, context for this CapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityPack.compile calls and related tests.
        """
        raise NotImplementedError

    def is_llm_intent_domain(self, goal: str, context: ClassificationContext) -> bool:
        """Is llm intent domain for CapabilityPack instances.

        Inputs:
            Receives goal, context for this CapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityPack.is_llm_intent_domain calls and related tests.
        """
        return False

    def try_llm_extract(self, goal: str, context: ClassificationContext, extractor: Any) -> IntentResult:
        """Try llm extract for CapabilityPack instances.

        Inputs:
            Receives goal, context, extractor for this CapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityPack.try_llm_extract calls and related tests.
        """
        return IntentResult(matched=False, reason=f"{self.name}_llm_intent_not_supported")
