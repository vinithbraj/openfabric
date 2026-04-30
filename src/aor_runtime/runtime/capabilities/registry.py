"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.registry

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
from aor_runtime.runtime.capabilities.compound import CompoundCapabilityPack
from aor_runtime.runtime.capabilities.fetch import FetchCapabilityPack
from aor_runtime.runtime.capabilities.filesystem import FilesystemCapabilityPack
from aor_runtime.runtime.capabilities.shell import ShellCapabilityPack
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.capabilities.sql import SqlCapabilityPack
from aor_runtime.runtime.capabilities.text_transform import TextTransformCapabilityPack
from aor_runtime.runtime.intents import IntentResult


class CapabilityRegistry:
    """Represent capability registry within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CapabilityRegistry.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.registry.CapabilityRegistry and related tests.
    """
    def __init__(self) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityRegistry.__init__ calls and related tests.
        """
        self._packs: list[CapabilityPack] = []

    @property
    def packs(self) -> tuple[CapabilityPack, ...]:
        """Packs for CapabilityRegistry instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityRegistry.packs calls and related tests.
        """
        return tuple(self._packs)

    def register(self, pack: CapabilityPack) -> None:
        """Register for CapabilityRegistry instances.

        Inputs:
            Receives pack for this CapabilityRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityRegistry.register calls and related tests.
        """
        self._packs.append(pack)

    def classify(self, goal: str, context: ClassificationContext, *, extractor: Any | None = None) -> IntentResult:
        """Classify for CapabilityRegistry instances.

        Inputs:
            Receives goal, context, extractor for this CapabilityRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityRegistry.classify calls and related tests.
        """
        for pack in self._packs:
            result = pack.classify(goal, context)
            if result.matched:
                return result
        if extractor is not None:
            for pack in self._packs:
                if not getattr(pack, "supports_llm_intent_extraction", False):
                    continue
                if not pack.is_llm_intent_domain(goal, context):
                    continue
                result = pack.try_llm_extract(goal, context, extractor)
                if result.matched:
                    return result
        return IntentResult(matched=False, reason="no_deterministic_intent")

    def compile_result(self, intent: Any, context: CompileContext) -> CompiledIntentPlan:
        """Compile result for CapabilityRegistry instances.

        Inputs:
            Receives intent, context for this CapabilityRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityRegistry.compile_result calls and related tests.
        """
        for pack in self._packs:
            compiled = pack.compile(intent, context)
            if compiled is not None:
                return compiled
        raise ValueError(f"No capability pack can compile intent type: {type(intent).__name__}")

    def compile(self, intent: Any, context: CompileContext):
        """Compile for CapabilityRegistry instances.

        Inputs:
            Receives intent, context for this CapabilityRegistry method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityRegistry.compile calls and related tests.
        """
        return self.compile_result(intent, context).plan


def build_default_capability_registry() -> CapabilityRegistry:
    """Build default capability registry for the surrounding runtime workflow.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.registry.build_default_capability_registry.
    """
    registry = CapabilityRegistry()
    registry.register(CompoundCapabilityPack())
    registry.register(FilesystemCapabilityPack())
    registry.register(TextTransformCapabilityPack())
    registry.register(SqlCapabilityPack())
    registry.register(SlurmCapabilityPack())
    registry.register(ShellCapabilityPack())
    registry.register(FetchCapabilityPack())
    return registry
