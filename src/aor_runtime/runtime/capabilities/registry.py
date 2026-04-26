from __future__ import annotations

from typing import Any

from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.compound import CompoundCapabilityPack
from aor_runtime.runtime.capabilities.fetch import FetchCapabilityPack
from aor_runtime.runtime.capabilities.filesystem import FilesystemCapabilityPack
from aor_runtime.runtime.capabilities.shell import ShellCapabilityPack
from aor_runtime.runtime.capabilities.sql import SqlCapabilityPack
from aor_runtime.runtime.capabilities.text_transform import TextTransformCapabilityPack
from aor_runtime.runtime.intents import IntentResult


class CapabilityRegistry:
    def __init__(self) -> None:
        self._packs: list[CapabilityPack] = []

    @property
    def packs(self) -> tuple[CapabilityPack, ...]:
        return tuple(self._packs)

    def register(self, pack: CapabilityPack) -> None:
        self._packs.append(pack)

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        for pack in self._packs:
            result = pack.classify(goal, context)
            if result.matched:
                return result
        return IntentResult(matched=False, reason="no_deterministic_intent")

    def compile(self, intent: Any, context: CompileContext):
        for pack in self._packs:
            compiled = pack.compile(intent, context)
            if compiled is not None:
                return compiled.plan
        raise ValueError(f"No capability pack can compile intent type: {type(intent).__name__}")


def build_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(CompoundCapabilityPack())
    registry.register(FilesystemCapabilityPack())
    registry.register(TextTransformCapabilityPack())
    registry.register(SqlCapabilityPack())
    registry.register(ShellCapabilityPack())
    registry.register(FetchCapabilityPack())
    return registry
