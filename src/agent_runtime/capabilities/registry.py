"""Capability registry for execution-time lookup and LLM-safe export."""

from __future__ import annotations

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import CapabilityNotFoundError


class CapabilityRegistry:
    """In-memory registry of manifest-driven capabilities."""

    def __init__(self) -> None:
        self._capabilities: dict[str, BaseCapability] = {}

    def register(self, capability: BaseCapability) -> None:
        """Register or replace a capability by capability id."""

        self._capabilities[capability.manifest.capability_id] = capability

    def get(self, capability_id: str) -> BaseCapability:
        """Return a capability or raise a typed error."""

        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise CapabilityNotFoundError(f"capability not registered: {capability_id}") from exc

    def list_manifests(self) -> list[CapabilityManifest]:
        """Return all registered manifests."""

        return [capability.manifest for capability in self._capabilities.values()]

    def find_by_domain(self, domain: str) -> list[CapabilityManifest]:
        """Return manifests in the requested domain."""

        normalized = str(domain or "").strip().lower()
        return [manifest for manifest in self.list_manifests() if manifest.domain.lower() == normalized]

    def find_by_semantic_verb(self, verb: str) -> list[CapabilityManifest]:
        """Return manifests supporting the requested semantic verb."""

        normalized = str(verb or "").strip().lower()
        return [
            manifest
            for manifest in self.list_manifests()
            if normalized in {item.strip().lower() for item in manifest.semantic_verbs}
        ]

    def export_llm_manifest(self) -> list[dict[str, object]]:
        """Return a compact LLM-safe manifest without implementation details."""

        exported: list[dict[str, object]] = []
        for manifest in self.list_manifests():
            exported.append(
                {
                    "capability_id": manifest.capability_id,
                    "operation_id": manifest.operation_id,
                    "domain": manifest.domain,
                    "description": manifest.description,
                    "semantic_verbs": list(manifest.semantic_verbs),
                    "object_types": list(manifest.object_types),
                    "required_arguments": list(manifest.required_arguments),
                    "optional_arguments": list(manifest.optional_arguments),
                    "risk_level": manifest.risk_level,
                    "read_only": manifest.read_only,
                    "examples": list(manifest.examples),
                }
            )
        return exported
