from __future__ import annotations

from agent_runtime.capabilities import (
    CapabilityRegistry,
    ListDirectoryCapability,
    MarkdownRenderCapability,
    ReadFileCapability,
    ReadQueryCapability,
    SearchFilesCapability,
    TransformTableCapability,
    WriteFileCapability,
)


def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(ListDirectoryCapability())
    registry.register(ReadFileCapability())
    registry.register(SearchFilesCapability())
    registry.register(WriteFileCapability())
    registry.register(ReadQueryCapability())
    registry.register(TransformTableCapability())
    registry.register(MarkdownRenderCapability())
    return registry


def test_register_and_lookup_capability() -> None:
    registry = _registry()

    capability = registry.get("filesystem.list_directory")

    assert capability.manifest.name == "List Directory"
    assert capability.manifest.operation_id == "list_directory"


def test_list_manifests_returns_registered_manifests() -> None:
    manifests = _registry().list_manifests()

    assert len(manifests) == 7
    assert {manifest.capability_id for manifest in manifests} >= {
        "filesystem.list_directory",
        "filesystem.write_file",
        "sql.read_query",
        "markdown.render",
    }


def test_find_by_domain_filters_manifests() -> None:
    manifests = _registry().find_by_domain("filesystem")

    assert {manifest.capability_id for manifest in manifests} == {
        "filesystem.list_directory",
        "filesystem.read_file",
        "filesystem.search_files",
        "filesystem.write_file",
    }


def test_find_by_semantic_verb_filters_manifests() -> None:
    manifests = _registry().find_by_semantic_verb("render")

    assert {manifest.capability_id for manifest in manifests} == {
        "filesystem.write_file",
        "markdown.render",
    }


def test_export_llm_manifest_is_compact_and_safe() -> None:
    exported = _registry().export_llm_manifest()

    first = next(item for item in exported if item["capability_id"] == "filesystem.list_directory")
    assert set(first) == {
        "capability_id",
        "operation_id",
        "domain",
        "description",
        "semantic_verbs",
        "object_types",
        "required_arguments",
        "optional_arguments",
        "risk_level",
        "read_only",
        "examples",
    }
    assert "argument_schema" not in first
    assert "output_schema" not in first
    assert "safety_notes" not in first
    assert "execution_backend" not in first
    assert "backend_operation" not in first
    assert first["read_only"] is True
