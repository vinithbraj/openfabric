"""Safety checks for the execution pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.config import RuntimeConfig
from agent_runtime.core.errors import CapabilityNotFoundError, SafetyError
from agent_runtime.core.types import ActionDAG, ActionNode


class SafetyDecision(BaseModel):
    """Typed result of deterministic DAG safety evaluation."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    requires_confirmation: bool
    blocked_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sanitized_dag: ActionDAG | None = None


class SafetyPolicy:
    """Backward-compatible per-node safety helper for the execution engine."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()

    def assert_allowed(self, capability: BaseCapability, action: str) -> None:
        """Reject mismatched or mutating actions in the legacy single-node path."""

        if action != capability.manifest.operation_id:
            raise SafetyError(
                f"action not declared on capability: {capability.manifest.capability_id}.{action}"
            )
        if (
            not capability.manifest.read_only
            and not self.config.allow_mutating_capabilities
            and not self.config.confirmation_granted
        ):
            raise SafetyError(
                f"mutating action is disabled: {capability.manifest.capability_id}.{action}"
            )


def _coerce_policy_config(policy_config: RuntimeConfig | dict[str, Any] | None) -> RuntimeConfig:
    """Normalize policy input into a RuntimeConfig instance."""

    if isinstance(policy_config, RuntimeConfig):
        return policy_config
    if policy_config is None:
        return RuntimeConfig()
    return RuntimeConfig.model_validate(policy_config)


def _supports_argument(capability: BaseCapability, argument_name: str) -> bool:
    """Return whether a capability manifest declares one argument name."""

    manifest = capability.manifest
    allowed = set(manifest.required_arguments) | set(manifest.optional_arguments)
    return argument_name in allowed


def _path_argument_names(arguments: dict[str, Any]) -> list[str]:
    """Return argument names that should be treated as filesystem paths."""

    names: list[str] = []
    for key in arguments:
        normalized = key.strip().lower()
        if normalized == "path" or normalized.endswith("_path") or normalized in {
            "file",
            "filepath",
            "directory",
            "dir",
        }:
            names.append(key)
    return names


def _normalize_path(path_value: str, workspace_root: Path) -> tuple[str | None, str | None]:
    """Normalize a path and reject traversal outside the workspace root."""

    raw = str(path_value or "").strip()
    if not raw:
        return None, "Path value must be a non-empty string."
    if raw == ".":
        return ".", None

    candidate = Path(raw)
    resolved = (
        (workspace_root / candidate).resolve(strict=False)
        if not candidate.is_absolute()
        else candidate.resolve(strict=False)
    )
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError:
        return None, f"Path {path_value!r} resolves outside the workspace root."
    return relative.as_posix() or ".", None


def _is_secret_path(path_value: str) -> bool:
    """Return whether a normalized path targets an obvious secret file."""

    name = Path(path_value).name.lower()
    if name in {".env", "id_rsa", "id_ed25519", "credentials.json"}:
        return True
    return name.startswith("secrets.")


def _is_delete_operation(node: ActionNode, capability: BaseCapability) -> bool:
    """Return whether a node semantically represents deletion."""

    manifest_verbs = {verb.strip().lower() for verb in capability.manifest.semantic_verbs}
    return (
        node.semantic_verb == "delete"
        or node.operation_id.startswith("delete")
        or "delete" in manifest_verbs
    )


def _is_update_or_create(node: ActionNode, capability: BaseCapability) -> bool:
    """Return whether a node semantically represents create/update mutation."""

    manifest_verbs = {verb.strip().lower() for verb in capability.manifest.semantic_verbs}
    return node.semantic_verb in {"update", "create"} or bool(
        {"update", "create"} & manifest_verbs
    )


def _always_requires_confirmation(capability: BaseCapability) -> bool:
    """Return whether one capability must always pause for explicit confirmation."""

    return capability.manifest.capability_id == "filesystem.write_file"


def _is_network_operation(capability: BaseCapability, node: ActionNode) -> bool:
    """Return whether a capability should be treated as a network operation."""

    manifest = capability.manifest
    object_types = {item.strip().lower() for item in manifest.object_types}
    labels = {label.strip().lower() for label in node.safety_labels}
    return (
        manifest.domain.strip().lower() == "network"
        or manifest.capability_id.startswith("network.")
        or "network" in object_types
        or "network" in labels
    )


def _is_environment_facing(manifest) -> bool:
    """Return whether a capability touches the remote environment rather than in-memory data."""

    return manifest.domain.strip().lower() in {"filesystem", "shell"}


def _detect_sql_write(query: str) -> bool:
    """Return whether a SQL string appears to perform mutation."""

    normalized = re.sub(r"^\s+", "", str(query or "")).lower()
    write_keywords = (
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "truncate",
        "merge",
        "grant",
        "revoke",
        "call",
        "copy",
    )
    return normalized.startswith(write_keywords)


def _detect_query_intent_write(query_intent: dict[str, Any]) -> bool:
    """Return whether a structured query intent appears to request mutation."""

    template = str(query_intent.get("template") or "").strip().lower()
    mode = str(query_intent.get("mode") or "").strip().lower()
    return mode == "write" or template.startswith(
        ("insert", "update", "delete", "drop", "create", "alter", "truncate", "merge")
    )


def _clamp_int_argument(
    arguments: dict[str, Any],
    key: str,
    max_value: int,
    warnings: list[str],
    warning_message: str,
) -> None:
    """Clamp one integer argument to a configured maximum."""

    if key not in arguments:
        return
    try:
        value = int(arguments[key])
    except (TypeError, ValueError):
        return
    if value > max_value:
        arguments[key] = max_value
        warnings.append(warning_message)


def evaluate_dag_safety(
    dag: ActionDAG,
    registry: CapabilityRegistry,
    policy_config: RuntimeConfig | dict[str, Any] | None,
) -> SafetyDecision:
    """Evaluate whether a DAG is safe to execute and return a sanitized copy."""

    config = _coerce_policy_config(policy_config)
    workspace_root = Path(config.workspace_root).resolve()
    blocked_reasons: list[str] = []
    warnings: list[str] = []
    requires_confirmation = dag.requires_confirmation
    sanitized_nodes: list[ActionNode] = []

    for node in dag.nodes:
        try:
            capability = registry.get(node.capability_id)
        except CapabilityNotFoundError:
            blocked_reasons.append(f"Unknown capability: {node.capability_id}.")
            continue

        manifest = capability.manifest
        if node.operation_id != manifest.operation_id:
            blocked_reasons.append(
                f"Operation {node.operation_id} is not declared by capability {manifest.capability_id}."
            )
            continue

        if _is_environment_facing(manifest) and manifest.execution_backend != "gateway":
            blocked_reasons.append(
                f"Environment-facing capability must be gateway-backed: {manifest.capability_id}."
            )
            continue

        if manifest.execution_backend == "gateway" and not (
            config.gateway_url or config.gateway_endpoints
        ):
            blocked_reasons.append(
                f"Gateway routing is not configured for {manifest.capability_id}."
            )
            continue

        if manifest.domain.strip().lower() == "shell" and not config.allow_shell_execution:
            blocked_reasons.append(
                f"Shell execution is disabled by policy for {manifest.capability_id}."
            )

        if _is_network_operation(capability, node) and not config.allow_network_operations:
            blocked_reasons.append(
                f"Network operations are disabled by policy for {manifest.capability_id}."
            )

        sanitized_arguments = dict(node.arguments)
        for argument_name in _path_argument_names(sanitized_arguments):
            value = sanitized_arguments.get(argument_name)
            if not isinstance(value, str):
                blocked_reasons.append(
                    f"Path argument {argument_name} on node {node.id} must be a string."
                )
                continue
            normalized_path, error = _normalize_path(value, workspace_root)
            if error:
                blocked_reasons.append(error)
                continue
            assert normalized_path is not None
            if _is_secret_path(normalized_path):
                blocked_reasons.append(
                    f"Access to secret path {normalized_path!r} is blocked by policy."
                )
                continue
            sanitized_arguments[argument_name] = normalized_path

        if manifest.domain.strip().lower() == "sql" and isinstance(
            sanitized_arguments.get("query"), str
        ):
            query = sanitized_arguments["query"]
            if _detect_sql_write(query):
                if manifest.read_only or not config.confirmation_granted:
                    blocked_reasons.append(
                        f"SQL write operations are blocked for capability {manifest.capability_id}."
                    )
            if _supports_argument(capability, "limit"):
                limit = sanitized_arguments.get("limit")
                if limit is None:
                    sanitized_arguments["limit"] = config.max_rows_returned
                    warnings.append(
                        f"Applied default SQL row limit {config.max_rows_returned} to node {node.id}."
                    )
                else:
                    _clamp_int_argument(
                        sanitized_arguments,
                        "limit",
                        config.max_rows_returned,
                        warnings,
                        f"Clamped SQL row limit to {config.max_rows_returned} on node {node.id}.",
                    )
        if manifest.domain.strip().lower() == "sql" and isinstance(
            sanitized_arguments.get("query_intent"), dict
        ):
            query_intent = sanitized_arguments["query_intent"]
            if _detect_query_intent_write(query_intent):
                if manifest.read_only or not config.confirmation_granted:
                    blocked_reasons.append(
                        f"SQL write operations are blocked for capability {manifest.capability_id}."
                    )
            limit = sanitized_arguments.get("limit")
            if limit is None:
                sanitized_arguments["limit"] = config.max_rows_returned
                warnings.append(
                    f"Applied default SQL row limit {config.max_rows_returned} to node {node.id}."
                )
            else:
                _clamp_int_argument(
                    sanitized_arguments,
                    "limit",
                    config.max_rows_returned,
                    warnings,
                    f"Clamped SQL row limit to {config.max_rows_returned} on node {node.id}.",
                )

        if manifest.capability_id in {"filesystem.list_directory", "filesystem.search_files"} and _supports_argument(
            capability, "limit"
        ):
            limit = sanitized_arguments.get("limit")
            if limit is None:
                sanitized_arguments["limit"] = config.max_files_listed
                warnings.append(
                    f"Applied default filesystem list limit {config.max_files_listed} to node {node.id}."
                )
            else:
                _clamp_int_argument(
                    sanitized_arguments,
                    "limit",
                    config.max_files_listed,
                    warnings,
                    f"Clamped filesystem list limit to {config.max_files_listed} on node {node.id}.",
                )

        preview_key = None
        if "preview_bytes" in sanitized_arguments:
            preview_key = "preview_bytes"
        elif "output_preview_bytes" in sanitized_arguments:
            preview_key = "output_preview_bytes"
        if preview_key is not None:
            _clamp_int_argument(
                sanitized_arguments,
                preview_key,
                config.max_output_preview_bytes,
                warnings,
                f"Clamped preview bytes to {config.max_output_preview_bytes} on node {node.id}.",
            )

        if _always_requires_confirmation(capability):
            requires_confirmation = True
        elif _is_delete_operation(node, capability):
            requires_confirmation = True
        elif _is_update_or_create(node, capability):
            allowlisted = (
                manifest.risk_level == "low"
                and manifest.capability_id in set(config.low_risk_mutation_allowlist)
            )
            if not allowlisted:
                requires_confirmation = True

        safety_labels = list(node.safety_labels)
        if requires_confirmation and "requires-confirmation" not in safety_labels:
            safety_labels.append("requires-confirmation")
        if manifest.read_only and "read-only" not in safety_labels:
            safety_labels.append("read-only")
        if manifest.mutates_state and "mutates-state" not in safety_labels:
            safety_labels.append("mutates-state")

        sanitized_nodes.append(
            ActionNode(
                id=node.id,
                task_id=node.task_id,
                description=node.description,
                semantic_verb=node.semantic_verb,
                capability_id=node.capability_id,
                operation_id=node.operation_id,
                arguments=sanitized_arguments,
                depends_on=list(node.depends_on),
                safety_labels=safety_labels,
                dry_run=node.dry_run,
            )
        )

    if blocked_reasons:
        return SafetyDecision(
            allowed=False,
            requires_confirmation=requires_confirmation,
            blocked_reasons=blocked_reasons,
            warnings=warnings,
            sanitized_dag=None,
        )

    global_constraints = dict(dag.global_constraints)
    preview_bytes = global_constraints.get("max_output_preview_bytes")
    if preview_bytes is None:
        global_constraints["max_output_preview_bytes"] = config.max_output_preview_bytes
    else:
        try:
            preview_value = int(preview_bytes)
        except (TypeError, ValueError):
            global_constraints["max_output_preview_bytes"] = config.max_output_preview_bytes
        else:
            if preview_value > config.max_output_preview_bytes:
                global_constraints["max_output_preview_bytes"] = config.max_output_preview_bytes
                warnings.append(
                    f"Clamped global preview bytes to {config.max_output_preview_bytes}."
                )

    try:
        sanitized_dag = ActionDAG(
            dag_id=dag.dag_id,
            nodes=sanitized_nodes,
            edges=list(dag.edges),
            global_constraints=global_constraints,
            requires_confirmation=requires_confirmation,
        )
    except PydanticValidationError as exc:
        return SafetyDecision(
            allowed=False,
            requires_confirmation=requires_confirmation,
            blocked_reasons=[str(exc)],
            warnings=warnings,
            sanitized_dag=None,
        )

    return SafetyDecision(
        allowed=True,
        requires_confirmation=requires_confirmation,
        blocked_reasons=[],
        warnings=warnings,
        sanitized_dag=sanitized_dag,
    )
