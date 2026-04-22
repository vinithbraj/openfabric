from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - lightweight test stubs
    from pydantic import BaseModel  # type: ignore

    def Field(default: Any = None, default_factory: Any | None = None, **_: Any) -> Any:  # type: ignore
        if default_factory is not None:
            return default_factory()
        return default


AGENT_CONTRACT_VERSION = "agent_contract_v1"
SHARED_REQUEST_CONTRACT_NAME = "agent_execution_request"
SHARED_RESULT_CONTRACT_NAME = "agent_execution_result"
REQUEST_ENVELOPE_FIELDS = [
    "node",
    "task",
    "original_task",
    "instruction",
    "context",
    "policy",
    "artifacts",
]
RESULT_ENVELOPE_FIELDS = [
    "node",
    "status",
    "api",
    "detail",
    "raw_output",
    "structured_output",
    "artifacts",
    "reduction_request",
    "error",
    "metrics",
]


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    if hasattr(model, "dict"):
        return model.dict()  # type: ignore[attr-defined]
    return deepcopy(getattr(model, "__dict__", {}))


def _model_json_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model_type, "model_json_schema"):
        return model_type.model_json_schema()  # type: ignore[attr-defined]
    if hasattr(model_type, "schema"):
        return model_type.schema()  # type: ignore[attr-defined]
    return {}


def _metadata_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return deepcopy(raw)
    if isinstance(raw, BaseModel):
        return _model_dump(raw)
    return {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _event_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return _string_list(value)


def _unique_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def shared_request_field_names() -> list[str]:
    return list(REQUEST_ENVELOPE_FIELDS)


def shared_result_field_names() -> list[str]:
    return list(RESULT_ENVELOPE_FIELDS)


class AgentInstructionContract(BaseModel):
    api: str = ""
    input: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class AgentExecutionRequestContract(BaseModel):
    node: dict[str, Any] = Field(default_factory=dict)
    task: str = ""
    original_task: str = ""
    instruction: AgentInstructionContract = Field(default_factory=AgentInstructionContract)
    context: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        extra = "allow"


class AgentExecutionResultContract(BaseModel):
    node: dict[str, Any] = Field(default_factory=dict)
    status: str = ""
    api: str = ""
    detail: str = ""
    raw_output: dict[str, Any] = Field(default_factory=dict)
    structured_output: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reduction_request: dict[str, Any] = Field(default_factory=dict)
    error: Any = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


def shared_request_schema() -> dict[str, Any]:
    return _model_json_schema(AgentExecutionRequestContract)


def shared_result_schema() -> dict[str, Any]:
    return _model_json_schema(AgentExecutionResultContract)


class AgentApiSpec(BaseModel):
    name: str
    event: str = ""
    trigger_event: str = ""
    emits: list[str] = Field(default_factory=list)
    summary: str = ""
    when: str = ""
    intent_tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    risk_level: str | None = None
    deterministic: bool | None = None
    side_effect_level: str | None = None
    request_contract: str = SHARED_REQUEST_CONTRACT_NAME
    result_contract: str = SHARED_RESULT_CONTRACT_NAME
    request_envelope_fields: list[str] = Field(default_factory=shared_request_field_names)
    result_envelope_fields: list[str] = Field(default_factory=shared_result_field_names)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class AgentDescriptor(BaseModel):
    name: str
    contract_version: str = AGENT_CONTRACT_VERSION
    role: str = ""
    description: str = ""
    capability_domains: list[str] = Field(default_factory=list)
    action_verbs: list[str] = Field(default_factory=list)
    side_effect_policy: str = "unspecified"
    safety_enforced_by_agent: bool = False
    routing_notes: list[str] = Field(default_factory=list)
    apis: list[AgentApiSpec] = Field(default_factory=list)
    request_contract: str = SHARED_REQUEST_CONTRACT_NAME
    result_contract: str = SHARED_RESULT_CONTRACT_NAME
    request_envelope_fields: list[str] = Field(default_factory=shared_request_field_names)
    result_envelope_fields: list[str] = Field(default_factory=shared_result_field_names)
    request_schema: dict[str, Any] = Field(default_factory=shared_request_schema)
    result_schema: dict[str, Any] = Field(default_factory=shared_result_schema)

    class Config:
        extra = "allow"


def build_agent_api(
    *,
    name: str,
    event: str | None = None,
    trigger_event: str | None = None,
    emits: list[str] | str | None = None,
    summary: str = "",
    when: str = "",
    intent_tags: list[str] | None = None,
    examples: list[str] | None = None,
    anti_patterns: list[str] | None = None,
    risk_level: str | None = None,
    deterministic: bool | None = None,
    side_effect_level: str | None = None,
    request_contract: str = SHARED_REQUEST_CONTRACT_NAME,
    result_contract: str = SHARED_RESULT_CONTRACT_NAME,
    request_envelope_fields: list[str] | None = None,
    result_envelope_fields: list[str] | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    effective_trigger_event = str(trigger_event or event or "").strip()
    emitted_events = _event_list(emits)
    for legacy_key in ("emitted_event", "output_event"):
        legacy_value = extra.pop(legacy_key, None)
        if not emitted_events:
            emitted_events = _event_list(legacy_value)
    api = AgentApiSpec(
        name=name,
        event=effective_trigger_event,
        trigger_event=effective_trigger_event,
        emits=_unique_list(emitted_events),
        summary=summary or when,
        when=when or summary,
        intent_tags=intent_tags or [],
        examples=examples or [],
        anti_patterns=anti_patterns or [],
        risk_level=risk_level,
        deterministic=deterministic,
        side_effect_level=side_effect_level,
        request_contract=request_contract or SHARED_REQUEST_CONTRACT_NAME,
        result_contract=result_contract or SHARED_RESULT_CONTRACT_NAME,
        request_envelope_fields=list(request_envelope_fields or shared_request_field_names()),
        result_envelope_fields=list(result_envelope_fields or shared_result_field_names()),
        input_schema=deepcopy(input_schema) if isinstance(input_schema, dict) else {},
        output_schema=deepcopy(output_schema) if isinstance(output_schema, dict) else {},
        **extra,
    )
    return _model_dump(api)


def _legacy_method_alias(api: dict[str, Any]) -> dict[str, Any]:
    trigger_event = api_trigger_event(api)
    payload = {
        "name": str(api.get("name") or "").strip(),
        "event": trigger_event,
        "trigger_event": trigger_event,
    }
    emitted_events = api_emit_events(api)
    if emitted_events:
        payload["emits"] = emitted_events
    when = str(api.get("when") or api.get("summary") or "").strip()
    if when:
        payload["when"] = when
    for key in ("intent_tags", "examples", "anti_patterns"):
        values = _string_list(api.get(key))
        if values:
            payload[key] = values
    for key in ("risk_level",):
        value = api.get(key)
        if isinstance(value, str) and value.strip():
            payload[key] = value.strip()
    return payload


def api_trigger_event(api: Any) -> str:
    if not isinstance(api, dict):
        return ""
    value = api.get("trigger_event")
    if not isinstance(value, str) or not value.strip():
        value = api.get("event")
    return value.strip() if isinstance(value, str) else ""


def api_emit_events(api: Any) -> list[str]:
    if not isinstance(api, dict):
        return []
    emitted = _event_list(api.get("emits"))
    if emitted:
        return _unique_list(emitted)
    for key in ("emitted_event", "output_event"):
        emitted = _event_list(api.get(key))
        if emitted:
            return _unique_list(emitted)
    return []


def metadata_api_trigger_events(raw_metadata: Any) -> list[str]:
    raw = _metadata_dict(raw_metadata)
    api_source = raw.get("apis")
    if not isinstance(api_source, list):
        api_source = raw.get("methods")
    events = [api_trigger_event(item) for item in api_source or [] if isinstance(item, dict)]
    return _unique_list([item for item in events if item])


def metadata_api_emit_events(raw_metadata: Any) -> list[str]:
    raw = _metadata_dict(raw_metadata)
    api_source = raw.get("apis")
    if not isinstance(api_source, list):
        api_source = raw.get("methods")
    emitted: list[str] = []
    for item in api_source or []:
        if not isinstance(item, dict):
            continue
        emitted.extend(api_emit_events(item))
    return _unique_list(emitted)


def normalize_agent_metadata(agent_name: str, raw_metadata: Any) -> dict[str, Any]:
    raw = _metadata_dict(raw_metadata)
    if not raw:
        return {}

    api_source = raw.get("apis")
    if not isinstance(api_source, list):
        api_source = raw.get("methods")
    normalized_apis: list[dict[str, Any]] = []
    if isinstance(api_source, list):
        for item in api_source:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            trigger_event = api_trigger_event(item)
            if not name or not trigger_event:
                continue
            normalized_apis.append(
                build_agent_api(
                    name=name,
                    trigger_event=trigger_event,
                    emits=api_emit_events(item),
                    summary=str(item.get("summary") or item.get("when") or "").strip(),
                    when=str(item.get("when") or item.get("summary") or "").strip(),
                    intent_tags=_string_list(item.get("intent_tags")),
                    examples=_string_list(item.get("examples")),
                    anti_patterns=_string_list(item.get("anti_patterns")),
                    risk_level=str(item.get("risk_level") or "").strip() or None,
                    deterministic=item.get("deterministic") if isinstance(item.get("deterministic"), bool) else None,
                    side_effect_level=str(item.get("side_effect_level") or "").strip() or None,
                    request_contract=str(item.get("request_contract") or "").strip() or SHARED_REQUEST_CONTRACT_NAME,
                    result_contract=str(item.get("result_contract") or "").strip() or SHARED_RESULT_CONTRACT_NAME,
                    request_envelope_fields=_string_list(item.get("request_envelope_fields")) or shared_request_field_names(),
                    result_envelope_fields=_string_list(item.get("result_envelope_fields")) or shared_result_field_names(),
                    input_schema=deepcopy(item.get("input_schema")) if isinstance(item.get("input_schema"), dict) else None,
                    output_schema=deepcopy(item.get("output_schema")) if isinstance(item.get("output_schema"), dict) else None,
                )
            )

    payload = deepcopy(raw)
    payload.setdefault("name", agent_name)
    payload.setdefault("contract_version", AGENT_CONTRACT_VERSION)
    payload["apis"] = normalized_apis
    payload["methods"] = [_legacy_method_alias(item) for item in normalized_apis]
    payload.setdefault("request_contract", SHARED_REQUEST_CONTRACT_NAME)
    payload.setdefault("result_contract", SHARED_RESULT_CONTRACT_NAME)
    payload.setdefault("request_envelope_fields", shared_request_field_names())
    payload.setdefault("result_envelope_fields", shared_result_field_names())
    payload.setdefault("request_schema", shared_request_schema())
    payload.setdefault("result_schema", shared_result_schema())

    descriptor = AgentDescriptor(**payload)
    normalized = _model_dump(descriptor)
    normalized["apis"] = normalized_apis
    normalized["methods"] = [_legacy_method_alias(item) for item in normalized_apis]
    return normalized


def build_agent_descriptor(
    *,
    name: str,
    role: str,
    description: str,
    capability_domains: list[str] | None = None,
    action_verbs: list[str] | None = None,
    side_effect_policy: str = "unspecified",
    safety_enforced_by_agent: bool = False,
    routing_notes: list[str] | None = None,
    apis: list[dict[str, Any]] | None = None,
    request_contract: str = SHARED_REQUEST_CONTRACT_NAME,
    result_contract: str = SHARED_RESULT_CONTRACT_NAME,
    request_envelope_fields: list[str] | None = None,
    result_envelope_fields: list[str] | None = None,
    request_schema: dict[str, Any] | None = None,
    result_schema: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "role": role,
        "description": description,
        "capability_domains": capability_domains or [],
        "action_verbs": action_verbs or [],
        "side_effect_policy": side_effect_policy,
        "safety_enforced_by_agent": safety_enforced_by_agent,
        "routing_notes": routing_notes or [],
        "apis": apis or [],
        "request_contract": request_contract or SHARED_REQUEST_CONTRACT_NAME,
        "result_contract": result_contract or SHARED_RESULT_CONTRACT_NAME,
        "request_envelope_fields": list(request_envelope_fields or shared_request_field_names()),
        "result_envelope_fields": list(result_envelope_fields or shared_result_field_names()),
        "request_schema": deepcopy(request_schema) if isinstance(request_schema, dict) else shared_request_schema(),
        "result_schema": deepcopy(result_schema) if isinstance(result_schema, dict) else shared_result_schema(),
        **extra,
    }
    return normalize_agent_metadata(name, payload)
