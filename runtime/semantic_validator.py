from urllib.parse import urlparse

from agent_library.contracts import api_emit_events, api_trigger_event


def _validate_contract_name(
    contracts: dict,
    agent_name: str,
    scope: str,
    key: str,
    value,
):
    if value in (None, ""):
        return
    if not isinstance(value, str):
        raise ValueError(
            f"Agent '{agent_name}' {scope} has non-string '{key}'"
        )
    if value not in contracts:
        raise ValueError(
            f"Agent '{agent_name}' {scope} references unknown contract '{value}' via '{key}'"
        )


def _validate_field_list(agent_name: str, scope: str, key: str, value):
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(
            f"Agent '{agent_name}' {scope} must define '{key}' as a list"
        )
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Agent '{agent_name}' {scope} has invalid '{key}' entry"
            )


def _validate_method_like(
    *,
    contracts: dict,
    events: dict,
    agent_name: str,
    scope_name: str,
    method: dict,
):
    if not method.get("name") or not isinstance(method.get("name"), str):
        raise ValueError(
            f"Agent '{agent_name}' {scope_name} is missing string 'name'"
        )
    legacy_event = method.get("event")
    if legacy_event is not None and (
        not isinstance(legacy_event, str) or not legacy_event.strip()
    ):
        raise ValueError(
            f"Agent '{agent_name}' {scope_name} has invalid string 'event'"
        )
    trigger_event = api_trigger_event(method)
    if not trigger_event:
        raise ValueError(
            f"Agent '{agent_name}' {scope_name} is missing string 'trigger_event' or legacy 'event'"
        )
    if legacy_event and legacy_event.strip() != trigger_event:
        raise ValueError(
            f"Agent '{agent_name}' {scope_name} defines mismatched 'event' and 'trigger_event'"
        )
    if trigger_event not in events:
        raise ValueError(
            f"Agent '{agent_name}' {scope_name} references undefined trigger event '{trigger_event}'"
        )
    emitted_events = api_emit_events(method)
    for event_name in emitted_events:
        if event_name not in events:
            raise ValueError(
                f"Agent '{agent_name}' {scope_name} emits undefined event '{event_name}'"
            )
    if "when" in method and not isinstance(method["when"], str):
        raise ValueError(
            f"Agent '{agent_name}' {scope_name} has non-string 'when'"
        )
    _validate_contract_name(contracts, agent_name, scope_name, "request_contract", method.get("request_contract"))
    _validate_contract_name(contracts, agent_name, scope_name, "result_contract", method.get("result_contract"))
    _validate_field_list(agent_name, scope_name, "request_envelope_fields", method.get("request_envelope_fields"))
    _validate_field_list(agent_name, scope_name, "result_envelope_fields", method.get("result_envelope_fields"))


def validate_semantics(spec: dict):

    contracts = spec.get("contracts", {})
    events = spec.get("events", {})
    agents = spec.get("agents", {})

    # Validate event contract references
    for event_name, event in events.items():
        contract_name = event["contract"]
        if contract_name not in contracts:
            raise ValueError(
                f"Event '{event_name}' references unknown contract '{contract_name}'"
            )

    # Validate agent subscriptions and emissions
    for agent_name, agent in agents.items():
        runtime_cfg = agent.get("runtime", {})
        methods = agent.get("methods", [])
        apis = agent.get("apis", [])

        if methods is None:
            methods = []
        if not isinstance(methods, list):
            raise ValueError(
                f"Agent '{agent_name}' must define 'methods' as a list"
            )
        for idx, method in enumerate(methods):
            if not isinstance(method, dict):
                raise ValueError(
                    f"Agent '{agent_name}' method at index {idx} must be an object"
                )
            _validate_method_like(
                contracts=contracts,
                events=events,
                agent_name=agent_name,
                scope_name=f"method at index {idx}",
                method=method,
            )
        if apis is None:
            apis = []
        if not isinstance(apis, list):
            raise ValueError(
                f"Agent '{agent_name}' must define 'apis' as a list"
            )
        for idx, api in enumerate(apis):
            if not isinstance(api, dict):
                raise ValueError(
                    f"Agent '{agent_name}' api at index {idx} must be an object"
                )
            _validate_method_like(
                contracts=contracts,
                events=events,
                agent_name=agent_name,
                scope_name=f"api at index {idx}",
                method=api,
            )

        for event in agent.get("subscribes_to", []):
            if event not in events:
                raise ValueError(
                    f"Agent '{agent_name}' subscribes to undefined event '{event}'"
                )

        for event in agent.get("emits", []):
            if event not in events:
                raise ValueError(
                    f"Agent '{agent_name}' emits undefined event '{event}'"
                )
        _validate_contract_name(contracts, agent_name, "descriptor", "request_contract", agent.get("request_contract"))
        _validate_contract_name(contracts, agent_name, "descriptor", "result_contract", agent.get("result_contract"))
        _validate_field_list(agent_name, "descriptor", "request_envelope_fields", agent.get("request_envelope_fields"))
        _validate_field_list(agent_name, "descriptor", "result_envelope_fields", agent.get("result_envelope_fields"))

        if runtime_cfg.get("adapter") == "http":
            endpoint = runtime_cfg.get("endpoint")
            if not endpoint:
                raise ValueError(
                    f"HTTP agent '{agent_name}' is missing runtime.endpoint"
                )

            parsed = urlparse(endpoint)
            if not parsed.scheme or not parsed.hostname or parsed.port is None:
                raise ValueError(
                    f"HTTP agent '{agent_name}' endpoint must include scheme, host, and port"
                )

            autostart_cfg = runtime_cfg.get("autostart")
            if autostart_cfg and not autostart_cfg.get("app"):
                raise ValueError(
                    f"HTTP agent '{agent_name}' runtime.autostart requires 'app'"
                )
