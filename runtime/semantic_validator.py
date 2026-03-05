from urllib.parse import urlparse


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
