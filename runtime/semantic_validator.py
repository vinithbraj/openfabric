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
