from .contracts import ContractRegistry
from .event_bus import EventBus
from .registry import ADAPTER_REGISTRY


class Engine:

    def __init__(self, spec: dict):
        self.spec = spec
        self.contracts = ContractRegistry(spec["contracts"])
        self.bus = EventBus()
        self.agents = {}

    def setup(self):

        # Instantiate agents
        for name, config in self.spec["agents"].items():
            adapter_type = config["runtime"]["adapter"]

            if adapter_type not in ADAPTER_REGISTRY:
                raise ValueError(f"Unknown adapter type: {adapter_type}")

            adapter_cls = ADAPTER_REGISTRY[adapter_type]
            adapter = adapter_cls(config["runtime"])

            self.agents[name] = {
                "adapter": adapter,
                "subscribes_to": config.get("subscribes_to", [])
            }

        # Register subscriptions
        for agent_name, agent in self.agents.items():
            for event in agent["subscribes_to"]:
                self.bus.subscribe(event, agent_name)

    def emit(self, event_name: str, payload: dict, depth: int = 0):

        indent = "  " * depth
        print(f"{indent}[EVENT] {event_name} -> {payload}")

        contract_name = self.spec["events"][event_name]["contract"]
        self.contracts.validate_payload(contract_name, payload)

        subscribers = self.bus.get_subscribers(event_name)

        for agent_name in subscribers:
            print(f"{indent}  ↳ handled by: {agent_name}")

            agent = self.agents[agent_name]
            results = agent["adapter"].handle(event_name, payload)

            for new_event, new_payload in results:
                self.emit(new_event, new_payload, depth + 1)
