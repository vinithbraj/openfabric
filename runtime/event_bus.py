from collections import defaultdict


class EventBus:

    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event_name: str, agent_name: str):
        self.subscribers[event_name].append(agent_name)

    def get_subscribers(self, event_name: str):
        return self.subscribers.get(event_name, [])
