from .base import Adapter


class MockRetrieval(Adapter):

    def handle(self, event_name, payload):
        query = payload["query"]
        return [
            ("research.result", {"content": f"Result for '{query}'"})
        ]
