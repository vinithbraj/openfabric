from .base import Adapter


class MockLLM(Adapter):

    def handle(self, event_name, payload):

        if event_name == "user.ask":
            question = payload["question"]
            return [
                ("research.query", {"query": f"{question} overview"}),
                ("research.query", {"query": f"{question} history"}),
                ("research.query", {"query": f"{question} applications"})
            ]

        if event_name == "research.result":
            content = payload["content"]
            return [
                ("answer.final", {"answer": f"Final synthesized answer from: {content}"})
            ]

        return []
