import requests
from .base import Adapter


class HTTPAdapter(Adapter):

    def handle(self, event_name, payload):

        endpoint = self.config["endpoint"]

        response = requests.post(
            endpoint,
            json={
                "event": event_name,
                "payload": payload
            },
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        if "emits" not in data:
            raise ValueError(
                f"Invalid response from HTTP agent at {endpoint}"
            )

        return [
            (item["event"], item["payload"])
            for item in data["emits"]
        ]
