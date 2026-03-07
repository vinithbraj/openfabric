import requests
from .base import Adapter


class HTTPAdapter(Adapter):

    def handle(self, event_name, payload):

        endpoint = self.config["endpoint"]
        timeout_seconds = float(self.config.get("timeout_seconds", 30))

        try:
            response = requests.post(
                endpoint,
                json={
                    "event": event_name,
                    "payload": payload
                },
                timeout=timeout_seconds
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "HTTP adapter could not reach endpoint "
                f"'{endpoint}'. Ensure the sample services are running on "
                "ports 8001/8002/8003. You can start them with: "
                "'./scripts/start_http_agents.sh'"
            ) from exc

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
