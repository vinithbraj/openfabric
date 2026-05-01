"""LLM client interfaces.

The foundation defines the boundary and includes a small OpenAI-compatible JSON
client so the agent runtime can call a real model without extra dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClient(Protocol):
    """Protocol for structured LLM calls."""

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-compatible object matching the requested schema."""


class StaticLLMClient:
    """Test placeholder that always returns a configured payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return the static payload without contacting an external model."""

        return dict(self.payload)


def _coerce_message_content(content: Any) -> str:
    """Normalize OpenAI-compatible message content into one text string."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _strip_json_fences(text: str) -> str:
    """Remove common markdown code fences around JSON output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output text."""

    candidate = _strip_json_fences(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(candidate[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a valid JSON object.")


class OpenAICompatLLMClient:
    """Tiny OpenAI-compatible structured JSON client using stdlib HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)

    def _request_payload(self, prompt: str, schema: dict[str, Any], include_response_format: bool) -> dict[str, Any]:
        """Build an OpenAI-compatible chat completion payload."""

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return one valid JSON object only. "
                        "Do not include markdown fences, explanations, or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt + "\n\nTarget JSON schema:\n" + json.dumps(schema, separators=(",", ":")),
                },
            ],
        }
        if include_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one payload to the configured chat-completions endpoint."""

        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Call the model and parse a JSON object from the assistant message."""

        response_payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for include_response_format in (True, False):
            try:
                response_payload = self._post(self._request_payload(prompt, schema, include_response_format))
                break
            except HTTPError as exc:
                last_error = exc
                if include_response_format and exc.code in {400, 404, 415, 422, 500}:
                    continue
                raise RuntimeError(f"LLM request failed with HTTP {exc.code}.") from exc
            except URLError as exc:
                raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        if response_payload is None:
            raise RuntimeError("LLM request failed.") from last_error

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response did not include choices.")
        message = choices[0].get("message", {})
        content = _coerce_message_content(message.get("content"))
        return _extract_json_object(content)
