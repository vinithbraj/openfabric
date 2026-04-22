from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.utils import extract_json_object


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._discovered_models: list[str] | None = None

    def available_models(self) -> list[str]:
        if self._discovered_models is not None:
            return self._discovered_models
        try:
            response = httpx.get(f"{self.settings.llm_base_url.rstrip('/')}/models", timeout=self.settings.llm_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            self._discovered_models = [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
        except Exception:  # noqa: BLE001
            self._discovered_models = []
        return self._discovered_models

    def resolve_model(self, requested: str | None = None) -> str:
        requested_model = requested or self.settings.default_model
        available = self.available_models()
        if not available:
            return requested_model
        if requested_model in available:
            return requested_model
        lowered_requested = requested_model.lower()
        for candidate in available:
            lowered_candidate = candidate.lower()
            if lowered_requested in lowered_candidate or lowered_candidate.endswith(lowered_requested):
                return candidate
        return available[0]

    def _build_model(self, *, model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.resolve_model(model),
            temperature=self.settings.default_temperature if temperature is None else temperature,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=2,
        )

    def complete(self, *, system_prompt: str, user_prompt: str, model: str | None = None, temperature: float | None = None) -> str:
        llm = self._build_model(model=model, temperature=temperature)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return str(response.content)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        raw = self.complete(system_prompt=system_prompt, user_prompt=user_prompt, model=model, temperature=temperature)
        data = extract_json_object(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object response from model")
        return data

    def load_prompt(self, path: str | None, fallback: str) -> str:
        if not path:
            return fallback
        prompt_path = Path(path)
        if prompt_path.exists():
            return prompt_path.read_text()
        return fallback
