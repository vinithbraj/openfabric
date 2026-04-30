"""OpenFABRIC Runtime Module: aor_runtime.llm.client

Purpose:
    Wrap LLM provider calls, structured response handling, and token accounting.

Responsibilities:
    Normalize configured provider access, model selection, token accounting, and structured response handling.

Data flow / Interfaces:
    Receives compact planner/presentation prompts and returns model responses plus usage metadata.

Boundaries:
    Must not be handed raw result rows or sensitive payloads except through explicitly sanitized prompt construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.utils import extract_json_object


class LLMClient:
    """Represent l l m client within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by LLMClient.

    Data flow / Interfaces:
        Instances are created and consumed by LLM provider integration code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.llm.client.LLMClient and related tests.
    """
    def __init__(self, settings: Settings | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by LLM provider integration through LLMClient.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        self._discovered_models: list[str] | None = None
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0

    def available_models(self) -> list[str]:
        """Available models for LLMClient instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient.available_models calls and related tests.
        """
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
        """Resolve model for LLMClient instances.

        Inputs:
            Receives requested for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient.resolve_model calls and related tests.
        """
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
        """Handle the internal build model helper path for this module.

        Inputs:
            Receives model, temperature for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient._build_model calls and related tests.
        """
        return ChatOpenAI(
            model=self.resolve_model(model),
            temperature=self.settings.default_temperature if temperature is None else temperature,
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=2,
        )

    def complete(self, *, system_prompt: str, user_prompt: str, model: str | None = None, temperature: float | None = None) -> str:
        """Complete for LLMClient instances.

        Inputs:
            Receives system_prompt, user_prompt, model, temperature for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient.complete calls and related tests.
        """
        llm = self._build_model(model=model, temperature=temperature)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        self._record_usage(response)
        return str(response.content)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Complete json for LLMClient instances.

        Inputs:
            Receives system_prompt, user_prompt, model, temperature for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient.complete_json calls and related tests.
        """
        raw = self.complete(system_prompt=system_prompt, user_prompt=user_prompt, model=model, temperature=temperature)
        data = extract_json_object(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object response from model")
        return data

    def load_prompt(self, path: str | None, fallback: str) -> str:
        """Load prompt for LLMClient instances.

        Inputs:
            Receives path, fallback for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient.load_prompt calls and related tests.
        """
        if not path:
            return fallback
        prompt_path = Path(path)
        if prompt_path.exists():
            return prompt_path.read_text()
        return fallback

    def _record_usage(self, response: Any) -> None:
        """Handle the internal record usage helper path for this module.

        Inputs:
            Receives response for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by LLM provider integration through LLMClient._record_usage calls and related tests.
        """
        usage = self._extract_usage(response)
        self.last_usage = usage
        self.total_prompt_tokens += usage["prompt_tokens"]
        self.total_completion_tokens += usage["completion_tokens"]
        self.total_tokens += usage["total_tokens"]

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Handle the internal extract usage helper path for this module.

        Inputs:
            Receives response for this LLMClient method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by LLM provider integration through LLMClient._extract_usage calls and related tests.
        """
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict) and usage_metadata:
            prompt_tokens = _coerce_int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens"))
            completion_tokens = _coerce_int(usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens"))
            total_tokens = _coerce_int(usage_metadata.get("total_tokens"))
            if total_tokens <= 0 and (prompt_tokens or completion_tokens):
                total_tokens = prompt_tokens + completion_tokens
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
            if isinstance(token_usage, dict):
                prompt_tokens = _coerce_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
                completion_tokens = _coerce_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
                total_tokens = _coerce_int(token_usage.get("total_tokens"))
                if total_tokens <= 0 and (prompt_tokens or completion_tokens):
                    total_tokens = prompt_tokens + completion_tokens
                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _coerce_int(value: Any) -> int:
    """Handle the internal coerce int helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by LLM provider integration code paths that import or call aor_runtime.llm.client._coerce_int.
    """
    try:
        return max(0, int(value or 0))
    except Exception:  # noqa: BLE001
        return 0
