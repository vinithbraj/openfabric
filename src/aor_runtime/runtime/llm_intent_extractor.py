from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.utils import extract_json_object
from aor_runtime.llm.client import LLMClient


LLM_INTENT_FIXTURE_PATH_ENV = "AOR_LLM_INTENT_FIXTURE_PATH"
DISALLOWED_PAYLOAD_KEYS = {
    "steps",
    "action",
    "command",
    "tool",
    "tool_name",
    "execution_plan",
    "gateway_command",
    "argv",
    "shell",
    "bash",
    "exec",
    "code",
}
DISALLOWED_STRING_RE = (
    r"\b(?:shell\.exec|python\.exec|runtime\.return|slurm\.[a-z_]+|squeue|sacct|sinfo|scontrol|sbatch|scancel|ExecutionPlan)\b"
    r"|(?:^|\b)(?:bash|sh)\s+-c\b"
    r"|\bcurl\s+-[A-Za-z]"
    r"|\bwget\b"
    r"|\bgateway\s+(?:command|exec)\b"
)


class LLMIntentExtractionResult(BaseModel):
    matched: bool
    intent: Any | None = None
    intent_type: str | None = None
    confidence: float = 0.0
    reason: str | None = None
    error: str | None = None


class LLMIntentExtractor:
    def __init__(
        self,
        *,
        llm: LLMClient,
        settings: Settings | None = None,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.confidence_threshold = confidence_threshold

    def extract_intent(
        self,
        goal: str,
        capability_name: str,
        allowed_intents: list[type],
        context: dict[str, Any] | None = None,
    ) -> LLMIntentExtractionResult:
        model_lookup = {intent_type.__name__: intent_type for intent_type in allowed_intents}
        if not model_lookup:
            return LLMIntentExtractionResult(
                matched=False,
                reason="No intent models are available for extraction.",
                error="no_allowed_intents",
            )

        extraction_context = dict(context or {})
        system_prompt = str(
            extraction_context.get("system_prompt")
            or self._default_system_prompt(capability_name)
        ).strip()
        user_prompt = self._build_user_prompt(goal, capability_name, allowed_intents, extraction_context)

        try:
            raw_output = self._complete(goal, capability_name, system_prompt, user_prompt, extraction_context)
        except Exception as exc:  # noqa: BLE001
            return LLMIntentExtractionResult(
                matched=False,
                reason="LLM intent extraction failed.",
                error=str(exc),
            )

        try:
            payload = extract_json_object(raw_output)
        except Exception as exc:  # noqa: BLE001
            return LLMIntentExtractionResult(
                matched=False,
                reason="Malformed JSON response from LLM intent extractor.",
                error=f"malformed_json:{exc}",
            )

        if not isinstance(payload, dict):
            return LLMIntentExtractionResult(
                matched=False,
                reason="LLM intent extractor must return a JSON object.",
                error="non_object_json",
            )
        if self._looks_like_disallowed_top_level_payload(payload):
            return LLMIntentExtractionResult(
                matched=False,
                reason="LLM intent extractor returned an unsafe payload shape.",
                error="unsafe_payload_shape",
            )

        matched = bool(payload.get("matched"))
        intent_type_name = payload.get("intent_type")
        confidence = self._coerce_confidence(payload.get("confidence"))
        reason = str(payload.get("reason") or "").strip() or None
        arguments = payload.get("arguments")

        if not matched:
            return LLMIntentExtractionResult(
                matched=False,
                confidence=confidence,
                reason=reason or "The request did not map safely to a typed intent.",
            )

        if not isinstance(intent_type_name, str) or intent_type_name not in model_lookup:
            return LLMIntentExtractionResult(
                matched=False,
                confidence=confidence,
                reason=reason or "The LLM selected an unknown intent type.",
                error="unknown_intent_type",
            )
        if confidence < float(extraction_context.get("confidence_threshold", self.confidence_threshold)):
            return LLMIntentExtractionResult(
                matched=False,
                intent_type=intent_type_name,
                confidence=confidence,
                reason=reason or "The LLM confidence was too low.",
                error="low_confidence",
            )
        if not isinstance(arguments, dict):
            return LLMIntentExtractionResult(
                matched=False,
                intent_type=intent_type_name,
                confidence=confidence,
                reason=reason or "The LLM did not return an argument object.",
                error="invalid_arguments_shape",
            )
        if self._looks_like_disallowed_output(arguments):
            return LLMIntentExtractionResult(
                matched=False,
                intent_type=intent_type_name,
                confidence=confidence,
                reason=reason or "The LLM returned command-like or tool-like arguments.",
                error="unsafe_arguments",
            )

        intent_model = model_lookup[intent_type_name]
        try:
            intent = intent_model.model_validate(arguments)
        except ValidationError as exc:
            return LLMIntentExtractionResult(
                matched=False,
                intent_type=intent_type_name,
                confidence=confidence,
                reason=reason or "The LLM returned invalid intent arguments.",
                error=f"validation_error:{exc}",
            )

        return LLMIntentExtractionResult(
            matched=True,
            intent=intent,
            intent_type=intent_type_name,
            confidence=confidence,
            reason=reason,
        )

    def _complete(
        self,
        goal: str,
        capability_name: str,
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any],
    ) -> str:
        fixture_output = _fixture_response(capability_name, goal)
        if fixture_output is not None:
            return fixture_output
        return self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=context.get("model"),
            temperature=float(context.get("temperature", 0.0)),
        )

    def _default_system_prompt(self, capability_name: str) -> str:
        return (
            f"You convert user requests into one typed {capability_name} intent.\n"
            "You must output JSON only.\n"
            "You may only choose one of the allowed intent types.\n"
            "You must not create shell commands.\n"
            "You must not create tool calls.\n"
            "You must not create execution plans.\n"
            "You must not use python.\n"
            "If the request is ambiguous, choose a safe intent or return matched=false.\n"
            "Use the exact JSON keys: matched, intent_type, confidence, arguments, reason."
        )

    def _build_user_prompt(
        self,
        goal: str,
        capability_name: str,
        allowed_intents: list[type],
        context: dict[str, Any],
    ) -> str:
        schema_payload = []
        for intent_type in allowed_intents:
            json_schema = intent_type.model_json_schema()
            properties = dict(json_schema.get("properties", {}))
            properties.pop("gateway_node", None)
            required = [field for field in json_schema.get("required", []) if field != "gateway_node"]
            schema_payload.append(
                {
                    "intent_type": intent_type.__name__,
                    "properties": properties,
                    "required": required,
                }
            )
        prompt_payload: dict[str, Any] = {
            "capability": capability_name,
            "goal": str(goal or "").strip(),
            "allowed_intents": schema_payload,
        }
        for key in ("current_user", "notes", "examples"):
            if key in context:
                prompt_payload[key] = context[key]
        return json.dumps(prompt_payload, indent=2, ensure_ascii=False)

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:  # noqa: BLE001
            return 0.0

    def _looks_like_disallowed_output(self, value: Any) -> bool:
        if isinstance(value, dict):
            if DISALLOWED_PAYLOAD_KEYS.intersection(value.keys()):
                return True
            return any(self._looks_like_disallowed_output(item) for item in value.values())
        if isinstance(value, list):
            return any(self._looks_like_disallowed_output(item) for item in value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return False
            return bool(re.search(DISALLOWED_STRING_RE, text))
        return False

    def _looks_like_disallowed_top_level_payload(self, payload: dict[str, Any]) -> bool:
        if DISALLOWED_PAYLOAD_KEYS.intersection(payload.keys()):
            return True
        for key in ("matched", "intent_type", "confidence", "arguments", "reason"):
            if key not in payload:
                continue
            if key == "arguments":
                continue
            if self._looks_like_disallowed_output(payload.get(key)):
                return True
        return False


@lru_cache(maxsize=4)
def _load_fixture_payload(path_str: str) -> dict[str, Any]:
    path = Path(path_str).expanduser().resolve()
    return json.loads(path.read_text())


def _fixture_response(capability_name: str, goal: str) -> str | None:
    fixture_path = str(os.getenv(LLM_INTENT_FIXTURE_PATH_ENV, "")).strip()
    if not fixture_path:
        return None
    payload = _load_fixture_payload(fixture_path)
    key = str(goal or "").strip()
    capability_payload = payload.get(capability_name)
    if isinstance(capability_payload, dict) and key in capability_payload:
        return str(capability_payload[key])
    direct_value = payload.get(key)
    if direct_value is None:
        return None
    return str(direct_value)
