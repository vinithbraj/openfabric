"""Helpers for validating structured LLM output."""

from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from agent_runtime.llm.client import LLMClient, LLMClientError

ModelT = TypeVar("ModelT", bound=BaseModel)


StructuredCallErrorKind = Literal[
    "transport_error",
    "invalid_json",
    "schema_validation_error",
    "empty_response",
    "parsing_error",
]


class StructuredCallDiagnostics(BaseModel):
    """Typed diagnostics for a failed structured LLM call."""

    error_kind: StructuredCallErrorKind
    error_message: str
    raw_response_preview: str | None = None
    raw_payload_preview: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    schema_name: str


class StructuredCallError(RuntimeError):
    """Raised when a structured LLM call fails before producing a valid model."""

    def __init__(self, diagnostics: StructuredCallDiagnostics) -> None:
        super().__init__(diagnostics.error_message)
        self.diagnostics = diagnostics


def _truncate_preview(value: Any, max_length: int = 1000) -> str | None:
    """Return a safe truncated preview for diagnostics."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"


def _stable_preview(value: Any) -> str | None:
    """Serialize one structure stably for diagnostics previews."""

    try:
        serialized = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except Exception:
        serialized = repr(value)
    return _truncate_preview(serialized)


def structured_call(client: LLMClient, prompt: str, output_model: type[ModelT]) -> ModelT:
    """Call an LLM client and validate the JSON result as a Pydantic model."""

    try:
        payload: dict[str, Any] = client.complete_json(prompt, output_model.model_json_schema())
    except LLMClientError as exc:
        raise StructuredCallError(
            StructuredCallDiagnostics(
                error_kind=exc.error_kind,  # type: ignore[arg-type]
                error_message=exc.error_message,
                raw_response_preview=exc.raw_response_preview,
                raw_payload_preview=exc.raw_payload_preview or _truncate_preview(prompt),
                validation_errors=[],
                schema_name=output_model.__name__,
            )
        ) from exc
    except Exception as exc:
        raise StructuredCallError(
            StructuredCallDiagnostics(
                error_kind="parsing_error",
                error_message=str(exc),
                raw_response_preview=None,
                raw_payload_preview=_truncate_preview(prompt),
                validation_errors=[],
                schema_name=output_model.__name__,
            )
        ) from exc
    try:
        return output_model.model_validate(payload)
    except PydanticValidationError as exc:
        raise StructuredCallError(
            StructuredCallDiagnostics(
                error_kind="schema_validation_error",
                error_message="Structured LLM response did not match the expected schema.",
                raw_response_preview=None,
                raw_payload_preview=_stable_preview(payload),
                validation_errors=[str(exc)],
                schema_name=output_model.__name__,
            )
        ) from exc
