"""Helpers for validating structured LLM output."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from agent_runtime.llm.client import LLMClient

ModelT = TypeVar("ModelT", bound=BaseModel)


def structured_call(client: LLMClient, prompt: str, output_model: type[ModelT]) -> ModelT:
    """Call an LLM client and validate the JSON result as a Pydantic model."""

    payload: dict[str, Any] = client.complete_json(prompt, output_model.model_json_schema())
    return output_model.model_validate(payload)
