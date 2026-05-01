"""Prompt templates for typed LLM stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PromptTemplate(BaseModel):
    """A small typed prompt template."""

    model_config = ConfigDict(extra="forbid")

    name: str
    text: str


def semantic_frame_prompt(user_prompt: str) -> PromptTemplate:
    """Build a placeholder semantic-frame extraction prompt."""

    return PromptTemplate(
        name="semantic_frame",
        text=f"Return a typed semantic frame for this user prompt: {user_prompt}",
    )
