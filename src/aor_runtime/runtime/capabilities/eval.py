from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CapabilityEvalCase(BaseModel):
    id: str
    prompt: str
    expected: Any | None = None
    expected_contains: list[str] | None = None
    expected_regex: str | None = None
    expect_llm_calls: int | None = 0
    category: str | None = None
    setup: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "prompt")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("expected_contains")
    @classmethod
    def _validate_expected_contains(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("expected_contains must not be empty when provided")
        normalized = [str(item).strip() for item in value]
        if not all(normalized):
            raise ValueError("expected_contains items must be non-empty strings")
        return normalized

    @model_validator(mode="after")
    def _validate_expectation_shape(self) -> "CapabilityEvalCase":
        if self.expected is None and not self.expected_contains and not self.expected_regex:
            raise ValueError("each eval case must define expected, expected_contains, or expected_regex")
        return self


class CapabilityEvalPack(BaseModel):
    capability: str
    cases: list[CapabilityEvalCase]
    strict_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    semantic_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    max_llm_fallbacks: int = Field(default=0, ge=0)

    @field_validator("capability")
    @classmethod
    def _validate_capability(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("capability must be a non-empty string")
        return text

    @model_validator(mode="after")
    def _validate_unique_case_ids(self) -> "CapabilityEvalPack":
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id within pack '{self.capability}': {case.id}")
            seen.add(case.id)
        return self


class CapabilityEvalResult(BaseModel):
    capability: str
    total: int
    strict_pass: int
    semantic_pass: int
    llm_fallbacks: int
    failures: list[dict[str, Any]] = Field(default_factory=list)


def load_capability_eval_pack(path: Path) -> CapabilityEvalPack:
    return CapabilityEvalPack.model_validate_json(path.read_text())


def load_capability_eval_packs(directory: Path) -> list[CapabilityEvalPack]:
    return [load_capability_eval_pack(path) for path in sorted(directory.glob("*.json"))]


def ensure_unique_case_ids(packs: list[CapabilityEvalPack]) -> None:
    seen: dict[str, str] = {}
    for pack in packs:
        for case in pack.cases:
            prior_capability = seen.get(case.id)
            if prior_capability is not None:
                raise ValueError(
                    f"duplicate eval case id across packs: {case.id} appears in {prior_capability} and {pack.capability}"
                )
            seen[case.id] = pack.capability
