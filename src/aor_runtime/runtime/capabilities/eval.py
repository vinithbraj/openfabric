"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.eval

Purpose:
    Provide compatibility capability-pack helpers and fixtures for domain-specific tests and utilities.

Responsibilities:
    Classify or compile typed intents when called directly by tests or compatibility surfaces.

Data flow / Interfaces:
    Consumes compile contexts, allowed tools, and typed intents; returns execution-plan fragments or eval metadata.

Boundaries:
    These modules are not the active top-level natural-language planner; user prompts route through LLMActionPlanner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CapabilityEvalCase(BaseModel):
    """Represent capability eval case within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CapabilityEvalCase.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.eval.CapabilityEvalCase and related tests.
    """
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
        """Validate validate required text invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this CapabilityEvalCase method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityEvalCase._validate_required_text calls and related tests.
        """
        text = str(value).strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("expected_contains")
    @classmethod
    def _validate_expected_contains(cls, value: list[str] | None) -> list[str] | None:
        """Validate validate expected contains invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this CapabilityEvalCase method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityEvalCase._validate_expected_contains calls and related tests.
        """
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
        """Validate validate expectation shape invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityEvalCase._validate_expectation_shape calls and related tests.
        """
        if self.expected is None and not self.expected_contains and not self.expected_regex:
            raise ValueError("each eval case must define expected, expected_contains, or expected_regex")
        return self


class CapabilityEvalPack(BaseModel):
    """Represent capability eval pack within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CapabilityEvalPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.eval.CapabilityEvalPack and related tests.
    """
    capability: str
    cases: list[CapabilityEvalCase]
    strict_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    semantic_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    max_llm_fallbacks: int = Field(default=0, ge=0)

    @field_validator("capability")
    @classmethod
    def _validate_capability(cls, value: str) -> str:
        """Validate validate capability invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this CapabilityEvalPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityEvalPack._validate_capability calls and related tests.
        """
        text = str(value).strip()
        if not text:
            raise ValueError("capability must be a non-empty string")
        return text

    @model_validator(mode="after")
    def _validate_unique_case_ids(self) -> "CapabilityEvalPack":
        """Validate validate unique case ids invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through CapabilityEvalPack._validate_unique_case_ids calls and related tests.
        """
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id within pack '{self.capability}': {case.id}")
            seen.add(case.id)
        return self


class CapabilityEvalResult(BaseModel):
    """Represent capability eval result within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CapabilityEvalResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.eval.CapabilityEvalResult and related tests.
    """
    capability: str
    total: int
    strict_pass: int
    semantic_pass: int
    llm_fallbacks: int
    llm_intent_calls: int = 0
    raw_planner_llm_calls: int = 0
    deterministic_calls: int = 0
    failures: list[dict[str, Any]] = Field(default_factory=list)


def load_capability_eval_pack(path: Path) -> CapabilityEvalPack:
    """Load capability eval pack for the surrounding runtime workflow.

    Inputs:
        Receives path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.eval.load_capability_eval_pack.
    """
    return CapabilityEvalPack.model_validate_json(path.read_text())


def load_capability_eval_packs(directory: Path) -> list[CapabilityEvalPack]:
    """Load capability eval packs for the surrounding runtime workflow.

    Inputs:
        Receives directory for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.eval.load_capability_eval_packs.
    """
    return [load_capability_eval_pack(path) for path in sorted(directory.glob("*.json"))]


def ensure_unique_case_ids(packs: list[CapabilityEvalPack]) -> None:
    """Ensure unique case ids for the surrounding runtime workflow.

    Inputs:
        Receives packs for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.eval.ensure_unique_case_ids.
    """
    seen: dict[str, str] = {}
    for pack in packs:
        for case in pack.cases:
            prior_capability = seen.get(case.id)
            if prior_capability is not None:
                raise ValueError(
                    f"duplicate eval case id across packs: {case.id} appears in {prior_capability} and {pack.capability}"
                )
            seen[case.id] = pack.capability
