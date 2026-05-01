"""Helpers for evaluating and selecting among untrusted LLM planning proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

ModelT = TypeVar("ModelT")


@dataclass
class CandidateEvaluation(Generic[ModelT]):
    """One evaluated proposal candidate."""

    proposal: ModelT | None = None
    raw_response: object | None = None
    validation_errors: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    deterministic_normalizations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    capability_compatibility: int = 0
    risk_score: int = 0
    assumption_count: int = 0
    unresolved_count: int = 0
    missing_required_count: int = 0

    @property
    def is_valid(self) -> bool:
        """Return whether the candidate survived deterministic validation."""

        return self.proposal is not None and not self.validation_errors and not self.rejection_reasons


def _selection_key(candidate: CandidateEvaluation[ModelT]) -> tuple[int, int, int, int, float, int, int]:
    """Return the deterministic ranking key for one candidate."""

    return (
        1 if candidate.is_valid else 0,
        candidate.capability_compatibility,
        -candidate.missing_required_count,
        -candidate.risk_score,
        candidate.confidence,
        -candidate.assumption_count,
        -candidate.unresolved_count,
    )


def select_best_candidate(
    candidates: list[CandidateEvaluation[ModelT]],
) -> CandidateEvaluation[ModelT] | None:
    """Return the best candidate by deterministic runtime preference rules."""

    if not candidates:
        return None
    return max(candidates, key=_selection_key)
