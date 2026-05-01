"""OpenFABRIC Runtime Module: aor_runtime.runtime.llm_recursion

Purpose:
    Provide shared recursion-budget metadata for typed LLM planning stages.

Responsibilities:
    Cap recursive LLM calls, describe stage metadata, and keep depth accounting consistent across semantic planning and presentation.

Data flow / Interfaces:
    Consumes runtime settings and produces safe metadata dictionaries for LLM prompts and planner telemetry.

Boundaries:
    This module tracks structure-only recursion; it never carries raw tool output, PHI, command streams, or executable payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


LLMStageName = Literal["semantic_frame", "action_plan", "repair_plan", "presentation_intent"]


@dataclass(frozen=True)
class LLMStageRecursionState:
    """Represent one typed LLM recursion call site.

    Inputs:
        Created from stage name, parent id, depth, composition, and schema metadata.

    Returns:
        An immutable state object used to build safe LLM prompt metadata.

    Used by:
        LLMStageRecursionBudget and stage-specific prompt builders.
    """

    stage: LLMStageName
    parent_id: str | None = None
    depth: int = 0
    remaining_depth: int = 0
    composition: str = "single"
    allowed_schema: dict[str, Any] = field(default_factory=dict)

    def prompt_metadata(self) -> dict[str, Any]:
        """Return the safe metadata envelope sent to the LLM.

        Inputs:
            Uses the recursion state fields.

        Returns:
            A JSON-compatible dictionary with no operational payloads.

        Used by:
            Semantic-frame and intelligent-output prompt builders.
        """
        return {
            "stage": self.stage,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "remaining_depth": self.remaining_depth,
            "composition": self.composition,
            "allowed_schema": dict(self.allowed_schema or {}),
        }


class LLMStageRecursionBudget:
    """Enforce a shared maximum recursion depth for typed LLM stages.

    Responsibilities:
        Computes stage-specific depth caps and records maximum observed depth/call count.

    Data flow / Interfaces:
        Created from settings and used by LLM semantic extraction, repair/action planning, and presentation intent selection.

    Used by:
        Runtime LLM stage orchestration and tests.
    """

    def __init__(self, settings: Any, *, stage: LLMStageName, max_depth: int | None = None) -> None:
        """Initialize a recursion budget for one LLM stage.

        Inputs:
            Receives runtime settings, stage name, and optional explicit max depth.

        Returns:
            Initializes the budget with a positive cap.

        Used by:
            Stage-specific LLM callers before making recursive requests.
        """
        global_max = int(getattr(settings, "llm_stage_max_depth", 10) or 10)
        stage_max = int(max_depth if max_depth is not None else global_max)
        self.stage = stage
        self.max_depth = max(0, min(global_max, stage_max))
        self.calls = 0
        self.max_observed_depth = 0

    def state(
        self,
        *,
        depth: int,
        parent_id: str | None = None,
        composition: str = "single",
        allowed_schema: dict[str, Any] | None = None,
    ) -> LLMStageRecursionState:
        """Create metadata for one recursive LLM call.

        Inputs:
            Receives the intended depth, optional parent id, composition, and schema facts.

        Returns:
            LLMStageRecursionState when the call is within budget.

        Used by:
            Prompt builders immediately before invoking the LLM.
        """
        if depth > self.max_depth:
            raise ValueError(f"{self.stage} recursion depth {depth} exceeds maximum {self.max_depth}.")
        self.calls += 1
        self.max_observed_depth = max(self.max_observed_depth, depth)
        return LLMStageRecursionState(
            stage=self.stage,
            parent_id=parent_id,
            depth=depth,
            remaining_depth=max(0, self.max_depth - depth),
            composition=composition,
            allowed_schema=dict(allowed_schema or {}),
        )

    def metadata(self) -> dict[str, Any]:
        """Return safe telemetry for completed recursive LLM work.

        Inputs:
            Uses local budget counters.

        Returns:
            A compact metadata dictionary.

        Used by:
            Planner metadata and tests.
        """
        return {
            "stage": self.stage,
            "calls": self.calls,
            "max_depth": self.max_depth,
            "max_observed_depth": self.max_observed_depth,
        }
