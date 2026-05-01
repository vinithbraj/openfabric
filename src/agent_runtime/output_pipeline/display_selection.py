"""LLM-guided display strategy selection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import model_json_schema

from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import DisplayPlan
from agent_runtime.llm.proposals import DisplayPlanProposal
from agent_runtime.llm.structured_call import structured_call


class DisplaySelectionInput(BaseModel):
    """Safe LLM input for choosing a display plan."""

    model_config = ConfigDict(extra="forbid")

    original_prompt: str
    dag_summary: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    safe_previews: list[dict[str, Any]] = Field(default_factory=list)
    available_display_types: list[str] = Field(default_factory=list)


def _build_selection_prompt(selection_input: DisplaySelectionInput) -> str:
    """Build the strict JSON-only prompt for display-plan selection."""

    schema = model_json_schema(DisplayPlanProposal)
    return "\n".join(
        [
            "You are selecting a safe display plan for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not invent results.",
            "Do not summarize unavailable data.",
            "Use only the original prompt, DAG summary, result summary, and safe previews.",
            "Never assume data exists outside the provided result refs and safe previews.",
            "If multiple presentations are useful, choose display_type='multi_section' and use sections with explicit source refs.",
            "Allowed section keys include title, display_type, source_node_id, source_data_ref, and parameters.",
            "Display selection input:",
            str(selection_input.model_dump()),
            "JSON schema:",
            str(schema),
        ]
    )


def _validate_display_plan_references(
    display_plan: DisplayPlan,
    selection_input: DisplaySelectionInput,
) -> DisplayPlan:
    """Validate display types and section refs against safe available sources."""

    allowed_types = set(selection_input.available_display_types)
    if display_plan.display_type not in allowed_types:
        raise ValidationError(
            f"display type {display_plan.display_type!r} is not in the available display types."
        )

    available_node_ids = {
        str(item.get("node_id"))
        for item in selection_input.safe_previews
        if item.get("node_id") is not None
    }
    available_data_refs = {
        str(item.get("data_ref"))
        for item in selection_input.safe_previews
        if item.get("data_ref") is not None
    }

    for section in display_plan.sections:
        section_type = str(section.get("display_type") or display_plan.display_type)
        if section_type not in allowed_types:
            raise ValidationError(
                f"section display type {section_type!r} is not in the available display types."
            )
        source_node_id = section.get("source_node_id")
        if source_node_id is not None and str(source_node_id) not in available_node_ids:
            raise ValidationError(f"display plan references missing node_id: {source_node_id}")
        source_data_ref = section.get("source_data_ref")
        if source_data_ref is not None and str(source_data_ref) not in available_data_refs:
            raise ValidationError(f"display plan references missing data_ref: {source_data_ref}")

    return display_plan


def select_display_plan(selection_input: DisplaySelectionInput, llm_client) -> DisplayPlan:
    """Select a display plan from safe previews through a structured LLM call."""

    prompt = _build_selection_prompt(selection_input)
    proposal = structured_call(llm_client, prompt, DisplayPlanProposal)
    plan = DisplayPlan.model_validate(proposal.model_dump(mode="json"))
    return _validate_display_plan_references(plan, selection_input)


class DisplaySelector:
    """Compatibility selector for older orchestrator code paths."""

    def select(self, selection_input: DisplaySelectionInput) -> DisplayPlan:
        """Return a deterministic fallback display plan when no LLM is used."""

        def _preview_payload(preview: dict[str, Any]) -> dict[str, Any]:
            payload = preview.get("preview")
            return payload if isinstance(payload, dict) else {}

        default_type = (
            "table"
            if any(
                _preview_payload(preview).get("entries")
                or _preview_payload(preview).get("rows")
                or _preview_payload(preview).get("matches")
                or _preview_payload(preview).get("processes")
                or _preview_payload(preview).get("listeners")
                for preview in selection_input.safe_previews
            )
            else "markdown"
        )
        return DisplayPlan(
            display_type=default_type,
            title=selection_input.result_summary.get("title"),
            sections=[],
            constraints={},
            redaction_policy="standard",
        )
