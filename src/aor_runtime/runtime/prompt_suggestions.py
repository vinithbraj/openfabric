"""OpenFABRIC Runtime Module: aor_runtime.runtime.prompt_suggestions

Purpose:
    Generate optional prompt suggestions for debug or compatibility surfaces.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


PromptSuggestionErrorType = Literal[
    "ambiguous_file_reference",
    "missing_file_path",
    "ambiguous_database",
    "unsupported_compound_task",
    "unsupported_mutating_operation",
    "unsupported_output_shape",
    "tool_unavailable",
    "file_aggregate_not_matched",
    "validation_failure",
    "execution_failure",
    "llm_fallback_used",
    "sql_table_not_found",
    "sql_column_not_found",
    "sql_ambiguous_table",
    "sql_ambiguous_column",
    "sql_constraint_unresolved",
    "sql_constraint_uncovered",
    "sql_projection_unresolved",
    "sql_projection_uncovered",
    "sql_generation_failed",
    "sql_readonly_validation_failed",
    "sql_schema_unavailable",
    "slurm_request_unresolved",
    "slurm_request_uncovered",
    "slurm_constraint_unresolved",
    "slurm_constraint_uncovered",
    "slurm_tool_unavailable",
    "slurm_accounting_unavailable",
    "slurmdbd_unavailable",
    "slurm_mutation_unsupported",
    "slurm_ambiguous_request",
    "slurm_llm_intent_rejected",
    "unknown",
]


class PromptSuggestion(BaseModel):
    """Represent prompt suggestion within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PromptSuggestion.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.prompt_suggestions.PromptSuggestion and related tests.
    """
    title: str
    suggested_prompt: str
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("title", "suggested_prompt", "reason")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        """Validate validate text fields invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this PromptSuggestion method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through PromptSuggestion._validate_text_fields calls and related tests.
        """
        text = str(value).strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text


class PromptSuggestionResult(BaseModel):
    """Represent prompt suggestion result within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PromptSuggestionResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.prompt_suggestions.PromptSuggestionResult and related tests.
    """
    error_type: PromptSuggestionErrorType
    message: str
    suggestions: list[PromptSuggestion] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        """Validate validate message invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this PromptSuggestionResult method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through PromptSuggestionResult._validate_message calls and related tests.
        """
        text = str(value).strip()
        if not text:
            raise ValueError("message must be a non-empty string")
        return text

    def metadata_payload(self) -> dict[str, Any]:
        """Metadata payload for PromptSuggestionResult instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PromptSuggestionResult.metadata_payload calls and related tests.
        """
        return {
            "failure_type": self.error_type,
            "prompt_suggestions": [suggestion.model_dump() for suggestion in self.suggestions],
            "suggestion_count": len(self.suggestions),
        }


def render_prompt_suggestion_section(result: PromptSuggestionResult) -> str:
    """Render prompt suggestion section for the surrounding runtime workflow.

    Inputs:
        Receives result for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.prompt_suggestions.render_prompt_suggestion_section.
    """
    if not result.suggestions:
        return ""
    lines = ["Suggested prompts:"]
    for index, suggestion in enumerate(result.suggestions, start=1):
        lines.append(f"{index}. {suggestion.suggested_prompt}")
    return "\n".join(lines)


def append_prompt_suggestions(message: str, result: PromptSuggestionResult) -> str:
    """Append prompt suggestions for the surrounding runtime workflow.

    Inputs:
        Receives message, result for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.prompt_suggestions.append_prompt_suggestions.
    """
    base = str(message or "").strip()
    section = render_prompt_suggestion_section(result)
    if not section:
        return base
    if "Suggested prompts:" in base:
        return base
    if not base:
        return section
    return f"{base}\n\n{section}"
