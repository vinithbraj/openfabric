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
    title: str
    suggested_prompt: str
    reason: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("title", "suggested_prompt", "reason")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text


class PromptSuggestionResult(BaseModel):
    error_type: PromptSuggestionErrorType
    message: str
    suggestions: list[PromptSuggestion] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("message must be a non-empty string")
        return text

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "failure_type": self.error_type,
            "prompt_suggestions": [suggestion.model_dump() for suggestion in self.suggestions],
            "suggestion_count": len(self.suggestions),
        }


def render_prompt_suggestion_section(result: PromptSuggestionResult) -> str:
    if not result.suggestions:
        return ""
    lines = ["Suggested prompts:"]
    for index, suggestion in enumerate(result.suggestions, start=1):
        lines.append(f"{index}. {suggestion.suggested_prompt}")
    return "\n".join(lines)


def append_prompt_suggestions(message: str, result: PromptSuggestionResult) -> str:
    base = str(message or "").strip()
    section = render_prompt_suggestion_section(result)
    if not section:
        return base
    if "Suggested prompts:" in base:
        return base
    if not base:
        return section
    return f"{base}\n\n{section}"
