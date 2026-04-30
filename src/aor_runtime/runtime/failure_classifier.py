"""OpenFABRIC Runtime Module: aor_runtime.runtime.failure_classifier

Purpose:
    Classify runtime/eval failures into stable issue classes for regression reports.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.prompt_suggestions import PromptSuggestion, PromptSuggestionResult
from aor_runtime.runtime.text_extract import extract_quoted_content


AMBIGUOUS_FILE_REFERENCE_RE = re.compile(
    r"\b(?:the\s+file|the\s+meeting\s+notes|the\s+report|the\s+notes|the\s+document)\b",
    re.IGNORECASE,
)
FILE_OPERATION_RE = re.compile(
    r"\b(?:read|open|list|count|search|find|show|return|read line|scan)\b",
    re.IGNORECASE,
)
FILE_NOUN_RE = re.compile(r"\b(?:file|files|folder|directory|line|lines|\.txt|\.md|\*\.txt|\*\.md)\b", re.IGNORECASE)
FILE_AGGREGATE_RE = re.compile(
    r"\b(?:total\s+file\s+size|total\s+size|sum(?:\s+the)?\s+size|how\s+much\s+space|disk\s+space\s+used\s+by|size\s+of\s+all|total\s+bytes|count\s+and\s+total\s+size)\b",
    re.IGNORECASE,
)
FILE_AGGREGATE_FAILURE_RE = re.compile(
    r"(?:python\.exec\s+must\s+not\s+call|tool-call contract|fs\.size|shell\.exec)",
    re.IGNORECASE,
)
VAGUE_OUTPUT_RE = re.compile(r"\b(?:make it nice|format beautifully|format nicely|beautify|pretty format)\b", re.IGNORECASE)
MUTATING_OPERATION_RE = re.compile(
    r"\b(?:scancel|scontrol\s+update|drain|resume|delete|remove|kill|shutdown|poweroff|reboot)\b",
    re.IGNORECASE,
)
COMPOUND_HINT_RE = re.compile(r"\b(?:then|and then|after that|save .* return|write .* return)\b", re.IGNORECASE)
PATH_PREPOSITION_RE = re.compile(r"\b(?:in|under|from|inside|within|at|to)\s+([^\s,;:]+)", re.IGNORECASE)
BARE_FILE_RE = re.compile(r"\b(?!\*\.)[\w.-]+\.[A-Za-z0-9]{1,8}\b")
ABSOLUTE_PATH_RE = re.compile(r"(?:\.\.?/|~/|/|[A-Za-z]:\\)[^\s,;:]+")
DATABASE_HINT_RE = re.compile(r"\b(?:database|table|tables|schema|sql|query|queries|rows|select|join|columns?)\b", re.IGNORECASE)
DATABASE_NAME_RE = re.compile(r"\b(?:database\s+([A-Za-z_][\w-]*)|in\s+([A-Za-z_][\w-]*_db|dicom))\b", re.IGNORECASE)
TABLE_NAME_RE = re.compile(r"\b(?:from|in)\s+([A-Za-z_][\w-]*)\b", re.IGNORECASE)
SQL_TABLE_NOT_FOUND_RE = re.compile(r"(?:UndefinedTable|relation .+ does not exist)", re.IGNORECASE)
SQL_COLUMN_NOT_FOUND_RE = re.compile(
    r"(?:UndefinedColumn|column .+ does not exist|SQL references unknown column)",
    re.IGNORECASE,
)
SQL_AMBIGUOUS_RE = re.compile(r"\bambiguous\s+(?:column|table|alias)\b", re.IGNORECASE)
COMMAND_NOT_FOUND_RE = re.compile(
    r"(?:command not found|not installed|tool not found|executable file not found|no such file or directory)",
    re.IGNORECASE,
)
SLURM_HINT_RE = re.compile(
    r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|scheduler|accounting|partition|partitions|node|nodes|queue|cluster|gpu|gpus|gres|jobs?)\b",
    re.IGNORECASE,
)
GENERIC_PATH_TOKENS = {
    "the",
    "this",
    "that",
    "file",
    "files",
    "folder",
    "directory",
    "report",
    "notes",
    "meeting",
    "database",
    "table",
    "json",
    "csv",
    "text",
}


def classify_failure(
    goal: str,
    error: Exception | None = None,
    plan: ExecutionPlan | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Classify failure for the surrounding runtime workflow.

    Inputs:
        Receives goal, error, plan, metadata for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier.classify_failure.
    """
    text = str(goal or "").strip()
    lowered_goal = text.lower()
    metadata_payload = dict(metadata or {})
    detail = _metadata_text(metadata_payload)
    error_text = str(error) if error is not None else ""
    lowered_error = error_text.lower()
    lowered_detail = detail.lower()

    if bool(metadata_payload.get("llm_calls")) and str(metadata_payload.get("status") or "").lower() == "completed":
        return "llm_fallback_used"

    slurm_error_type = str(metadata_payload.get("slurm_error_type") or "").strip()
    if slurm_error_type:
        return slurm_error_type

    if MUTATING_OPERATION_RE.search(text):
        return "slurm_mutation_unsupported" if _looks_like_slurm_task(text, plan=plan, metadata=metadata_payload) else "unsupported_mutating_operation"

    if VAGUE_OUTPUT_RE.search(text):
        return "unsupported_output_shape"

    if AMBIGUOUS_FILE_REFERENCE_RE.search(text) and not _has_explicit_path(text):
        return "ambiguous_file_reference"

    if FILE_AGGREGATE_RE.search(text) and not _has_explicit_path(text):
        return "missing_file_path"

    if _looks_like_file_task(text) and not _has_explicit_path(text):
        return "missing_file_path"

    if FILE_AGGREGATE_RE.search(text) and (
        FILE_AGGREGATE_FAILURE_RE.search(lowered_error) or FILE_AGGREGATE_FAILURE_RE.search(lowered_detail)
    ):
        return "file_aggregate_not_matched"

    if _looks_like_sql_task(text, plan=plan, metadata=metadata_payload):
        sql_error_class = str(metadata_payload.get("sql_error_class") or "").strip()
        if sql_error_class:
            return sql_error_class
        if str(metadata_payload.get("error_source") or "").strip().lower() == "planner" and str(
            metadata_payload.get("error_kind") or ""
        ).strip().lower() in {"invalid_action_plan", "malformed_json"}:
            return "sql_generation_failed"
        if SQL_TABLE_NOT_FOUND_RE.search(error_text) or SQL_TABLE_NOT_FOUND_RE.search(detail):
            return "sql_table_not_found"
        if SQL_COLUMN_NOT_FOUND_RE.search(error_text) or SQL_COLUMN_NOT_FOUND_RE.search(detail):
            return "sql_column_not_found"
        if SQL_AMBIGUOUS_RE.search(error_text) or SQL_AMBIGUOUS_RE.search(detail):
            return "sql_ambiguous_column"
        if "read-only validation" in lowered_error or "read-only validation" in lowered_detail:
            return "sql_readonly_validation_failed"
        if "schema is unavailable" in lowered_error or "schema unavailable" in lowered_detail:
            return "sql_schema_unavailable"
        if not _has_explicit_database(text):
            return "ambiguous_database"

    if _looks_like_sql_task(text, plan=plan, metadata=metadata_payload) and not _has_explicit_database(text):
        return "ambiguous_database"

    if _looks_like_slurm_task(text, plan=plan, metadata=metadata_payload):
        if metadata_payload.get("slurm_requests_missing"):
            return "slurm_request_uncovered"
        if metadata_payload.get("slurm_constraints_missing"):
            return "slurm_constraint_uncovered"
        if COMMAND_NOT_FOUND_RE.search(lowered_error) or COMMAND_NOT_FOUND_RE.search(lowered_detail):
            return "slurm_tool_unavailable"

    if COMMAND_NOT_FOUND_RE.search(lowered_error) or COMMAND_NOT_FOUND_RE.search(lowered_detail):
        return "tool_unavailable"

    error_kind = str(metadata_payload.get("error_kind") or "").strip().lower()
    reason = str(metadata_payload.get("reason") or "").strip().lower()
    failed_step = str(metadata_payload.get("failed_step") or "").strip().lower()

    if (reason == "validation_failed" or "validation failed" in lowered_error or "validation failed" in lowered_detail) and (
        COMPOUND_HINT_RE.search(text) or _plan_has_multi_step_actions(plan)
    ):
        return "unsupported_compound_task"

    if reason == "validation_failed" or "validation failed" in lowered_error or "validation failed" in lowered_detail:
        return "validation_failure"

    if error_kind == "configuration" and failed_step == "shell.exec":
        return "tool_unavailable"

    if reason == "tool_execution_failed" or failed_step or error_kind:
        return "execution_failure"

    return "unknown"


def generate_prompt_suggestions(goal: str, error_type: str, context: dict[str, Any] | None = None) -> PromptSuggestionResult:
    """Generate prompt suggestions for the surrounding runtime workflow.

    Inputs:
        Receives goal, error_type, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier.generate_prompt_suggestions.
    """
    context_payload = dict(context or {})
    workspace_root = _workspace_root(context_payload)
    outputs_dir = _outputs_dir(context_payload)
    error_detail = str(context_payload.get("error_detail") or context_payload.get("error") or "").strip()

    if error_type == "ambiguous_file_reference":
        filename = _infer_named_file(goal)
        search_term = _infer_search_term(goal) or "agenda"
        suggestions = [
            PromptSuggestion(
                title="Use an explicit file path",
                suggested_prompt=f"Read line 2 from {workspace_root / filename} and return only the line.",
                reason="The request names a file conceptually but does not provide a concrete path.",
            ),
            PromptSuggestion(
                title="Find the file first",
                suggested_prompt=f"Find files named {filename} under {workspace_root}.",
                reason="A discovery step can identify the exact file path before reading it.",
            ),
            PromptSuggestion(
                title="Search by content",
                suggested_prompt=f"Search {workspace_root} for files containing '{search_term}' and return matching filenames.",
                reason="If the file name is uncertain, searching by content is more precise.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="ambiguous_file_reference",
            message="I could not determine which file you meant.",
            suggestions=suggestions,
        )

    if error_type == "missing_file_path":
        extension = ".md" if ".md" in goal.lower() or "markdown" in goal.lower() else ".txt"
        if FILE_AGGREGATE_RE.search(goal):
            aggregate_extension = _infer_aggregate_extension(goal) or ".mp4"
            suggestions = [
                PromptSuggestion(
                    title="Return bytes only",
                    suggested_prompt=f"Calculate total file size of *{aggregate_extension} files under {workspace_root} and return bytes only.",
                    reason="Aggregate file-size prompts need an explicit root path to stay deterministic.",
                ),
                PromptSuggestion(
                    title="Ask for a JSON summary",
                    suggested_prompt=f"Count and total size of {aggregate_extension} files under {workspace_root} as JSON.",
                    reason="A structured JSON form makes file-count and size aggregation explicit.",
                ),
                PromptSuggestion(
                    title="Restrict to the top level",
                    suggested_prompt=f"Total size of top-level {aggregate_extension} files in {workspace_root}.",
                    reason="If you only want direct children, say so explicitly to avoid recursive matching.",
                ),
            ]
            return PromptSuggestionResult(
                error_type="missing_file_path",
                message="I need an explicit folder path for that file-size aggregation request.",
                suggestions=suggestions,
            )
        suggestions = [
            PromptSuggestion(
                title="Provide a concrete folder",
                suggested_prompt=f"Count top-level {extension} files in {workspace_root} and return the count only.",
                reason="File tasks are deterministic when the target folder is explicit.",
            ),
            PromptSuggestion(
                title="Use a recursive list form",
                suggested_prompt=f"List {extension} files under {workspace_root} recursively as JSON.",
                reason="Recursive file prompts work best when both the folder and output shape are explicit.",
            ),
        ]
        needle = _infer_search_term(goal)
        if needle:
            suggestions.append(
                PromptSuggestion(
                    title="Search by content with an explicit root",
                    suggested_prompt=f"Find {extension} files containing '{needle}' under {workspace_root} and return matching filenames.",
                    reason="Adding a search root makes content search deterministic.",
                )
            )
        return PromptSuggestionResult(
            error_type="missing_file_path",
            message="I need an explicit file or folder path for that request.",
            suggestions=suggestions,
        )

    if error_type == "file_aggregate_not_matched":
        aggregate_extension = _infer_aggregate_extension(goal) or ".mp4"
        suggestions = [
            PromptSuggestion(
                title="Return bytes only",
                suggested_prompt=f"Calculate total file size of *{aggregate_extension} files under {workspace_root} and return bytes only.",
                reason="That request is closest to the deterministic filesystem aggregate path.",
            ),
            PromptSuggestion(
                title="Ask for JSON count and size",
                suggested_prompt=f"Count and total size of {aggregate_extension} files under {workspace_root} as JSON.",
                reason="A structured summary avoids planner ambiguity around aggregation and formatting.",
            ),
            PromptSuggestion(
                title="Limit matching to the top level",
                suggested_prompt=f"Total size of top-level {aggregate_extension} files in {workspace_root}.",
                reason="Top-level wording keeps non-recursive aggregation deterministic.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="file_aggregate_not_matched",
            message="I could not map that file-size aggregation request to a supported deterministic form.",
            suggestions=suggestions,
        )

    if error_type == "ambiguous_database":
        table = _infer_table_name(goal) or "patients"
        database = _infer_database_name(goal) or "database_name"
        column = _infer_column_name(goal, table)
        suggestions = [
            PromptSuggestion(
                title="Make the database explicit",
                suggested_prompt=f"Count rows in {table} from database {database} and return the count only.",
                reason="SQL tasks are deterministic when the database and table are named directly.",
            ),
            PromptSuggestion(
                title="Ask for one column as CSV",
                suggested_prompt=f"Query {column} from {table} in {database} as CSV.",
                reason="Explicit columns and output formats reduce planner ambiguity.",
            ),
            PromptSuggestion(
                title="Ask for JSON rows",
                suggested_prompt=f"Query {column} from {table} in {database} as JSON with rows.",
                reason="Structured row output is supported when the query shape is explicit.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="ambiguous_database",
            message="I could not determine the exact database query shape you wanted.",
            suggestions=suggestions,
        )

    if error_type in {
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
    }:
        database = _infer_database_name(goal) or "dicom"
        table = _infer_sql_relation_from_context(context_payload) or _infer_table_name(goal) or "table_name"
        suggestions = [
            PromptSuggestion(
                title="List SQL tables",
                suggested_prompt=f"List all tables in {database}.",
                reason="The schema catalog is the source of truth for available tables.",
            ),
            PromptSuggestion(
                title="Describe the table",
                suggested_prompt=f"Describe table {table} in {database}.",
                reason="Column names must match the database schema exactly, including case.",
            ),
            PromptSuggestion(
                title="Count rows explicitly",
                suggested_prompt=f"Count rows in {table} from {database}.",
                reason="A direct table count is a safe read-only SQL query.",
            ),
        ]
        message = {
            "sql_table_not_found": "The SQL query referenced a table that was not found.",
            "sql_column_not_found": "The SQL query referenced a column that was not found.",
            "sql_ambiguous_table": "The SQL request matched more than one table.",
            "sql_ambiguous_column": "The SQL request matched more than one column.",
            "sql_constraint_unresolved": "The SQL request contained constraints that could not be resolved against the schema.",
            "sql_constraint_uncovered": "The generated SQL did not cover every requested constraint.",
            "sql_projection_unresolved": "The SQL request contained projections that could not be resolved against the schema.",
            "sql_projection_uncovered": "The generated SQL did not cover every requested projection.",
            "sql_generation_failed": "The SQL generator could not produce a safe query for that request.",
            "sql_readonly_validation_failed": "The SQL query was rejected by read-only safety validation.",
            "sql_schema_unavailable": "The SQL schema catalog could not be loaded.",
        }.get(error_type, "The SQL request could not be completed safely.")
        return PromptSuggestionResult(error_type=error_type, message=message, suggestions=suggestions)

    if error_type in {
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
    }:
        suggestions = [
            PromptSuggestion(
                title="Show queue",
                suggested_prompt="Show SLURM queue as JSON.",
                reason="Queue inspection is read-only and deterministic.",
            ),
            PromptSuggestion(
                title="Count queue states",
                suggested_prompt="Count running and pending SLURM jobs.",
                reason="This covers two explicit queue facts without mutation.",
            ),
            PromptSuggestion(
                title="Inspect nodes",
                suggested_prompt="Show problematic SLURM nodes.",
                reason="Problematic node inspection uses read-only node status.",
            ),
            PromptSuggestion(
                title="Summarize health",
                suggested_prompt="Summarize queue, node, and GPU status.",
                reason="A compound health summary can cover several cluster facts together.",
            ),
        ]
        message = {
            "slurm_request_unresolved": "The SLURM request could not be resolved into safe read-only intents.",
            "slurm_request_uncovered": "The SLURM plan did not cover every requested fact.",
            "slurm_constraint_unresolved": "A SLURM filter or constraint could not be resolved safely.",
            "slurm_constraint_uncovered": "A SLURM filter or constraint was not represented in the plan.",
            "slurm_tool_unavailable": "A required SLURM inspection tool appears unavailable.",
            "slurm_accounting_unavailable": "SLURM accounting appears unavailable or inaccessible.",
            "slurmdbd_unavailable": "SLURMDBD appears unavailable or inaccessible.",
            "slurm_mutation_unsupported": "I can inspect SLURM, but I cannot cancel, drain, submit, or modify it.",
            "slurm_ambiguous_request": "The SLURM request is ambiguous.",
            "slurm_llm_intent_rejected": "The SLURM LLM intent output was rejected by safety or coverage validation.",
        }.get(error_type, "The SLURM request could not be completed safely.")
        return PromptSuggestionResult(error_type=error_type, message=message, suggestions=suggestions)

    if error_type == "unsupported_compound_task":
        suggestions = [
            PromptSuggestion(
                title="Split the task into the first supported step",
                suggested_prompt=f"First list top-level .txt files in {workspace_root} as CSV.",
                reason="Breaking the workflow into smaller deterministic prompts avoids unsupported multi-step phrasing.",
            ),
            PromptSuggestion(
                title="Then save the intermediate result",
                suggested_prompt=f"Then write that result to {outputs_dir / 'files.csv'} and return the file contents.",
                reason="Save-and-return flows are more reliable when expressed as a follow-up prompt.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="unsupported_compound_task",
            message="That request combines steps in a way the deterministic path does not handle well yet.",
            suggestions=suggestions,
        )

    if error_type == "unsupported_mutating_operation":
        suggestions = [
            PromptSuggestion(
                title="Use a non-mutating status request",
                suggested_prompt="Show running SLURM jobs.",
                reason="Inspection prompts are safer and supported before admin-style mutations.",
            ),
            PromptSuggestion(
                title="Inspect node state",
                suggested_prompt="Show SLURM node status as JSON.",
                reason="Status queries can usually answer the underlying operational question without mutation.",
            ),
            PromptSuggestion(
                title="Summarize queue health",
                suggested_prompt="Summarize pending jobs by partition.",
                reason="Queue summaries are deterministic and avoid administrative side effects.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="unsupported_mutating_operation",
            message="I can suggest a safer non-mutating form for that request.",
            suggestions=suggestions,
        )

    if error_type == "unsupported_output_shape":
        suggestions = [
            PromptSuggestion(
                title="Request CSV directly",
                suggested_prompt="Return the result as CSV only.",
                reason="The runtime supports a fixed set of deterministic output shapes.",
            ),
            PromptSuggestion(
                title="Request JSON directly",
                suggested_prompt="Return the result as JSON with matches.",
                reason="JSON output is reliable when the expected wrapper shape is explicit.",
            ),
            PromptSuggestion(
                title="Request a count directly",
                suggested_prompt="Return the count only.",
                reason="Count-only requests remove formatting ambiguity.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="unsupported_output_shape",
            message="I could not determine the output shape you wanted.",
            suggestions=suggestions,
        )

    if error_type == "tool_unavailable":
        suggestions = [
            PromptSuggestion(
                title="Prefer a native filesystem search",
                suggested_prompt=f"Use filesystem search instead of shell: find .txt files containing cinnamon under {workspace_root}.",
                reason="A native filesystem tool avoids dependence on missing shell utilities.",
            ),
            PromptSuggestion(
                title="Check installation first",
                suggested_prompt="Check whether the required tool is installed and retry.",
                reason="If the task depends on a missing binary, installation or a different tool path is required.",
            ),
        ]
        if error_detail:
            suggestions.append(
                PromptSuggestion(
                    title="Ask for an explicit shell fallback only if needed",
                    suggested_prompt="Using shell, run a command that is available on this machine and return stdout only.",
                    reason=f"The previous attempt appears to depend on an unavailable tool: {error_detail}",
                )
            )
        return PromptSuggestionResult(
            error_type="tool_unavailable",
            message="A required tool or binary appears to be unavailable.",
            suggestions=suggestions,
        )

    if error_type == "llm_fallback_used":
        suggestions = _llm_fallback_suggestions(goal, workspace_root, outputs_dir)
        return PromptSuggestionResult(
            error_type="llm_fallback_used",
            message="This request worked, but it fell back to LLM planning instead of using a deterministic form.",
            suggestions=suggestions,
        )

    if error_type == "validation_failure":
        suggestions = [
            PromptSuggestion(
                title="Make the request more explicit",
                suggested_prompt=f"List .txt files under {workspace_root} recursively as JSON.",
                reason="Validation failures are often caused by under-specified tool or output expectations.",
            ),
            PromptSuggestion(
                title="Specify output mode directly",
                suggested_prompt="Return the result as CSV only.",
                reason="Explicit output modes reduce plan/validation ambiguity.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="validation_failure",
            message="The request needs a more explicit supported form.",
            suggestions=suggestions,
        )

    if error_type == "execution_failure":
        suggestions = [
            PromptSuggestion(
                title="Reduce the request to one explicit step",
                suggested_prompt=f"List top-level .txt files in {workspace_root} as text.",
                reason="Execution failures are easier to diagnose when the task is narrowed to a single supported action.",
            ),
            PromptSuggestion(
                title="Ask for a concrete format",
                suggested_prompt="Return the result as JSON with matches.",
                reason="Structured output makes retries and verification more predictable.",
            ),
        ]
        return PromptSuggestionResult(
            error_type="execution_failure",
            message="The request executed unsuccessfully in its current form.",
            suggestions=suggestions,
        )

    suggestions = [
        PromptSuggestion(
            title="Add explicit inputs",
            suggested_prompt=f"Read line 2 from {workspace_root / 'meeting_notes.txt'} and return only the line.",
            reason="Concrete file paths and output constraints improve determinism.",
        ),
        PromptSuggestion(
            title="Make structured data explicit",
            suggested_prompt="Count rows in patients from database dicom and return the count only.",
            reason="Database tasks work best when the database, table, and output shape are named directly.",
        ),
    ]
    return PromptSuggestionResult(
        error_type="unknown",
        message="I could not confidently rewrite that request, but these supported forms are close.",
        suggestions=suggestions,
    )


def _looks_like_file_task(goal: str) -> bool:
    """Handle the internal looks like file task helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._looks_like_file_task.
    """
    return bool(FILE_OPERATION_RE.search(goal) and FILE_NOUN_RE.search(goal))


def _infer_aggregate_extension(goal: str) -> str | None:
    """Handle the internal infer aggregate extension helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_aggregate_extension.
    """
    match = re.search(r"\*\.(?P<ext>[A-Za-z0-9]+)\b", goal)
    if match is not None:
        return f".{match.group('ext').lower()}"
    match = re.search(r"\.(?P<ext>[A-Za-z0-9]+)\s+files?\b", goal, re.IGNORECASE)
    if match is not None:
        return f".{match.group('ext').lower()}"
    match = re.search(r"\b(?P<ext>[A-Za-z0-9]{2,8})\s+files?\b", goal, re.IGNORECASE)
    if match is not None:
        ext = match.group("ext").lower()
        if ext not in {"all", "file", "files", "total", "size", "top", "level"}:
            return f".{ext}"
    return None


def _has_explicit_path(goal: str) -> bool:
    """Handle the internal has explicit path helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._has_explicit_path.
    """
    if ABSOLUTE_PATH_RE.search(goal) or BARE_FILE_RE.search(goal):
        return True
    for match in PATH_PREPOSITION_RE.finditer(goal):
        candidate = _clean_token(match.group(1))
        if not candidate:
            continue
        if candidate.lower() in GENERIC_PATH_TOKENS:
            continue
        if candidate.startswith("*."):
            continue
        if "/" in candidate or "\\" in candidate or candidate.startswith(".") or "." in candidate:
            return True
        if re.fullmatch(r"[A-Za-z0-9_.-]+", candidate):
            return True
    return False


def _looks_like_sql_task(goal: str, *, plan: ExecutionPlan | None, metadata: dict[str, Any]) -> bool:
    """Handle the internal looks like sql task helper path for this module.

    Inputs:
        Receives goal, plan, metadata for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._looks_like_sql_task.
    """
    if plan is not None and any(step.action == "sql.query" for step in plan.steps):
        return True
    if str(metadata.get("error_source") or "").strip().lower() == "sql":
        return True
    if str(metadata.get("failed_step") or "").strip().lower() == "sql.query":
        return True
    lowered = goal.lower()
    if DATABASE_HINT_RE.search(goal):
        return True
    return any(token in lowered for token in ("patient", "patients", "study", "studies", "dicom"))


def _looks_like_slurm_task(goal: str, *, plan: ExecutionPlan | None, metadata: dict[str, Any]) -> bool:
    """Handle the internal looks like slurm task helper path for this module.

    Inputs:
        Receives goal, plan, metadata for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._looks_like_slurm_task.
    """
    if plan is not None and any(step.action.startswith("slurm.") for step in plan.steps):
        return True
    if str(metadata.get("capability") or metadata.get("capability_pack") or "").strip().lower() == "slurm":
        return True
    if any(key.startswith("slurm_") for key in metadata):
        return True
    return SLURM_HINT_RE.search(goal) is not None


def _has_explicit_database(goal: str) -> bool:
    """Handle the internal has explicit database helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._has_explicit_database.
    """
    return DATABASE_NAME_RE.search(goal) is not None


def _plan_has_multi_step_actions(plan: ExecutionPlan | None) -> bool:
    """Handle the internal plan has multi step actions helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._plan_has_multi_step_actions.
    """
    return plan is not None and len(plan.steps) > 1


def _metadata_text(metadata: dict[str, Any]) -> str:
    """Handle the internal metadata text helper path for this module.

    Inputs:
        Receives metadata for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._metadata_text.
    """
    parts: list[str] = []
    for key in ("error_source", "error_kind", "error_target", "error_detail", "reason", "failed_step"):
        value = metadata.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _workspace_root(context: dict[str, Any]) -> Path:
    """Handle the internal workspace root helper path for this module.

    Inputs:
        Receives context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._workspace_root.
    """
    value = str(context.get("workspace_root") or "/tmp/work").strip() or "/tmp/work"
    return Path(value)


def _outputs_dir(context: dict[str, Any]) -> Path:
    """Handle the internal outputs dir helper path for this module.

    Inputs:
        Receives context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._outputs_dir.
    """
    value = str(context.get("outputs_dir") or (_workspace_root(context) / "outputs")).strip()
    return Path(value)


def _infer_named_file(goal: str) -> str:
    """Handle the internal infer named file helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_named_file.
    """
    lowered = goal.lower()
    if "meeting notes" in lowered:
        return "meeting_notes.txt"
    if "report" in lowered:
        return "report.txt"
    if "notes" in lowered:
        return "notes.txt"
    if "studies" in lowered:
        return "studies.txt"
    match = BARE_FILE_RE.search(goal)
    if match is not None:
        return match.group(0)
    return "meeting_notes.txt"


def _infer_search_term(goal: str) -> str | None:
    """Handle the internal infer search term helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_search_term.
    """
    quoted = extract_quoted_content(goal)
    if quoted:
        return quoted
    lowered = goal.lower()
    for token in ("agenda", "cinnamon", "garden", "meeting", "report", "study"):
        if token in lowered:
            return token
    return None


def _infer_table_name(goal: str) -> str | None:
    """Handle the internal infer table name helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_table_name.
    """
    lowered = goal.lower()
    if "studies" in lowered or "study" in lowered:
        return "study"
    if "patients" in lowered or "patient" in lowered:
        return "patients"
    match = TABLE_NAME_RE.search(goal)
    if match is not None:
        candidate = _clean_token(match.group(1))
        if candidate and candidate.lower() not in {"database", "table", "json", "csv"}:
            return candidate
    return None


def _infer_database_name(goal: str) -> str | None:
    """Handle the internal infer database name helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_database_name.
    """
    match = DATABASE_NAME_RE.search(goal)
    if match is None:
        return None
    for group in match.groups():
        candidate = _clean_token(group or "")
        if candidate:
            return candidate
    return None


def _infer_column_name(goal: str, table: str) -> str:
    """Handle the internal infer column name helper path for this module.

    Inputs:
        Receives goal, table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_column_name.
    """
    lowered = goal.lower()
    if "name" in lowered:
        return "name"
    if table == "study":
        return "StudyInstanceUID"
    if table in {"patient", "patients"}:
        return "PatientName"
    return "name"


def _infer_sql_relation_from_context(context: dict[str, Any]) -> str | None:
    """Handle the internal infer sql relation from context helper path for this module.

    Inputs:
        Receives context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._infer_sql_relation_from_context.
    """
    for key in ("sql_table", "table", "relation"):
        value = str(context.get(key) or "").strip()
        if value:
            return value
    step = context.get("step")
    if isinstance(step, dict):
        args = step.get("args")
        if isinstance(args, dict):
            query = str(args.get("query") or "")
            match = re.search(r"\bfrom\s+([^\s,;]+)", query, re.IGNORECASE)
            if match is not None:
                return match.group(1)
    return None


def _llm_fallback_suggestions(goal: str, workspace_root: Path, outputs_dir: Path) -> list[PromptSuggestion]:
    """Handle the internal llm fallback suggestions helper path for this module.

    Inputs:
        Receives goal, workspace_root, outputs_dir for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._llm_fallback_suggestions.
    """
    lowered = goal.lower()
    if _looks_like_sql_task(goal, plan=None, metadata={}):
        table = _infer_table_name(goal) or "table_name"
        database = _infer_database_name(goal) or "database_name"
        column = _infer_column_name(goal, table)
        return [
            PromptSuggestion(
                title="Use an explicit SQL count form",
                suggested_prompt=f"Count rows in {table} from database {database} and return the count only.",
                reason="Naming the database and table keeps the task on the deterministic SQL path.",
            ),
            PromptSuggestion(
                title="Use an explicit SQL select form",
                suggested_prompt=f"Query {column} from {table} in {database} as CSV.",
                reason="Explicit column and output mode requests map cleanly to the SQL capability.",
            ),
            PromptSuggestion(
                title="Save the SQL output deterministically",
                suggested_prompt=f"Query {column} from {table} in {database} as CSV, save the result to {outputs_dir / 'result.csv'}, and return the file contents.",
                reason="Explicit save-and-return wording avoids planner-only formatting paths.",
            ),
        ]

    if "shell" in lowered or "command" in lowered or "printf" in lowered:
        return [
            PromptSuggestion(
                title="Make the shell intent explicit",
                suggested_prompt="Using shell, print alpha and beta on separate lines and return CSV.",
                reason="Explicit shell wording keeps the task on the deterministic shell capability.",
            ),
            PromptSuggestion(
                title="Ask for stdout only",
                suggested_prompt="Using shell, run `hostname` and return stdout only.",
                reason="Explicit shell output requests are less ambiguous than generic command prompts.",
            ),
        ]

    if any(token in lowered for token in ("write", "save", "create")):
        quoted = extract_quoted_content(goal) or "hello"
        return [
            PromptSuggestion(
                title="Use exact quoted content",
                suggested_prompt=f"Write the exact text '{quoted}' to {outputs_dir / 'result.txt'} and return it.",
                reason="Quoted write-and-return prompts map cleanly to deterministic filesystem intents.",
            ),
            PromptSuggestion(
                title="Separate save from other work",
                suggested_prompt=f"Save '{quoted}' to {outputs_dir / 'result.txt'} and return the saved file contents only.",
                reason="Explicit save-and-return phrasing avoids planner-only output shaping.",
            ),
        ]

    if _looks_like_file_task(goal):
        needle = _infer_search_term(goal)
        if "line" in lowered:
            filename = _infer_named_file(goal)
            return [
                PromptSuggestion(
                    title="Use an explicit file path",
                    suggested_prompt=f"Read line 2 from {workspace_root / filename} and return only the line.",
                    reason="Explicit file paths keep read requests on the deterministic filesystem path.",
                ),
                PromptSuggestion(
                    title="Find the file before reading",
                    suggested_prompt=f"Find files named {filename} under {workspace_root}.",
                    reason="When the file name is conceptual, a discovery step is more reliable.",
                ),
            ]
        if "count" in lowered:
            return [
                PromptSuggestion(
                    title="Use an explicit folder and extension",
                    suggested_prompt=f"Count top-level .txt files in {workspace_root} and return the count only.",
                    reason="Explicit folder and output wording maps directly to deterministic file counting.",
                ),
                PromptSuggestion(
                    title="Use a recursive file count form",
                    suggested_prompt=f"Count .txt files under {workspace_root} recursively and return the count only.",
                    reason="Recursive count prompts are supported when the search root is explicit.",
                ),
            ]
        if "search" in lowered or "find" in lowered:
            term = needle or "cinnamon"
            return [
                PromptSuggestion(
                    title="Use explicit content-search wording",
                    suggested_prompt=f"Find .txt files containing '{term}' under {workspace_root} and return matching filenames.",
                    reason="Explicit path, pattern, and output wording keep file search deterministic.",
                ),
                PromptSuggestion(
                    title="Ask for JSON matches directly",
                    suggested_prompt=f"Find .txt files containing '{term}' under {workspace_root} and return them as JSON only.",
                    reason="Structured output removes ambiguity from search-result formatting.",
                ),
            ]
        return [
            PromptSuggestion(
                title="Use an explicit list form",
                suggested_prompt=f"List top-level .txt files in {workspace_root} as CSV.",
                reason="Explicit list prompts are more likely to match the deterministic filesystem capability.",
            ),
            PromptSuggestion(
                title="Use explicit JSON output",
                suggested_prompt=f"List .txt files under {workspace_root} recursively as JSON.",
                reason="Adding both the folder and output mode avoids planner fallback.",
            ),
        ]

    return [
        PromptSuggestion(
            title="Make the request concrete",
            suggested_prompt=f"List .txt files under {workspace_root} recursively as JSON.",
            reason="Explicit paths and output modes help the runtime stay deterministic.",
        ),
        PromptSuggestion(
            title="Use an explicit SQL form when querying data",
            suggested_prompt="Count rows in patients from database dicom and return the count only.",
            reason="Explicit database phrasing avoids ambiguous natural-language query requests.",
        ),
    ]


def _clean_token(value: str) -> str:
    """Handle the internal clean token helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.failure_classifier._clean_token.
    """
    return str(value or "").strip().strip("\"'()[]{}")
