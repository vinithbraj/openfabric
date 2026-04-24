from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.dataflow import collect_step_references


TOKEN_RE = re.compile(r"[a-z0-9_]+")
PLACEHOLDER_TOKEN_RE = re.compile(r"\b(?:name|item|value|row|col|file)\d+\b", re.IGNORECASE)
PYTHON_INPUT_ACCESS_RE = re.compile(r"\binputs(?:\[[^\]]+\]|\.)")
MAX_ALLOWED_STEPS = 12


class PlanningPolicy(BaseModel):
    name: str
    description: str
    rules: list[str]


DEFAULT_POLICIES = [
    PlanningPolicy(
        name="sql_preference",
        description="Prefer SQL for structured data queries",
        rules=[
            "Use sql.query for filtering, aggregation, joins",
            "Do not use python.exec for simple filtering",
        ],
    ),
    PlanningPolicy(
        name="filesystem_preference",
        description="Use filesystem tools for file operations",
        rules=[
            "Use fs.* for file read/write/copy",
            "Do not use shell for file operations if fs exists",
        ],
    ),
    PlanningPolicy(
        name="python_usage",
        description="Restrict python.exec usage",
        rules=[
            "Use python.exec once for simple composition",
            "Combine logic into a single block when possible",
            "Use multiple python.exec steps only if necessary",
            "Avoid unnecessary python.exec usage",
        ],
    ),
    PlanningPolicy(
        name="efficiency",
        description="Minimize steps and transitions",
        rules=[
            "Generate the minimal number of steps",
            "Avoid unnecessary domain switching",
        ],
    ),
]


STRUCTURED_DATA_KEYWORDS = {
    "aggregate",
    "aggregation",
    "column",
    "columns",
    "count",
    "database",
    "databases",
    "filter",
    "filters",
    "group",
    "join",
    "joins",
    "query",
    "queries",
    "record",
    "records",
    "row",
    "rows",
    "select",
    "sum",
    "table",
    "tables",
    "top",
    "where",
}
FILESYSTEM_KEYWORDS = {
    "content",
    "contents",
    "copy",
    "copied",
    "copies",
    "directory",
    "directories",
    "file",
    "files",
    "folder",
    "folders",
    "line",
    "lines",
    "mkdir",
    "move",
    "moved",
    "overwrite",
    "path",
    "paths",
    "read",
    "rename",
    "write",
}
PYTHON_COMPOSITION_KEYWORDS = {
    "bulk",
    "each",
    "every",
    "iterate",
    "iterating",
    "loop",
    "loops",
    "multiple",
}
PYTHON_COMPOSITION_PHRASES = {
    "copy all",
    "for each",
    "for every",
    "move all",
    "per file",
    "rename all",
}


def select_policies(
    goal: str,
    allowed_tools: list[str],
    schema: dict[str, Any] | None = None,
) -> list[PlanningPolicy]:
    selected_names: list[str] = []
    goal_text = str(goal or "").lower()
    goal_tokens = _tokenize(goal_text)
    schema_tokens = _schema_tokens(schema)
    explicit_python_requested = "using python" in goal_text or "with python" in goal_text or "python" in goal_tokens

    if "sql.query" in allowed_tools and not explicit_python_requested and (
        goal_tokens & STRUCTURED_DATA_KEYWORDS or goal_tokens & schema_tokens
    ):
        selected_names.append("sql_preference")

    if any(tool_name.startswith("fs.") for tool_name in allowed_tools) and (
        goal_tokens & FILESYSTEM_KEYWORDS or _looks_path_like(goal_text)
    ):
        selected_names.append("filesystem_preference")

    if "python.exec" in allowed_tools and (
        explicit_python_requested
        or goal_tokens & PYTHON_COMPOSITION_KEYWORDS
        or any(phrase in goal_text for phrase in PYTHON_COMPOSITION_PHRASES)
    ):
        selected_names.append("python_usage")

    selected_names.append("efficiency")
    return [policy for policy in DEFAULT_POLICIES if policy.name in selected_names]


def render_policy_text(policies: list[PlanningPolicy]) -> str:
    return "\n\n".join(
        f"{policy.name}: {policy.description}\nRules:\n- " + "\n- ".join(policy.rules) for policy in policies
    )


def validate_plan(plan: ExecutionPlan, max_allowed_steps: int = MAX_ALLOWED_STEPS) -> None:
    if not plan.steps:
        raise ValueError("Empty plan")

    if len(plan.steps) > max_allowed_steps:
        raise ValueError(f"Plan too complex: {len(plan.steps)} steps exceeds limit {max_allowed_steps}.")

    validate_dataflow(plan)
    _validate_placeholder_outputs(plan)
    _validate_python_inputs_contract(plan)


def validate_plan_efficiency(plan: ExecutionPlan, max_allowed_steps: int = MAX_ALLOWED_STEPS) -> None:
    validate_plan(plan, max_allowed_steps=max_allowed_steps)


def validate_dataflow(plan: ExecutionPlan) -> None:
    outputs: set[str] = set()

    for step in plan.steps:
        referenced_outputs = collect_step_references(step.args)
        declared_inputs = {str(value).strip() for value in step.input if str(value).strip()}
        for dependency in step.input:
            dependency_name = str(dependency).strip()
            if dependency_name not in outputs:
                raise ValueError("Invalid data dependency")
        for dependency_name in referenced_outputs:
            if dependency_name not in outputs:
                raise ValueError("Invalid data dependency")
            if dependency_name not in declared_inputs:
                raise ValueError("Referenced outputs must also be declared in step.input")
        if declared_inputs and not declared_inputs.issubset(referenced_outputs):
            raise ValueError("Declared step.input entries must be referenced in step args")

        output_name = str(step.output or "").strip()
        if output_name:
            if output_name in outputs:
                raise ValueError(f"Duplicate output alias: {output_name}")
            outputs.add(output_name)


def _validate_placeholder_outputs(plan: ExecutionPlan) -> None:
    prior_data_steps = 0
    seen_outputs: set[str] = set()
    for step in plan.steps:
        if step.action == "fs.write":
            content = step.args.get("content")
            if isinstance(content, str) and PLACEHOLDER_TOKEN_RE.search(content) and seen_outputs and not collect_step_references(content):
                raise ValueError("Planner used a placeholder output instead of referenced upstream data.")
            if prior_data_steps and not step.input and not collect_step_references(step.args):
                raise ValueError("Write steps after data-producing steps must reference upstream outputs explicitly.")

        output_name = str(step.output or "").strip()
        if output_name:
            seen_outputs.add(output_name)
        if step.action in {"fs.read", "fs.list", "fs.find", "fs.size", "sql.query", "python.exec"}:
            prior_data_steps += 1


def _validate_python_inputs_contract(plan: ExecutionPlan) -> None:
    for step in plan.steps:
        if step.action != "python.exec":
            continue
        code = str(step.args.get("code", ""))
        inputs_mapping = step.args.get("inputs")
        if PYTHON_INPUT_ACCESS_RE.search(code):
            if not isinstance(inputs_mapping, dict) or not inputs_mapping:
                raise ValueError("python.exec steps that read inputs[...] must declare args.inputs.")


def _tokenize(value: str) -> set[str]:
    tokens = set(TOKEN_RE.findall(value))
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
        elif len(token) > 2:
            expanded.add(f"{token}s")
    return expanded


def _schema_tokens(schema: dict[str, Any] | None) -> set[str]:
    if not isinstance(schema, dict):
        return set()

    tokens: set[str] = set()
    for database in schema.get("databases", []):
        if not isinstance(database, dict):
            continue
        database_name = database.get("name")
        if isinstance(database_name, str):
            tokens.update(_tokenize(database_name.lower()))
        for table in database.get("tables", []):
            if not isinstance(table, dict):
                continue
            table_name = table.get("name")
            if isinstance(table_name, str):
                tokens.update(_tokenize(table_name.lower()))
            for column in table.get("columns", []):
                if not isinstance(column, dict):
                    continue
                column_name = column.get("name")
                if isinstance(column_name, str):
                    tokens.update(_tokenize(column_name.lower()))
    return tokens


def _looks_path_like(goal_text: str) -> bool:
    return "/" in goal_text or "\\" in goal_text or ".txt" in goal_text or ".md" in goal_text
