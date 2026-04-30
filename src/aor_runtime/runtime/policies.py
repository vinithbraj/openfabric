"""OpenFABRIC Runtime Module: aor_runtime.runtime.policies

Purpose:
    Define runtime policy checks used by validators and execution boundaries.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel

from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep
from aor_runtime.runtime.dataflow import collect_step_references
from aor_runtime.tools.sql import validate_safe_query


TOKEN_RE = re.compile(r"[a-z0-9_]+")
PLACEHOLDER_TOKEN_RE = re.compile(r"\b(?:name|item|value|row|col|file)\d+\b", re.IGNORECASE)
PYTHON_INPUT_ACCESS_RE = re.compile(r"\binputs(?:\[[^\]]+\]|\.)")
COUNT_ONLY_RE = re.compile(r"\b(?:count|how many|number of)\b", re.IGNORECASE)
CSV_ONLY_RE = re.compile(r"\bcsv\b", re.IGNORECASE)
JSON_ONLY_RE = re.compile(r"\bjson\b", re.IGNORECASE)
RETURN_REQUEST_RE = re.compile(r"\b(return|show|provide|list)\b", re.IGNORECASE)
SQL_SELECT_RE = re.compile(r"select\s+(?P<select>.+?)\s+from\s", re.IGNORECASE | re.DOTALL)
SQL_ALIAS_RE = re.compile(r"\bas\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", re.IGNORECASE)
PATH_ARG_KEYS = {"path", "src", "dst"}
MAX_ALLOWED_STEPS = 12
FORBIDDEN_IMPORT_MODULES = {"os", "subprocess"}
ALLOWED_PYTHON_IMPORT_MODULES = {"json", "re"}
FORBIDDEN_PYTHON_NAMES = {"__import__", "compile", "eval", "exec", "open"}
FORBIDDEN_NAME_CALLS = {"system", "popen", "spawn", "fork", "execv", "execve", "execl", "execvp"}
FORBIDDEN_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("shell", "exec"),
    ("sql", "query"),
}
TEXTUAL_RESULT_PATHS = {"content", "csv", "json", "markdown", "text", "value"}


class PlanningPolicy(BaseModel):
    """Represent planning policy within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanningPolicy.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.policies.PlanningPolicy and related tests.
    """
    name: str
    description: str
    rules: list[str]


@dataclass(slots=True)
class PlanViolation:
    """Represent plan violation within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanViolation.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.policies.PlanViolation and related tests.
    """
    tier: Literal["hard", "soft"]
    code: str
    message: str


@dataclass(slots=True)
class PlanViolations:
    """Represent plan violations within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanViolations.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.policies.PlanViolations and related tests.
    """
    hard: list[PlanViolation] = field(default_factory=list)
    soft: list[PlanViolation] = field(default_factory=list)

    def add(self, violation: PlanViolation) -> None:
        """Add for PlanViolations instances.

        Inputs:
            Receives violation for this PlanViolations method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through PlanViolations.add calls and related tests.
        """
        if violation.tier == "hard":
            self.hard.append(violation)
        else:
            self.soft.append(violation)

    def extend(self, violations: list[PlanViolation]) -> None:
        """Extend for PlanViolations instances.

        Inputs:
            Receives violations for this PlanViolations method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through PlanViolations.extend calls and related tests.
        """
        for violation in violations:
            self.add(violation)

    def first(self) -> PlanViolation | None:
        """First for PlanViolations instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanViolations.first calls and related tests.
        """
        if self.hard:
            return self.hard[0]
        if self.soft:
            return self.soft[0]
        return None

    def any(self) -> bool:
        """Any for PlanViolations instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanViolations.any calls and related tests.
        """
        return bool(self.hard or self.soft)


class PlanContractViolation(ValueError):
    """Represent plan contract violation within the OpenFABRIC runtime. It extends ValueError.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanContractViolation.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.policies.PlanContractViolation and related tests.
    """
    def __init__(self, message: str, *, tier: Literal["hard", "soft"], code: str, violations: list[PlanViolation]) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives message, tier, code, violations for this PlanContractViolation method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through PlanContractViolation.__init__ calls and related tests.
        """
        super().__init__(message)
        self.tier = tier
        self.code = code
        self.violations = list(violations)

    @classmethod
    def from_violations(cls, violations: PlanViolations) -> "PlanContractViolation":
        """From violations for PlanContractViolation instances.

        Inputs:
            Receives violations for this PlanContractViolation method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanContractViolation.from_violations calls and related tests.
        """
        first = violations.first()
        if first is None:
            raise ValueError("No plan violations were provided.")
        relevant = violations.hard if violations.hard else violations.soft
        return cls(first.message, tier=first.tier, code=first.code, violations=relevant)

    def as_metadata(self) -> dict[str, Any]:
        """As metadata for PlanContractViolation instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanContractViolation.as_metadata calls and related tests.
        """
        return {
            "contract_violation": True,
            "violation_tier": self.tier,
            "violation_code": self.code,
            "violations": [violation.message for violation in self.violations],
        }


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
    """Select policies for the surrounding runtime workflow.

    Inputs:
        Receives goal, allowed_tools, schema for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.select_policies.
    """
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
    """Render policy text for the surrounding runtime workflow.

    Inputs:
        Receives policies for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.render_policy_text.
    """
    return "\n\n".join(
        f"{policy.name}: {policy.description}\nRules:\n- " + "\n- ".join(policy.rules) for policy in policies
    )


def infer_output_mode(goal: str) -> Literal["json", "csv", "count", "text"]:
    """Infer output mode for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.infer_output_mode.
    """
    goal_text = str(goal or "").lower()
    if JSON_ONLY_RE.search(goal_text):
        return "json"
    if CSV_ONLY_RE.search(goal_text):
        return "csv"
    if COUNT_ONLY_RE.search(goal_text):
        return "count"
    return "text"


def goal_requests_return_value(goal: str) -> bool:
    """Goal requests return value for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.goal_requests_return_value.
    """
    return bool(RETURN_REQUEST_RE.search(str(goal or "")))


def validate_plan(plan: ExecutionPlan, max_allowed_steps: int = MAX_ALLOWED_STEPS, *, goal: str = "") -> None:
    """Validate plan for the surrounding runtime workflow.

    Inputs:
        Receives plan, max_allowed_steps, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.validate_plan.
    """
    validate_plan_contract(plan, goal=goal, max_allowed_steps=max_allowed_steps)


def validate_plan_efficiency(plan: ExecutionPlan, max_allowed_steps: int = MAX_ALLOWED_STEPS, *, goal: str = "") -> None:
    """Validate plan efficiency for the surrounding runtime workflow.

    Inputs:
        Receives plan, max_allowed_steps, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.validate_plan_efficiency.
    """
    validate_plan(plan, max_allowed_steps=max_allowed_steps, goal=goal)


def classify_plan_violations(
    plan: ExecutionPlan,
    *,
    goal: str = "",
    max_allowed_steps: int = MAX_ALLOWED_STEPS,
) -> PlanViolations:
    """Classify plan violations for the surrounding runtime workflow.

    Inputs:
        Receives plan, goal, max_allowed_steps for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.classify_plan_violations.
    """
    violations = PlanViolations()

    if not plan.steps:
        violations.add(PlanViolation("hard", "empty_plan", "Empty plan"))
        return violations

    if len(plan.steps) > max_allowed_steps:
        violations.add(
            PlanViolation(
                "hard",
                "plan_too_complex",
                f"Plan too complex: {len(plan.steps)} steps exceeds limit {max_allowed_steps}.",
            )
        )

    violations.extend(_classify_sql_contract_violations(plan))
    violations.extend(_classify_python_contract_violations(plan))
    violations.extend(_classify_dataflow_violations(plan))
    violations.extend(_classify_path_violations(plan))
    violations.extend(_classify_output_contract_violations(plan, goal))
    return violations


def validate_plan_contract(plan: ExecutionPlan, *, goal: str = "", max_allowed_steps: int = MAX_ALLOWED_STEPS) -> None:
    """Validate plan contract for the surrounding runtime workflow.

    Inputs:
        Receives plan, goal, max_allowed_steps for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.validate_plan_contract.
    """
    violations = classify_plan_violations(plan, goal=goal, max_allowed_steps=max_allowed_steps)
    if violations.any():
        raise PlanContractViolation.from_violations(violations)


def validate_dataflow(plan: ExecutionPlan) -> None:
    """Validate dataflow for the surrounding runtime workflow.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies.validate_dataflow.
    """
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
    """Handle the internal validate placeholder outputs helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._validate_placeholder_outputs.
    """
    prior_data_steps = 0
    seen_outputs: set[str] = set()
    for step in plan.steps:
        if step.action == "fs.write":
            content = step.args.get("content")
            if (
                isinstance(content, str)
                and PLACEHOLDER_TOKEN_RE.search(content)
                and seen_outputs
                and not collect_step_references(content)
            ):
                raise ValueError("Planner used a placeholder output instead of referenced upstream data.")
            if prior_data_steps and not step.input and not collect_step_references(step.args):
                raise ValueError("Write steps after data-producing steps must reference upstream outputs explicitly.")

        output_name = str(step.output or "").strip()
        if output_name:
            seen_outputs.add(output_name)
        if step.action in {"fs.read", "fs.list", "fs.find", "fs.size", "sql.query", "python.exec", "text.format"}:
            prior_data_steps += 1


def _validate_python_inputs_contract(plan: ExecutionPlan) -> None:
    """Handle the internal validate python inputs contract helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._validate_python_inputs_contract.
    """
    for step in plan.steps:
        if step.action != "python.exec":
            continue
        code = str(step.args.get("code", ""))
        inputs_mapping = step.args.get("inputs")
        if PYTHON_INPUT_ACCESS_RE.search(code):
            if not isinstance(inputs_mapping, dict) or not inputs_mapping:
                raise ValueError("python.exec steps that read inputs[...] must declare args.inputs.")


def _classify_sql_contract_violations(plan: ExecutionPlan) -> list[PlanViolation]:
    """Handle the internal classify sql contract violations helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_sql_contract_violations.
    """
    violations: list[PlanViolation] = []
    for step in plan.steps:
        if step.action != "sql.query":
            continue
        query = step.args.get("query")
        if not isinstance(query, str):
            continue
        try:
            validate_safe_query(query)
        except Exception as exc:  # noqa: BLE001
            violations.append(PlanViolation("hard", "unsafe_sql", str(exc)))
    return violations


def _classify_python_contract_violations(plan: ExecutionPlan) -> list[PlanViolation]:
    """Handle the internal classify python contract violations helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_python_contract_violations.
    """
    producer_by_output = {
        str(step.output).strip(): step for step in plan.steps if isinstance(step.output, str) and str(step.output).strip()
    }
    violations: list[PlanViolation] = []
    for step in plan.steps:
        if step.action != "python.exec":
            continue
        code = str(step.args.get("code", ""))
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            violations.append(PlanViolation("hard", "python_syntax_error", f"Invalid python.exec code: {exc.msg}."))
            continue

        violations.extend(_classify_python_ast_hard_violations(tree))
        if any(violation.tier == "hard" for violation in violations):
            continue

        if not _assigns_result_variable(tree):
            violations.append(
                PlanViolation("soft", "missing_result_assignment", "python.exec must assign the final value to result.")
            )

        row_aliases = _sql_row_aliases_for_step(step, producer_by_output)
        for alias in row_aliases:
            if _indexes_alias_without_guard(code, alias):
                violations.append(
                    PlanViolation(
                        "soft",
                        "unguarded_sql_rows",
                        f"python.exec indexes SQL rows via {alias}[0] without handling empty results safely.",
                    )
                )

        selected_fields_by_alias = _selected_sql_fields_for_step(step, producer_by_output)
        field_violations = _classify_sql_field_assumptions(tree, selected_fields_by_alias)
        violations.extend(field_violations)
    return violations


def _classify_python_ast_hard_violations(tree: ast.AST) -> list[PlanViolation]:
    """Handle the internal classify python ast hard violations helper path for this module.

    Inputs:
        Receives tree for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_python_ast_hard_violations.
    """
    violations: list[PlanViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = str(alias.name or "").split(".", 1)[0]
                if module_name not in ALLOWED_PYTHON_IMPORT_MODULES:
                    violations.append(
                        PlanViolation(
                            "hard",
                            "forbidden_python_import",
                            f"python.exec may only import json or re, not {module_name}.",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module_name = str(node.module or "").split(".", 1)[0]
            if module_name not in ALLOWED_PYTHON_IMPORT_MODULES:
                violations.append(
                    PlanViolation(
                        "hard",
                        "forbidden_python_import",
                        f"python.exec may only import json or re, not {module_name}.",
                    )
                )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_PYTHON_NAMES:
            violations.append(
                PlanViolation("hard", "forbidden_python_name", f"python.exec must not use {node.id}.")
            )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_IMPORT_MODULES:
            violations.append(
                PlanViolation("hard", "forbidden_python_module", f"python.exec must not use {node.id}.")
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAME_CALLS:
                violations.append(
                    PlanViolation("hard", "python_system_call", f"python.exec must not call {node.func.id}().")
                )
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                base_name = node.func.value.id
                attr_name = node.func.attr
                if (base_name, attr_name) in FORBIDDEN_ATTR_CALLS:
                    message = f"python.exec must not call {base_name}.{attr_name}()."
                    violations.append(PlanViolation("hard", "python_system_call", message))
        elif isinstance(node, ast.Subscript) and _is_nested_inputs_wrapper_access(node):
            nested_key = _subscript_literal(node)
            if isinstance(nested_key, str):
                violations.append(
                    PlanViolation(
                        "hard",
                        "nested_input_wrapper",
                        f'python.exec must use inputs[...] directly and must not access nested wrapper fields like ["{nested_key}"].',
                    )
                )
    return _dedupe_violations(violations)


def _classify_dataflow_violations(plan: ExecutionPlan) -> list[PlanViolation]:
    """Handle the internal classify dataflow violations helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_dataflow_violations.
    """
    violations: list[PlanViolation] = []
    for code, validator in (
        ("invalid_dataflow", validate_dataflow),
        ("placeholder_output", _validate_placeholder_outputs),
        ("python_inputs_contract", _validate_python_inputs_contract),
    ):
        try:
            validator(plan)
        except ValueError as exc:
            violations.append(PlanViolation("soft", code, str(exc)))
    return violations


def _classify_path_violations(plan: ExecutionPlan) -> list[PlanViolation]:
    """Handle the internal classify path violations helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_path_violations.
    """
    violations: list[PlanViolation] = []
    prior_paths: list[str] = []
    for step in plan.steps:
        for key in PATH_ARG_KEYS:
            value = step.args.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = value.strip()
            basename = PurePosixPath(normalized).name
            if basename == normalized:
                candidates = sorted(
                    {
                        prior
                        for prior in prior_paths
                        if PurePosixPath(prior).name == basename and PurePosixPath(prior).as_posix() != basename
                    }
                )
                if len(candidates) == 1:
                    violations.append(
                        PlanViolation(
                            "soft",
                            "path_inconsistency",
                            f"Path lost its directory prefix: {normalized}. Expected {candidates[0]}.",
                        )
                    )
            prior_paths.append(normalized)
    return _dedupe_violations(violations)


def _classify_output_contract_violations(plan: ExecutionPlan, goal: str) -> list[PlanViolation]:
    """Handle the internal classify output contract violations helper path for this module.

    Inputs:
        Receives plan, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_output_contract_violations.
    """
    violations: list[PlanViolation] = []
    if not plan.steps:
        return violations

    last_step = plan.steps[-1]
    if goal_requests_return_value(goal) and last_step.action == "fs.write":
        violations.append(
            PlanViolation(
                "soft",
                "missing_return_step",
                "Final step must surface the requested data instead of ending with fs.write.",
            )
        )
    return _dedupe_violations(violations)


def _assigns_result_variable(tree: ast.AST) -> bool:
    """Handle the internal assigns result variable helper path for this module.

    Inputs:
        Receives tree for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._assigns_result_variable.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                return True
    return False


def _sql_row_aliases_for_step(step: ExecutionStep, producer_by_output: dict[str, ExecutionStep]) -> list[str]:
    """Handle the internal sql row aliases for step helper path for this module.

    Inputs:
        Receives step, producer_by_output for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._sql_row_aliases_for_step.
    """
    inputs_mapping = step.args.get("inputs")
    if not isinstance(inputs_mapping, dict):
        return []
    aliases: list[str] = []
    for input_name, value in inputs_mapping.items():
        if not isinstance(input_name, str):
            continue
        if _ref_points_to_sql_rows(value, producer_by_output):
            aliases.append(input_name)
    return aliases


def _selected_sql_fields_for_step(
    step: ExecutionStep,
    producer_by_output: dict[str, ExecutionStep],
) -> dict[str, set[str]]:
    """Handle the internal selected sql fields for step helper path for this module.

    Inputs:
        Receives step, producer_by_output for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._selected_sql_fields_for_step.
    """
    inputs_mapping = step.args.get("inputs")
    if not isinstance(inputs_mapping, dict):
        return {}
    selected: dict[str, set[str]] = {}
    for input_name, value in inputs_mapping.items():
        if not isinstance(input_name, str) or not isinstance(value, dict):
            continue
        ref_name = str(value.get("$ref") or "").strip()
        ref_path = str(value.get("path") or "").strip()
        producer = producer_by_output.get(ref_name)
        if producer is None or producer.action != "sql.query":
            continue
        if ref_path and ref_path != "rows":
            continue
        fields = _extract_selected_sql_fields(str(producer.args.get("query", "")))
        if fields:
            selected[input_name] = fields
    return selected


def _classify_sql_field_assumptions(tree: ast.AST, selected_fields_by_alias: dict[str, set[str]]) -> list[PlanViolation]:
    """Handle the internal classify sql field assumptions helper path for this module.

    Inputs:
        Receives tree, selected_fields_by_alias for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._classify_sql_field_assumptions.
    """
    violations: list[PlanViolation] = []
    alias_assignments = _assignments_from_inputs(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        field_name = _subscript_literal(node)
        if not isinstance(field_name, str):
            continue
        base_name = _subscript_base_name(node)
        if not isinstance(base_name, str):
            continue
        input_alias = alias_assignments.get(base_name)
        if not input_alias:
            continue
        allowed_fields = selected_fields_by_alias.get(input_alias)
        if not allowed_fields:
            continue
        if field_name not in allowed_fields:
            violations.append(
                PlanViolation(
                    "soft",
                    "sql_field_assumption",
                    f'python.exec assumes SQL field "{field_name}" for {input_alias}, but that field was not explicitly selected.',
                )
            )
    return _dedupe_violations(violations)


def _assignments_from_inputs(tree: ast.AST) -> dict[str, str]:
    """Handle the internal assignments from inputs helper path for this module.

    Inputs:
        Receives tree for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._assignments_from_inputs.
    """
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Subscript) and _is_inputs_access(value):
            key = _subscript_literal(value)
            if isinstance(key, str):
                assignments[target.id] = key
    return assignments


def _ref_points_to_sql_rows(value: Any, producer_by_output: dict[str, ExecutionStep]) -> bool:
    """Handle the internal ref points to sql rows helper path for this module.

    Inputs:
        Receives value, producer_by_output for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._ref_points_to_sql_rows.
    """
    if not isinstance(value, dict):
        return False
    ref_name = str(value.get("$ref") or "").strip()
    ref_path = str(value.get("path") or "").strip()
    if not ref_name:
        return False
    producer = producer_by_output.get(ref_name)
    if producer is None or producer.action != "sql.query":
        return False
    return not ref_path or ref_path == "rows"


def _extract_selected_sql_fields(query: str) -> set[str]:
    """Handle the internal extract selected sql fields helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._extract_selected_sql_fields.
    """
    match = SQL_SELECT_RE.search(str(query or ""))
    if match is None:
        return set()
    select_clause = match.group("select")
    expressions = [part.strip() for part in select_clause.split(",") if part.strip()]
    fields: set[str] = set()
    for expression in expressions:
        alias_match = SQL_ALIAS_RE.search(expression)
        if alias_match is not None:
            fields.add(alias_match.group(1))
            continue
        normalized = expression.strip()
        if "(" in normalized or " " in normalized:
            continue
        fields.add(normalized.split(".")[-1].strip('"'))
    return fields


def _indexes_alias_without_guard(code: str, alias: str) -> bool:
    """Handle the internal indexes alias without guard helper path for this module.

    Inputs:
        Receives code, alias for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._indexes_alias_without_guard.
    """
    direct_index = re.search(rf"\b{re.escape(alias)}\s*\[\s*0\s*\]", code)
    if direct_index is None:
        return False
    guard_patterns = (
        rf"\bif\s+{re.escape(alias)}\b",
        rf"\bif\s+len\(\s*{re.escape(alias)}\s*\)\b",
        rf"\b{re.escape(alias)}\s+if\s+{re.escape(alias)}\s+else\b",
        rf"\bif\s+not\s+{re.escape(alias)}\b",
    )
    return not any(re.search(pattern, code) for pattern in guard_patterns)


def _is_inputs_access(node: ast.Subscript) -> bool:
    """Handle the internal is inputs access helper path for this module.

    Inputs:
        Receives node for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._is_inputs_access.
    """
    return isinstance(node.value, ast.Name) and node.value.id == "inputs"


def _is_nested_inputs_wrapper_access(node: ast.Subscript) -> bool:
    """Handle the internal is nested inputs wrapper access helper path for this module.

    Inputs:
        Receives node for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._is_nested_inputs_wrapper_access.
    """
    if not isinstance(node.value, ast.Subscript):
        return False
    if not _is_inputs_access(node.value):
        return False
    nested_key = _subscript_literal(node)
    return isinstance(nested_key, str)


def _subscript_literal(node: ast.Subscript) -> Any:
    """Handle the internal subscript literal helper path for this module.

    Inputs:
        Receives node for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._subscript_literal.
    """
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant):
        return slice_node.value
    return None


def _subscript_base_name(node: ast.Subscript) -> str | None:
    """Handle the internal subscript base name helper path for this module.

    Inputs:
        Receives node for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._subscript_base_name.
    """
    value = node.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Subscript):
        return _subscript_base_name(value)
    return None


def _dedupe_violations(violations: list[PlanViolation]) -> list[PlanViolation]:
    """Handle the internal dedupe violations helper path for this module.

    Inputs:
        Receives violations for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._dedupe_violations.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[PlanViolation] = []
    for violation in violations:
        key = (violation.tier, violation.code, violation.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(violation)
    return unique


def _tokenize(value: str) -> set[str]:
    """Handle the internal tokenize helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._tokenize.
    """
    tokens = set(TOKEN_RE.findall(value))
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
        elif len(token) > 2:
            expanded.add(f"{token}s")
    return expanded


def _schema_tokens(schema: dict[str, Any] | None) -> set[str]:
    """Handle the internal schema tokens helper path for this module.

    Inputs:
        Receives schema for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._schema_tokens.
    """
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
    """Handle the internal looks path like helper path for this module.

    Inputs:
        Receives goal_text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.policies._looks_path_like.
    """
    return "/" in goal_text or "\\" in goal_text or ".txt" in goal_text or ".md" in goal_text
