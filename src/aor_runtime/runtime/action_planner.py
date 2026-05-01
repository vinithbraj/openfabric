"""OpenFABRIC Runtime Module: aor_runtime.runtime.action_planner

Purpose:
    Convert natural-language goals into validator-enforced action plans using the LLM.

Responsibilities:
    Build planner prompts, parse action JSON, canonicalize tool arguments/dataflow, apply repairs, and compile ExecutionPlans.

Data flow / Interfaces:
    Receives user goals, tool registry metadata, runtime date context, and compact repair facts; returns deterministic ExecutionPlan steps.

Boundaries:
    The LLM may propose actions, but this module enforces schema, domain, temporal, SQL, SLURM, shell, dataflow, and output-shape boundaries before execution.
"""

from __future__ import annotations

import json
import re
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep, PlannerConfig
from aor_runtime.core.utils import dumps_json, extract_json_object
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.dataflow import collect_step_references, normalize_execution_plan_dataflow
from aor_runtime.runtime.diagnostic_orchestration import diagnostic_plan_for_goal
from aor_runtime.runtime.llm_recursion import LLMStageRecursionBudget
from aor_runtime.runtime.output_shape import (
    grouped_count_field_for_goal,
    infer_goal_output_contract,
    resolve_output_intent,
    is_scalar_count_goal as _shared_is_scalar_count_goal,
    is_shell_status_goal,
    scalar_field_for_tool,
)
from aor_runtime.runtime.semantic_obligations import apply_semantic_obligations_to_actions
from aor_runtime.runtime.shell_safety import classify_shell_command
from aor_runtime.runtime.sql_ast_validation import normalize_and_validate_sql_ast
from aor_runtime.runtime.sql_safety import ensure_read_only_sql, normalize_pg_relation_quoting, validate_read_only_sql
from aor_runtime.runtime.temporal import TemporalArgumentCanonicalizer, TemporalNormalizationError
from aor_runtime.runtime.tool_output_contracts import (
    TOOL_OUTPUT_CONTRACTS,
    available_paths_for_tool,
    default_path_for_tool,
    formatter_source_path_for_tool,
    normalize_tool_ref_path,
    path_is_declared_for_tool,
    return_value_path_for_tool,
)
from aor_runtime.runtime import temporal as temporal_runtime
from aor_runtime.tools.base import ToolRegistry
from aor_runtime.tools.filesystem import resolve_path
from aor_runtime.tools.sql import get_all_sql_catalogs, get_sql_catalog, resolve_sql_databases


ACTION_PLANNER_SYSTEM_PROMPT = """You are a compact action planner for a validator-enforced local runtime.

Return JSON only matching ActionPlan:
{
  "goal": "string",
  "actions": [
    {
      "id": "short_id",
      "tool": "registered.tool",
      "purpose": "why this action exists",
      "inputs": {},
      "depends_on": ["previous_id"],
      "output_binding": "optional_alias",
      "expected_result_shape": {"kind": "scalar|table|file|text|json|status"}
    }
  ],
  "expected_final_shape": {"kind": "scalar|table|file|text|json|status"},
  "notes": []
}

Use only listed tools. Do not emit ExecutionPlan. Do not use python.exec.
Use sql.query for database reads. SQL must be SELECT-only.
For requests to generate, validate, or explain a SQL query, emit the SQL action you would use; the runtime may convert it to non-executing sql.validate.
Use text.format for data formatting. Use fs.write for file writes. Never use shell redirection for writes.
For count/how-many/number-of questions, produce one aggregate scalar row, not one row per entity.
If a count requires GROUP BY/HAVING, wrap the grouped query and SELECT COUNT(*) from it.
For SQL questions, prefer one complete sql.query. Do not compute arithmetic by combining multiple SQL outputs in text.format or runtime.return.
For database count questions, always use sql.query; never return literal placeholder values such as {patient_count}.
For "no/without/missing related rows" SQL questions, use NOT EXISTS or LEFT JOIN with IS NULL in one SELECT.
Use exact SQL identifiers from schema. For PostgreSQL, double-quote mixed-case identifiers exactly.
For missing dates or timestamps, use IS NULL; never compare date/timestamp columns to empty strings.
Use runtime.return as the final action.
Use canonical shape kinds when possible: scalar, table, file, text, json, status. If you use synonyms such as count, rows, or records, the runtime will normalize them before validation.
Output JSON only."""


REFERENCE_RE = re.compile(r"^\$([A-Za-z0-9_-]+)(?:\.([A-Za-z0-9_.-]+))?$")
EXPORT_GOAL_RE = re.compile(r"\b(?:save|write|export)\b.+\b[\w.-]+\.(?:txt|csv|json|md|markdown)\b", re.IGNORECASE)
SQL_GOAL_RE = re.compile(r"\b(?:sql|database|table|tables|row|rows|patient|patients|study|studies|series|dicom)\b", re.IGNORECASE)
DATABASE_NAME_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*_db\b")


ExpectedKind = Literal["scalar", "table", "file", "text", "json", "status", "unknown"]

SHAPE_KIND_ALIASES = {
    "bool": "status",
    "boolean": "status",
    "count": "scalar",
    "csv": "text",
    "integer": "scalar",
    "int": "scalar",
    "list": "table",
    "markdown": "text",
    "md": "text",
    "number": "scalar",
    "numeric": "scalar",
    "record": "table",
    "records": "table",
    "row": "table",
    "rows": "table",
    "string": "text",
}
FORMAT_ALIASES = {
    "md": "markdown",
    "markdown_table": "markdown",
    "text": "txt",
}
TOOL_ALIASES = {
    "format": "text.format",
    "fs": "fs.list",
    "read": "fs.read",
    "return": "runtime.return",
    "runtime": "runtime.return",
    "shell": "shell.exec",
    "sql": "sql.query",
    "sql.select": "sql.query",
    "sql_query": "sql.query",
    "write": "fs.write",
}
DATA_REF_PATHS = {
    tool: contract.default_path
    for tool, contract in TOOL_OUTPUT_CONTRACTS.items()
    if tool not in {"python.exec", "runtime.return"} and contract.default_path
}
EXPORT_PATH_RE = re.compile(r"\b(?:to|as|at|into)\s+([^\s,;]+?\.(?:txt|csv|json|md|markdown))\b", re.IGNORECASE)


class ExpectedResultShape(BaseModel):
    """Represent expected result shape within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ExpectedResultShape.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ExpectedResultShape and related tests.
    """
    kind: ExpectedKind = "unknown"
    columns: list[str] = Field(default_factory=list)
    format: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_shape(cls, value: Any) -> Any:
        """Validate coerce shape invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this ExpectedResultShape method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through ExpectedResultShape.coerce_shape calls and related tests.
        """
        if isinstance(value, str):
            return {"kind": _normalize_shape_kind(value)}
        if isinstance(value, dict):
            copied = dict(value)
            copied["kind"] = _normalize_shape_kind(copied.get("kind", "unknown"))
            if "format" in copied:
                copied["format"] = _normalize_format(copied.get("format"))
            return copied
        return value


class RawPlannedAction(BaseModel):
    """Represent raw planned action within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RawPlannedAction.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.RawPlannedAction and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    id: Any = None
    tool: Any = None
    action: Any = None
    purpose: Any = ""
    inputs: Any = Field(default_factory=dict)
    args: Any = None
    depends_on: Any = Field(default_factory=list)
    output_binding: Any = None
    output: Any = None
    expected_result_shape: Any = None


class RawActionPlan(BaseModel):
    """Represent raw action plan within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RawActionPlan.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.RawActionPlan and related tests.
    """
    model_config = ConfigDict(extra="forbid")

    goal: Any = ""
    actions: list[RawPlannedAction] = Field(default_factory=list)
    expected_final_shape: Any = Field(default_factory=dict)
    notes: Any = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_execution_plan_shape(cls, value: Any) -> Any:
        """Validate reject execution plan shape invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this RawActionPlan method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through RawActionPlan.reject_execution_plan_shape calls and related tests.
        """
        if isinstance(value, dict) and "steps" in value and "actions" not in value:
            raise ValueError("LLM returned ExecutionPlan shape; expected ActionPlan.actions.")
        return value


class PlannedAction(BaseModel):
    """Represent planned action within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlannedAction.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.PlannedAction and related tests.
    """
    id: str
    tool: str
    purpose: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    output_binding: str | None = None
    expected_result_shape: ExpectedResultShape = Field(default_factory=ExpectedResultShape)

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_keys(cls, value: Any) -> Any:
        """Validate coerce legacy keys invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this PlannedAction method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through PlannedAction.coerce_legacy_keys calls and related tests.
        """
        if not isinstance(value, dict):
            return value
        copied = dict(value)
        if "tool" not in copied and "action" in copied:
            copied["tool"] = copied.pop("action")
        if "inputs" not in copied and "args" in copied:
            copied["inputs"] = copied.pop("args")
        if "output_binding" not in copied and "output" in copied:
            copied["output_binding"] = copied.pop("output")
        return copied

    @model_validator(mode="after")
    def normalize(self) -> "PlannedAction":
        """Validate normalize invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through PlannedAction.normalize calls and related tests.
        """
        self.id = _normalize_name(self.id)
        self.tool = str(self.tool or "").strip()
        self.depends_on = [_normalize_name(item) for item in self.depends_on if _normalize_name(item)]
        if self.output_binding is None:
            self.output_binding = f"{self.id}_output"
        else:
            self.output_binding = _normalize_name(self.output_binding) or f"{self.id}_output"
        return self


class ActionPlan(BaseModel):
    """Represent action plan within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ActionPlan.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ActionPlan and related tests.
    """
    goal: str
    actions: list[PlannedAction] = Field(default_factory=list)
    expected_final_shape: ExpectedResultShape = Field(default_factory=ExpectedResultShape)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_execution_plan_shape(cls, value: Any) -> Any:
        """Validate reject execution plan shape invariants before runtime data crosses this boundary.

        Inputs:
            Receives value for this ActionPlan method; type hints and validators define accepted shapes.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlan.reject_execution_plan_shape calls and related tests.
        """
        if isinstance(value, dict) and "steps" in value and "actions" not in value:
            raise ValueError("LLM returned ExecutionPlan shape; expected ActionPlan.actions.")
        return value

    @model_validator(mode="after")
    def validate_actions(self) -> "ActionPlan":
        """Validate validate actions invariants before runtime data crosses this boundary.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the validated value or model instance after enforcing the declared invariant.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlan.validate_actions calls and related tests.
        """
        if not self.actions:
            raise ValueError("ActionPlan requires at least one action.")
        return self


ActionPlanProposal = ActionPlan


class RepairPlanProposal(BaseModel):
    """Represent a typed LLM repair decision for failed planning attempts.

    Inputs:
        Created from compact failure facts only, never raw rows or payloads.

    Returns:
        A strict repair instruction envelope that can choose replan or fail.

    Used by:
        Repair-aware planner prompts and future typed repair orchestration.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["replan", "fail"] = "replan"
    reason: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ActionValidationResult:
    """Represent action validation result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ActionValidationResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ActionValidationResult and related tests.
    """
    valid: bool
    errors: list[str] = field(default_factory=list)
    repairable: bool = True


@dataclass(frozen=True)
class PlanIssue:
    """Represent plan issue within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanIssue.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.PlanIssue and related tests.
    """
    code: str
    message: str
    severity: Literal["info", "repair", "error"] = "repair"


@dataclass(frozen=True)
class CanonicalizationResult:
    """Represent canonicalization result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CanonicalizationResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.CanonicalizationResult and related tests.
    """
    plan: ActionPlan
    repairs: list[str] = field(default_factory=list)
    issues: list[PlanIssue] = field(default_factory=list)
    symbol_table: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PlanSymbol(BaseModel):
    """Represent plan symbol within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanSymbol.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.PlanSymbol and related tests.
    """
    name: str
    action_id: str
    output_binding: str
    tool: str
    default_path: str | None = None


class PlanSymbolTable(BaseModel):
    """Represent plan symbol table within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanSymbolTable.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.PlanSymbolTable and related tests.
    """
    symbols: dict[str, PlanSymbol] = Field(default_factory=dict)

    @classmethod
    def from_actions(cls, actions: list[dict[str, Any]]) -> "PlanSymbolTable":
        """From actions for PlanSymbolTable instances.

        Inputs:
            Receives actions for this PlanSymbolTable method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanSymbolTable.from_actions calls and related tests.
        """
        symbols: dict[str, PlanSymbol] = {}
        for action in actions:
            action_id = _normalize_name(action.get("id") or "")
            output_binding = _normalize_name(action.get("output_binding") or "")
            tool = _normalize_tool_name(action.get("tool"))
            default_path = default_path_for_tool(tool)
            canonical_name = output_binding or action_id
            if not action_id or not canonical_name:
                continue
            symbol = PlanSymbol(
                name=canonical_name,
                action_id=action_id,
                output_binding=canonical_name,
                tool=tool,
                default_path=default_path,
            )
            symbols[action_id] = symbol
            symbols[canonical_name] = symbol
        return cls(symbols=symbols)

    def lookup(self, value: str) -> PlanSymbol | None:
        """Lookup for PlanSymbolTable instances.

        Inputs:
            Receives value for this PlanSymbolTable method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanSymbolTable.lookup calls and related tests.
        """
        normalized = _normalize_name(value)
        return self.symbols.get(normalized)


class PlanDataflowValidator:
    """Represent plan dataflow validator within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by PlanDataflowValidator.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.PlanDataflowValidator and related tests.
    """
    def validate(self, plan: ActionPlan) -> list[str]:
        """Validate for PlanDataflowValidator instances.

        Inputs:
            Receives plan for this PlanDataflowValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through PlanDataflowValidator.validate calls and related tests.
        """
        symbols = PlanSymbolTable.from_actions([action.model_dump() for action in plan.actions])
        errors: list[str] = []
        for action in plan.actions:
            for alias, path in _iter_action_refs_with_paths(action.inputs):
                symbol = symbols.lookup(alias)
                if symbol is None:
                    continue
                if path_is_declared_for_tool(symbol.tool, path):
                    continue
                available = ", ".join(available_paths_for_tool(symbol.tool)) or "<whole result>"
                suggested = default_path_for_tool(symbol.tool)
                suffix = f" Suggested path: {suggested}." if suggested else ""
                errors.append(
                    f"Invalid reference path for {alias}: {path}. "
                    f"Producer {symbol.output_binding} uses {symbol.tool} and exposes: {available}.{suffix}"
                )
        return errors


@dataclass(frozen=True)
class ToolArgumentCanonicalizationResult:
    """Represent tool argument canonicalization result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolArgumentCanonicalizationResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ToolArgumentCanonicalizationResult and related tests.
    """
    repairs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolArgumentCanonicalizer:
    """Represent tool argument canonicalizer within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ToolArgumentCanonicalizer.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ToolArgumentCanonicalizer and related tests.
    """
    def __init__(self, tools: ToolRegistry) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives tools for this ToolArgumentCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through ToolArgumentCanonicalizer.__init__ calls and related tests.
        """
        self.tools = tools

    def canonicalize(self, actions: list[dict[str, Any]]) -> ToolArgumentCanonicalizationResult:
        """Canonicalize for ToolArgumentCanonicalizer instances.

        Inputs:
            Receives actions for this ToolArgumentCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ToolArgumentCanonicalizer.canonicalize calls and related tests.
        """
        repairs: list[str] = []
        scrubbed: list[dict[str, Any]] = []
        for action in actions:
            tool_name = str(action.get("tool") or "")
            try:
                tool = self.tools.get(tool_name)
            except KeyError:
                continue
            allowed = set(getattr(tool.args_model, "model_fields", {}).keys())
            if not allowed:
                continue
            inputs = action.setdefault("inputs", {})
            if not isinstance(inputs, dict):
                continue
            extra_keys = [
                key
                for key in list(inputs.keys())
                if str(key) not in allowed and not str(key).startswith("__")
            ]
            if not extra_keys:
                continue
            for key in extra_keys:
                inputs.pop(key, None)
            action_id = str(action.get("id") or tool_name)
            scrubbed.append({"action_id": action_id, "tool": tool_name, "keys": sorted(str(key) for key in extra_keys)})
            repairs.append(
                f"Removed metadata-only unsupported argument(s) from {tool_name}: {', '.join(sorted(str(key) for key in extra_keys))}."
            )
        metadata = {"scrubbed_arguments": scrubbed} if scrubbed else {}
        return ToolArgumentCanonicalizationResult(repairs=repairs, metadata=metadata)


class LLMActionPlanner:
    """Represent l l m action planner within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by LLMActionPlanner.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.LLMActionPlanner and related tests.
    """
    def __init__(self, *, llm: LLMClient, tools: ToolRegistry, settings: Settings) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives llm, tools, settings for this LLMActionPlanner method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through LLMActionPlanner.__init__ calls and related tests.
        """
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.last_raw_output: str | None = None
        self.last_prompt: dict[str, Any] | None = None
        self.last_raw_action_plan: dict[str, Any] | None = None
        self.last_normalized_action_plan: dict[str, Any] | None = None
        self.last_canonicalized_action_plan: dict[str, Any] | None = None
        self.last_canonicalization_repairs: list[str] = []
        self.last_database_propagation_repairs: list[str] = []
        self.last_dataflow_canonicalization_repairs: list[str] = []
        self.last_temporal_normalization_repairs: list[str] = []
        self.last_temporal_normalization_metadata: dict[str, Any] = {}
        self.last_temporal_llm_calls: int = 0
        self.last_tool_argument_canonicalization_metadata: dict[str, Any] = {}
        self.last_contract_validation_errors: list[str] = []
        self.last_domain_validation_errors: list[str] = []
        self.last_validation_errors: list[str] = []

    def build_plan(
        self,
        *,
        goal: str,
        planner: PlannerConfig,
        allowed_tools: list[str],
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Build plan for LLMActionPlanner instances.

        Inputs:
            Receives goal, planner, allowed_tools, input_payload, failure_context for this LLMActionPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through LLMActionPlanner.build_plan calls and related tests.
        """
        planning_now = temporal_runtime.current_local_datetime(self.settings)
        context = self._build_context(goal, allowed_tools, input_payload, failure_context, now=planning_now)
        self.last_prompt = context
        raw = self.llm.complete(
            system_prompt=ACTION_PLANNER_SYSTEM_PROMPT,
            user_prompt=dumps_json(context, indent=2),
            model=planner.model,
            temperature=planner.temperature,
        )
        self.last_raw_output = raw
        payload = extract_json_object(raw)
        raw_plan = RawActionPlan.model_validate(payload)
        self.last_raw_action_plan = json.loads(json.dumps(raw_plan.model_dump(), default=str))
        normalized_payload = normalize_raw_action_plan(raw_plan, fallback_goal=goal)
        self.last_normalized_action_plan = json.loads(json.dumps(normalized_payload, default=str))
        plan = ActionPlan.model_validate(normalized_payload)
        canonicalization = ActionPlanCanonicalizer(
            goal=goal,
            settings=self.settings,
            failure_context=failure_context,
            llm=self.llm,
            now=planning_now,
            tools=self.tools,
        ).canonicalize(plan)
        plan = canonicalization.plan
        self.last_canonicalization_repairs = list(canonicalization.repairs)
        self.last_canonicalized_action_plan = plan.model_dump()
        self.last_temporal_normalization_repairs = [
            repair
            for repair in canonicalization.repairs
            if "time range" in repair.lower() or "temporal" in repair.lower() or "date" in repair.lower()
        ]
        self.last_temporal_normalization_metadata = dict(canonicalization.metadata.get("temporal_normalization") or {})
        self.last_temporal_llm_calls = int(canonicalization.metadata.get("temporal_llm_calls") or 0)
        self.last_tool_argument_canonicalization_metadata = dict(
            canonicalization.metadata.get("tool_argument_canonicalization") or {}
        )
        self.last_database_propagation_repairs = [
            repair for repair in canonicalization.repairs if "database" in repair.lower()
        ]
        self.last_dataflow_canonicalization_repairs = [
            repair for repair in canonicalization.repairs if "database" not in repair.lower()
        ]
        validation = ActionPlanValidator(settings=self.settings, tools=self.tools, allowed_tools=allowed_tools).validate(plan, goal=goal)
        self.last_validation_errors = list(validation.errors)
        self.last_contract_validation_errors, self.last_domain_validation_errors = _split_validation_errors(validation.errors)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return ActionPlanCompiler(goal=goal).compile(plan)

    def _build_context(
        self,
        goal: str,
        allowed_tools: list[str],
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        """Handle the internal build context helper path for this module.

        Inputs:
            Receives goal, allowed_tools, input_payload, failure_context, now for this LLMActionPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through LLMActionPlanner._build_context calls and related tests.
        """
        manifest = build_tool_manifest(self.tools, allowed_tools)
        stage_name = "repair_plan" if failure_context else "action_plan"
        budget = LLMStageRecursionBudget(self.settings, stage=stage_name)
        stage_state = budget.state(
            depth=0,
            parent_id=None,
            composition="single",
            allowed_schema={
                "proposal": "RepairPlanProposal" if failure_context else "ActionPlanProposal",
                "tools": allowed_tools,
            },
        )
        context: dict[str, Any] = {
            "goal": goal,
            "input": input_payload,
            "runtime_date": temporal_runtime.runtime_date_context(self.settings, now=now),
            "llm_stage": stage_state.prompt_metadata(),
            "available_tools": manifest,
            "runtime_rules": {
                "sql": "SELECT-only. Use exact schema identifiers. Use an outer COUNT(*) for count prompts over grouped/HAVING sets. Do not add LIMIT/FETCH row caps when the user asks for all/list/show rows unless they explicitly requested top/first/sample/a limit.",
                "formatting": "Use text.format. Do not ask the LLM to format row data. Do not put placeholder strings like {rows}; use structured $refs or one SQL query.",
                "filesystem": "Use fs.write only for explicit save/write/export file goals.",
                "shell": "Use shell.exec only for explicit commands or system inspection; commands are safety-classified. Do not add head/tail/row limits when the user asks for all/list/show rows unless they requested top/first/sample/a limit.",
                "domains": "Use SLURM tools only for SLURM cluster/jobs/nodes/partitions. Use shell.exec for operating-system process/port/computer inspection.",
                "temporal_args": "For SLURM start/end/date/time_range arguments, use ISO-like YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Interpret relative phrases such as past 7 days relative to runtime_date.current_local_date; the runtime will validate and normalize temporal arguments before execution.",
            },
            "failure_context": _compact_failure_context(failure_context or {}),
        }
        diagnostic_plan = diagnostic_plan_for_goal(goal)
        if diagnostic_plan is not None:
            context["diagnostic_orchestration"] = {
                **diagnostic_plan.model_dump(),
                "rules": [
                    "Use compact staged inspection actions only.",
                    "Return summarized facts per section; do not stream raw lists or file contents.",
                    "If the budget is not enough, return completed sections and a not-completed note.",
                ],
            }
        if _should_include_sql_schema(goal, allowed_tools, self.settings):
            context["sql_schema"] = _compact_sql_schema(self.settings)
        return context


class ActionPlanValidator:
    """Represent action plan validator within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ActionPlanValidator.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ActionPlanValidator and related tests.
    """
    def __init__(self, *, settings: Settings, tools: ToolRegistry, allowed_tools: list[str]) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings, tools, allowed_tools for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator.__init__ calls and related tests.
        """
        self.settings = settings
        self.tools = tools
        self.allowed_tools = set(allowed_tools) | {"runtime.return", "text.format", "sql.schema", "sql.validate"}

    def validate(self, plan: ActionPlan, *, goal: str) -> ActionValidationResult:
        """Validate for ActionPlanValidator instances.

        Inputs:
            Receives plan, goal for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator.validate calls and related tests.
        """
        errors: list[str] = []
        if len(plan.actions) > 12:
            errors.append("Action plan has too many actions; maximum is 12.")
        ids = {action.id for action in plan.actions}
        aliases = {str(action.output_binding or "") for action in plan.actions if action.output_binding}
        for action in plan.actions:
            if action.tool == "python.exec":
                errors.append("python.exec is not allowed for LLM-planned orchestration.")
            if action.tool not in self.allowed_tools:
                errors.append(f"Unknown or disallowed tool: {action.tool}")
            else:
                errors.extend(self._validate_tool_args(action))
            for dep in action.depends_on:
                if dep not in ids:
                    errors.append(f"Action {action.id} depends on unknown action {dep}.")
                if dep == action.id:
                    errors.append(f"Action {action.id} depends on itself.")
            for ref in _collect_action_refs(action.inputs):
                if ref not in ids and ref not in aliases:
                    errors.append(f"Action {action.id} references unknown output {ref}.")
            placeholder_aliases = _placeholder_aliases(action.inputs, ids | aliases)
            if placeholder_aliases:
                errors.append(
                    f"Action {action.id} contains unresolved output placeholders: {', '.join(sorted(placeholder_aliases))}."
                )
            errors.extend(self._validate_action_policy(action, goal=goal))
        if _has_cycle(plan):
            errors.append("Action plan contains a dependency cycle.")
        errors.extend(PlanDataflowValidator().validate(plan))
        errors.extend(self._validate_goal_flow(plan, goal=goal))
        return ActionValidationResult(valid=not errors, errors=errors, repairable=True)

    def _validate_tool_args(self, action: PlannedAction) -> list[str]:
        """Handle the internal validate tool args helper path for this module.

        Inputs:
            Receives action for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator._validate_tool_args calls and related tests.
        """
        if _collect_action_refs(action.inputs):
            return []
        try:
            self.tools.validate_step(action.tool, action.inputs)
        except Exception as exc:  # noqa: BLE001
            return [f"Invalid inputs for {action.tool}: {exc}"]
        return []

    def _validate_action_policy(self, action: PlannedAction, *, goal: str) -> list[str]:
        """Handle the internal validate action policy helper path for this module.

        Inputs:
            Receives action, goal for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator._validate_action_policy calls and related tests.
        """
        if action.tool in {"sql.query", "sql.validate"}:
            return self._validate_sql_action(action, goal=goal)
        if action.tool == "shell.exec":
            command = str(action.inputs.get("command") or "")
            policy = classify_shell_command(
                command,
                mode=self.settings.shell_mode,
                allow_mutation_with_approval=self.settings.shell_allow_mutation_with_approval or self.settings.allow_destructive_shell,
            )
            if not policy.allowed:
                return [f"Unsafe shell command: {policy.reason}"]
            if _goal_requires_unlimited_rows(goal) and not _is_explicit_shell_command_goal(goal) and _shell_command_has_unrequested_limit(command):
                return ["Shell command adds an unrequested row limit; the user asked for all rows."]
        if action.tool.startswith("fs."):
            return self._validate_fs_action(action, goal=goal)
        if action.tool.startswith("slurm.") and action.tool not in _READ_ONLY_SLURM_TOOLS:
            return [f"SLURM mutation/admin tool is not allowed: {action.tool}"]
        if action.tool == "text.format":
            output_format = str(action.inputs.get("format") or "txt").strip().lower()
            if output_format not in {"txt", "csv", "json", "markdown"}:
                return [f"Unsupported text.format output format: {output_format}"]
        if action.tool == "runtime.return":
            mode = str(action.inputs.get("mode") or "text").strip().lower()
            if mode not in {"text", "csv", "json", "count"}:
                return [f"Unsupported runtime.return mode: {mode}"]
            if mode == "json":
                return ["runtime.return mode=json is not user-facing; use markdown/text presentation or write JSON to a file."]
        return []

    def _validate_sql_action(self, action: PlannedAction, *, goal: str) -> list[str]:
        """Handle the internal validate sql action helper path for this module.

        Inputs:
            Receives action, goal for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator._validate_sql_action calls and related tests.
        """
        query = str(action.inputs.get("query") or "")
        try:
            safe_query = ensure_read_only_sql(query)
        except ValueError as exc:
            return [str(exc)]
        if _goal_requires_unlimited_rows(goal) and _sql_query_has_unrequested_limit(safe_query):
            return ["SQL query adds an unrequested row limit; the user asked for all rows."]
        database = action.inputs.get("database")
        configured = resolve_sql_databases(self.settings)
        if configured and database:
            normalized = _normalize_database_name(str(database))
            if normalized not in {_normalize_database_name(name) for name in configured}:
                return [f"Unknown SQL database: {database}"]
        if configured and len(configured) > 1 and not str(database or "").strip():
            return ["SQL action must include database when multiple databases are configured."]
        if _mentioned_database(goal, configured) and not str(database or "").strip():
            return ["SQL action must include database when the goal names a configured database."]
        try:
            catalog = get_sql_catalog(self.settings, str(database) if database else None)
        except Exception:
            catalog = None
        if catalog is not None and catalog.dialect == "postgresql":
            normalized = normalize_pg_relation_quoting(safe_query, catalog)
            normalized = _repair_pg_relation_qualified_columns(normalized, catalog)
            normalized = _repair_sql_empty_date_comparisons(normalized, catalog)
            normalized = _repair_sql_date_year_extraction(normalized, catalog)
            normalized = _repair_sql_age_argument_order(normalized, catalog)
            normalized = _repair_sql_relationship_query_from_goal(normalized, catalog, goal=goal) or normalized
            concept_errors = _validate_sql_goal_concepts(normalized, catalog, goal=goal)
            if concept_errors:
                return concept_errors
            ast_validation = normalize_and_validate_sql_ast(normalized, catalog)
            if not ast_validation.valid:
                return ast_validation.messages
            normalized = ast_validation.normalized_sql
            validation = validate_read_only_sql(normalized)
            if not validation.valid:
                return [validation.reason or "SQL failed validation."]
            reference_errors = _validate_sql_catalog_references(normalized, catalog)
            if reference_errors:
                return reference_errors
            action.inputs["query"] = normalized
        else:
            action.inputs["query"] = safe_query
        return []

    def _validate_fs_action(self, action: PlannedAction, *, goal: str) -> list[str]:
        """Handle the internal validate fs action helper path for this module.

        Inputs:
            Receives action, goal for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator._validate_fs_action calls and related tests.
        """
        path_keys = [key for key in ("path", "src", "dst") if key in action.inputs]
        errors: list[str] = []
        for key in path_keys:
            raw = action.inputs.get(key)
            if not isinstance(raw, str):
                continue
            if ".." in Path(raw).parts:
                errors.append(f"Unsafe path traversal in {key}: {raw}")
                continue
            try:
                resolved = resolve_path(self.settings, raw)
                resolved.relative_to(self.settings.workspace_root.resolve())
            except Exception:
                errors.append(f"Path for {action.tool} is outside the workspace root: {raw}")
        if action.tool == "fs.write" and not EXPORT_GOAL_RE.search(goal):
            errors.append("fs.write requires explicit save/write/export file intent.")
        return errors

    def _validate_goal_flow(self, plan: ActionPlan, *, goal: str) -> list[str]:
        """Handle the internal validate goal flow helper path for this module.

        Inputs:
            Receives plan, goal for this ActionPlanValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanValidator._validate_goal_flow calls and related tests.
        """
        actions = [action.tool for action in plan.actions]
        errors: list[str] = []
        output_intent = resolve_output_intent(goal, planner_expected_shape=plan.expected_final_shape.model_dump())
        if not actions or actions[-1] != "runtime.return":
            errors.append("Final action must be runtime.return.")
        if _looks_like_sql_goal(goal) and not any(action.tool in {"sql.query", "sql.schema", "sql.validate"} for action in plan.actions):
            errors.append("Database-shaped goals must include a SQL action, not literal or precomputed data.")
        sql_query_count = sum(1 for action in plan.actions if action.tool == "sql.query")
        if output_intent.kind == "scalar" and output_intent.cardinality == "single" and sql_query_count > 1:
            errors.append("Single-scalar SQL goals must use one aggregate sql.query, not multiple SQL queries combined later.")
        if output_intent.cardinality == "multi_scalar" and sql_query_count > 1:
            errors.append("Multi-scalar SQL goals must use one aggregate sql.query with multiple aggregate columns, not multiple SQL queries combined later.")
        if "sql.query" in actions and "text.format" not in actions:
            errors.append("SQL results must be formatted locally with text.format before returning or writing.")
        data_seen = False
        for action in plan.actions:
            if action.tool == "fs.write" and data_seen and not _collect_action_refs(action.inputs.get("content")):
                errors.append("fs.write content must reference upstream formatted output.")
            if action.tool in {
                "sql.query",
                "fs.read",
                "fs.list",
                "fs.find",
                "fs.glob",
                "fs.search_content",
                "fs.aggregate",
                "text.format",
                "shell.exec",
                "slurm.queue",
                "slurm.nodes",
                "slurm.partitions",
                "slurm.accounting",
            }:
                data_seen = True
        if EXPORT_GOAL_RE.search(goal):
            if "fs.write" not in actions:
                errors.append("Export/save goals must include fs.write.")
            if "sql.query" in actions and "text.format" not in actions:
                errors.append("Export/save goals must include text.format before fs.write.")
        elif "fs.write" in actions:
            errors.append("Display-only goals must not write files.")
        errors.extend(_validate_shell_scalar_flow(plan, goal=goal))
        errors.extend(_validate_goal_domain_contract(plan, goal=goal))
        return errors


class ActionPlanCanonicalizer:
    """Represent action plan canonicalizer within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ActionPlanCanonicalizer.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ActionPlanCanonicalizer and related tests.
    """
    def __init__(
        self,
        *,
        goal: str,
        settings: Settings | None = None,
        failure_context: dict[str, Any] | None = None,
        llm: Any | None = None,
        now: Any | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives goal, settings, failure_context, llm, now, tools for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer.__init__ calls and related tests.
        """
        self.goal = goal
        self.settings = settings
        self.failure_context = failure_context or {}
        self.llm = llm
        self.now = now
        self.tools = tools
        self.repairs: list[str] = []
        self.issues: list[PlanIssue] = []
        self.symbol_table: PlanSymbolTable | None = None
        self.metadata: dict[str, Any] = {}

    def canonicalize(self, plan: ActionPlan) -> CanonicalizationResult:
        """Canonicalize for ActionPlanCanonicalizer instances.

        Inputs:
            Receives plan for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer.canonicalize calls and related tests.
        """
        payload = plan.model_dump()
        actions = [dict(action) for action in payload.get("actions", [])]
        if not actions:
            return CanonicalizationResult(plan=plan)

        for index, action in enumerate(actions, start=1):
            action["id"] = _normalize_name(action.get("id") or f"action_{index}") or f"action_{index}"
            action["tool"] = _normalize_tool_name(action.get("tool"))
            action["inputs"] = _normalize_action_inputs(dict(action.get("inputs") or {}), tool=action["tool"])
            action["output_binding"] = _normalize_name(action.get("output_binding") or f"{action['id']}_output")
            action["depends_on"] = [_normalize_name(dep) for dep in list(action.get("depends_on") or []) if _normalize_name(dep)]

        if len(actions) == 1 and actions[0]["tool"] == "runtime.return":
            return CanonicalizationResult(plan=ActionPlan.model_validate({**payload, "actions": actions}))

        actions = [action for action in actions if action["tool"] != "runtime.return"]
        self._propagate_sql_database(actions)
        self._normalize_temporal_arguments(actions)
        self._apply_semantic_obligations(actions)
        self._canonicalize_tool_arguments(actions)
        self._repair_grouped_slurm_queue_counts(actions)
        self._bound_broad_diagnostic_actions(actions)
        self._rewrite_schema_introspection_queries(actions)
        self._rewrite_explain_only_sql(actions)
        self._rewrite_bare_data_refs(actions)
        export_path = _extract_export_path(self.goal)
        export_goal = bool(export_path or EXPORT_GOAL_RE.search(self.goal))
        self._repair_missing_scalar_count_sql(actions)
        self._repair_scalar_count_absence_query(actions)
        self._repair_count_shape_query(actions)

        if export_goal:
            actions = self._canonicalize_export(actions, export_path=export_path)
        else:
            actions = self._canonicalize_display(actions)

        final_plan = ActionPlan.model_validate(
            {
                "goal": payload.get("goal") or self.goal,
                "actions": self._prune_unused_actions(actions),
                "expected_final_shape": payload.get("expected_final_shape") or {},
                "notes": payload.get("notes") or [],
            }
        )
        return CanonicalizationResult(
            plan=final_plan,
            repairs=list(self.repairs),
            issues=list(self.issues),
            symbol_table=(self.symbol_table.model_dump() if self.symbol_table is not None else {}),
            metadata=dict(self.metadata),
        )

    def _canonicalize_display(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Handle the internal canonicalize display helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._canonicalize_display calls and related tests.
        """
        if not actions:
            return actions
        scalar_actions = self._canonicalize_scalar_display(actions)
        if scalar_actions is not None:
            return scalar_actions
        producer = _last_data_action(actions)
        formatter = producer if producer is not None and producer.get("tool") == "text.format" else None
        if producer is not None and formatter is None:
            formatter = _formatter_after_producer(actions, producer)
        if producer is not None and formatter is None:
            formatter = self._make_formatter(producer)
            actions.append(formatter)
            self._repair("inserted_text_format", f"Inserted text.format after {producer['tool']}.")
        elif formatter is not None:
            self._ensure_formatter_source(formatter, actions)
        self._rewrite_bare_data_refs(actions)
        final = self._make_return(formatter or _last_data_action(actions), dependency=actions[-1] if actions else None)
        actions.append(final)
        self._repair("ensured_runtime_return", "Ensured plan ends with runtime.return.")
        return actions

    def _canonicalize_scalar_display(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Handle the internal canonicalize scalar display helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._canonicalize_scalar_display calls and related tests.
        """
        contract = infer_goal_output_contract(self.goal)
        if contract.kind != "scalar":
            return None
        producer = _last_scalar_action(actions, goal=self.goal)
        if producer is None:
            return None
        final = self._make_scalar_return(producer)
        actions.append(final)
        self._repair(
            "repaired_scalar_final_output",
            f"Repaired scalar final output to use {producer['tool']} count field.",
        )
        return actions

    def _canonicalize_export(self, actions: list[dict[str, Any]], *, export_path: str | None) -> list[dict[str, Any]]:
        """Handle the internal canonicalize export helper path for this module.

        Inputs:
            Receives actions, export_path for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._canonicalize_export calls and related tests.
        """
        if not actions:
            return actions
        write_action = _first_action_with_tool(actions, "fs.write")
        formatter = _last_action_with_tool(actions, "text.format")
        producer = formatter or _last_data_action([action for action in actions if action["tool"] != "fs.write"])
        if producer is not None and formatter is None:
            formatter = self._make_formatter(producer, output_path=export_path)
            write_index = actions.index(write_action) if write_action in actions else len(actions)
            actions.insert(write_index, formatter)
            self._repair("inserted_text_format", f"Inserted text.format before export write for {producer['tool']}.")
        elif formatter is not None:
            self._ensure_formatter_source(formatter, actions)
            if export_path:
                formatter.setdefault("inputs", {})["output_path"] = export_path

        if write_action is None and export_path:
            dependency = formatter or _last_data_action(actions)
            write_action = {
                "id": _unique_action_id(actions, "write_file"),
                "tool": "fs.write",
                "purpose": f"Write formatted output to {export_path}.",
                "inputs": {"path": export_path, "content": f"${dependency['id']}.content" if dependency else ""},
                "depends_on": [dependency["id"]] if dependency else [],
                "output_binding": "written_file",
                "expected_result_shape": {"kind": "file"},
            }
            actions.append(write_action)
            self._repair("inserted_fs_write", f"Inserted fs.write for explicit export path {export_path}.")
        elif write_action is not None and formatter is not None:
            write_action.setdefault("inputs", {})
            if export_path and not str(write_action["inputs"].get("path") or "").strip():
                write_action["inputs"]["path"] = export_path
            if not _collect_action_refs(write_action["inputs"].get("content")):
                write_action["inputs"]["content"] = f"${formatter['id']}.content"
                self._repair("repaired_fs_write_content", "Repaired fs.write content to reference formatted output.")
            if formatter["id"] not in write_action.get("depends_on", []):
                write_action.setdefault("depends_on", []).append(formatter["id"])

        final_source = formatter or write_action or _last_data_action(actions)
        dependency = write_action or final_source
        self._rewrite_bare_data_refs(actions)
        actions.append(self._make_return(final_source, dependency=dependency))
        self._repair("ensured_runtime_return", "Ensured export plan ends with runtime.return.")
        return actions

    def _make_formatter(
        self,
        producer: dict[str, Any],
        *,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Handle the internal make formatter helper path for this module.

        Inputs:
            Receives producer, output_path for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._make_formatter calls and related tests.
        """
        producer_id = str(producer["id"])
        producer_tool = str(producer["tool"])
        path = formatter_source_path_for_tool(producer_tool)
        source = f"${producer_id}.{path}" if path else f"${producer_id}"
        output_format = _format_for_goal(self.goal, producer=producer, output_path=output_path)
        inputs: dict[str, Any] = {"source": source, "format": output_format}
        if output_path:
            inputs["output_path"] = output_path
        if producer_tool == "sql.query":
            query = dict(producer.get("inputs") or {}).get("query")
            if isinstance(query, str) and query.strip():
                inputs["query_used"] = query
        return {
            "id": _unique_action_id([], f"{producer_id}_format"),
            "tool": "text.format",
            "purpose": "Format tool output locally.",
            "inputs": inputs,
            "depends_on": [producer_id],
            "output_binding": f"{producer_id}_formatted",
            "expected_result_shape": {"kind": "text", "format": output_format},
        }

    def _ensure_formatter_source(self, formatter: dict[str, Any], actions: list[dict[str, Any]]) -> None:
        """Handle the internal ensure formatter source helper path for this module.

        Inputs:
            Receives formatter, actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._ensure_formatter_source calls and related tests.
        """
        inputs = formatter.setdefault("inputs", {})
        try:
            formatter_index = actions.index(formatter)
        except ValueError:
            formatter_index = len(actions)
        producer = _last_data_action(actions[:formatter_index])
        format_basis = producer or formatter
        current_format = _normalize_format(inputs.get("format") or _format_for_goal(self.goal, producer=format_basis))
        desired_format = _format_for_goal(self.goal, producer=format_basis)
        if desired_format == "markdown" and current_format in {"txt", "json"}:
            inputs["format"] = desired_format
            if current_format == "json":
                self._repair("repaired_user_visible_json_format", "Repaired user-visible JSON formatting to Markdown.")
        else:
            inputs["format"] = current_format
        rewritten = self._rewrite_data_value(inputs.get("source"), preferred_path=None)
        if rewritten != inputs.get("source"):
            inputs["source"] = rewritten
            self._ensure_depends_on_ref(formatter, rewritten)
            self._repair("repaired_text_format_source", "Repaired text.format source reference.")
            return
        if inputs.get("source"):
            return
        if producer is None:
            return
        path = formatter_source_path_for_tool(str(producer["tool"]))
        inputs["source"] = f"${producer['id']}.{path}" if path else f"${producer['id']}"
        formatter.setdefault("depends_on", []).append(producer["id"])
        self._repair("repaired_text_format_source", "Repaired text.format source reference.")

    def _make_return(self, source: dict[str, Any] | None, *, dependency: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle the internal make return helper path for this module.

        Inputs:
            Receives source, dependency for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._make_return calls and related tests.
        """
        value = ""
        if source is not None:
            source_tool = str(source.get("tool") or "")
            if source_tool == "text.format":
                value = f"${source['id']}.content"
            elif source_tool == "fs.write":
                value = f"${source['id']}.path"
            else:
                path = return_value_path_for_tool(source_tool)
                value = f"${source['id']}.{path}" if path else f"${source['id']}"
        depends_on = [dependency["id"]] if dependency is not None else ([source["id"]] if source is not None else [])
        return {
            "id": _unique_action_id([], "return_result"),
            "tool": "runtime.return",
            "purpose": "Return the final user-facing result.",
            "inputs": {"value": value, "mode": "text"},
            "depends_on": depends_on,
            "output_binding": "runtime_return_result",
            "expected_result_shape": {"kind": "text"},
        }

    def _make_scalar_return(self, source: dict[str, Any]) -> dict[str, Any]:
        """Handle the internal make scalar return helper path for this module.

        Inputs:
            Receives source for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._make_scalar_return calls and related tests.
        """
        source_tool = str(source.get("tool") or "")
        field = scalar_field_for_tool(source_tool, goal=self.goal)
        value = f"${source['id']}.{field}" if field else f"${source['id']}"
        return {
            "id": _unique_action_id([], "return_count"),
            "tool": "runtime.return",
            "purpose": "Return the requested scalar count.",
            "inputs": {"value": value, "mode": "count"},
            "depends_on": [source["id"]],
            "output_binding": "runtime_return_result",
            "expected_result_shape": {"kind": "scalar"},
        }

    def _repair(self, code: str, message: str) -> None:
        """Handle the internal repair helper path for this module.

        Inputs:
            Receives code, message for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._repair calls and related tests.
        """
        self.repairs.append(message)
        self.issues.append(PlanIssue(code=code, message=message))

    def _symbol_table(self, actions: list[dict[str, Any]]) -> "PlanSymbolTable":
        """Handle the internal symbol table helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._symbol_table calls and related tests.
        """
        self.symbol_table = PlanSymbolTable.from_actions(actions)
        return self.symbol_table

    def _propagate_sql_database(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal propagate sql database helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._propagate_sql_database calls and related tests.
        """
        if self.settings is None:
            return
        configured = resolve_sql_databases(self.settings)
        if not configured:
            return
        selected = _database_for_goal(self.goal, configured)
        if selected is None and len(configured) == 1:
            selected = next(iter(configured))
        if not selected:
            return
        for action in actions:
            if action.get("tool") not in {"sql.query", "sql.schema"}:
                continue
            inputs = action.setdefault("inputs", {})
            if str(inputs.get("database") or "").strip():
                continue
            inputs["database"] = selected
            self._repair("propagated_sql_database", f"Set {action.get('tool')} database to {selected}.")

    def _normalize_temporal_arguments(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal normalize temporal arguments helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._normalize_temporal_arguments calls and related tests.
        """
        if self.settings is None:
            return
        try:
            result = TemporalArgumentCanonicalizer(goal=self.goal, settings=self.settings, llm=self.llm, now=self.now).canonicalize(actions)
        except TemporalNormalizationError:
            raise
        for repair in result.repairs:
            self._repair("normalized_temporal_arguments", repair)
        if result.metadata:
            self.metadata["temporal_normalization"] = dict(result.metadata)
            self.issues.append(
                PlanIssue(
                    code="temporal_normalization_metadata",
                    message=json.dumps(result.metadata, sort_keys=True, default=str),
                    severity="info",
                )
            )
        if result.llm_calls:
            self.metadata["temporal_llm_calls"] = result.llm_calls

    def _apply_semantic_obligations(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal apply semantic obligations helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._apply_semantic_obligations calls and related tests.
        """
        result = apply_semantic_obligations_to_actions(self.goal, actions)
        if result.metadata["obligations"]:
            self.metadata["semantic_obligations"] = result.metadata
            self.issues.append(
                PlanIssue(
                    code="semantic_obligations",
                    message=json.dumps(result.metadata, sort_keys=True, default=str),
                    severity="info",
                )
            )
        for repair in result.repairs:
            self._repair("applied_semantic_obligation", repair)

    def _canonicalize_tool_arguments(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal canonicalize tool arguments helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._canonicalize_tool_arguments calls and related tests.
        """
        if self.tools is None:
            return
        result = ToolArgumentCanonicalizer(self.tools).canonicalize(actions)
        if result.metadata:
            self.metadata["tool_argument_canonicalization"] = result.metadata
        for repair in result.repairs:
            self._repair("canonicalized_tool_arguments", repair)

    def _repair_grouped_slurm_queue_counts(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal repair grouped slurm queue counts helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._repair_grouped_slurm_queue_counts calls and related tests.
        """
        group_by = grouped_count_field_for_goal(self.goal)
        if group_by is None:
            return
        for action in actions:
            if action.get("tool") != "slurm.queue":
                continue
            inputs = action.setdefault("inputs", {})
            current = str(inputs.get("group_by") or "").strip().lower().replace(" ", "_")
            if current != group_by:
                inputs["group_by"] = group_by
                self._repair("repaired_slurm_grouped_count", f"Set slurm.queue group_by={group_by} for grouped count request.")
            if "limit" not in inputs:
                inputs["limit"] = None
                self._repair("repaired_slurm_grouped_count", "Removed default queue limit for grouped count request.")
            if group_by == "partition" and str(inputs.get("partition") or "").strip():
                inputs.pop("partition", None)
                self._repair("repaired_slurm_grouped_count", "Removed single-partition filter from grouped partition count.")
            expected_shape = dict(action.get("expected_result_shape") or {})
            if _normalize_shape_kind(expected_shape.get("kind")) == "scalar":
                expected_shape["kind"] = "table"
                action["expected_result_shape"] = expected_shape
                self._repair("repaired_grouped_count_shape", "Repaired grouped count expected shape to table.")

    def _bound_broad_diagnostic_actions(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal bound broad diagnostic actions helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._bound_broad_diagnostic_actions calls and related tests.
        """
        diagnostic_plan = diagnostic_plan_for_goal(self.goal)
        if diagnostic_plan is None:
            return
        allowed = {tool for section in diagnostic_plan.sections for tool in section.allowed_tools}
        allowed.update({"text.format", "runtime.return"})
        bounded: list[dict[str, Any]] = []
        seen_section_tools: set[str] = set()
        pruned: list[str] = []
        for action in actions:
            tool = str(action.get("tool") or "")
            action_id = str(action.get("id") or tool)
            if tool not in allowed:
                pruned.append(action_id)
                continue
            if tool not in {"text.format", "runtime.return"}:
                if tool in seen_section_tools and tool != "fs.search_content":
                    pruned.append(action_id)
                    continue
                seen_section_tools.add(tool)
            bounded.append(action)
            if len([item for item in bounded if item.get("tool") not in {"text.format", "runtime.return"}]) >= diagnostic_plan.budget.max_actions - 2:
                break
        if len(bounded) != len(actions):
            if not bounded:
                bounded.append(
                    {
                        "id": "workspace_summary",
                        "tool": "fs.list",
                        "purpose": "Summarize the workspace as the first diagnostic section.",
                        "inputs": {"path": "."},
                        "depends_on": [],
                        "output_binding": "workspace_entries",
                        "expected_result_shape": {"kind": "table"},
                    }
                )
            actions[:] = bounded
            self.metadata["diagnostic_orchestration"] = {
                "budget": diagnostic_plan.budget.model_dump(),
                "pruned_actions": pruned,
            }
            self._repair("bounded_diagnostic_plan", "Bounded broad diagnostic request to compact staged inspection actions.")

    def _rewrite_schema_introspection_queries(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal rewrite schema introspection queries helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._rewrite_schema_introspection_queries calls and related tests.
        """
        if not _schema_question_goal(self.goal):
            return
        for action in actions:
            if action.get("tool") != "sql.query":
                continue
            query = str(action.get("inputs", {}).get("query") or "")
            if not re.search(r"(?i)\binformation_schema\b|\bpg_catalog\b", query):
                continue
            inputs = {"database": action.get("inputs", {}).get("database")}
            action["tool"] = "sql.schema"
            action["inputs"] = {key: value for key, value in inputs.items() if str(value or "").strip()}
            action["expected_result_shape"] = {"kind": "table"}
            self._repair("rewrote_schema_introspection_query", "Rewrote system-catalog SQL query to sql.schema.")

    def _rewrite_explain_only_sql(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal rewrite explain only sql helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._rewrite_explain_only_sql calls and related tests.
        """
        if not _is_sql_explain_only_goal(self.goal):
            return
        for action in actions:
            if action.get("tool") != "sql.query":
                continue
            action["tool"] = "sql.validate"
            action["expected_result_shape"] = {"kind": "text"}
            self._repair("rewrote_sql_execute_to_validate", "Rewrote SQL execution to non-executing sql.validate for explain-only request.")

    def _rewrite_bare_data_refs(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal rewrite bare data refs helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._rewrite_bare_data_refs calls and related tests.
        """
        self._symbol_table(actions)
        for action in actions:
            tool = str(action.get("tool") or "")
            inputs = action.setdefault("inputs", {})
            if tool == "text.format" and "source" in inputs:
                rewritten = self._rewrite_data_value(inputs.get("source"), preferred_path=None)
                if rewritten != inputs.get("source"):
                    inputs["source"] = rewritten
                    self._ensure_depends_on_ref(action, rewritten)
                    self._repair("rewrote_formatter_source_ref", "Rewrote text.format source to structured output reference.")
            elif tool == "runtime.return" and "value" in inputs:
                rewritten = self._rewrite_data_value(inputs.get("value"), preferred_path="content")
                if rewritten != inputs.get("value"):
                    inputs["value"] = rewritten
                    self._ensure_depends_on_ref(action, rewritten)
                    self._repair("rewrote_return_value_ref", "Rewrote runtime.return value to structured output reference.")
            elif tool == "fs.write" and "content" in inputs:
                rewritten = self._rewrite_data_value(inputs.get("content"), preferred_path="content")
                if rewritten != inputs.get("content"):
                    inputs["content"] = rewritten
                    self._ensure_depends_on_ref(action, rewritten)
                    self._repair("rewrote_write_content_ref", "Rewrote fs.write content to structured output reference.")

    def _rewrite_data_value(self, value: Any, *, preferred_path: str | None) -> Any:
        """Handle the internal rewrite data value helper path for this module.

        Inputs:
            Receives value, preferred_path for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._rewrite_data_value calls and related tests.
        """
        if isinstance(value, dict) and "$ref" in value:
            ref = _normalize_name(str(value.get("$ref") or ""))
            symbol = self.symbol_table.lookup(ref) if self.symbol_table is not None else None
            if symbol is None:
                return value
            rewritten = dict(value)
            rewritten["$ref"] = symbol.name
            path = str(rewritten.get("path") or "").strip()
            expected_path = _ref_path_for_symbol(symbol, preferred_path)
            if not path or _is_generic_wrong_ref_path(path, symbol):
                rewritten["path"] = expected_path
            elif path_is_declared_for_tool(symbol.tool, path):
                rewritten["path"] = normalize_tool_ref_path(symbol.tool, path)
            return rewritten
        if isinstance(value, dict):
            return {key: self._rewrite_data_value(nested, preferred_path=preferred_path) for key, nested in value.items()}
        if isinstance(value, list):
            return [self._rewrite_data_value(nested, preferred_path=preferred_path) for nested in value]
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text.startswith("$"):
            return value
        placeholders = re.findall(r"\{([A-Za-z0-9_-]+)\}", text)
        if placeholders and self.symbol_table is not None:
            symbols = [self.symbol_table.lookup(item) for item in placeholders]
            known_symbols = [symbol for symbol in symbols if symbol is not None]
            if len(known_symbols) == 1:
                symbol = known_symbols[0]
                return {"$ref": symbol.name, "path": _ref_path_for_symbol(symbol, preferred_path)}
        symbol = self.symbol_table.lookup(text) if self.symbol_table is not None else None
        if symbol is None:
            return value
        return {"$ref": symbol.name, "path": _ref_path_for_symbol(symbol, preferred_path)}

    def _ensure_depends_on_ref(self, action: dict[str, Any], value: Any) -> None:
        """Handle the internal ensure depends on ref helper path for this module.

        Inputs:
            Receives action, value for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._ensure_depends_on_ref calls and related tests.
        """
        refs = _collect_action_refs(value)
        if not refs or self.symbol_table is None:
            return
        depends = action.setdefault("depends_on", [])
        for ref in refs:
            symbol = self.symbol_table.lookup(ref)
            if symbol is None or symbol.action_id == action.get("id"):
                continue
            if symbol.action_id not in depends:
                depends.append(symbol.action_id)

    def _prune_unused_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Handle the internal prune unused actions helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._prune_unused_actions calls and related tests.
        """
        if not actions:
            return actions
        required_ids: set[str] = set()
        by_id = {str(action.get("id")): action for action in actions}

        def mark(action: dict[str, Any]) -> None:
            """Mark for the surrounding runtime workflow.

            Inputs:
                Receives action for this function; type hints and validators define accepted shapes.

            Returns:
                Returns None; side effects are limited to the local runtime operation described above.

            Used by:
                Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner.mark.
            """
            action_id = str(action.get("id") or "")
            if not action_id or action_id in required_ids:
                return
            required_ids.add(action_id)
            for dep in action.get("depends_on") or []:
                dep_action = by_id.get(str(dep))
                if dep_action is not None:
                    mark(dep_action)
            for ref in _collect_action_refs(action.get("inputs")):
                symbol = self.symbol_table.lookup(ref) if self.symbol_table is not None else None
                dep_action = by_id.get(symbol.action_id if symbol is not None else ref)
                if dep_action is not None:
                    mark(dep_action)

        for action in actions:
            if action.get("tool") in {"runtime.return", "fs.write"}:
                mark(action)
        if not required_ids:
            return actions
        pruned = [action for action in actions if str(action.get("id")) in required_ids]
        if len(pruned) != len(actions):
            self._repair("pruned_unused_actions", "Removed unused read-only exploratory actions from final DAG.")
        return pruned

    def _repair_count_shape_query(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal repair count shape query helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._repair_count_shape_query calls and related tests.
        """
        repaired_query = _count_shape_repair_query(self.goal, self.failure_context)
        if not repaired_query:
            return
        sql_action = _first_action_with_tool(actions, "sql.query")
        if sql_action is None:
            return
        inputs = sql_action.setdefault("inputs", {})
        current_query = str(inputs.get("query") or "")
        if _looks_like_outer_count_query(current_query):
            return
        inputs["query"] = repaired_query
        sql_action["expected_result_shape"] = {"kind": "scalar"}
        self._repair("repaired_count_shape_query", "Repaired grouped count SQL to return one numeric scalar.")

    def _repair_scalar_count_absence_query(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal repair scalar count absence query helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._repair_scalar_count_absence_query calls and related tests.
        """
        if self.settings is None or _is_sql_explain_only_goal(self.goal) or not _is_scalar_count_goal(self.goal):
            return
        if sum(1 for action in actions if action.get("tool") == "sql.query") <= 1:
            return
        configured = resolve_sql_databases(self.settings)
        database = _database_for_goal(self.goal, configured) if configured else None
        if database is None and len(configured) == 1:
            database = next(iter(configured))
        if not database:
            return
        try:
            catalog = get_sql_catalog(self.settings, database)
        except Exception:
            return
        query = _scalar_count_absence_query(self.goal, catalog)
        if not query:
            return
        actions[:] = [
            {
                "id": "count_query",
                "tool": "sql.query",
                "purpose": "Count base rows that have no matching related rows.",
                "inputs": {"database": database, "query": query},
                "depends_on": [],
                "output_binding": "count_result",
                "expected_result_shape": {"kind": "scalar"},
            }
        ]
        self._repair("repaired_absence_count_query", "Repaired multi-query count arithmetic into one anti-join SQL aggregate.")

    def _repair_missing_scalar_count_sql(self, actions: list[dict[str, Any]]) -> None:
        """Handle the internal repair missing scalar count sql helper path for this module.

        Inputs:
            Receives actions for this ActionPlanCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCanonicalizer._repair_missing_scalar_count_sql calls and related tests.
        """
        if self.settings is None or _is_sql_explain_only_goal(self.goal) or not _is_scalar_count_goal(self.goal) or not _looks_like_sql_goal(self.goal):
            return
        if any(action.get("tool") == "sql.query" for action in actions):
            return
        configured = resolve_sql_databases(self.settings)
        database = _database_for_goal(self.goal, configured) if configured else None
        if database is None and len(configured) == 1:
            database = next(iter(configured))
        if not database:
            return
        try:
            catalog = get_sql_catalog(self.settings, database)
        except Exception:
            return
        query = _scalar_count_query_from_goal(self.goal, catalog)
        if not query:
            return
        actions[:] = [
            {
                "id": "count_query",
                "tool": "sql.query",
                "purpose": "Count requested database rows.",
                "inputs": {"database": database, "query": query},
                "depends_on": [],
                "output_binding": "count_result",
                "expected_result_shape": {"kind": "scalar"},
            }
        ]
        self._repair("inserted_scalar_count_query", "Inserted missing SQL aggregate for scalar database count goal.")

class ActionPlanCompiler:
    """Represent action plan compiler within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ActionPlanCompiler.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.ActionPlanCompiler and related tests.
    """
    def __init__(self, *, goal: str) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives goal for this ActionPlanCompiler method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCompiler.__init__ calls and related tests.
        """
        self.goal = goal

    def compile(self, plan: ActionPlan) -> ExecutionPlan:
        """Compile for ActionPlanCompiler instances.

        Inputs:
            Receives plan for this ActionPlanCompiler method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through ActionPlanCompiler.compile calls and related tests.
        """
        aliases_by_action: dict[str, str] = {}
        steps: list[ExecutionStep] = []
        for index, action in enumerate(plan.actions, start=1):
            output = str(action.output_binding or f"{action.id}_output")
            aliases_by_action[action.id] = output
            args = _normalize_refs(action.inputs, aliases_by_action)
            steps.append(
                ExecutionStep(
                    id=index,
                    action=action.tool,
                    args=args,
                    input=sorted(collect_step_references(args)),
                    output=output,
                )
            )
        execution_plan = ExecutionPlan(steps=steps)
        normalize_execution_plan_dataflow(execution_plan)
        return execution_plan


def build_tool_manifest(tools: ToolRegistry, allowed_tools: list[str]) -> list[dict[str, Any]]:
    """Build tool manifest for the surrounding runtime workflow.

    Inputs:
        Receives tools, allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner.build_tool_manifest.
    """
    manifest: list[dict[str, Any]] = []
    for spec in tools.specs(allowed_tools):
        name = str(spec["name"])
        if name == "python.exec":
            continue
        manifest.append(
            {
                "name": name,
                "description": spec.get("description"),
                "input_schema": spec.get("arguments_schema"),
                "constraints": _constraints_for_tool(name),
            }
        )
    if "text.format" not in {item["name"] for item in manifest}:
        try:
            spec = tools.get("text.format").spec.model_dump()
            manifest.append(
                {
                    "name": "text.format",
                    "description": spec.get("description"),
                    "input_schema": spec.get("arguments_schema"),
                    "constraints": _constraints_for_tool("text.format"),
                }
            )
        except Exception:
            pass
    if "sql.schema" not in {item["name"] for item in manifest}:
        try:
            spec = tools.get("sql.schema").spec.model_dump()
            manifest.append(
                {
                    "name": "sql.schema",
                    "description": spec.get("description"),
                    "input_schema": spec.get("arguments_schema"),
                    "constraints": _constraints_for_tool("sql.schema"),
                }
            )
        except Exception:
            pass
    if "sql.validate" not in {item["name"] for item in manifest}:
        try:
            spec = tools.get("sql.validate").spec.model_dump()
            manifest.append(
                {
                    "name": "sql.validate",
                    "description": spec.get("description"),
                    "input_schema": spec.get("arguments_schema"),
                    "constraints": _constraints_for_tool("sql.validate"),
                }
            )
        except Exception:
            pass
    if "runtime.return" not in {item["name"] for item in manifest}:
        try:
            spec = tools.get("runtime.return").spec.model_dump()
            manifest.append(
                {
                    "name": "runtime.return",
                    "description": spec.get("description"),
                    "input_schema": spec.get("arguments_schema"),
                    "constraints": _constraints_for_tool("runtime.return"),
                }
            )
        except Exception:
            pass
    return manifest


def _constraints_for_tool(name: str) -> list[str]:
    """Handle the internal constraints for tool helper path for this module.

    Inputs:
        Receives name for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._constraints_for_tool.
    """
    if name == "sql.query":
        return ["SELECT-only", "known database", "known schema identifiers", "PostgreSQL mixed-case identifiers must be quoted"]
    if name == "sql.schema":
        return ["read-only schema inspection", "no row data"]
    if name == "sql.validate":
        return ["validate and explain SQL without executing", "SELECT-only", "known schema identifiers"]
    if name == "shell.exec":
        return ["read-only shell safety classifier", "destructive/admin commands blocked"]
    if name == "fs.write":
        return ["workspace-root only", "requires explicit file artifact intent", "content must come from upstream formatted output"]
    if name == "text.format":
        return ["local deterministic formatting only", "format must be txt/csv/json/markdown"]
    if name.startswith("slurm."):
        return ["read-only SLURM inspection only", "start/end time arguments must be ISO-like dates or datetimes"]
    return []


def _should_include_sql_schema(goal: str, allowed_tools: list[str], settings: Settings) -> bool:
    """Handle the internal should include sql schema helper path for this module.

    Inputs:
        Receives goal, allowed_tools, settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._should_include_sql_schema.
    """
    if "sql.query" not in allowed_tools:
        return False
    configured = resolve_sql_databases(settings)
    text = str(goal or "").lower()
    if any(name.lower() in text for name in configured):
        return True
    return bool(SQL_GOAL_RE.search(text))


def _compact_sql_schema(settings: Settings) -> dict[str, Any]:
    """Handle the internal compact sql schema helper path for this module.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._compact_sql_schema.
    """
    databases: list[dict[str, Any]] = []
    for catalog in get_all_sql_catalogs(settings):
        databases.append(
            {
                "database": catalog.database,
                "dialect": catalog.dialect,
                "error": catalog.error,
                "tables": [
                    {
                        "schema": table.schema_name,
                        "table": table.table_name,
                        "qualified_name": table.qualified_name,
                        "columns": [
                            {
                                "name": column.column_name,
                                "type": column.data_type,
                                "primary_key": column.primary_key,
                                "foreign_key": column.foreign_key,
                            }
                            for column in table.columns
                        ],
                        "primary_key_columns": list(table.primary_key_columns),
                        "foreign_keys": list(table.foreign_keys),
                    }
                    for table in catalog.tables
                ],
            }
        )
    return {
        "rules": [
            "Use exact identifiers from schema.",
            'For PostgreSQL, quote mixed-case identifiers exactly, e.g. "PatientID".',
            "Do not invent columns or tables.",
        ],
        "databases": databases,
    }


def _compact_failure_context(context: dict[str, Any]) -> dict[str, Any]:
    """Handle the internal compact failure context helper path for this module.

    Inputs:
        Receives context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._compact_failure_context.
    """
    if not context:
        return {}
    compact = {
        key: context.get(key)
        for key in (
            "reason",
            "error",
            "failed_step",
            "retryable",
            "shape_error",
            "expected_shape",
            "actual_shape",
            "failed_sql",
            "validation",
            "validation_checks",
            "summary",
        )
        if key in context
    }
    return json.loads(json.dumps(compact, default=str)) if compact else {}


def normalize_raw_action_plan(raw_plan: RawActionPlan, *, fallback_goal: str) -> dict[str, Any]:
    """Normalize raw action plan for the surrounding runtime workflow.

    Inputs:
        Receives raw_plan, fallback_goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner.normalize_raw_action_plan.
    """
    actions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(raw_plan.actions, start=1):
        raw_inputs = raw_action.inputs if raw_action.inputs is not None else raw_action.args
        if raw_action.args is not None and raw_action.inputs == {}:
            raw_inputs = raw_action.args
        inputs = dict(raw_inputs) if isinstance(raw_inputs, dict) else {}
        action_id = _normalize_name(raw_action.id or f"action_{index}") or f"action_{index}"
        actions.append(
            {
                "id": action_id,
                "tool": _normalize_tool_name(raw_action.tool if raw_action.tool is not None else raw_action.action),
                "purpose": str(raw_action.purpose or ""),
                "inputs": _normalize_action_inputs(inputs, tool=_normalize_tool_name(raw_action.tool if raw_action.tool is not None else raw_action.action)),
                "depends_on": _normalize_depends_on(raw_action.depends_on),
                "output_binding": _normalize_name(raw_action.output_binding or raw_action.output or f"{action_id}_output"),
                "expected_result_shape": _normalize_shape_payload(raw_action.expected_result_shape),
            }
        )
    notes = raw_plan.notes if isinstance(raw_plan.notes, list) else []
    return {
        "goal": str(raw_plan.goal or fallback_goal),
        "actions": actions,
        "expected_final_shape": _normalize_shape_payload(raw_plan.expected_final_shape),
        "notes": [str(note) for note in notes],
    }


def _normalize_shape_payload(value: Any) -> dict[str, Any]:
    """Handle the internal normalize shape payload helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_shape_payload.
    """
    if isinstance(value, str):
        return {"kind": _normalize_shape_kind(value)}
    if isinstance(value, dict):
        copied = dict(value)
        copied["kind"] = _normalize_shape_kind(copied.get("kind", "unknown"))
        if "format" in copied:
            copied["format"] = _normalize_format(copied.get("format"))
        return copied
    return {"kind": "unknown"}


def _normalize_shape_kind(value: Any) -> str:
    """Handle the internal normalize shape kind helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_shape_kind.
    """
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "unknown").strip().lower()).strip("_")
    return SHAPE_KIND_ALIASES.get(normalized, normalized if normalized in {"scalar", "table", "file", "text", "json", "status"} else "unknown")


def _normalize_format(value: Any) -> str:
    """Handle the internal normalize format helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_format.
    """
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "txt").strip().lower()).strip("_")
    normalized = FORMAT_ALIASES.get(normalized, normalized)
    return normalized if normalized in {"txt", "csv", "json", "markdown"} else "txt"


def _normalize_tool_name(value: Any) -> str:
    """Handle the internal normalize tool name helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_tool_name.
    """
    raw = str(value or "").strip()
    lowered = raw.lower().replace("_", ".")
    return TOOL_ALIASES.get(lowered, raw)


def _normalize_action_inputs(inputs: dict[str, Any], *, tool: str) -> dict[str, Any]:
    """Handle the internal normalize action inputs helper path for this module.

    Inputs:
        Receives inputs, tool for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_action_inputs.
    """
    normalized = dict(inputs)
    if "output_format" in normalized and "format" not in normalized:
        normalized["format"] = normalized.pop("output_format")
    if "format" in normalized:
        normalized["format"] = _normalize_format(normalized.get("format"))
    if tool == "sql.query" and isinstance(normalized.get("query"), str):
        normalized["query"] = html.unescape(str(normalized["query"])).strip().rstrip(";")
    if tool == "runtime.return":
        output_contract = normalized.get("output_contract")
        if isinstance(output_contract, dict) and "kind" in output_contract and "mode" not in output_contract:
            kind = _normalize_shape_kind(output_contract.get("kind"))
            normalized["mode"] = "count" if kind == "scalar" else ("json" if kind == "json" else "text")
            normalized.pop("output_contract", None)
        if "mode" in normalized and str(normalized.get("mode")).strip().lower() not in {"text", "csv", "json", "count"}:
            normalized["mode"] = "text"
    return normalized


def _normalize_depends_on(value: Any) -> list[str]:
    """Handle the internal normalize depends on helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_depends_on.
    """
    if isinstance(value, str):
        return [_normalize_name(value)] if _normalize_name(value) else []
    if isinstance(value, list):
        return [_normalize_name(item) for item in value if _normalize_name(item)]
    return []


def _extract_export_path(goal: str) -> str | None:
    """Handle the internal extract export path helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._extract_export_path.
    """
    match = EXPORT_PATH_RE.search(str(goal or ""))
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def _format_for_goal(goal: str, *, producer: dict[str, Any], output_path: str | None = None) -> str:
    """Handle the internal format for goal helper path for this module.

    Inputs:
        Receives goal, producer, output_path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._format_for_goal.
    """
    if output_path:
        suffix = Path(output_path).suffix.lower().lstrip(".")
        return _normalize_format("markdown" if suffix == "md" else suffix)
    if "json" in str(goal or "").lower():
        return "markdown"
    if "csv" in str(goal or "").lower():
        return "csv"
    if _normalize_shape_kind((producer.get("expected_result_shape") or {}).get("kind")) == "scalar":
        return "txt"
    output_intent = resolve_output_intent(goal)
    if output_intent.kind == "scalar" and output_intent.cardinality == "single":
        return "txt"
    if _is_table_display_goal(goal) and str(producer.get("tool")) in {
        "shell.exec",
        "sql.query",
        "slurm.metrics",
        "slurm.nodes",
        "slurm.queue",
        "fs.list",
        "fs.find",
        "fs.glob",
    }:
        return "markdown"
    return (
        "markdown"
        if str(producer.get("tool")) in {"sql.query", "slurm.metrics", "slurm.nodes", "slurm.queue", "fs.list", "fs.find", "fs.glob"}
        else "txt"
    )


def _is_table_display_goal(goal: str) -> bool:
    """Handle the internal is table display goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._is_table_display_goal.
    """
    text = str(goal or "").lower()
    output_intent = resolve_output_intent(text)
    if EXPORT_GOAL_RE.search(text) or (output_intent.kind == "scalar" and output_intent.cardinality == "single"):
        return False
    return bool(re.search(r"\b(?:list|show|display|return|give\s+me|all)\b", text))


def _goal_requires_unlimited_rows(goal: str) -> bool:
    """Handle the internal goal requires unlimited rows helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._goal_requires_unlimited_rows.
    """
    text = str(goal or "").lower()
    if not _is_table_display_goal(text):
        return False
    if re.search(r"\b(?:top|first|last|sample|limit|head|tail)\b", text):
        return False
    if re.search(r"\btop\s+\d+\b|\bfirst\s+\d+\b|\blast\s+\d+\b|\b\d+\s+(?:rows|records|processes|jobs|files|lines)\b", text):
        return False
    return bool(re.search(r"\b(?:all|list|show|display)\b", text))


def _is_explicit_shell_command_goal(goal: str) -> bool:
    """Handle the internal is explicit shell command goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._is_explicit_shell_command_goal.
    """
    text = str(goal or "").strip().lower()
    return bool(
        re.search(r"^(?:run|execute)\b", text)
        or re.search(r"^shell\s*:", text)
        or re.search(r"\b(?:using|with)\s+shell\b", text)
        or re.search(r"\bshell\s+command\b", text)
    )


def _shell_command_has_unrequested_limit(command: str) -> bool:
    """Handle the internal shell command has unrequested limit helper path for this module.

    Inputs:
        Receives command for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._shell_command_has_unrequested_limit.
    """
    text = str(command or "")
    lowered = text.lower()
    if re.search(r"(?:^|[|;&]\s*)head(?:\s+-n)?\s+\d+\b", lowered):
        return True
    if re.search(r"(?:^|[|;&]\s*)tail(?:\s+-n)?\s+\d+\b", lowered):
        return True
    if re.search(r"\bsed\s+-n\s+['\"]?1\s*,\s*\d+\s*p['\"]?", lowered):
        return True
    if re.search(r"\bawk\b[^|;&]*(?:nr|NR)\s*(?:<=|<)\s*\d+", text):
        return True
    if re.search(r"\blimit\s+\d+\b", lowered):
        return True
    return False


def _sql_query_has_unrequested_limit(query: str) -> bool:
    """Handle the internal sql query has unrequested limit helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._sql_query_has_unrequested_limit.
    """
    text = str(query or "")
    lowered = text.lower()
    if re.search(r"\blimit\s+\d+\b", lowered):
        return True
    if re.search(r"\bfetch\s+(?:first|next)\s+\d+\s+rows?\s+only\b", lowered):
        return True
    return False


def _count_shape_repair_query(goal: str, failure_context: dict[str, Any]) -> str | None:
    """Handle the internal count shape repair query helper path for this module.

    Inputs:
        Receives goal, failure_context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._count_shape_repair_query.
    """
    output_intent = resolve_output_intent(goal)
    if not (output_intent.kind == "scalar" and output_intent.cardinality == "single"):
        return None
    reason = str(failure_context.get("reason") or "")
    shape_error = str(failure_context.get("shape_error") or failure_context.get("error") or "")
    if reason != "result_shape_failed" and "expected one numeric aggregate" not in shape_error:
        return None
    failed_sql = str(failure_context.get("failed_sql") or "").strip()
    if not failed_sql or "[truncated]" in failed_sql:
        return None
    if not re.match(r"(?is)^(?:with|select)\b", failed_sql):
        return None
    inner_sql = failed_sql.rstrip().rstrip(";").strip()
    if not inner_sql or _looks_like_outer_count_query(inner_sql):
        return None
    return f"SELECT COUNT(*) AS count_value FROM (\n{inner_sql}\n) matched_rows"


def _looks_like_outer_count_query(query: str) -> bool:
    """Handle the internal looks like outer count query helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._looks_like_outer_count_query.
    """
    text = str(query or "").strip().rstrip(";")
    if not text:
        return False
    if re.search(r"(?is)\bgroup\s+by\b", text):
        return False
    return bool(re.match(r"(?is)^select\s+count\s*\(", text)) or bool(
        re.match(r"(?is)^with\b.+\bselect\s+count\s*\(", text)
    )


def _is_scalar_count_goal(goal: str) -> bool:
    """Handle the internal is scalar count goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._is_scalar_count_goal.
    """
    return _shared_is_scalar_count_goal(goal)


def _looks_like_sql_goal(goal: str) -> bool:
    """Handle the internal looks like sql goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._looks_like_sql_goal.
    """
    return bool(SQL_GOAL_RE.search(str(goal or "")))


def _is_sql_explain_only_goal(goal: str) -> bool:
    """Handle the internal is sql explain only goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._is_sql_explain_only_goal.
    """
    text = str(goal or "").lower()
    if not re.search(r"\b(?:generate|write|draft|produce)\b", text):
        return False
    if not re.search(r"\bsql\b|\bquery\b", text):
        return False
    return bool(re.search(r"\b(?:validate|explain|what\s+it\s+would\s+return|would\s+return)\b", text))


def _explicit_json_goal(goal: str) -> bool:
    """Handle the internal explicit json goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._explicit_json_goal.
    """
    return bool(re.search(r"\b(?:json|raw\s+json)\b", str(goal or ""), re.IGNORECASE))


def _validate_goal_domain_contract(plan: ActionPlan, *, goal: str) -> list[str]:
    """Handle the internal validate goal domain contract helper path for this module.

    Inputs:
        Receives plan, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._validate_goal_domain_contract.
    """
    domain = _infer_goal_domain(goal)
    if domain != "system_process":
        return []
    errors: list[str] = []
    tools = [action.tool for action in plan.actions]
    if any(tool.startswith("slurm.") for tool in tools):
        errors.append(
            "System process requests must use shell.exec or another system-inspection tool; SLURM queue/jobs do not answer OS process requests."
        )
    if not any(tool == "shell.exec" for tool in tools):
        errors.append("System process requests must include a safe shell.exec process inspection action.")
    return errors


def _infer_goal_domain(goal: str) -> str:
    """Handle the internal infer goal domain helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._infer_goal_domain.
    """
    text = str(goal or "").lower()
    if re.search(r"\bslurm\b|\bsqueue\b|\bsinfo\b|\bsacct\b|\bcluster\b|\bpartition(?:s)?\b|\bnode(?:s)?\b", text):
        return "slurm"
    if re.search(r"\b(?:process|processes|processing)\b", text) and re.search(
        r"\b(?:system|computer|machine|cpu|memory|ram|port|list|running)\b",
        text,
    ):
        return "system_process"
    if re.search(r"\b(?:pid|port|listening|lsof|ss|netstat|ps\s+aux)\b", text):
        return "system_process"
    return "unknown"


def _scalar_count_absence_query(goal: str, catalog: Any) -> str | None:
    """Handle the internal scalar count absence query helper path for this module.

    Inputs:
        Receives goal, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._scalar_count_absence_query.
    """
    related_concept = _absence_related_concept(goal)
    if not related_concept:
        return None
    related_table = _table_for_concept(catalog, related_concept)
    if related_table is None:
        return None
    base_table = _base_table_for_absence_goal(goal, catalog, related_concept)
    if base_table is None:
        return None
    join_column = _shared_join_column(base_table, related_table)
    if join_column is None:
        return None
    base_relation = _quote_sql_relation(base_table.schema_name, base_table.table_name, dialect=catalog.dialect)
    related_relation = _quote_sql_relation(related_table.schema_name, related_table.table_name, dialect=catalog.dialect)
    base_column = _quote_sql_identifier(join_column, dialect=catalog.dialect)
    related_column = _quote_sql_identifier(join_column, dialect=catalog.dialect)
    return (
        "SELECT COUNT(*) AS count_value\n"
        f"FROM {base_relation} b\n"
        "WHERE NOT EXISTS (\n"
        f"  SELECT 1 FROM {related_relation} r WHERE r.{related_column} = b.{base_column}\n"
        ")"
    )


def _scalar_count_query_from_goal(goal: str, catalog: Any) -> str | None:
    """Handle the internal scalar count query from goal helper path for this module.

    Inputs:
        Receives goal, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._scalar_count_query_from_goal.
    """
    dicom_query = _dicom_patient_semantic_count_query(goal, catalog)
    if dicom_query:
        return dicom_query
    table = _first_table_mentioned_in_goal(goal, catalog)
    if table is None:
        return None
    relation = _quote_sql_relation(table.schema_name, table.table_name, dialect=catalog.dialect)
    return f"SELECT COUNT(*) AS count_value FROM {relation}"


def _dicom_patient_semantic_count_query(goal: str, catalog: Any) -> str | None:
    """Build one scalar patient-count query for DICOM concepts.

    Inputs:
        Receives user goal and SQL catalog.

    Returns:
        A schema-grounded aggregate SQL query, or None when unsupported.

    Used by:
        Scalar count insertion fallback when semantic compilation did not already supply SQL.
    """
    text = str(goal or "")
    if not re.search(r"\bpatients?\b|\bpatieints?\b", text, re.IGNORECASE):
        return None
    patient = catalog.table_by_name(None, "Patient")
    study = catalog.table_by_name(None, "Study")
    series = catalog.table_by_name(None, "Series")
    if patient is None or study is None or series is None:
        return None
    required = [
        (patient, "PatientID"),
        (study, "PatientID"),
        (study, "StudyInstanceUID"),
        (series, "StudyInstanceUID"),
    ]
    if any(table.column_by_name(column) is None for table, column in required):
        return None
    concept_terms = _dicom_text_concepts_from_goal(text)
    modalities = _dicom_modalities_from_goal(text)
    has_text_columns = bool(study.column_by_name("StudyDescription") and series.column_by_name("SeriesDescription"))
    if concept_terms and not has_text_columns:
        return None
    if not concept_terms and not modalities:
        return None
    if _dicom_all_terms_policy(text) and len(concept_terms) > 1:
        clauses = [
            _dicom_patient_exists_clause_for_action_planner(
                term=term,
                modalities=modalities,
                study_alias=f"st_{index}",
                series_alias=f"se_{index}",
            )
            for index, term in enumerate(concept_terms, start=1)
        ]
        return 'SELECT COUNT(DISTINCT p."PatientID") AS patient_count\nFROM flathr."Patient" p\nWHERE ' + "\n  AND ".join(clauses)
    predicates: list[str] = []
    if modalities:
        quoted = ", ".join(f"'{value}'" for value in modalities)
        predicates.append(f'se."Modality" IN ({quoted})')
    if concept_terms:
        predicates.append("(" + " OR ".join(_dicom_description_predicate(term, study_alias="st", series_alias="se") for term in concept_terms) + ")")
    if not predicates:
        return None
    return (
        'SELECT COUNT(DISTINCT p."PatientID") AS patient_count\n'
        'FROM flathr."Patient" p\n'
        'JOIN flathr."Study" st ON st."PatientID" = p."PatientID"\n'
        'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"\n'
        "WHERE " + "\n  AND ".join(predicates)
    )


def _dicom_patient_exists_clause_for_action_planner(*, term: str, modalities: list[str], study_alias: str, series_alias: str) -> str:
    """Build one correlated DICOM text EXISTS clause for patient counts.

    Inputs:
        Receives concept term, optional modalities, and aliases.

    Returns:
        SQL EXISTS clause correlated to patient alias p.

    Used by:
        _dicom_patient_semantic_count_query.
    """
    predicates = [_dicom_description_predicate(term, study_alias=study_alias, series_alias=series_alias)]
    if modalities:
        quoted = ", ".join(f"'{value}'" for value in modalities)
        predicates.append(f'{series_alias}."Modality" IN ({quoted})')
    return (
        "EXISTS (\n"
        "    SELECT 1\n"
        f'    FROM flathr."Study" {study_alias}\n'
        f'    JOIN flathr."Series" {series_alias} ON {series_alias}."StudyInstanceUID" = {study_alias}."StudyInstanceUID"\n'
        f'    WHERE {study_alias}."PatientID" = p."PatientID"\n'
        "      AND " + "\n      AND ".join(predicates) + "\n"
        "  )"
    )


def _dicom_description_predicate(term: str, *, study_alias: str, series_alias: str) -> str:
    """Build a predicate over DICOM study and series descriptions.

    Inputs:
        Receives concept term and aliases.

    Returns:
        SQL predicate string.

    Used by:
        DICOM scalar-count SQL builders.
    """
    escaped = str(term or "").replace("'", "''").lower()
    return (
        f"(LOWER({study_alias}.\"StudyDescription\") LIKE '%{escaped}%' "
        f"OR LOWER({series_alias}.\"SeriesDescription\") LIKE '%{escaped}%')"
    )


def _dicom_text_concepts_from_goal(goal: str) -> list[str]:
    """Extract supported DICOM description concepts from a goal.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered concept terms.

    Used by:
        DICOM scalar-count SQL builders.
    """
    return [term for term in ("brain", "breast") if re.search(rf"\b{re.escape(term)}\b", goal, re.IGNORECASE)]


def _dicom_modalities_from_goal(goal: str) -> list[str]:
    """Extract DICOM modality targets from a goal.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered uppercase modality names.

    Used by:
        DICOM scalar-count SQL builders.
    """
    return [value for value in ("CT", "MR", "RTSTRUCT", "RTPLAN", "RTDOSE", "PT") if re.search(rf"\b{re.escape(value)}\b", goal, re.IGNORECASE)]


def _dicom_all_terms_policy(goal: str) -> bool:
    """Detect all-terms semantics for DICOM text concepts.

    Inputs:
        Receives the user goal.

    Returns:
        True when both/all/and wording requires separate concept matches.

    Used by:
        DICOM scalar-count SQL builders.
    """
    text = str(goal or "").lower()
    return bool(re.search(r"\b(?:both|all)\b", text) or re.search(r"\band\b", text))


def _first_table_mentioned_in_goal(goal: str, catalog: Any) -> Any:
    """Handle the internal first table mentioned in goal helper path for this module.

    Inputs:
        Receives goal, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._first_table_mentioned_in_goal.
    """
    text = str(goal or "").lower()
    candidates: list[tuple[int, Any]] = []
    for table in catalog.tables:
        concept = _singular_concept(table.table_name)
        match = re.search(_concept_pattern(concept), text)
        if match:
            candidates.append((match.start(), table))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else None


def _absence_related_concept(goal: str) -> str | None:
    """Handle the internal absence related concept helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._absence_related_concept.
    """
    match = re.search(r"\b(?:no|without|missing)\s+([A-Za-z_][A-Za-z0-9_]*)", str(goal or ""), re.IGNORECASE)
    if not match:
        return None
    return _singular_concept(match.group(1))


def _base_table_for_absence_goal(goal: str, catalog: Any, related_concept: str) -> Any:
    """Handle the internal base table for absence goal helper path for this module.

    Inputs:
        Receives goal, catalog, related_concept for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._base_table_for_absence_goal.
    """
    text = str(goal or "").lower()
    related_pos_match = re.search(r"\b(?:no|without|missing)\b", text)
    related_pos = related_pos_match.start() if related_pos_match else len(text)
    candidates: list[tuple[int, Any]] = []
    for table in catalog.tables:
        concept = _singular_concept(table.table_name)
        if concept == related_concept:
            continue
        match = re.search(_concept_pattern(concept), text)
        if match and match.start() <= related_pos:
            candidates.append((match.start(), table))
    candidates.sort(key=lambda item: item[0])
    if candidates:
        return candidates[0][1]
    related_table = _table_for_concept(catalog, related_concept)
    if related_table is None:
        return None
    for table in catalog.tables:
        if table is related_table:
            continue
        if _shared_join_column(table, related_table):
            return table
    return None


def _table_for_concept(catalog: Any, concept: str) -> Any:
    """Handle the internal table for concept helper path for this module.

    Inputs:
        Receives catalog, concept for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._table_for_concept.
    """
    normalized = _singular_concept(concept)
    exact = [table for table in catalog.tables if _singular_concept(table.table_name) == normalized]
    if exact:
        return exact[0]
    contains = [table for table in catalog.tables if normalized in _singular_concept(table.table_name)]
    return contains[0] if contains else None


def _shared_join_column(base_table: Any, related_table: Any) -> str | None:
    """Handle the internal shared join column helper path for this module.

    Inputs:
        Receives base_table, related_table for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._shared_join_column.
    """
    base_columns = {column.column_name for column in base_table.columns}
    related_columns = {column.column_name for column in related_table.columns}
    shared = base_columns.intersection(related_columns)
    if not shared:
        return None
    preferred: list[str] = []
    preferred.extend([column for column in base_table.primary_key_columns if column in shared])
    preferred.extend([column for column in related_table.primary_key_columns if column in shared])
    preferred.extend(sorted(column for column in shared if column.lower().endswith("id") or column.lower().endswith("uid")))
    preferred.extend(sorted(shared))
    return preferred[0] if preferred else None


def _singular_concept(value: str) -> str:
    """Handle the internal singular concept helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._singular_concept.
    """
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if text == "series":
        return text
    if text.endswith("ies") and len(text) > 3:
        return f"{text[:-3]}y"
    if text.endswith("s") and len(text) > 1:
        return text[:-1]
    return text


def _concept_pattern(concept: str) -> str:
    """Handle the internal concept pattern helper path for this module.

    Inputs:
        Receives concept for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._concept_pattern.
    """
    escaped = re.escape(concept)
    if concept == "series":
        return rf"\b{escaped}\b"
    if concept.endswith("y"):
        return rf"\b(?:{re.escape(concept[:-1])}ies|{escaped}s?)\b"
    return rf"\b{escaped}s?\b"


def _quote_sql_relation(schema: str, table: str, *, dialect: str | None) -> str:
    """Handle the internal quote sql relation helper path for this module.

    Inputs:
        Receives schema, table, dialect for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._quote_sql_relation.
    """
    return f"{_quote_sql_identifier(schema, dialect=dialect)}.{_quote_sql_identifier(table, dialect=dialect)}"


def _quote_sql_identifier(identifier: str, *, dialect: str | None) -> str:
    """Handle the internal quote sql identifier helper path for this module.

    Inputs:
        Receives identifier, dialect for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._quote_sql_identifier.
    """
    if str(dialect or "").lower() == "postgresql":
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    return str(identifier)


def _first_action_with_tool(actions: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    """Handle the internal first action with tool helper path for this module.

    Inputs:
        Receives actions, tool for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._first_action_with_tool.
    """
    for action in actions:
        if action.get("tool") == tool:
            return action
    return None


def _last_action_with_tool(actions: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    """Handle the internal last action with tool helper path for this module.

    Inputs:
        Receives actions, tool for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._last_action_with_tool.
    """
    for action in reversed(actions):
        if action.get("tool") == tool:
            return action
    return None


def _formatter_after_producer(actions: list[dict[str, Any]], producer: dict[str, Any]) -> dict[str, Any] | None:
    """Handle the internal formatter after producer helper path for this module.

    Inputs:
        Receives actions, producer for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._formatter_after_producer.
    """
    try:
        producer_index = actions.index(producer)
    except ValueError:
        return None
    producer_symbols = {_normalize_name(producer.get("id") or ""), _normalize_name(producer.get("output_binding") or "")}
    producer_symbols.discard("")
    for action in actions[producer_index + 1 :]:
        if action.get("tool") != "text.format":
            continue
        refs = _collect_action_refs(action.get("inputs"))
        if refs and refs.intersection(producer_symbols):
            return action
        source = action.get("inputs", {}).get("source")
        if isinstance(source, str) and _normalize_name(source) in producer_symbols:
            return action
    return None


def _last_data_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Handle the internal last data action helper path for this module.

    Inputs:
        Receives actions for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._last_data_action.
    """
    for action in reversed(actions):
        if str(action.get("tool") or "") in DATA_REF_PATHS:
            return action
    return None


def _last_scalar_action(actions: list[dict[str, Any]], *, goal: str) -> dict[str, Any] | None:
    """Handle the internal last scalar action helper path for this module.

    Inputs:
        Receives actions, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._last_scalar_action.
    """
    for action in reversed(actions):
        tool = str(action.get("tool") or "")
        if scalar_field_for_tool(tool, goal=goal):
            return action
    return None


def _unique_action_id(actions: list[dict[str, Any]], preferred: str) -> str:
    """Handle the internal unique action id helper path for this module.

    Inputs:
        Receives actions, preferred for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._unique_action_id.
    """
    existing = {str(action.get("id") or "") for action in actions}
    base = _normalize_name(preferred) or "action"
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _database_for_goal(goal: str, configured: dict[str, str]) -> str | None:
    """Handle the internal database for goal helper path for this module.

    Inputs:
        Receives goal, configured for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._database_for_goal.
    """
    goal_text = str(goal or "").lower()
    normalized_goal = re.sub(r"[^a-z0-9_]+", " ", goal_text)
    tokens = set(normalized_goal.split())
    for name in configured:
        lowered = name.lower()
        if lowered in tokens or lowered in goal_text:
            return name
    return None


def _ref_path_for_symbol(symbol: PlanSymbol, preferred_path: str | None) -> str | None:
    """Handle the internal ref path for symbol helper path for this module.

    Inputs:
        Receives symbol, preferred_path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._ref_path_for_symbol.
    """
    if preferred_path is None:
        return default_path_for_tool(symbol.tool)
    if path_is_declared_for_tool(symbol.tool, preferred_path):
        return preferred_path
    return default_path_for_tool(symbol.tool)


def _is_generic_wrong_ref_path(path: str, symbol: PlanSymbol) -> bool:
    """Handle the internal is generic wrong ref path helper path for this module.

    Inputs:
        Receives path, symbol for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._is_generic_wrong_ref_path.
    """
    if not default_path_for_tool(symbol.tool):
        return False
    normalized = str(path or "").strip()
    if not normalized:
        return True
    if path_is_declared_for_tool(symbol.tool, normalized):
        return False
    return normalized in {"value", "result", "output", "data", "content", "text"}


def _schema_question_goal(goal: str) -> bool:
    """Handle the internal schema question goal helper path for this module.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._schema_question_goal.
    """
    return bool(re.search(r"\b(?:schema|schemas|table|tables|column|columns|describe|identify|link|links|related)\b", str(goal or ""), re.IGNORECASE))


@dataclass(frozen=True)
class SqlRelationshipEdge:
    """Represent sql relationship edge within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlRelationshipEdge.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.SqlRelationshipEdge and related tests.
    """
    left_table: str
    right_table: str
    left_column: str
    right_column: str


class SqlRelationshipGraph:
    """Represent sql relationship graph within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlRelationshipGraph.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.action_planner.SqlRelationshipGraph and related tests.
    """
    def __init__(self, catalog: Any) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives catalog for this SqlRelationshipGraph method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through SqlRelationshipGraph.__init__ calls and related tests.
        """
        self.catalog = catalog
        self.edges = self._build_edges(catalog)

    def join_columns(self, left_table: Any, right_table: Any) -> tuple[str, str] | None:
        """Join columns for SqlRelationshipGraph instances.

        Inputs:
            Receives left_table, right_table for this SqlRelationshipGraph method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlRelationshipGraph.join_columns calls and related tests.
        """
        left_name = left_table.qualified_name
        right_name = right_table.qualified_name
        for edge in self.edges:
            if edge.left_table == left_name and edge.right_table == right_name:
                return edge.left_column, edge.right_column
            if edge.left_table == right_name and edge.right_table == left_name:
                return edge.right_column, edge.left_column
        shared = _shared_join_column(left_table, right_table)
        if shared:
            return shared, shared
        return None

    @staticmethod
    def _build_edges(catalog: Any) -> list[SqlRelationshipEdge]:
        """Handle the internal build edges helper path for this module.

        Inputs:
            Receives catalog for this SqlRelationshipGraph method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlRelationshipGraph._build_edges calls and related tests.
        """
        edges: list[SqlRelationshipEdge] = []
        tables_by_qualified = {table.qualified_name.lower(): table for table in catalog.tables}
        for table in catalog.tables:
            for fk in table.foreign_keys:
                constrained = list(fk.get("constrained_columns") or [])
                referred = list(fk.get("referred_columns") or [])
                referred_schema = str(fk.get("referred_schema") or table.schema_name)
                referred_table = str(fk.get("referred_table") or "")
                target = tables_by_qualified.get(f"{referred_schema}.{referred_table}".lower())
                if target is None:
                    continue
                for left_column, right_column in zip(constrained, referred):
                    edges.append(
                        SqlRelationshipEdge(
                            left_table=table.qualified_name,
                            right_table=target.qualified_name,
                            left_column=str(left_column),
                            right_column=str(right_column),
                        )
                    )
        for left_name, right_name, column in (
            ("Study", "Series", "StudyInstanceUID"),
            ("Series", "Instance", "SeriesInstanceUID"),
            ("Patient", "Study", "PatientID"),
        ):
            left = catalog.table_by_name(None, left_name)
            right = catalog.table_by_name(None, right_name)
            if left is not None and right is not None and left.column_by_name(column) and right.column_by_name(column):
                edge = SqlRelationshipEdge(left.qualified_name, right.qualified_name, column, column)
                if edge not in edges:
                    edges.append(edge)
        return edges


def _repair_sql_relationship_query_from_goal(query: str, catalog: Any, *, goal: str) -> str | None:
    """Handle the internal repair sql relationship query from goal helper path for this module.

    Inputs:
        Receives query, catalog, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._repair_sql_relationship_query_from_goal.
    """
    goal_text = str(goal or "").lower()
    aggregate_repair = _aggregate_over_related_counts_query(catalog, goal=goal)
    if aggregate_repair:
        return aggregate_repair
    if _is_study_count_by_modality_goal(goal_text):
        repaired = _study_count_by_modality_query(catalog, goal=goal)
        if repaired:
            return repaired
    if re.search(r"\binstances?\b", goal_text) and re.search(r"\bstud(?:y|ies)\b", goal_text):
        repaired = _study_instance_count_query(catalog, goal=goal)
        if repaired:
            return repaired
    return None


def _aggregate_over_related_counts_query(catalog: Any, *, goal: str) -> str | None:
    """Handle the internal aggregate over related counts query helper path for this module.

    Inputs:
        Receives catalog, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._aggregate_over_related_counts_query.
    """
    text = str(goal or "").lower()
    if not (re.search(r"\bmin(?:imum)?\b", text) and re.search(r"\bmax(?:imum)?\b", text) and re.search(r"\b(?:average|avg)\b", text)):
        return None
    relation: tuple[str, str, str] | None = None
    if re.search(r"\bstud(?:y|ies)\s+per\s+patient\b|\bnumber\s+of\s+stud(?:y|ies)\s+per\s+patient\b", text):
        relation = ("Patient", "Study", "studies_per_patient")
    elif re.search(r"\bseries\s+per\s+stud(?:y|ies)\b|\bnumber\s+of\s+series\s+per\s+stud", text):
        relation = ("Study", "Series", "series_per_study")
    elif re.search(r"\binstances?\s+per\s+series\b|\bnumber\s+of\s+instances?\s+per\s+series\b", text):
        relation = ("Series", "Instance", "instances_per_series")
    if relation is None:
        return None
    base_name, related_name, label = relation
    base = catalog.table_by_name(None, base_name)
    related = catalog.table_by_name(None, related_name)
    if base is None or related is None:
        return None
    graph = SqlRelationshipGraph(catalog)
    join = graph.join_columns(base, related)
    if join is None:
        return None
    base_join, related_join = join
    base_id = _first_existing_column(base, [base_join, *base.primary_key_columns]) or base_join
    base_relation = _quote_sql_relation(base.schema_name, base.table_name, dialect=catalog.dialect)
    related_relation = _quote_sql_relation(related.schema_name, related.table_name, dialect=catalog.dialect)
    base_join_q = _quote_sql_identifier(base_join, dialect=catalog.dialect)
    related_join_q = _quote_sql_identifier(related_join, dialect=catalog.dialect)
    base_id_q = _quote_sql_identifier(base_id, dialect=catalog.dialect)
    return (
        "WITH grouped_counts AS (\n"
        f"  SELECT b.{base_id_q} AS entity_id, COUNT(r.{related_join_q}) AS related_count\n"
        f"  FROM {base_relation} b\n"
        f"  LEFT JOIN {related_relation} r ON r.{related_join_q} = b.{base_join_q}\n"
        f"  GROUP BY b.{base_id_q}\n"
        ")\n"
        f"SELECT MIN(related_count) AS min_{label},\n"
        f"       MAX(related_count) AS max_{label},\n"
        f"       AVG(related_count) AS average_{label}\n"
        "FROM grouped_counts"
    )


def _is_study_count_by_modality_goal(goal_text: str) -> bool:
    """Handle the internal is study count by modality goal helper path for this module.

    Inputs:
        Receives goal_text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._is_study_count_by_modality_goal.
    """
    if not (re.search(r"\bmodalit(?:y|ies)\b", goal_text) and re.search(r"\bstud(?:y|ies)\b", goal_text)):
        return False
    return bool(
        re.search(r"\bgroup(?:ed)?\s+by\s+modalit", goal_text)
        or re.search(r"\bby\s+modalit", goal_text)
        or re.search(r"\bnumber\s+of\s+stud(?:y|ies)\b", goal_text)
        or re.search(r"\bstud(?:y|ies)\s+per\s+modalit", goal_text)
    )


def _study_count_by_modality_query(catalog: Any, *, goal: str) -> str | None:
    """Handle the internal study count by modality query helper path for this module.

    Inputs:
        Receives catalog, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._study_count_by_modality_query.
    """
    study = catalog.table_by_name(None, "Study")
    series = catalog.table_by_name(None, "Series")
    if study is None or series is None:
        return None
    graph = SqlRelationshipGraph(catalog)
    join = graph.join_columns(study, series)
    study_uid = join[0] if join else _first_existing_column(study, ["StudyInstanceUID", "StudyUID"])
    series_study_uid = join[1] if join else _first_existing_column(series, ["StudyInstanceUID", "StudyUID"])
    modality = _first_existing_column(series, ["Modality"])
    if not study_uid or not series_study_uid or not modality:
        return None
    study_relation = _quote_sql_relation(study.schema_name, study.table_name, dialect=catalog.dialect)
    series_relation = _quote_sql_relation(series.schema_name, series.table_name, dialect=catalog.dialect)
    study_uid_q = _quote_sql_identifier(study_uid, dialect=catalog.dialect)
    series_study_uid_q = _quote_sql_identifier(series_study_uid, dialect=catalog.dialect)
    modality_q = _quote_sql_identifier(modality, dialect=catalog.dialect)
    count_alias = "study_count" if re.search(r"\bstud(?:y|ies)\b", str(goal or ""), re.IGNORECASE) else "count_value"
    return (
        f"SELECT se.{modality_q} AS modality, COUNT(DISTINCT st.{study_uid_q}) AS {count_alias}\n"
        f"FROM {study_relation} st\n"
        f"JOIN {series_relation} se ON se.{series_study_uid_q} = st.{study_uid_q}\n"
        f"GROUP BY se.{modality_q}\n"
        f"ORDER BY {count_alias} DESC"
    )


def _study_instance_count_query(catalog: Any, *, goal: str) -> str | None:
    """Handle the internal study instance count query helper path for this module.

    Inputs:
        Receives catalog, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._study_instance_count_query.
    """
    study = catalog.table_by_name(None, "Study")
    series = catalog.table_by_name(None, "Series")
    instance = catalog.table_by_name(None, "Instance")
    if study is None or series is None or instance is None:
        return None
    graph = SqlRelationshipGraph(catalog)
    study_series_join = graph.join_columns(study, series)
    series_instance_join = graph.join_columns(series, instance)
    study_uid = study_series_join[0] if study_series_join else _first_existing_column(study, ["StudyInstanceUID", "StudyUID"])
    series_study_uid = study_series_join[1] if study_series_join else _first_existing_column(series, ["StudyInstanceUID", "StudyUID"])
    series_uid = series_instance_join[0] if series_instance_join else _first_existing_column(series, ["SeriesInstanceUID", "SeriesUID"])
    instance_series_uid = series_instance_join[1] if series_instance_join else _first_existing_column(instance, ["SeriesInstanceUID", "SeriesUID"])
    if not all((study_uid, series_study_uid, series_uid, instance_series_uid)):
        return None
    limit_match = re.search(r"\btop\s+(\d+)|\bfirst\s+(\d+)|\blimit\s+(\d+)", str(goal or ""), re.IGNORECASE)
    limit = next((int(item) for item in (limit_match.groups() if limit_match else ()) if item), None)
    study_relation = _quote_sql_relation(study.schema_name, study.table_name, dialect=catalog.dialect)
    series_relation = _quote_sql_relation(series.schema_name, series.table_name, dialect=catalog.dialect)
    instance_relation = _quote_sql_relation(instance.schema_name, instance.table_name, dialect=catalog.dialect)
    study_uid_q = _quote_sql_identifier(str(study_uid), dialect=catalog.dialect)
    series_study_uid_q = _quote_sql_identifier(str(series_study_uid), dialect=catalog.dialect)
    series_uid_q = _quote_sql_identifier(str(series_uid), dialect=catalog.dialect)
    instance_series_uid_q = _quote_sql_identifier(str(instance_series_uid), dialect=catalog.dialect)
    query = (
        f"SELECT st.{study_uid_q} AS study_instance_uid, COUNT(*) AS instance_count\n"
        f"FROM {study_relation} st\n"
        f"JOIN {series_relation} se ON se.{series_study_uid_q} = st.{study_uid_q}\n"
        f"JOIN {instance_relation} i ON i.{instance_series_uid_q} = se.{series_uid_q}\n"
        f"GROUP BY st.{study_uid_q}\n"
        "ORDER BY instance_count DESC"
    )
    if limit is not None:
        query = f"{query}\nLIMIT {limit}"
    return query


def _first_existing_column(table: Any, candidates: list[str]) -> str | None:
    """Handle the internal first existing column helper path for this module.

    Inputs:
        Receives table, candidates for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._first_existing_column.
    """
    for candidate in candidates:
        column = table.column_by_name(candidate)
        if column is not None:
            return column.column_name
    return None


def _validate_sql_catalog_references(query: str, catalog: Any) -> list[str]:
    """Handle the internal validate sql catalog references helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._validate_sql_catalog_references.
    """
    if re.search(r"(?i)\binformation_schema\b|\bpg_catalog\b", query):
        return ["Use sql.schema for schema/catalog questions instead of querying system catalog tables directly."]

    alias_columns = _sql_alias_columns(query, catalog)
    alias_tables = _sql_alias_tables(query, catalog)
    referenced_tables = list(alias_columns.values())
    all_referenced_columns = {column for columns in alias_columns.values() for column in columns}
    all_catalog_columns = {column.column_name for table in catalog.tables for column in table.columns}
    output_aliases = _sql_output_aliases(query)
    errors: list[str] = []

    for alias, column in re.findall(r'(?<!")\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"([^"]+)"', query):
        columns = alias_columns.get(alias.lower())
        if columns is not None and column not in columns:
            errors.append(_unknown_column_error(column, columns, alias=alias))

    for alias, column in re.findall(r'(?<!")\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()', query):
        if alias.lower() not in alias_tables:
            continue
        columns = alias_columns.get(alias.lower())
        if columns is not None and column not in columns:
            errors.append(_unknown_column_error(column, columns, alias=alias))

    for alias, table in alias_tables.items():
        table_columns = {column.column_name for column in table.columns}
        referenced = re.findall(rf'(?<!")\b{re.escape(alias)}\s*\.\s*"([^"]+)"', query, flags=re.IGNORECASE)
        referenced.extend(
            re.findall(rf'(?<!")\b{re.escape(alias)}\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()', query, flags=re.IGNORECASE)
        )
        for column in referenced:
            if column not in table_columns:
                candidate = _nearest_column_match(column, table_columns)
                if candidate:
                    errors.append(
                        f"Column {column} is not on table {table.qualified_name}; "
                        f"use {alias}.{_quote_sql_identifier(candidate, dialect=catalog.dialect)} or join the table that owns it."
                    )

    for column in re.findall(r'"([^"]+)"', query):
        if "." in column:
            continue
        if column in output_aliases or column.lower() in {alias.lower() for alias in output_aliases}:
            continue
        if column in _sql_relation_identifier_names(query):
            continue
        if column in all_referenced_columns or column in all_catalog_columns:
            continue
        if column.lower() in {"count", "sum", "avg", "min", "max"}:
            continue
        candidate_columns = all_referenced_columns or all_catalog_columns
        errors.append(_unknown_column_error(column, candidate_columns))

    errors.extend(_validate_sql_literal_type_mismatches(query, catalog))
    return list(dict.fromkeys(errors))


def _validate_sql_goal_concepts(query: str, catalog: Any, *, goal: str) -> list[str]:
    """Handle the internal validate sql goal concepts helper path for this module.

    Inputs:
        Receives query, catalog, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._validate_sql_goal_concepts.
    """
    goal_text = str(goal or "").lower()
    if not re.search(r"\bbody\s+parts?\b|\bbodypart\b|\bbody\s+part\s+examined\b", goal_text):
        return []
    all_columns = {column.column_name for table in catalog.tables for column in table.columns}
    body_part_columns = [
        column
        for column in sorted(all_columns)
        if "bodypart" in re.sub(r"[^a-z0-9]+", "", column.lower())
        or ("body" in column.lower() and "part" in column.lower())
    ]
    if not body_part_columns:
        if re.search(r"\bif\s+such\s+a\s+column\s+exists\b|\bif\s+available\b|\bif\s+present\b", goal_text):
            return ["Requested optional schema concept body part/anatomical region is not present in the SQL catalog."]
        series_table = catalog.table_by_name(None, "Series")
        candidates = {column.column_name for column in series_table.columns} if series_table is not None else all_columns
        return [_unknown_column_error("BodyPartExamined", candidates)]
    if not any(re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(column)}"?(?![A-Za-z0-9_])', query) for column in body_part_columns):
        return [f"SQL does not include requested body-part column. Candidate columns: {', '.join(body_part_columns[:8])}."]
    return []


def _repair_sql_empty_date_comparisons(query: str, catalog: Any) -> str:
    """Handle the internal repair sql empty date comparisons helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._repair_sql_empty_date_comparisons.
    """
    date_columns = {
        column.column_name
        for table in catalog.tables
        for column in table.columns
        if "date" in str(column.data_type or column.column_name).lower()
    }
    repaired = query
    for column in sorted(date_columns, key=len, reverse=True):
        quoted = re.escape(f'"{column}"')
        unquoted = re.escape(column)
        identifier = rf'(?:\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?(?:{quoted}|{unquoted})'
        repaired = re.sub(rf'\s+(?i:or)\s+{identifier}\s*=\s*\'\'', "", repaired)
        repaired = re.sub(rf'(?P<ident>{identifier})\s*=\s*\'\'\s+(?i:or)\s+', "", repaired)
        repaired = re.sub(rf'\s+(?i:and)\s+{identifier}\s*(?:!=|<>)\s*\'\'', "", repaired)
        repaired = re.sub(rf'(?P<ident>{identifier})\s*(?:!=|<>)\s*\'\'\s+(?i:and)\s+', "", repaired)
        repaired = re.sub(rf'(?P<ident>{identifier})\s*=\s*\'\'', r'\g<ident> IS NULL', repaired)
        repaired = re.sub(rf'(?P<ident>{identifier})\s*(?:!=|<>)\s*\'\'', r'\g<ident> IS NOT NULL', repaired)
    return repaired


def _repair_sql_date_year_extraction(query: str, catalog: Any) -> str:
    """Handle the internal repair sql date year extraction helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._repair_sql_date_year_extraction.
    """
    date_columns = {
        column.column_name
        for table in catalog.tables
        for column in table.columns
        if "date" in str(column.data_type or column.column_name).lower()
    }
    repaired = query
    for column in sorted(date_columns, key=len, reverse=True):
        quoted = re.escape(f'"{column}"')
        unquoted = re.escape(column)
        identifier = rf'(?:\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?(?:{quoted}|{unquoted})'
        repaired = re.sub(
            rf'(?is)substring\s*\(\s*(?P<ident>{identifier})\s+from\s+1\s+for\s+4\s*\)',
            r'EXTRACT(YEAR FROM \g<ident>)',
            repaired,
        )
        repaired = re.sub(
            rf'(?is)substring\s*\(\s*(?P<ident>{identifier})\s*,\s*1\s*,\s*4\s*\)',
            r'EXTRACT(YEAR FROM \g<ident>)',
            repaired,
        )
    return repaired


def _repair_sql_age_argument_order(query: str, catalog: Any) -> str:
    """Handle the internal repair sql age argument order helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._repair_sql_age_argument_order.
    """
    date_columns = {
        column.column_name
        for table in catalog.tables
        for column in table.columns
        if "date" in str(column.data_type or column.column_name).lower()
    }
    repaired = query
    for column in sorted(date_columns, key=len, reverse=True):
        quoted = re.escape(f'"{column}"')
        unquoted = re.escape(column)
        bare_identifier = rf'(?:\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?(?:{quoted}|{unquoted})'
        cast_identifier = rf'CAST\s*\(\s*{bare_identifier}\s+AS\s+DATE\s*\)'
        identifier = rf'(?:{cast_identifier}|{bare_identifier})'
        repaired = re.sub(
            rf'(?is)age\s*\(\s*(?P<birth>{identifier})\s*,\s*current_date\s*\)',
            r'AGE(CURRENT_DATE, \g<birth>)',
            repaired,
        )
    return repaired


def _repair_pg_relation_qualified_columns(query: str, catalog: Any) -> str:
    """Handle the internal repair pg relation qualified columns helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._repair_pg_relation_qualified_columns.
    """
    if str(catalog.dialect or "").lower() != "postgresql":
        return query
    repaired = query
    for table in catalog.tables:
        schema_variants = [re.escape(table.schema_name), re.escape(f'"{table.schema_name}"')]
        table_variants = [re.escape(table.table_name), re.escape(f'"{table.table_name}"')]
        replacement = f'{_quote_sql_identifier(table.table_name, dialect=catalog.dialect)}.'
        for schema_part in schema_variants:
            for table_part in table_variants:
                repaired = re.sub(
                    rf'(?<!["A-Za-z0-9_]){schema_part}\s*\.\s*{table_part}\s*\.',
                    replacement,
                    repaired,
                )
    return repaired


def _sql_output_aliases(query: str) -> set[str]:
    """Handle the internal sql output aliases helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._sql_output_aliases.
    """
    aliases: set[str] = set()
    select_part = _top_level_select_part(query)
    if not select_part:
        return aliases
    for match in re.finditer(r'(?is)\bas\s+("[^"]+"|[A-Za-z_][A-Za-z0-9_]*)', select_part):
        aliases.add(_strip_sql_identifier(match.group(1)))
    return aliases


def _top_level_select_part(query: str) -> str:
    """Handle the internal top level select part helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._top_level_select_part.
    """
    match = re.search(r"(?is)\bselect\b", query)
    if not match:
        return ""
    start = match.end()
    depth = 0
    quote = False
    index = start
    while index < len(query):
        char = query[index]
        if char == '"':
            quote = not quote
        elif not quote:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            elif depth == 0 and query[index : index + 4].lower() == "from":
                before = query[index - 1] if index > 0 else " "
                after = query[index + 4] if index + 4 < len(query) else " "
                if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                    return query[start:index]
        index += 1
    return query[start:]


def _sql_alias_columns(query: str, catalog: Any) -> dict[str, set[str]]:
    """Handle the internal sql alias columns helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._sql_alias_columns.
    """
    aliases: dict[str, set[str]] = {}
    relation_re = re.compile(
        r'(?is)\b(?:from|join)\s+(?!\()((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?)(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?'
    )
    for match in relation_re.finditer(query):
        relation = match.group(1)
        alias = match.group(2)
        schema, table = _split_sql_relation(relation)
        table_ref = catalog.table_by_name(schema, table)
        if table_ref is None:
            continue
        columns = {column.column_name for column in table_ref.columns}
        aliases[table_ref.table_name.lower()] = columns
        aliases[table.lower()] = columns
        if alias and alias.lower() not in CLAUSE_ALIAS_STOP_WORDS:
            aliases[alias.lower()] = columns
    return aliases


def _sql_alias_tables(query: str, catalog: Any) -> dict[str, Any]:
    """Handle the internal sql alias tables helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._sql_alias_tables.
    """
    aliases: dict[str, Any] = {}
    relation_re = re.compile(
        r'(?is)\b(?:from|join)\s+(?!\()((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?)(?:\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*))?'
    )
    for match in relation_re.finditer(query):
        relation = match.group(1)
        alias = match.group(2)
        schema, table = _split_sql_relation(relation)
        table_ref = catalog.table_by_name(schema, table)
        if table_ref is None:
            continue
        aliases[table_ref.table_name.lower()] = table_ref
        aliases[table.lower()] = table_ref
        if alias and alias.lower() not in CLAUSE_ALIAS_STOP_WORDS:
            aliases[alias.lower()] = table_ref
    return aliases


CLAUSE_ALIAS_STOP_WORDS = {
    "where",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "full",
    "group",
    "order",
    "having",
    "limit",
    "on",
}


def _split_sql_relation(relation: str) -> tuple[str | None, str]:
    """Handle the internal split sql relation helper path for this module.

    Inputs:
        Receives relation for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._split_sql_relation.
    """
    parts = [_strip_sql_identifier(part) for part in re.split(r"\s*\.\s*", relation.strip()) if part.strip()]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, parts[0] if parts else ""


def _strip_sql_identifier(value: str) -> str:
    """Handle the internal strip sql identifier helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._strip_sql_identifier.
    """
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text


def _sql_relation_identifier_names(query: str) -> set[str]:
    """Handle the internal sql relation identifier names helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._sql_relation_identifier_names.
    """
    names: set[str] = set()
    for relation in re.findall(
        r'(?is)\b(?:from|join)\s+((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?)',
        query,
    ):
        schema, table = _split_sql_relation(relation)
        if schema:
            names.add(schema)
        if table:
            names.add(table)
    return names


def _unknown_column_error(column: str, candidates: set[str], *, alias: str | None = None) -> str:
    """Handle the internal unknown column error helper path for this module.

    Inputs:
        Receives column, candidates, alias for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._unknown_column_error.
    """
    ranked = _rank_column_candidates(column, candidates)
    target = f"{alias}.{column}" if alias else column
    suffix = f" Candidate columns: {', '.join(ranked[:8])}." if ranked else ""
    return f"SQL references unknown column: {target}.{suffix}"


def _rank_column_candidates(column: str, candidates: set[str]) -> list[str]:
    """Handle the internal rank column candidates helper path for this module.

    Inputs:
        Receives column, candidates for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._rank_column_candidates.
    """
    needle = re.sub(r"[^a-z0-9]+", "", column.lower())
    scored: list[tuple[int, str]] = []
    for candidate in sorted(candidates):
        normalized = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        score = 0
        if normalized == needle:
            score = 100
        elif needle and (needle in normalized or normalized in needle):
            score = 50
        elif needle and normalized and needle[:4] == normalized[:4]:
            score = 20
        if score:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    return [candidate for _, candidate in scored] or sorted(candidates)[:8]


def _nearest_column_match(column: str, candidates: set[str]) -> str | None:
    """Handle the internal nearest column match helper path for this module.

    Inputs:
        Receives column, candidates for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._nearest_column_match.
    """
    ranked = _rank_column_candidates(column, candidates)
    return ranked[0] if ranked else None


def _validate_sql_literal_type_mismatches(query: str, catalog: Any) -> list[str]:
    """Handle the internal validate sql literal type mismatches helper path for this module.

    Inputs:
        Receives query, catalog for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._validate_sql_literal_type_mismatches.
    """
    date_columns = {
        column.column_name
        for table in catalog.tables
        for column in table.columns
        if "date" in str(column.data_type or column.column_name).lower()
    }
    errors: list[str] = []
    for column in re.findall(r'"([^"]+)"\s*=\s*\'\'', query):
        if column in date_columns:
            errors.append(f"SQL compares date column {column} to an empty string; use IS NULL for missing dates.")
    return errors


def _split_validation_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """Handle the internal split validation errors helper path for this module.

    Inputs:
        Receives errors for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._split_validation_errors.
    """
    contract_markers = (
        "Action plan",
        "depends on",
        "references unknown",
        "dependency cycle",
        "Final action",
        "Unknown or disallowed tool",
        "Invalid inputs",
    )
    contract: list[str] = []
    domain: list[str] = []
    for error in errors:
        if any(marker in error for marker in contract_markers):
            contract.append(error)
        else:
            domain.append(error)
    return contract, domain


def _collect_action_refs(value: Any) -> set[str]:
    """Handle the internal collect action refs helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._collect_action_refs.
    """
    refs = collect_step_references(_normalize_refs(value, {}))
    if isinstance(value, str):
        match = REFERENCE_RE.match(value.strip())
        if match:
            refs.add(_normalize_name(match.group(1)))
    elif isinstance(value, dict):
        for nested in value.values():
            refs.update(_collect_action_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            refs.update(_collect_action_refs(nested))
    return refs


def _iter_action_refs_with_paths(value: Any) -> list[tuple[str, str | None]]:
    """Handle the internal iter action refs with paths helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._iter_action_refs_with_paths.
    """
    refs: list[tuple[str, str | None]] = []
    normalized = _normalize_refs(value, {})
    if isinstance(normalized, dict):
        raw_ref = normalized.get("$ref")
        if isinstance(raw_ref, str) and raw_ref.strip():
            raw_path = normalized.get("path")
            refs.append((_normalize_name(raw_ref), None if raw_path is None else str(raw_path).strip()))
            return refs
        for nested in normalized.values():
            refs.extend(_iter_action_refs_with_paths(nested))
    elif isinstance(normalized, list):
        for nested in normalized:
            refs.extend(_iter_action_refs_with_paths(nested))
    elif isinstance(value, str):
        match = REFERENCE_RE.match(value.strip())
        if match:
            refs.append((_normalize_name(match.group(1)), str(match.group(2) or "").strip() or None))
    return refs


def _validate_shell_scalar_flow(plan: ActionPlan, *, goal: str) -> list[str]:
    """Handle the internal validate shell scalar flow helper path for this module.

    Inputs:
        Receives plan, goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._validate_shell_scalar_flow.
    """
    output_intent = resolve_output_intent(goal)
    if not (output_intent.kind == "scalar" and output_intent.cardinality == "single") or is_shell_status_goal(goal):
        return []
    alias_to_tool: dict[str, str] = {}
    for action in plan.actions:
        alias_to_tool[action.id] = action.tool
        if action.output_binding:
            alias_to_tool[str(action.output_binding)] = action.tool
    errors: list[str] = []
    for action in plan.actions:
        if action.tool != "runtime.return":
            continue
        for alias, path in _iter_action_refs_with_paths(action.inputs.get("value")):
            if alias_to_tool.get(alias) != "shell.exec":
                continue
            root_path = str(path or "").split(".", 1)[0].strip()
            if root_path in {"exit_code", "returncode"}:
                errors.append("Shell count requests must return stdout, not command exit status.")
    return errors


def _placeholder_aliases(value: Any, aliases: set[str]) -> set[str]:
    """Handle the internal placeholder aliases helper path for this module.

    Inputs:
        Receives value, aliases for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._placeholder_aliases.
    """
    found: set[str] = set()
    normalized_aliases = {_normalize_name(alias) for alias in aliases if _normalize_name(alias)}
    if isinstance(value, str):
        for match in re.finditer(r"\{([A-Za-z0-9_-]+)\}", value):
            alias = _normalize_name(match.group(1))
            if alias in normalized_aliases:
                found.add(alias)
    elif isinstance(value, dict):
        for nested in value.values():
            found.update(_placeholder_aliases(nested, aliases))
    elif isinstance(value, list):
        for nested in value:
            found.update(_placeholder_aliases(nested, aliases))
    return found


def _normalize_refs(value: Any, aliases_by_action: dict[str, str]) -> Any:
    """Handle the internal normalize refs helper path for this module.

    Inputs:
        Receives value, aliases_by_action for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_refs.
    """
    if isinstance(value, str):
        match = REFERENCE_RE.match(value.strip())
        if match:
            raw_ref = _normalize_name(match.group(1))
            path = str(match.group(2) or "").strip()
            ref = aliases_by_action.get(raw_ref, raw_ref)
            payload: dict[str, Any] = {"$ref": ref}
            if path:
                payload["path"] = path
            return payload
        return value
    if isinstance(value, dict):
        if "$ref" in value:
            copied = dict(value)
            copied["$ref"] = aliases_by_action.get(_normalize_name(str(value["$ref"])), _normalize_name(str(value["$ref"])))
            return copied
        return {key: _normalize_refs(nested, aliases_by_action) for key, nested in value.items()}
    if isinstance(value, list):
        return [_normalize_refs(item, aliases_by_action) for item in value]
    return value


def _has_cycle(plan: ActionPlan) -> bool:
    """Handle the internal has cycle helper path for this module.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._has_cycle.
    """
    graph = {action.id: list(action.depends_on) for action in plan.actions}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        """Visit for the surrounding runtime workflow.

        Inputs:
            Receives node for this function; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner.visit.
        """
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in graph.get(node, []):
            if visit(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _mentioned_database(goal: str, configured: dict[str, str]) -> bool:
    """Handle the internal mentioned database helper path for this module.

    Inputs:
        Receives goal, configured for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._mentioned_database.
    """
    text = str(goal or "").lower()
    if any(name.lower() in text or name.replace("_", " ").lower() in text for name in configured):
        return True
    return bool(DATABASE_NAME_RE.search(text))


def _normalize_database_name(value: str) -> str:
    """Handle the internal normalize database name helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_database_name.
    """
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _normalize_name(value: Any) -> str:
    """Handle the internal normalize name helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.action_planner._normalize_name.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")


_READ_ONLY_SLURM_TOOLS = {
    "slurm.queue",
    "slurm.job_detail",
    "slurm.nodes",
    "slurm.node_detail",
    "slurm.partitions",
    "slurm.accounting",
    "slurm.accounting_aggregate",
    "slurm.metrics",
    "slurm.slurmdbd_health",
}
