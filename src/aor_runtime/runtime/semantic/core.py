"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.core

Purpose:
    Provide the implementation behind the semantic-frame compatibility facade.

Responsibilities:
    Extract, canonicalize, validate, compile, and project semantic frames for compound and multi-target tasks.

Data flow / Interfaces:
    Consumes user goals, runtime settings, LLM output, tool capability facts, and tool results; returns ExecutionPlans and projected results.

Boundaries:
    The LLM may describe intent, but this module forbids executable payloads and keeps strategy selection deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.core.utils import dumps_json, extract_json_object
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.diagnostic_orchestration import diagnostic_plan_for_goal
from aor_runtime.runtime.llm_recursion import LLMStageRecursionBudget, LLMStageRecursionState
from aor_runtime.runtime.output_shape import resolve_output_intent
from aor_runtime.runtime.semantic.action_matrix import (
    SemanticActionRow,
    select_semantic_action,
    select_semantic_action_decision,
    semantic_action_prompt_metadata,
)
from aor_runtime.runtime.semantic.ontology import (
    detect_prompt_entities,
    normalize_semantic_dimension,
    normalize_semantic_entity,
    normalize_semantic_filter_field,
    normalize_semantic_token,
    ontology_prompt_metadata,
)
from aor_runtime.runtime.slurm_aggregations import format_elapsed_seconds
from aor_runtime.runtime.temporal import (
    TemporalRange,
    current_local_datetime,
    parse_temporal_range,
    runtime_date_context,
)


SemanticDomain = Literal["slurm", "sql", "filesystem", "shell", "text", "diagnostic", "unknown"]
SemanticComposition = Literal["single", "sequence", "parallel_safe", "compare", "dashboard", "diagnostic_sections"]
SemanticIntent = Literal[
    "count",
    "list",
    "aggregate_metric",
    "status",
    "schema_answer",
    "validate_or_explain",
    "diagnostic",
    "transform",
    "unknown",
]
SemanticOutputKind = Literal["scalar", "table", "file", "text", "json", "status", "unknown"]
SemanticOutputCardinality = Literal["single", "multi_scalar", "grouped", "collection", "sectioned", "unknown"]
SemanticRenderStyle = Literal["scalar", "metric_table", "record_table", "key_value", "bullets", "sectioned", "unknown"]
SemanticOutputSource = Literal["llm", "deterministic_backup", "planner_expected", "merged", "unknown"]
SemanticStrategy = Literal[
    "single_target_pushdown",
    "grouped_pushdown",
    "multi_target_pushdown",
    "fan_out",
    "local_projection",
    "fallback_action_planner",
    "unsupported",
]
SemanticFilterValue = str | int | float | bool | list[str | int | float | bool] | None

DISALLOWED_FRAME_KEYS = {
    "action",
    "argv",
    "bash",
    "code",
    "command",
    "execution_plan",
    "gateway_command",
    "shell",
    "sql",
    "steps",
    "tool",
    "tool_name",
}
DISALLOWED_FRAME_TEXT_RE = re.compile(
    r"\b(?:shell\.exec|python\.exec|runtime\.return|sql\.query|slurm\.[a-z_]+|ExecutionPlan|"
    r"squeue|sacct|sinfo|scontrol|sbatch|scancel|SELECT\s+.+\s+FROM|bash\s+-c|sh\s+-c)\b",
    re.IGNORECASE | re.DOTALL,
)
SLURM_AGGREGATE_RE = re.compile(
    r"\b(?:average|avg|mean|min(?:imum)?|max(?:imum)?|sum|total)\b[^.?!]*\b(?:job|jobs|runtime|elapsed|times?|duration)\b"
    r"|\b(?:job|jobs)\b[^.?!]*\b(?:average|avg|mean|min(?:imum)?|max(?:imum)?|sum|total)\b[^.?!]*\b(?:time|runtime|elapsed|duration)",
    re.IGNORECASE,
)
SLURM_DOMAIN_RE = re.compile(r"\b(?:slurm|partition|partitions|sacct|squeue|cluster|jobs?)\b", re.IGNORECASE)
SLURM_ALL_STATES_RE = re.compile(
    r"\bdo\s+not\s+filter\s+by\s+completed\b|"
    r"\bdon't\s+filter\s+by\s+completed\b|"
    r"\bdo\s+not\s+restrict\s+to\s+completed\b|"
    r"\bdon't\s+restrict\s+to\s+completed\b|"
    r"\bnot\s+just\s+completed\b|"
    r"\ball\s+jobs?\b|"
    r"\bany\s+state\b|"
    r"\ball\s+job\s+states\b|"
    r"\bacross\s+all\s+states\b|"
    r"\bregardless\s+of\s+state\b|"
    r"\bget\s+all\s+jobs?\b|"
    r"\binclude\s+all\s+job\s+states\b|"
    r"\binclude\s+failed\s+jobs?\s+too\b",
    re.IGNORECASE,
)
PARTITION_TARGETS_RE = re.compile(
    r"\b([A-Za-z0-9._-]+(?:\s*,\s*[A-Za-z0-9._-]+)+)\s+partitions?\b",
    re.IGNORECASE,
)
SINGLE_PARTITION_RE = re.compile(
    r"\b(?:in|on|for)\s+([A-Za-z0-9._-]+)\s+partitions?\b|\b([A-Za-z0-9._-]+)\s+partitions?\b",
    re.IGNORECASE,
)
COUNT_GOAL_RE = re.compile(r"\b(?:count|how\s+many|number\s+of)\b", re.IGNORECASE)
COMPOUND_SPLIT_RE = re.compile(r"\b(?:then|and then|after that|also|plus summarize|as well as)\b|[;]", re.IGNORECASE)
EXPLICIT_MULTI_TARGET_RE = re.compile(r"\b(?:by|per|each|for each|separately)\b", re.IGNORECASE)
FILESYSTEM_DOMAIN_RE = re.compile(r"\b(?:file|files|folder|folders|directory|directories|csv|json|markdown|md|py|python)\b", re.IGNORECASE)
SHELL_DOMAIN_RE = re.compile(r"\b(?:process|processes|port|ports|disk|memory|cpu|resource|resources|hostname|uptime|mounted|filesystem)\b", re.IGNORECASE)
DIAGNOSTIC_DOMAIN_RE = re.compile(r"\b(?:diagnostic|capabilities|workspace|configuration|config flags|end-to-end)\b", re.IGNORECASE)
SQL_VALIDATE_RE = re.compile(r"\b(?:generate|validate|explain)\b.+\bsql\b|\bsql\b.+\b(?:validate|explain)\b", re.IGNORECASE)
EXTENSION_RE = re.compile(r"\b(?:csv|json|txt|md|markdown|py|python)\b", re.IGNORECASE)


class SemanticTargetSet(BaseModel):
    """Describe named target values for one semantic dimension.

    Inputs:
        Created from LLM or deterministic extraction with a dimension and requested values.

    Returns:
        Pydantic model used by frame canonicalization, strategy selection, and projection.

    Used by:
        SemanticFrame, SemanticFrameCanonicalizer, and SemanticFrameCompiler.
    """
    model_config = ConfigDict(extra="forbid")

    dimension: str
    values: list[str] = Field(default_factory=list)
    mode: Literal["explicit", "all", "none"] = "none"

    @field_validator("dimension")
    @classmethod
    def normalize_dimension(cls, value: str) -> str:
        """Normalize a target dimension into a stable lower-case identifier.

        Inputs:
            Receives the raw dimension string.

        Returns:
            The normalized dimension name.

        Used by:
            Pydantic validation for SemanticTargetSet.
        """
        return _normalize_token(value)

    @field_validator("values", mode="before")
    @classmethod
    def normalize_values(cls, value: Any) -> list[str]:
        """Normalize target values into unique display-preserving tokens.

        Inputs:
            Receives raw scalar, comma-separated string, or list values.

        Returns:
            A de-duplicated list of target values.

        Used by:
            Pydantic validation for SemanticTargetSet.
        """
        if isinstance(value, SemanticTargetSet):
            return _target_values(value.model_dump().get("values"))
        return _target_values(value)

    @model_validator(mode="after")
    def infer_mode(self) -> "SemanticTargetSet":
        """Infer explicit/none mode from the presence of target values.

        Inputs:
            Uses the validated instance state.

        Returns:
            The updated target-set model.

        Used by:
            Pydantic validation for SemanticTargetSet.
        """
        if _target_set_values(self) and self.mode == "none":
            self.mode = "explicit"
        return self


class SemanticFilter(BaseModel):
    """Represent a safe semantic filter requested by the user.

    Inputs:
        Created from LLM or deterministic extraction with field/operator/value data.

    Returns:
        A typed filter used by coverage validation and plan compilation.

    Used by:
        SemanticFrame and future domain compilers.
    """
    model_config = ConfigDict(extra="forbid")

    field: str
    value: SemanticFilterValue = None
    operator: Literal["eq", "neq", "in", "gt", "gte", "lt", "lte", "contains"] = "eq"

    @field_validator("field")
    @classmethod
    def normalize_field(cls, value: str) -> str:
        """Normalize a filter field into a stable lower-case identifier.

        Inputs:
            Receives the raw filter field.

        Returns:
            Normalized filter field text.

        Used by:
            Pydantic validation for SemanticFilter.
        """
        return _normalize_token(value)

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: Any) -> Any:
        """Normalize filter values while preserving typed target sets.

        Inputs:
            Receives scalar, list, or null values from an LLM semantic frame.

        Returns:
            A cleaned scalar, list, or None value.

        Used by:
            Pydantic validation for SemanticFilter.
        """
        if isinstance(value, list):
            cleaned: list[Any] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        cleaned.append(text)
                    continue
                cleaned.append(item)
            return cleaned
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_list_operator(self) -> "SemanticFilter":
        """Require list-valued filters to use the typed `in` operator.

        Inputs:
            Uses the validated filter operator and value.

        Returns:
            The filter when its value/operator contract is valid.

        Used by:
            Pydantic validation for SemanticFilter.
        """
        if isinstance(self.value, list) and self.operator != "in":
            raise ValueError("List-valued semantic filters must use operator='in'.")
        return self


class SemanticMetric(BaseModel):
    """Describe the metric requested by an aggregate semantic frame.

    Inputs:
        Created from semantic extraction with metric name and optional unit.

    Returns:
        A typed metric object used by strategy selection.

    Used by:
        SemanticFrame and SemanticFrameCompiler.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    unit: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize metric aliases into runtime metric identifiers.

        Inputs:
            Receives metric names such as avg, average time, or average_elapsed.

        Returns:
            The canonical metric identifier.

        Used by:
            Pydantic validation for SemanticMetric.
        """
        return normalize_metric_name(value)


class SemanticTimeWindow(BaseModel):
    """Represent a user-requested temporal window.

    Inputs:
        Created from LLM or deterministic extraction with phrase/start/end values.

    Returns:
        A normalized temporal object for tool-safe arguments and metadata.

    Used by:
        SemanticFrameCanonicalizer and SemanticFrameCompiler.
    """
    model_config = ConfigDict(extra="forbid")

    phrase: str | None = None
    start: str | None = None
    end: str | None = None
    label: str | None = None


class SemanticOutputContract(BaseModel):
    """Represent the requested user-visible output shape.

    Inputs:
        Created from semantic extraction or canonicalization.

    Returns:
        A typed output kind for strategy and result-shape decisions.

    Used by:
        SemanticFrame and SemanticFrameCompiler.
    """
    model_config = ConfigDict(extra="forbid")

    kind: SemanticOutputKind = "unknown"
    format: str | None = None
    cardinality: SemanticOutputCardinality = "unknown"
    render_style: SemanticRenderStyle = "unknown"
    result_entities: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    source: SemanticOutputSource = "unknown"
    corrections: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_output_payload(cls, value: Any) -> Any:
        """Normalize nullable list fields in LLM output-intent payloads.

        Inputs:
            Receives raw output contract payloads from semantic extraction.

        Returns:
            A payload with list fields defaulted to empty lists.

        Used by:
            Pydantic validation for SemanticOutputContract.
        """
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for key in ("result_entities", "group_by", "corrections"):
            if normalized.get(key) is None:
                normalized[key] = []
        return normalized


@dataclass(frozen=True)
class SlurmAccountingStatePolicy:
    """Describe how a SLURM accounting aggregate should constrain job states.

    Inputs:
        Created from typed semantic-frame state policy or canonical filters.

    Returns:
        A compact state policy consumed by the SLURM aggregate compiler.

    Used by:
        SemanticFrameCompiler._compile_slurm_aggregate.
    """

    state: str | None = None
    include_all_states: bool = False
    default_state_applied: bool = False


class SemanticFrame(BaseModel):
    """Describe the user's request before choosing executable tools.

    Inputs:
        Created from LLM output, deterministic backup extraction, or child frames.

    Returns:
        A typed semantic frame that can be canonicalized, validated, and compiled.

    Used by:
        SemanticFramePlanner, SemanticFrameCompiler, and semantic coverage validation.
    """

    model_config = ConfigDict(extra="forbid")

    frame_id: str | None = None
    parent_id: str | None = None
    depth: int = 0
    composition: SemanticComposition = "single"
    domain: SemanticDomain = "unknown"
    intent: SemanticIntent = "unknown"
    entity: str | None = None
    metric: SemanticMetric | None = None
    dimensions: list[str] = Field(default_factory=list)
    targets: dict[str, SemanticTargetSet] = Field(default_factory=dict)
    filters: list[SemanticFilter] = Field(default_factory=list)
    state_policy: SlurmAccountingStatePolicy | None = None
    time_window: SemanticTimeWindow | None = None
    output: SemanticOutputContract = Field(default_factory=SemanticOutputContract)
    children: list["SemanticFrame"] = Field(default_factory=list)
    allow_action_fallback: bool = False
    confidence: float = 1.0
    source: Literal["llm", "deterministic", "merged"] = "deterministic"
    rationale: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_frame_payload(cls, value: Any) -> Any:
        """Normalize common semantic-frame payload aliases before validation.

        Inputs:
            Receives raw decoded LLM or deterministic frame data.

        Returns:
            A payload with canonical field names.

        Used by:
            Pydantic validation for SemanticFrame.
        """
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "output_contract" in normalized and "output" not in normalized:
            normalized["output"] = normalized.pop("output_contract")
        if "metrics" in normalized and "metric" not in normalized:
            metrics = normalized.pop("metrics")
            if isinstance(metrics, list) and metrics:
                normalized["metric"] = metrics[0]
            elif isinstance(metrics, (str, dict)):
                normalized["metric"] = metrics
        for key in ("dimensions", "filters", "children"):
            if normalized.get(key) is None:
                normalized[key] = []
        if isinstance(normalized.get("output"), dict):
            output = dict(normalized["output"])
            for key in ("result_entities", "group_by", "corrections"):
                if output.get(key) is None:
                    output[key] = []
            normalized["output"] = output
        return normalized

    @model_validator(mode="after")
    def validate_recursive_limits(self) -> "SemanticFrame":
        """Validate recursive semantic-frame invariants.

        Inputs:
            Uses validated frame depth and child-frame metadata.

        Returns:
            The frame when recursion metadata is internally consistent.

        Used by:
            Pydantic validation before canonicalization and compilation.
        """
        if self.depth < 0:
            raise ValueError("Semantic frame depth must be zero or greater.")
        for child in self.children:
            if child.depth <= self.depth:
                child.depth = self.depth + 1
            if not child.parent_id and self.frame_id:
                child.parent_id = self.frame_id
        return self

    @field_validator("dimensions", mode="before")
    @classmethod
    def normalize_dimensions(cls, value: Any) -> list[str]:
        """Normalize frame dimensions into unique identifiers.

        Inputs:
            Receives raw dimension values from LLM or deterministic extraction.

        Returns:
            De-duplicated dimension identifiers.

        Used by:
            Pydantic validation for SemanticFrame.
        """
        items = value if isinstance(value, list) else [value] if value else []
        return list(dict.fromkeys(_normalize_token(item) for item in items if _normalize_token(item)))

    @field_validator("targets", mode="before")
    @classmethod
    def normalize_targets(cls, value: Any) -> dict[str, Any]:
        """Normalize raw target dictionaries into SemanticTargetSet payloads.

        Inputs:
            Receives a target mapping from the LLM or deterministic extraction.

        Returns:
            A mapping keyed by normalized dimensions.

        Used by:
            Pydantic validation for SemanticFrame.
        """
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key, raw_target in value.items():
            dimension = _normalize_token(key)
            if not dimension:
                continue
            if isinstance(raw_target, SemanticTargetSet):
                payload = raw_target.model_dump()
                payload.setdefault("dimension", dimension)
            elif isinstance(raw_target, dict):
                payload = dict(raw_target)
                payload.setdefault("dimension", dimension)
            else:
                payload = {"dimension": dimension, "values": raw_target}
            normalized[dimension] = payload
        return normalized


class SemanticCompound(BaseModel):
    """Represent a compound request as an ordered set of semantic frames.

    Inputs:
        Created when a user prompt contains independent tasks rather than one repeated operation.

    Returns:
        A typed container for future compound orchestration.

    Used by:
        Semantic frame tests and future orchestration compilers.
    """

    children: list[SemanticFrame] = Field(default_factory=list)
    output: SemanticOutputContract = Field(default_factory=lambda: SemanticOutputContract(kind="text"))


SemanticFrameProposal = SemanticFrame


@dataclass
class SemanticFrameExtractionResult:
    """Carry the outcome of semantic-frame extraction.

    Inputs:
        Built by deterministic extraction or the LLMSemanticFrameExtractor.

    Returns:
        Matched frame, metadata, and failure reason for callers.

    Used by:
        SemanticFramePlanner and tests.
    """

    matched: bool
    frame: SemanticFrame | None = None
    reason: str | None = None
    llm_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCompilationResult:
    """Carry a compiled semantic frame plan and its strategy metadata.

    Inputs:
        Created by SemanticFrameCompiler when a frame can be represented safely.

    Returns:
        ExecutionPlan and metadata for planner telemetry.

    Used by:
        TaskPlanner and tests.
    """

    plan: ExecutionPlan
    strategy: SemanticStrategy
    frame: SemanticFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCoverageResult:
    """Describe whether an execution plan covers a semantic frame.

    Inputs:
        Created from a frame and candidate ExecutionPlan.

    Returns:
        Boolean coverage result plus compact repair facts.

    Used by:
        SemanticFramePlanner and action-planner fallback validation.
    """

    covered: bool
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticCapabilitySpec:
    """Declare semantic strategies supported by one domain/tool family.

    Inputs:
        Built from static runtime facts about supported dimensions, metrics, and fallbacks.

    Returns:
        Immutable capability metadata used by strategy selection.

    Used by:
        SemanticStrategySelector and semantic-frame tests.
    """

    domain: str
    tools: tuple[str, ...]
    intents: tuple[str, ...]
    metrics: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    multi_target_pushdown: bool = False
    fan_out: bool = False
    local_projection: bool = False


@dataclass(frozen=True)
class SemanticStrategyDecision:
    """Describe the selected execution strategy for a semantic frame.

    Inputs:
        Created from a canonical semantic frame and domain capability spec.

    Returns:
        Strategy name plus compact facts for compilation metadata.

    Used by:
        SemanticFrameCompiler and coverage tests.
    """

    strategy: SemanticStrategy
    reason: str
    group_by: str | None = None
    fanout_dimension: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


SEMANTIC_CAPABILITIES: dict[str, SemanticCapabilitySpec] = {
    "slurm": SemanticCapabilitySpec(
        domain="slurm",
        tools=(
            "slurm.queue",
            "slurm.accounting",
            "slurm.accounting_aggregate",
            "slurm.metrics",
            "slurm.nodes",
            "slurm.partitions",
        ),
        intents=("count", "list", "aggregate_metric", "status"),
        metrics=("average_elapsed", "min_elapsed", "max_elapsed", "sum_elapsed", "count", "runtime_summary"),
        filters=("partition", "state", "user", "start", "end", "include_all_states"),
        dimensions=("partition", "state", "user", "node", "job_name"),
        group_by=("partition", "state", "user", "job_name"),
        multi_target_pushdown=True,
        fan_out=True,
        local_projection=True,
    ),
    "filesystem": SemanticCapabilitySpec(
        domain="filesystem",
        tools=("fs.aggregate", "fs.glob", "fs.find", "fs.list", "fs.read"),
        intents=("count", "list", "transform"),
        metrics=("count", "total_size", "count_and_total_size"),
        filters=("path", "extension", "pattern", "type"),
        dimensions=("extension", "path", "type"),
        fan_out=True,
        local_projection=True,
    ),
    "shell": SemanticCapabilitySpec(
        domain="shell",
        tools=("shell.exec",),
        intents=("count", "list", "status"),
        metrics=("count",),
        filters=("process", "port", "filesystem"),
        dimensions=("process", "port", "resource"),
        fan_out=True,
    ),
    "sql": SemanticCapabilitySpec(
        domain="sql",
        tools=("sql.query", "sql.schema", "sql.validate"),
        intents=("count", "list", "aggregate_metric", "schema_answer", "validate_or_explain"),
        metrics=("count", "average", "min", "max"),
        filters=("database", "modality", "patient", "study", "series", "age", "patient_age"),
        dimensions=("modality", "patient", "study", "series"),
        group_by=("modality", "patient", "study", "series"),
        local_projection=True,
    ),
    "diagnostic": SemanticCapabilitySpec(
        domain="diagnostic",
        tools=("fs.list", "fs.search_content", "sql.schema", "shell.exec"),
        intents=("diagnostic",),
        dimensions=("section",),
        fan_out=True,
    ),
}


class SemanticStrategySelector:
    """Select deterministic execution strategy for canonical semantic frames.

    Inputs:
        Constructed from semantic capability metadata.

    Returns:
        Strategy decisions used by domain compilers.

    Used by:
        SemanticFrameCompiler and tests.
    """

    def __init__(self, capabilities: dict[str, SemanticCapabilitySpec] | None = None) -> None:
        """Initialize the strategy selector.

        Inputs:
            Receives optional capability metadata for tests or custom domains.

        Returns:
            Initializes the selector instance.

        Used by:
            SemanticFrameCompiler.
        """
        self.capabilities = capabilities or SEMANTIC_CAPABILITIES

    def select(self, frame: SemanticFrame) -> SemanticStrategyDecision:
        """Choose grouped, pushdown, fan-out, local projection, or fallback strategy.

        Inputs:
            Receives a canonical semantic frame.

        Returns:
            A semantic strategy decision with compact metadata.

        Used by:
            SemanticFrameCompiler.
        """
        spec = self.capabilities.get(frame.domain)
        if spec is None or frame.intent not in spec.intents:
            return SemanticStrategyDecision("unsupported", "domain_or_intent_not_supported")
        target_dimensions = [
            dimension
            for dimension, target in frame.targets.items()
            if target.mode == "all" or len(_target_set_values(target)) > 1
        ]
        for dimension in target_dimensions:
            if dimension in spec.group_by:
                return SemanticStrategyDecision("grouped_pushdown", "multi_target_group_by", group_by=dimension)
        for dimension in target_dimensions:
            if spec.multi_target_pushdown and dimension in spec.dimensions:
                return SemanticStrategyDecision("multi_target_pushdown", "multi_target_supported", group_by=dimension)
        for dimension in target_dimensions:
            if spec.fan_out and dimension in spec.dimensions:
                return SemanticStrategyDecision("fan_out", "bounded_fan_out", fanout_dimension=dimension)
        if any(target.mode == "all" for target in frame.targets.values()) and spec.local_projection:
            return SemanticStrategyDecision("local_projection", "all_targets_local_projection")
        return SemanticStrategyDecision("single_target_pushdown", "single_target_or_no_target")


class LLMSemanticFrameExtractor:
    """Extract typed semantic frames from LLM output without executable payloads.

    Inputs:
        Constructed with an LLM client and settings.

    Returns:
        SemanticFrameExtractionResult objects for planner integration.

    Used by:
        SemanticFramePlanner and focused semantic-frame tests.
    """

    def __init__(self, *, llm: LLMClient, settings: Settings) -> None:
        """Initialize the semantic-frame extractor.

        Inputs:
            Receives the LLM client and runtime settings.

        Returns:
            Initializes the extractor instance.

        Used by:
            TaskPlanner and tests that exercise safe frame extraction.
        """
        self.llm = llm
        self.settings = settings
        self.last_prompt: dict[str, Any] | None = None
        self.last_raw_output: str | None = None

    def extract(self, *, goal: str, allowed_tools: list[str], model: str | None = None, temperature: float = 0.0) -> SemanticFrameExtractionResult:
        """Ask the LLM to produce a constrained semantic frame.

        Inputs:
            Receives the user goal, allowed tools, and optional LLM settings.

        Returns:
            A matched frame or a safe failure reason without executable payloads.

        Used by:
            SemanticFramePlanner when LLM semantic extraction is enabled for a candidate request.
        """
        budget = LLMStageRecursionBudget(
            self.settings,
            stage="semantic_frame",
            max_depth=int(getattr(self.settings, "semantic_frame_max_depth", 10) or 10),
        )
        result = self._extract_recursive(
            goal=goal,
            allowed_tools=allowed_tools,
            model=model,
            temperature=temperature,
            budget=budget,
            depth=0,
            parent_id=None,
            composition="single",
            parent_frame=None,
        )
        result.llm_calls = budget.calls
        result.metadata.update(
            {
                "source": "llm",
                "llm_stage_calls": {"semantic_frame": budget.calls},
                "llm_recursion_depth": budget.max_observed_depth,
                "llm_stage_budget": budget.metadata(),
            }
        )
        return result

    def _extract_recursive(
        self,
        *,
        goal: str,
        allowed_tools: list[str],
        model: str | None,
        temperature: float,
        budget: LLMStageRecursionBudget,
        depth: int,
        parent_id: str | None,
        composition: str,
        parent_frame: SemanticFrame | None,
    ) -> SemanticFrameExtractionResult:
        """Ask the LLM for one semantic frame and recursively complete children.

        Inputs:
            Receives the user/child goal, tool allow-list, LLM settings, recursion budget, and parent metadata.

        Returns:
            A complete semantic frame or a safe typed failure reason.

        Used by:
            LLMSemanticFrameExtractor.extract.
        """
        try:
            stage_state = budget.state(
                depth=depth,
                parent_id=parent_id,
                composition=composition,
                allowed_schema=_semantic_frame_allowed_schema(),
            )
        except ValueError as exc:
            return SemanticFrameExtractionResult(False, reason=str(exc), llm_calls=budget.calls)
        prompt = build_semantic_frame_prompt(
            goal=goal,
            settings=self.settings,
            allowed_tools=allowed_tools,
            recursion_state=stage_state,
            parent_frame=parent_frame,
        )
        self.last_prompt = prompt
        try:
            raw = self.llm.complete(
                system_prompt=SEMANTIC_FRAME_SYSTEM_PROMPT,
                user_prompt=dumps_json(prompt, indent=2),
                model=model,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            return SemanticFrameExtractionResult(False, reason=f"semantic_frame_llm_failed:{exc}", llm_calls=1)
        self.last_raw_output = raw
        try:
            payload = extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001
            return SemanticFrameExtractionResult(False, reason=f"semantic_frame_malformed_json:{exc}", llm_calls=1)
        if _contains_executable_payload(payload):
            return SemanticFrameExtractionResult(False, reason="semantic_frame_contains_executable_payload", llm_calls=1)
        try:
            frame = SemanticFrame.model_validate(payload)
        except ValidationError as exc:
            repaired_payload = self._repair_invalid_payload(
                goal=goal,
                payload=payload,
                validation_error=exc,
                allowed_tools=allowed_tools,
                model=model,
                temperature=temperature,
                budget=budget,
                depth=depth,
                parent_id=parent_id,
                composition=composition,
                parent_frame=parent_frame,
            )
            if repaired_payload is None:
                return SemanticFrameExtractionResult(
                    False,
                    reason="semantic_frame_contract_invalid",
                    llm_calls=budget.calls,
                    metadata={"validation_facts": _semantic_validation_facts(exc)},
                )
            try:
                frame = SemanticFrame.model_validate(repaired_payload)
            except ValidationError as repair_exc:
                return SemanticFrameExtractionResult(
                    False,
                    reason="semantic_frame_repair_failed",
                    llm_calls=budget.calls,
                    metadata={"validation_facts": _semantic_validation_facts(repair_exc)},
                )
        frame.source = "llm"
        if not frame.frame_id:
            frame.frame_id = f"llm_frame_{depth}_{budget.calls}"
        if parent_id and not frame.parent_id:
            frame.parent_id = parent_id
        frame.depth = depth
        expanded = self._expand_frame_children(
            frame,
            goal=goal,
            allowed_tools=allowed_tools,
            model=model,
            temperature=temperature,
            budget=budget,
        )
        if not expanded.matched or expanded.frame is None:
            return expanded
        frame = expanded.frame
        contract_error = _validate_llm_semantic_frame_contract(frame)
        if contract_error:
            return SemanticFrameExtractionResult(False, reason=contract_error, llm_calls=1)
        canonicalized = SemanticFrameCanonicalizer(self.settings).canonicalize(frame, goal=goal, allow_prompt_inference=False)
        contract_error = _validate_llm_semantic_frame_contract(canonicalized)
        if contract_error:
            return SemanticFrameExtractionResult(False, reason=contract_error, llm_calls=1)
        return SemanticFrameExtractionResult(True, frame=canonicalized, llm_calls=1, metadata={"source": "llm"})

    def _repair_invalid_payload(
        self,
        *,
        goal: str,
        payload: Any,
        validation_error: ValidationError,
        allowed_tools: list[str],
        model: str | None,
        temperature: float,
        budget: LLMStageRecursionBudget,
        depth: int,
        parent_id: str | None,
        composition: str,
        parent_frame: SemanticFrame | None,
    ) -> dict[str, Any] | None:
        """Ask for one schema-only repair of an invalid semantic-frame payload.

        Inputs:
            Receives the original prompt, invalid JSON, validation facts, and recursion metadata.

        Returns:
            A repaired semantic-frame payload, or None when repair is unsafe or unavailable.

        Used by:
            LLMSemanticFrameExtractor._extract_recursive.
        """
        try:
            stage_state = budget.state(
                depth=depth,
                parent_id=parent_id,
                composition=composition,
                allowed_schema=_semantic_frame_allowed_schema(),
            )
        except ValueError:
            return None
        prompt = build_semantic_frame_repair_prompt(
            goal=goal,
            invalid_payload=payload,
            validation_facts=_semantic_validation_facts(validation_error),
            settings=self.settings,
            allowed_tools=allowed_tools,
            recursion_state=stage_state,
            parent_frame=parent_frame,
        )
        try:
            raw = self.llm.complete(
                system_prompt=SEMANTIC_FRAME_REPAIR_SYSTEM_PROMPT,
                user_prompt=dumps_json(prompt, indent=2),
                model=model,
                temperature=temperature,
            )
        except Exception:  # noqa: BLE001
            return None
        try:
            repaired = extract_json_object(raw)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(repaired, dict) or _contains_executable_payload(repaired):
            return None
        return repaired

    def _expand_frame_children(
        self,
        frame: SemanticFrame,
        *,
        goal: str,
        allowed_tools: list[str],
        model: str | None,
        temperature: float,
        budget: LLMStageRecursionBudget,
    ) -> SemanticFrameExtractionResult:
        """Recursively complete compound semantic frames.

        Inputs:
            Receives a parent frame, original goal, tool allow-list, LLM settings, and recursion budget.

        Returns:
            The frame with complete children, or a safe typed failure.

        Used by:
            LLMSemanticFrameExtractor._extract_recursive.
        """
        max_children = int(getattr(self.settings, "semantic_frame_max_children", 8) or 8)
        if len(frame.children) > max_children:
            return SemanticFrameExtractionResult(
                False,
                reason=f"Semantic frame has {len(frame.children)} children; maximum is {max_children}.",
                llm_calls=budget.calls,
            )
        if frame.composition != "single" and not frame.children:
            result = self._extract_recursive(
                goal=_semantic_child_expansion_goal(goal=goal, frame=frame),
                allowed_tools=allowed_tools,
                model=model,
                temperature=temperature,
                budget=budget,
                depth=frame.depth + 1,
                parent_id=frame.frame_id,
                composition=frame.composition,
                parent_frame=frame,
            )
            if not result.matched or result.frame is None:
                return result
            frame.children = result.frame.children or [result.frame]
        completed_children: list[SemanticFrame] = []
        for index, child in enumerate(frame.children):
            prepared = _prepare_child_frame(child, parent=frame, index=index)
            if _llm_child_frame_needs_expansion(prepared):
                result = self._extract_recursive(
                    goal=_semantic_child_goal(goal=goal, child=prepared, index=index),
                    allowed_tools=allowed_tools,
                    model=model,
                    temperature=temperature,
                    budget=budget,
                    depth=prepared.depth,
                    parent_id=frame.frame_id,
                    composition=prepared.composition,
                    parent_frame=frame,
                )
                if not result.matched or result.frame is None:
                    return result
                completed_children.append(result.frame)
            else:
                completed_children.append(prepared)
        frame.children = completed_children
        return SemanticFrameExtractionResult(True, frame=frame, llm_calls=budget.calls, metadata={"source": "llm"})


class SemanticFrameCanonicalizer:
    """Canonicalize semantic frames using deterministic runtime rules.

    Inputs:
        Constructed with runtime settings for date/time handling.

    Returns:
        Canonical SemanticFrame instances.

    Used by:
        SemanticFramePlanner, LLMSemanticFrameExtractor, and tests.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the canonicalizer.

        Inputs:
            Receives runtime settings used for temporal normalization.

        Returns:
            Initializes the canonicalizer instance.

        Used by:
            Semantic frame extraction and deterministic backup extraction.
        """
        self.settings = settings

    def canonicalize(self, frame: SemanticFrame, *, goal: str, allow_prompt_inference: bool = True) -> SemanticFrame:
        """Canonicalize aliases, target sets, time windows, and output shape.

        Inputs:
            Receives a frame, original user goal, and whether prompt-derived semantic inference is allowed.

        Returns:
            A canonical semantic frame.

        Used by:
            SemanticFramePlanner before compilation or coverage checks.
        """
        data = frame.model_dump()
        canonical = SemanticFrame.model_validate(data)
        inference_goal = goal if allow_prompt_inference else ""
        canonical.metric = _canonical_metric(canonical.metric, inference_goal)
        if canonical.domain == "unknown" and allow_prompt_inference:
            canonical.domain = _detect_domain(goal)
        if canonical.intent == "unknown" and allow_prompt_inference:
            canonical.intent = _detect_intent(goal)
        if allow_prompt_inference:
            canonical.targets = _merge_target_backups(canonical.targets, goal)
            canonical.targets = _canonical_targets_for_domain(canonical, goal)
        canonical.filters = _canonical_filters_for_domain(canonical, inference_goal)
        canonical.targets, canonical.filters = _promote_list_filters_to_targets(canonical.targets, canonical.filters, canonical.domain)
        canonical = _canonicalize_frame_ontology(canonical, goal=goal)
        canonical.time_window = _canonical_time_window(canonical.time_window, inference_goal, self.settings)
        canonical.output = _canonical_output(canonical, allow_goal_backup=allow_prompt_inference)
        canonical = _canonicalize_frame_ontology(canonical, goal=goal)
        canonical.entity = _normalize_entity(canonical.entity, inference_goal, domain=canonical.domain)
        canonical.composition = _canonical_composition(canonical, inference_goal)
        canonical.children = [
            self.canonicalize(
                _prepare_child_frame(child, parent=canonical, index=index),
                goal=goal,
                allow_prompt_inference=allow_prompt_inference,
            )
            for index, child in enumerate(canonical.children)
        ]
        _validate_frame_runtime_limits(canonical, self.settings)
        return canonical


def _validate_llm_semantic_frame_contract(frame: SemanticFrame) -> str | None:
    """Validate that an LLM semantic frame is complete enough to drive runtime.

    Inputs:
        Receives a canonical LLM semantic frame.

    Returns:
        None for valid frames, otherwise a stable safe failure reason.

    Used by:
        LLMSemanticFrameExtractor before the frame can reach compilation.
    """
    if frame.domain == "unknown":
        return "semantic_frame_missing_domain"
    if frame.intent == "unknown":
        return "semantic_frame_missing_intent"
    if frame.output.kind == "unknown":
        return "semantic_frame_missing_output_kind"
    if frame.output.cardinality == "unknown":
        return "semantic_frame_missing_output_cardinality"
    if frame.output.render_style == "unknown":
        return "semantic_frame_missing_output_render_style"
    for index, child in enumerate(frame.children):
        child_error = _validate_llm_semantic_frame_contract(child)
        if child_error:
            return f"semantic_frame_child_{index}_{child_error}"
    return None


def _llm_child_frame_needs_expansion(child: SemanticFrame) -> bool:
    """Decide whether a child frame needs another typed LLM pass.

    Inputs:
        Receives a child semantic frame emitted by the LLM.

    Returns:
        True when the child is compound, has incomplete typed output intent, or contains incomplete descendants.

    Used by:
        LLMSemanticFrameExtractor._expand_frame_children.
    """
    if child.composition != "single" and not child.children:
        return True
    return _validate_llm_semantic_frame_contract(child) is not None


def _semantic_child_expansion_goal(*, goal: str, frame: SemanticFrame) -> str:
    """Build a safe request for expanding a compound parent frame.

    Inputs:
        Receives the original goal and the typed parent frame.

    Returns:
        A prompt fragment containing only user text and sanitized semantic metadata.

    Used by:
        LLMSemanticFrameExtractor when a parent frame has no children.
    """
    return (
        "Expand this compound semantic frame into complete typed child frames. "
        f"Original user goal: {goal}. "
        f"Parent composition: {frame.composition}. Parent intent: {frame.intent}. Parent entity: {frame.entity or ''}."
    )


def _semantic_child_goal(*, goal: str, child: SemanticFrame, index: int) -> str:
    """Build a safe request for completing one child semantic frame.

    Inputs:
        Receives the original goal, incomplete child frame, and child index.

    Returns:
        A prompt fragment containing no raw operational data.

    Used by:
        LLMSemanticFrameExtractor when a child frame is incomplete or compound.
    """
    return (
        f"Complete child semantic frame {index + 1} for the original user goal: {goal}. "
        f"Known child domain: {child.domain}. Known child intent: {child.intent}. "
        f"Known child entity: {child.entity or ''}. Child rationale: {child.rationale or ''}."
    )


class SemanticFrameCompiler:
    """Compile supported semantic frames into deterministic ExecutionPlans.

    Inputs:
        Constructed with settings and the tool names allowed for the current runtime spec.

    Returns:
        SemanticCompilationResult when a frame can be executed safely.

    Used by:
        SemanticFramePlanner before falling back to LLM action planning.
    """

    def __init__(self, *, settings: Settings, allowed_tools: list[str]) -> None:
        """Initialize the compiler with settings and tool allow-list.

        Inputs:
            Receives runtime settings and allowed tool names.

        Returns:
            Initializes the compiler instance.

        Used by:
            SemanticFramePlanner and tests.
        """
        self.settings = settings
        self.allowed_tools = set(allowed_tools) | {"text.format", "runtime.return"}
        self.strategy_selector = SemanticStrategySelector()

    def compile(self, frame: SemanticFrame) -> SemanticCompilationResult | None:
        """Compile a supported semantic frame to a deterministic execution plan.

        Inputs:
            Receives a canonical semantic frame.

        Returns:
            A compilation result, or None when the frame should fall back to action planning.

        Used by:
            SemanticFramePlanner.
        """
        if frame.children:
            return self._compile_compound(frame)
        semantic_action = select_semantic_action(frame)
        if semantic_action is None or not semantic_action.active:
            return None
        if not set(semantic_action.tools).issubset(self.allowed_tools):
            return None
        if frame.domain == "slurm" and semantic_action.semantic_action_id in {
            "slurm.accounting_elapsed_aggregate",
            "slurm.accounting_grouped_aggregate",
        }:
            return self._compile_slurm_aggregate(frame, semantic_action=semantic_action)
        if frame.domain == "slurm":
            return self._compile_slurm_inspection(frame, semantic_action=semantic_action)
        if frame.domain == "filesystem":
            return self._compile_filesystem(frame, semantic_action=semantic_action)
        if frame.domain == "shell":
            return self._compile_shell(frame, semantic_action=semantic_action)
        if frame.domain == "sql" and semantic_action.semantic_action_id in {
            "sql.schema_list_tables",
            "sql.schema_describe_entity",
            "sql.schema_join_path",
            "sql.validate_sql",
            "sql.explain_sql",
        }:
            return self._compile_sql_metadata(frame, semantic_action=semantic_action)
        if frame.domain == "sql":
            return self._compile_sql_query(frame, semantic_action=semantic_action)
        if frame.domain == "diagnostic":
            return self._compile_diagnostic(frame, semantic_action=semantic_action)
        return None

    def _compile_slurm_aggregate(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile SLURM aggregate frames with grouped partition support.

        Inputs:
            Receives a canonical SLURM aggregate frame.

        Returns:
            A grouped-pushdown execution plan when supported.

        Used by:
            SemanticFrameCompiler.compile.
        """
        if "slurm.accounting_aggregate" not in self.allowed_tools:
            return None
        metric = str(frame.metric.name if frame.metric else "average_elapsed")
        if metric not in {"average_elapsed", "min_elapsed", "max_elapsed", "sum_elapsed", "count", "count_longer_than", "runtime_summary"}:
            return None
        partition_targets = _target_set_values(frame.targets.get("partition"))
        group_by = "partition" if len(partition_targets) > 1 or _target_mode(frame, "partition") == "all" else None
        partition = partition_targets[0] if len(partition_targets) == 1 and group_by is None else None
        time_window = frame.time_window or SemanticTimeWindow()
        args: dict[str, Any] = {
            "metric": metric,
            "partition": partition,
            "group_by": group_by,
            "start": time_window.start,
            "end": time_window.end,
            "time_window_label": time_window.label,
        }
        state_policy = _slurm_accounting_state_policy(frame)
        if state_policy.include_all_states:
            args["include_all_states"] = True
        elif state_policy.state:
            args["state"] = state_policy.state
            if state_policy.default_state_applied:
                args["default_state_applied"] = True
        step_metadata: dict[str, Any] = {}
        if partition_targets and group_by == "partition":
            step_metadata["semantic_projection"] = {
                "field": "partition",
                "values": partition_targets,
                "source": "semantic_frame",
            }
        args = {key: value for key, value in args.items() if value is not None}
        plan = ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "slurm.accounting_aggregate",
                        "args": args,
                        "metadata": step_metadata,
                        "output": "semantic_slurm_aggregate",
                    },
                    {
                        "id": 2,
                        "action": "text.format",
                        "input": ["semantic_slurm_aggregate"],
                        "args": {
                            "source": {"$ref": "semantic_slurm_aggregate", "path": "groups" if group_by else "value_human"},
                            "format": "markdown",
                        },
                        "output": "semantic_formatted_output",
                    },
                    {
                        "id": 3,
                        "action": "runtime.return",
                        "input": ["semantic_formatted_output"],
                        "args": {
                            "value": {"$ref": "semantic_formatted_output", "path": "content"},
                            "mode": "text",
                        },
                        "output": "runtime_return_result",
                    },
                ]
            }
        )
        return SemanticCompilationResult(
            plan=plan,
            strategy="grouped_pushdown" if group_by else "single_target_pushdown",
            frame=frame,
            metadata={
                "semantic_frame": frame.model_dump(),
                "semantic_action": semantic_action.semantic_action_id if semantic_action else "slurm.accounting_elapsed_aggregate",
                "semantic_strategy": "grouped_pushdown" if group_by else "single_target_pushdown",
                "semantic_targets": {"partition": partition_targets} if partition_targets else {},
            },
        )

    def _compile_slurm_inspection(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile SLURM count, list, and status frames.

        Inputs:
            Receives a canonical SLURM frame.

        Returns:
            A deterministic SLURM inspection plan or None when unsupported.

        Used by:
            SemanticFrameCompiler.compile.
        """
        action_id = semantic_action.semantic_action_id if semantic_action else ""
        if action_id == "slurm.accounting_health":
            return self._compile_single_tool_frame(
                frame,
                action="slurm.slurmdbd_health",
                args={},
                output_alias="semantic_slurm_accounting_health",
                source_path="text_lines",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "slurm.list_partitions":
            return self._compile_single_tool_frame(
                frame,
                action="slurm.partitions",
                args=_slurm_partition_args(frame),
                output_alias="semantic_slurm_partitions",
                source_path="partitions",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "slurm.count_partitions":
            return self._compile_single_tool_frame(
                frame,
                action="slurm.partitions",
                args=_slurm_partition_args(frame),
                output_alias="semantic_slurm_partitions",
                source_path="count",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "slurm.list_nodes":
            return self._compile_single_tool_frame(
                frame,
                action="slurm.nodes",
                args=_slurm_node_args(frame),
                output_alias="semantic_slurm_nodes",
                source_path="nodes",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "slurm.count_nodes":
            return self._compile_single_tool_frame(
                frame,
                action="slurm.nodes",
                args=_slurm_node_args(frame),
                output_alias="semantic_slurm_nodes",
                source_path="unique_count",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "slurm.node_detail":
            node = _filter_value(frame, "node") or _filter_value(frame, "node_name")
            if not node:
                return None
            return self._compile_single_tool_frame(
                frame,
                action="slurm.node_detail",
                args={"node": node},
                output_alias="semantic_slurm_node_detail",
                source_path="field_rows",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "slurm.accounting_list_jobs":
            args = _slurm_filter_args(frame)
            time_window = frame.time_window or SemanticTimeWindow()
            args.update({"start": time_window.start, "end": time_window.end, "limit": None})
            group_by = _first_group_dimension(frame, {"partition", "state", "user", "job_name"})
            if group_by:
                args["group_by"] = group_by
            return self._compile_single_tool_frame(
                frame,
                action="slurm.accounting",
                args=args,
                output_alias="semantic_slurm_accounting",
                source_path="grouped" if group_by else "jobs",
                strategy="grouped_pushdown" if group_by else "single_target_pushdown",
                semantic_action=semantic_action,
            )
        if frame.intent == "status":
            return self._compile_single_tool_frame(
                frame,
                action="slurm.metrics",
                args={"metric_group": _slurm_metric_group(frame)},
                output_alias="semantic_slurm_status",
                source_path="payload",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if frame.intent == "list":
            if "slurm.queue" not in self.allowed_tools:
                return None
            args = _slurm_filter_args(frame)
            args["limit"] = None
            return self._compile_single_tool_frame(
                frame,
                action="slurm.queue",
                args=args,
                output_alias="semantic_slurm_list",
                source_path="jobs",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if frame.intent != "count" or "slurm.queue" not in self.allowed_tools:
            return None
        decision = self.strategy_selector.select(frame)
        group_by = decision.group_by or _first_group_dimension(frame, {"partition", "state", "user"})
        args = _slurm_filter_args(frame)
        args["limit"] = None
        source: Any = {"$ref": "semantic_slurm_count", "path": "count"}
        source_path = "count"
        strategy: SemanticStrategy = "single_target_pushdown"
        if group_by:
            args["group_by"] = group_by
            source = {"$ref": "semantic_slurm_count", "path": "grouped"}
            source_path = "grouped"
            strategy = "grouped_pushdown"
        return self._compile_single_tool_frame(
            frame,
            action="slurm.queue",
            args=args,
            output_alias="semantic_slurm_count",
            source_path=source_path,
            source_override=source,
            strategy=strategy,
            semantic_action=semantic_action,
        )

    def _compile_filesystem(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile filesystem list and grouped count frames.

        Inputs:
            Receives a canonical filesystem frame.

        Returns:
            A filesystem plan using fs.glob/fs.aggregate or None.

        Used by:
            SemanticFrameCompiler.compile.
        """
        path = str(_filter_value(frame, "path") or ".")
        action_id = semantic_action.semantic_action_id if semantic_action else ""
        if action_id == "filesystem.list_directory" and "fs.list" in self.allowed_tools:
            return self._compile_single_tool_frame(
                frame,
                action="fs.list",
                args={"path": path},
                output_alias="semantic_fs_directory",
                source_path="entries",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "filesystem.search_content" and "fs.search_content" in self.allowed_tools:
            needle = _filter_value(frame, "needle") or _filter_value(frame, "query") or _filter_value(frame, "term")
            if not needle:
                return None
            return self._compile_single_tool_frame(
                frame,
                action="fs.search_content",
                args={
                    "path": path,
                    "needle": str(needle),
                    "pattern": str(_filter_value(frame, "pattern") or "*"),
                    "recursive": True,
                    "file_only": True,
                    "path_style": "relative",
                },
                output_alias="semantic_fs_search",
                source_path="matches",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "filesystem.read_file" and "fs.read" in self.allowed_tools:
            file_path = _filter_value(frame, "file") or _filter_value(frame, "path")
            if not file_path:
                return None
            return self._compile_single_tool_frame(
                frame,
                action="fs.read",
                args={"path": str(file_path)},
                output_alias="semantic_fs_read",
                source_path="content",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if action_id == "filesystem.size_path" and "fs.aggregate" in self.allowed_tools:
            return self._compile_single_tool_frame(
                frame,
                action="fs.aggregate",
                args={
                    "path": path,
                    "pattern": str(_filter_value(frame, "pattern") or "*"),
                    "recursive": True,
                    "file_only": True,
                    "include_matches": False,
                    "aggregate": "count_and_total_size",
                    "size_unit": "auto",
                },
                output_alias="semantic_fs_size",
                source_path="summary_text",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        extensions = _target_set_values(frame.targets.get("extension"))
        if not extensions and _target_mode(frame, "extension") == "all":
            extensions = []
        if frame.intent == "list" and "fs.glob" in self.allowed_tools:
            pattern = _filesystem_pattern(frame, extensions[:1])
            return self._compile_single_tool_frame(
                frame,
                action="fs.glob",
                args={"path": path, "pattern": pattern, "recursive": True, "file_only": True, "path_style": "relative"},
                output_alias="semantic_fs_list",
                source_path="matches",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        if frame.intent != "count" or "fs.aggregate" not in self.allowed_tools:
            return None
        if len(extensions) > 1:
            return self._compile_filesystem_extension_fanout(frame, path=path, extensions=extensions, semantic_action=semantic_action)
        pattern = _filesystem_pattern(frame, extensions[:1])
        return self._compile_single_tool_frame(
            frame,
            action="fs.aggregate",
            args={
                "path": path,
                "pattern": pattern,
                "recursive": True,
                "file_only": True,
                "include_matches": False,
                "aggregate": "count",
            },
            output_alias="semantic_fs_count",
            source_path="file_count",
            strategy="single_target_pushdown",
            semantic_action=semantic_action,
        )

    def _compile_filesystem_extension_fanout(
        self,
        frame: SemanticFrame,
        *,
        path: str,
        extensions: list[str],
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile one fs.aggregate step per requested extension.

        Inputs:
            Receives a filesystem frame, root path, and extension target set.

        Returns:
            A fan-out plan whose formatter assembles one table row per extension.

        Used by:
            SemanticFrameCompiler._compile_filesystem.
        """
        steps: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for index, extension in enumerate(extensions, start=1):
            alias = f"semantic_fs_{_normalize_token(extension) or index}_count"
            steps.append(
                {
                    "id": index,
                    "action": "fs.aggregate",
                    "args": {
                        "path": path,
                        "pattern": _extension_pattern(extension),
                        "recursive": True,
                        "file_only": True,
                        "include_matches": False,
                        "aggregate": "count",
                    },
                    "output": alias,
                }
            )
            rows.append(
                {
                    "extension": _display_extension(extension),
                    "file_count": {"$ref": alias, "path": "file_count"},
                    "pattern": {"$ref": alias, "path": "pattern"},
                }
            )
        format_step_id = len(steps) + 1
        steps.append(
            {
                "id": format_step_id,
                "action": "text.format",
                "input": [str(step["output"]) for step in steps],
                "args": {"source": rows, "format": "markdown"},
                "output": "semantic_formatted_output",
            }
        )
        steps.append(
            {
                "id": format_step_id + 1,
                "action": "runtime.return",
                "input": ["semantic_formatted_output"],
                "args": {"value": {"$ref": "semantic_formatted_output", "path": "content"}, "mode": "text"},
                "output": "runtime_return_result",
            }
        )
        plan = ExecutionPlan.model_validate({"steps": steps})
        return SemanticCompilationResult(
            plan=plan,
            strategy="fan_out",
            frame=frame,
            metadata={
                "semantic_frame": frame.model_dump(),
                "semantic_action": semantic_action.semantic_action_id if semantic_action else "filesystem.group_count_by_extension",
                "semantic_strategy": "fan_out",
                "semantic_targets": {"extension": extensions},
            },
        )

    def _compile_shell(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile safe shell/system inspection frames.

        Inputs:
            Receives a shell/system semantic frame.

        Returns:
            A shell.exec plan with a deterministic read-only command or None.

        Used by:
            SemanticFrameCompiler.compile.
        """
        if "shell.exec" not in self.allowed_tools:
            return None
        command = _shell_command_for_frame(frame)
        if not command:
            return None
        return self._compile_single_tool_frame(
            frame,
            action="shell.exec",
            args={"command": command},
            output_alias="semantic_shell_output",
            source_path="stdout",
            strategy="single_target_pushdown",
            semantic_action=semantic_action,
        )

    def _compile_sql_metadata(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile SQL metadata and validate/explain frames.

        Inputs:
            Receives a SQL semantic frame.

        Returns:
            A safe SQL metadata plan, or None for executable SQL requests that need the action planner.

        Used by:
            SemanticFrameCompiler.compile.
        """
        if frame.intent == "schema_answer" and "sql.schema" in self.allowed_tools:
            database = _filter_value(frame, "database")
            return self._compile_single_tool_frame(
                frame,
                action="sql.schema",
                args={"database": database} if database else {},
                output_alias="semantic_sql_schema",
                source_path="catalog",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        query_text = str(_filter_value(frame, "query") or "").strip()
        if frame.intent == "validate_or_explain" and query_text and "sql.validate" in self.allowed_tools:
            database = _filter_value(frame, "database")
            args = {"query": query_text}
            if database:
                args["database"] = database
            return self._compile_single_tool_frame(
                frame,
                action="sql.validate",
                args=args,
                output_alias="semantic_sql_validation",
                source_path="explanation",
                strategy="single_target_pushdown",
                semantic_action=semantic_action,
            )
        return None

    def _compile_sql_query(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile narrow schema-grounded SQL data frames.

        Inputs:
            Receives a SQL count or aggregate semantic frame.

        Returns:
            A read-only sql.query plan for supported DICOM templates, otherwise None.

        Used by:
            SemanticFrameCompiler.compile.
        """
        if "sql.query" not in self.allowed_tools:
            return None
        query = _sql_query_for_frame(frame)
        if not query:
            return None
        database = _filter_value(frame, "database") or "dicom"
        return self._compile_single_tool_frame(
            frame,
            action="sql.query",
            args={"database": database, "query": query},
            output_alias="semantic_sql_query",
            source_path="rows",
            strategy="single_target_pushdown",
            semantic_action=semantic_action,
        )

    def _compile_diagnostic(
        self,
        frame: SemanticFrame,
        *,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile broad diagnostics into bounded section metadata.

        Inputs:
            Receives a diagnostic semantic frame.

        Returns:
            A compact deterministic diagnostic-section answer or None.

        Used by:
            SemanticFrameCompiler.compile.
        """
        diagnostic = diagnostic_plan_for_goal(frame.rationale or frame.entity or "diagnostic")
        if diagnostic is None:
            diagnostic = diagnostic_plan_for_goal("diagnostic workspace config sql filesystem shell capabilities")
        if diagnostic is None:
            return None
        rows = [
            {
                "section": section.name,
                "allowed_tools": ", ".join(section.allowed_tools),
                "instruction": section.instruction,
            }
            for section in diagnostic.sections
        ]
        return self._compile_literal_rows_frame(
            frame,
            rows=rows,
            strategy="fan_out",
            metadata={
                "semantic_action": semantic_action.semantic_action_id if semantic_action else "diagnostic.multi_domain_sections",
                "semantic_diagnostic_budget": diagnostic.budget.model_dump(),
            },
        )

    def _compile_compound(self, frame: SemanticFrame) -> SemanticCompilationResult | None:
        """Compile recursive child frames into a sectioned composite plan.

        Inputs:
            Receives a parent frame with child frames.

        Returns:
            A composite plan when every child can be compiled safely.

        Used by:
            SemanticFrameCompiler.compile.
        """
        child_results: list[SemanticCompilationResult] = []
        for child in frame.children:
            compiled = self.compile(child)
            if compiled is None:
                return None
            child_results.append(compiled)
        steps: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        next_id = 1
        for index, child_result in enumerate(child_results, start=1):
            alias_map: dict[str, str] = {}
            for step in child_result.plan.steps:
                copied = step.model_dump()
                old_output = str(copied.get("output") or "")
                if old_output:
                    alias_map[old_output] = f"child_{index}_{old_output}"
                    copied["output"] = alias_map[old_output]
                copied["id"] = next_id
                copied["input"] = [alias_map.get(str(item), str(item)) for item in copied.get("input") or []]
                copied["args"] = _rewrite_refs(copied.get("args") or {}, alias_map)
                steps.append(copied)
                next_id += 1
            final_alias = alias_map.get("runtime_return_result")
            rows.append(
                {
                    "section": child_result.frame.rationale or child_result.frame.intent,
                    "result": {"$ref": final_alias, "path": "output"} if final_alias else "",
                }
            )
        steps.append(
            {
                "id": next_id,
                "action": "text.format",
                "input": [row["result"]["$ref"] for row in rows if isinstance(row.get("result"), dict)],
                "args": {"source": rows, "format": "markdown"},
                "output": "semantic_composite_output",
            }
        )
        steps.append(
            {
                "id": next_id + 1,
                "action": "runtime.return",
                "input": ["semantic_composite_output"],
                "args": {"value": {"$ref": "semantic_composite_output", "path": "content"}, "mode": "text"},
                "output": "runtime_return_result",
            }
        )
        return SemanticCompilationResult(
            plan=ExecutionPlan.model_validate({"steps": steps}),
            strategy="fan_out",
            frame=frame,
            metadata={"semantic_frame": frame.model_dump(), "semantic_strategy": "recursive_composite"},
        )

    def _compile_single_tool_frame(
        self,
        frame: SemanticFrame,
        *,
        action: str,
        args: dict[str, Any],
        output_alias: str,
        source_path: str,
        strategy: SemanticStrategy,
        source_override: Any | None = None,
        semantic_action: SemanticActionRow | None = None,
    ) -> SemanticCompilationResult | None:
        """Compile a one-tool, format, return execution plan.

        Inputs:
            Receives tool identity, arguments, output alias, and formatting source.

        Returns:
            A semantic compilation result when the tool is allowed.

        Used by:
            Domain-specific semantic compilers.
        """
        if action not in self.allowed_tools:
            return None
        source = source_override if source_override is not None else {"$ref": output_alias, "path": source_path}
        plan = ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": action, "args": {key: value for key, value in args.items() if value is not None}, "output": output_alias},
                    {
                        "id": 2,
                        "action": "text.format",
                        "input": [output_alias],
                        "args": {"source": source, "format": "markdown"},
                        "output": "semantic_formatted_output",
                    },
                    {
                        "id": 3,
                        "action": "runtime.return",
                        "input": ["semantic_formatted_output"],
                        "args": {"value": {"$ref": "semantic_formatted_output", "path": "content"}, "mode": "text"},
                        "output": "runtime_return_result",
                    },
                ]
            }
        )
        return SemanticCompilationResult(
            plan=plan,
            strategy=strategy,
            frame=frame,
            metadata={
                "semantic_frame": frame.model_dump(),
                "semantic_action": semantic_action.semantic_action_id if semantic_action else None,
                "semantic_strategy": strategy,
                "semantic_targets": _frame_targets(frame),
            },
        )

    def _compile_literal_rows_frame(
        self,
        frame: SemanticFrame,
        *,
        rows: list[dict[str, Any]],
        strategy: SemanticStrategy,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticCompilationResult:
        """Compile a deterministic literal row result through formatter/return.

        Inputs:
            Receives display rows and semantic metadata.

        Returns:
            A compilation result without data-domain tool execution.

        Used by:
            Diagnostic and schema-limitation semantic compilers.
        """
        plan = ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "text.format",
                        "args": {"source": rows, "format": "markdown"},
                        "output": "semantic_formatted_output",
                    },
                    {
                        "id": 2,
                        "action": "runtime.return",
                        "input": ["semantic_formatted_output"],
                        "args": {"value": {"$ref": "semantic_formatted_output", "path": "content"}, "mode": "text"},
                        "output": "runtime_return_result",
                    },
                ]
            }
        )
        combined_metadata = {"semantic_frame": frame.model_dump(), "semantic_strategy": strategy, "semantic_targets": _frame_targets(frame)}
        combined_metadata.update(metadata or {})
        return SemanticCompilationResult(plan=plan, strategy=strategy, frame=frame, metadata=combined_metadata)


class SemanticCoverageValidator:
    """Validate that a candidate ExecutionPlan covers a semantic frame.

    Inputs:
        Constructed without external state.

    Returns:
        Coverage results with compact repair facts.

    Used by:
        SemanticFramePlanner and fallback action-planner validation.
    """

    def validate(self, frame: SemanticFrame, plan: ExecutionPlan) -> SemanticCoverageResult:
        """Validate metric, target, and domain coverage for supported frames.

        Inputs:
            Receives a semantic frame and candidate execution plan.

        Returns:
            SemanticCoverageResult describing missing coverage.

        Used by:
            SemanticFramePlanner.
        """
        decision = select_semantic_action_decision(frame)
        if decision.row is not None:
            expected_tools = set(decision.row.tools)
            actual_tools = {step.action for step in plan.steps}
            if not actual_tools.intersection(expected_tools):
                return SemanticCoverageResult(
                    False,
                    errors=[
                        (
                            f"Semantic action {decision.row.semantic_action_id} requires one of "
                            f"{sorted(expected_tools)}, got {sorted(actual_tools)}."
                        )
                    ],
                    metadata={
                        "semantic_action": decision.row.semantic_action_id,
                        "expected_actions": sorted(expected_tools),
                        "actual_actions": sorted(actual_tools),
                        "expected_output": decision.row.output.__dict__,
                    },
                )
            if frame.domain == "slurm" and frame.intent == "aggregate_metric":
                aggregate = self._validate_slurm_aggregate(frame, plan)
                if not aggregate.covered:
                    return aggregate
            return SemanticCoverageResult(
                True,
                metadata={
                    "coverage": "semantic_action_matrix",
                    "semantic_action": decision.row.semantic_action_id,
                    "expected_actions": sorted(expected_tools),
                    "actual_actions": sorted(actual_tools),
                    "expected_output": decision.row.output.__dict__,
                },
            )
        if frame.domain == "slurm" and frame.intent == "aggregate_metric":
            return self._validate_slurm_aggregate(frame, plan)
        if frame.domain == "slurm" and frame.intent == "count":
            return self._validate_action_present(frame, plan, {"slurm.queue", "slurm.nodes", "slurm.partitions"})
        if frame.domain == "filesystem" and frame.intent in {"count", "list"}:
            return self._validate_action_present(frame, plan, {"fs.aggregate", "fs.glob", "fs.find", "fs.list"})
        if frame.domain == "shell" and frame.intent in {"count", "list", "status"}:
            return self._validate_action_present(frame, plan, {"shell.exec"})
        if frame.domain == "sql" and frame.intent in {"schema_answer", "validate_or_explain"}:
            return self._validate_action_present(frame, plan, {"sql.schema", "sql.validate"})
        if frame.domain == "sql" and frame.intent in {"count", "aggregate_metric"}:
            return self._validate_action_present(frame, plan, {"sql.query"})
        if frame.domain == "diagnostic" and frame.intent == "diagnostic":
            return SemanticCoverageResult(True, metadata={"coverage": "diagnostic_sections"})
        return SemanticCoverageResult(True, metadata={"coverage": "not_required_for_frame"})

    def _validate_action_present(
        self,
        frame: SemanticFrame,
        plan: ExecutionPlan,
        allowed_actions: set[str],
    ) -> SemanticCoverageResult:
        """Validate that a frame is covered by at least one expected action.

        Inputs:
            Receives a frame, plan, and acceptable action names.

        Returns:
            Coverage result for generic semantic frames.

        Used by:
            SemanticCoverageValidator.validate.
        """
        actions = {step.action for step in plan.steps}
        if actions & allowed_actions:
            return SemanticCoverageResult(True, metadata={"coverage": f"{frame.domain}_{frame.intent}"})
        return SemanticCoverageResult(
            False,
            errors=[f"Semantic frame requires one of {sorted(allowed_actions)}, got {sorted(actions)}."],
            metadata={"expected_actions": sorted(allowed_actions), "actual_actions": sorted(actions)},
        )

    def _validate_slurm_aggregate(self, frame: SemanticFrame, plan: ExecutionPlan) -> SemanticCoverageResult:
        """Validate SLURM aggregate coverage.

        Inputs:
            Receives a SLURM aggregate frame and candidate plan.

        Returns:
            Coverage result that rejects queue/listing plans for runtime metrics.

        Used by:
            SemanticCoverageValidator.validate.
        """
        aggregate_steps = [step for step in plan.steps if step.action == "slurm.accounting_aggregate"]
        if not aggregate_steps:
            return SemanticCoverageResult(
                False,
                errors=["Semantic frame requires slurm.accounting_aggregate for aggregate_metric."],
                metadata={"missing_metric": str(frame.metric.name if frame.metric else "average_elapsed")},
            )
        metric = str(frame.metric.name if frame.metric else "average_elapsed")
        for step in aggregate_steps:
            if str(step.args.get("metric") or "average_elapsed") != metric:
                return SemanticCoverageResult(
                    False,
                    errors=[f"Semantic frame expected metric {metric}, got {step.args.get('metric')}."],
                    metadata={"expected_metric": metric, "actual_metric": step.args.get("metric")},
                )
        targets = _target_set_values(frame.targets.get("partition"))
        if len(targets) > 1:
            grouped = any(str(step.args.get("group_by") or "") == "partition" for step in aggregate_steps)
            fanout = {str(step.args.get("partition") or "") for step in aggregate_steps}
            if not grouped and not set(targets).issubset(fanout):
                return SemanticCoverageResult(
                    False,
                    errors=["Semantic frame target set is not covered by group_by=partition or per-partition fan-out."],
                    metadata={"expected_targets": targets, "actual_partitions": sorted(value for value in fanout if value)},
                )
        return SemanticCoverageResult(True, metadata={"coverage": "slurm_aggregate"})


class SemanticFramePlanner:
    """Coordinate semantic-frame extraction, compilation, and coverage validation.

    Inputs:
        Constructed with settings, LLM, and allowed tools.

    Returns:
        SemanticCompilationResult for supported semantic frames.

    Used by:
        TaskPlanner before the regular LLM action planner.
    """

    def __init__(self, *, settings: Settings, llm: LLMClient, allowed_tools: list[str]) -> None:
        """Initialize the semantic-frame planner.

        Inputs:
            Receives runtime settings, LLM client, and allowed tool names.

        Returns:
            Initializes the planner instance.

        Used by:
            TaskPlanner.
        """
        self.settings = settings
        self.llm = llm
        self.allowed_tools = allowed_tools
        self.last_extraction: SemanticFrameExtractionResult | None = None

    def try_build_plan(self, *, goal: str, model: str | None = None, temperature: float = 0.0) -> SemanticCompilationResult | None:
        """Try to build a plan from a semantic frame.

        Inputs:
            Receives the user goal and optional LLM settings.

        Returns:
            A semantic compilation result, or None when action planning should handle the request.

        Used by:
            TaskPlanner.build_plan.
        """
        mode = semantic_frame_mode(self.settings)
        if mode == "off":
            return None
        extraction = LLMSemanticFrameExtractor(llm=self.llm, settings=self.settings).extract(
            goal=goal,
            allowed_tools=self.allowed_tools,
            model=model,
            temperature=temperature,
        )
        self.last_extraction = extraction
        if not extraction.matched or extraction.frame is None:
            if mode == "enforce":
                raise ValueError(f"Semantic frame extraction failed: {extraction.reason or 'invalid semantic frame'}")
            return None
        compiled = SemanticFrameCompiler(settings=self.settings, allowed_tools=self.allowed_tools).compile(extraction.frame)
        if compiled is None:
            if extraction.frame.allow_action_fallback:
                return None
            if mode == "enforce":
                raise ValueError("Semantic frame could not be compiled by typed lowerers.")
            return None
        coverage = SemanticCoverageValidator().validate(extraction.frame, compiled.plan)
        if not coverage.covered:
            if mode == "enforce":
                raise ValueError("; ".join(coverage.errors))
            return None
        compiled.metadata.update(
            {
                "semantic_frame_mode": mode,
                "semantic_frame_source": extraction.frame.source,
                "semantic_frame_extraction": extraction.metadata,
                "semantic_coverage": coverage.metadata,
                "semantic_llm_calls": extraction.llm_calls,
                "llm_stage_calls": extraction.metadata.get("llm_stage_calls", {"semantic_frame": extraction.llm_calls}),
                "llm_recursion_depth": extraction.metadata.get("llm_recursion_depth", 0),
                "semantic_frame_tree": extraction.frame.model_dump(),
            }
        )
        return compiled


SEMANTIC_FRAME_SYSTEM_PROMPT = """Return JSON only. Extract the user's semantic request into the provided SemanticFrame schema.
Do not emit tools, shell commands, SQL text, code, execution plans, or raw data. Describe intent only."""

SEMANTIC_FRAME_REPAIR_SYSTEM_PROMPT = """Return JSON only. Repair the provided semantic-frame JSON so it matches the SemanticFrame schema.
Do not add tools, shell commands, SQL text, code, execution plans, or raw data. Preserve the user's intent."""


def build_semantic_frame_prompt(
    *,
    goal: str,
    settings: Settings,
    allowed_tools: list[str],
    recursion_state: LLMStageRecursionState | None = None,
    parent_frame: SemanticFrame | None = None,
) -> dict[str, Any]:
    """Build a safe prompt for semantic-frame extraction.

    Inputs:
        Receives the user goal, settings, allowed tools, and optional recursive stage metadata.

    Returns:
        A compact prompt dictionary with schema and capability facts only.

    Used by:
        LLMSemanticFrameExtractor and privacy tests.
    """
    schema = _semantic_frame_allowed_schema()
    return {
        "goal": goal,
        "runtime_date": runtime_date_context(settings),
        "llm_stage": (recursion_state.prompt_metadata() if recursion_state is not None else None),
        "parent_frame": (parent_frame.model_dump() if parent_frame is not None else None),
        "allowed_domains": ["slurm", "sql", "filesystem", "shell", "text", "diagnostic"],
        "allowed_intents": [
            "count",
            "list",
            "aggregate_metric",
            "status",
            "schema_answer",
            "validate_or_explain",
            "diagnostic",
            "transform",
        ],
        "capability_metadata": {
            domain: {
                "intents": list(spec.intents),
                "metrics": list(spec.metrics),
                "filters": list(spec.filters),
                "dimensions": list(spec.dimensions),
                "group_by": list(spec.group_by),
                "multi_target_pushdown": spec.multi_target_pushdown,
                "fan_out": spec.fan_out,
                "local_projection": spec.local_projection,
            }
            for domain, spec in SEMANTIC_CAPABILITIES.items()
        },
        "semantic_ontology": ontology_prompt_metadata(),
        "semantic_action_matrix": semantic_action_prompt_metadata(),
        "allowed_dimensions": ["partition", "state", "user", "node", "job_name", "modality", "extension", "path", "type", "section"],
        "allowed_output_kinds": ["scalar", "table", "file", "text", "json", "status"],
        "allowed_output_cardinalities": ["single", "multi_scalar", "grouped", "collection", "sectioned"],
        "allowed_render_styles": ["scalar", "metric_table", "record_table", "key_value", "bullets", "sectioned"],
        "output_intent_examples": [
            {
                "prompt": "count patients in dicom",
                "output": {"kind": "scalar", "cardinality": "single", "render_style": "scalar", "result_entities": ["patients"]},
            },
            {
                "prompt": "count patients, studies, series, RTPLANS in dicom",
                "output": {
                    "kind": "table",
                    "cardinality": "multi_scalar",
                    "render_style": "metric_table",
                    "result_entities": ["patients", "studies", "series", "rtplan"],
                },
            },
            {
                "prompt": "count jobs by partition",
                "output": {"kind": "table", "cardinality": "grouped", "render_style": "metric_table", "group_by": ["partition"]},
            },
            {
                "prompt": "list all patients",
                "output": {"kind": "table", "cardinality": "collection", "render_style": "record_table", "result_entities": ["patients"]},
            },
        ],
        "allowed_state_policy_filters": ["state", "include_all_states"],
        "allowed_metrics": [
            "average_elapsed",
            "min_elapsed",
            "max_elapsed",
            "sum_elapsed",
            "count",
            "runtime_summary",
            "total_size",
            "count_and_total_size",
        ],
        "schema": {
            "frame_id": "string|null",
            "parent_id": "string|null",
            "depth": "integer >= 0",
            "composition": "single|sequence|parallel_safe|compare|dashboard|diagnostic_sections",
            "domain": "slurm|sql|filesystem|shell|text|diagnostic",
            "intent": "count|list|aggregate_metric|status|schema_answer|validate_or_explain|diagnostic|transform",
            "entity": "string|null",
            "metric": {"name": "string", "unit": "string|null"},
            "dimensions": ["string"],
            "targets": {"dimension": ["values"]},
            "filters": [
                {
                    "field": "string",
                    "operator": "eq|neq|in|gt|gte|lt|lte|contains",
                    "value": "scalar|null, or a list only when operator is in",
                }
            ],
            "state_policy": {
                "state": "string|null; use for explicit completed/running/pending/failed wording",
                "include_all_states": "boolean; true for all jobs, any state, or regardless of state",
                "default_state_applied": "boolean; normally false in LLM output",
            },
            "time_window": {"phrase": "string|null", "start": "string|null", "end": "string|null", "label": "string|null"},
            "output": {
                "kind": "scalar|table|file|text|json|status",
                "format": "string|null",
                "cardinality": "single|multi_scalar|grouped|collection|sectioned",
                "render_style": "scalar|metric_table|record_table|key_value|bullets|sectioned",
                "result_entities": ["string"],
                "group_by": ["string"],
                "reason": "short explanation",
            },
            "children": [],
            "allow_action_fallback": "boolean, default false; true only when typed semantic lowering is not enough and action-plan fallback is acceptable",
            "confidence": "0..1",
        },
        "allowed_schema": schema,
        "recursive_instructions": [
            "Return one parent frame with children for true compound requests.",
            "Return one frame with target sets for same-operation multi-target requests.",
            "If expanding a parent_frame, return complete child frames only; every leaf requires output kind, cardinality, and render_style.",
            "Use operator='in' for list-valued filters, or represent repeated values as targets.",
            "Use state_policy.include_all_states=true for all-state SLURM accounting intent.",
            "Never include executable tool names, SQL text, shell commands, raw rows, file contents, stdout/stderr, or operational payloads.",
        ],
        "privacy_contract": "Return intent only. Do not include tools, executable commands, SQL text, operational payloads, command output streams, file contents, SLURM job details, or sensitive values.",
    }


def build_semantic_frame_repair_prompt(
    *,
    goal: str,
    invalid_payload: Any,
    validation_facts: list[dict[str, Any]],
    settings: Settings,
    allowed_tools: list[str],
    recursion_state: LLMStageRecursionState,
    parent_frame: SemanticFrame | None = None,
) -> dict[str, Any]:
    """Build a safe prompt for repairing invalid semantic-frame JSON.

    Inputs:
        Receives user text, sanitized invalid JSON, compact validation facts, and schema metadata.

    Returns:
        A JSON-compatible repair prompt containing no operational payloads.

    Used by:
        LLMSemanticFrameExtractor._repair_invalid_payload.
    """
    base = build_semantic_frame_prompt(
        goal=goal,
        settings=settings,
        allowed_tools=allowed_tools,
        recursion_state=recursion_state,
        parent_frame=parent_frame,
    )
    return {
        "task": "repair_semantic_frame_json",
        "goal": goal,
        "runtime_date": base["runtime_date"],
        "llm_stage": base["llm_stage"],
        "validation_facts": validation_facts,
        "invalid_payload": _sanitize_semantic_payload_for_repair(invalid_payload),
        "allowed_schema": base["allowed_schema"],
        "schema": base["schema"],
        "repair_rules": [
            "Return a complete SemanticFrame JSON object only.",
            "Replace unknown fields with legal typed fields when the meaning is clear.",
            "Use state_policy for SLURM accounting state semantics; never emit slurm_state_policy.",
            "Use [] instead of null for list fields such as filters, children, dimensions, output.group_by, and output.result_entities.",
            "Use operator='in' for list-valued filters, or move repeated values into targets.",
            "Do not add tools, SQL, shell commands, code, raw rows, file contents, stdout/stderr, SLURM jobs, or PHI.",
        ],
        "privacy_contract": base["privacy_contract"],
    }


def _semantic_frame_allowed_schema() -> dict[str, Any]:
    """Return compact schema metadata for recursive semantic-frame prompts.

    Inputs:
        Uses module constants only.

    Returns:
        A JSON-compatible schema summary safe for LLM prompts.

    Used by:
        LLMSemanticFrameExtractor and build_semantic_frame_prompt.
    """
    return {
        "required_leaf_fields": ["domain", "intent", "output.kind", "output.cardinality", "output.render_style"],
        "composition": ["single", "sequence", "parallel_safe", "compare", "dashboard", "diagnostic_sections"],
        "output": {
            "kind": ["scalar", "table", "file", "text", "json", "status"],
            "cardinality": ["single", "multi_scalar", "grouped", "collection", "sectioned"],
            "render_style": ["scalar", "metric_table", "record_table", "key_value", "bullets", "sectioned"],
        },
        "filters": {
            "operators": ["eq", "neq", "in", "gt", "gte", "lt", "lte", "contains"],
            "list_values": "Allowed only with operator='in'.",
        },
        "state_policy": {
            "state": "string|null",
            "include_all_states": "boolean",
            "default_state_applied": "boolean",
        },
        "privacy_forbidden": sorted(DISALLOWED_FRAME_KEYS),
    }


def deterministic_semantic_frame(goal: str, settings: Settings) -> SemanticFrameExtractionResult:
    """Extract a semantic frame with deterministic backup rules.

    Inputs:
        Receives the user goal and runtime settings.

    Returns:
        A matched semantic frame for known safe patterns, otherwise a non-match.

    Used by:
        SemanticFramePlanner before optional LLM extraction.
    """
    text = str(goal or "")
    frame: SemanticFrame | None = None
    compound = _deterministic_compound_frame(text, settings=settings, depth=0)
    if compound is not None:
        frame = compound
    else:
        frame = _deterministic_single_frame(text)
    if frame is not None:
        canonical = SemanticFrameCanonicalizer(settings).canonicalize(frame, goal=text)
        return SemanticFrameExtractionResult(True, frame=canonical, metadata={"source": "deterministic"})
    return SemanticFrameExtractionResult(False, reason="no_deterministic_semantic_frame")


def _deterministic_single_frame(text: str) -> SemanticFrame | None:
    """Extract one non-recursive deterministic semantic frame.

    Inputs:
        Receives one prompt clause.

    Returns:
        A frame for supported domains, or None.

    Used by:
        deterministic_semantic_frame and recursive compound extraction.
    """
    if _looks_like_slurm_aggregate(text):
        return _deterministic_slurm_aggregate_frame(text)
    if _looks_like_slurm_count(text):
        return _deterministic_slurm_count_frame(text)
    if _looks_like_slurm_status_or_list(text):
        return _deterministic_slurm_inspection_frame(text)
    if _looks_like_filesystem_frame(text):
        return _deterministic_filesystem_frame(text)
    if _looks_like_shell_frame(text):
        return _deterministic_shell_frame(text)
    if _looks_like_sql_metadata_frame(text):
        return _deterministic_sql_metadata_frame(text)
    if _looks_like_sql_frame(text):
        return _deterministic_sql_frame(text)
    if diagnostic_plan_for_goal(text) is not None:
        return _deterministic_diagnostic_frame(text)
    return None


def _deterministic_slurm_aggregate_frame(text: str) -> SemanticFrame:
    """Build a deterministic SLURM aggregate semantic frame.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A semantic frame for SLURM job runtime aggregation.

    Used by:
        deterministic_semantic_frame.
    """
    metric = normalize_metric_name(text)
    targets = _extract_partition_targets(text)
    target_mapping: dict[str, SemanticTargetSet] = {}
    group_dimension = _extract_group_dimension(text, {"partition", "state", "user", "job_name"})
    if targets:
        target_mapping["partition"] = SemanticTargetSet(dimension="partition", values=targets, mode="explicit")
    elif group_dimension:
        target_mapping[group_dimension] = SemanticTargetSet(dimension=group_dimension, values=[], mode="all")
    filters = _extract_common_filters(text)
    return SemanticFrame(
        domain="slurm",
        intent="aggregate_metric",
        entity="jobs",
        metric=SemanticMetric(name=metric, unit="seconds"),
        dimensions=[group_dimension] if group_dimension else ["partition"] if target_mapping else [],
        targets=target_mapping,
        filters=filters,
        time_window=SemanticTimeWindow(phrase=_extract_time_phrase(text)),
        output=SemanticOutputContract(kind="table" if targets or group_dimension else "scalar"),
        confidence=0.95,
        source="deterministic",
        rationale="Detected SLURM runtime aggregate request.",
    )


def _deterministic_compound_frame(text: str, *, settings: Settings, depth: int) -> SemanticFrame | None:
    """Build a recursive parent frame from independently supported clauses.

    Inputs:
        Receives prompt text, settings, and current recursion depth.

    Returns:
        Parent frame with child frames, or None when the prompt is not a true compound.

    Used by:
        deterministic_semantic_frame.
    """
    max_depth = int(getattr(settings, "semantic_frame_max_depth", 10) or 10)
    if depth >= max_depth or not COMPOUND_SPLIT_RE.search(text):
        return None
    parts = [part.strip(" .") for part in COMPOUND_SPLIT_RE.split(text) if part.strip(" .")]
    if len(parts) < 2:
        return None
    children: list[SemanticFrame] = []
    for index, part in enumerate(parts[: int(getattr(settings, "semantic_frame_max_children", 8) or 8)]):
        child = _deterministic_compound_frame(part, settings=settings, depth=depth + 1) or _deterministic_single_frame(part)
        if child is None:
            return None
        child.depth = depth + 1
        child.parent_id = "root" if depth == 0 else f"compound_{depth}"
        child.frame_id = f"{child.parent_id}.{index + 1}"
        children.append(child)
    if len(children) < 2:
        return None
    return SemanticFrame(
        frame_id="root" if depth == 0 else f"compound_{depth}",
        depth=depth,
        domain="diagnostic",
        intent="diagnostic",
        composition="sequence",
        children=children,
        output=SemanticOutputContract(kind="text"),
        confidence=min(child.confidence for child in children),
        source="deterministic",
        rationale="Detected compound request with independently supported child frames.",
    )


def _deterministic_slurm_count_frame(text: str) -> SemanticFrame:
    """Build a deterministic SLURM queue count frame.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A semantic frame for SLURM queue counting.

    Used by:
        deterministic_semantic_frame.
    """
    targets = _extract_partition_targets(text)
    group_dimension = _extract_group_dimension(text, {"partition", "state", "user"})
    target_mapping: dict[str, SemanticTargetSet] = {}
    if targets:
        target_mapping["partition"] = SemanticTargetSet(dimension="partition", values=targets, mode="explicit")
    elif group_dimension:
        target_mapping[group_dimension] = SemanticTargetSet(dimension=group_dimension, values=[], mode="all")
    return SemanticFrame(
        domain="slurm",
        intent="count",
        entity="jobs",
        metric=SemanticMetric(name="count"),
        dimensions=[group_dimension] if group_dimension else [],
        targets=target_mapping,
        filters=_extract_common_filters(text),
        output=SemanticOutputContract(kind="table" if group_dimension or len(targets) > 1 else "scalar"),
        confidence=0.92,
        source="deterministic",
        rationale="Detected SLURM queue count request.",
    )


def _deterministic_slurm_inspection_frame(text: str) -> SemanticFrame:
    """Build a deterministic SLURM status/list frame.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A semantic frame for SLURM status or queue listing.

    Used by:
        deterministic_semantic_frame.
    """
    intent: SemanticIntent = "status" if re.search(r"\b(?:status|health|summary|overview)\b", text, re.IGNORECASE) else "list"
    return SemanticFrame(
        domain="slurm",
        intent=intent,
        entity="cluster" if intent == "status" else "jobs",
        filters=_extract_common_filters(text),
        output=SemanticOutputContract(kind="status" if intent == "status" else "table"),
        confidence=0.85,
        source="deterministic",
        rationale="Detected SLURM status/list request.",
    )


def _deterministic_filesystem_frame(text: str) -> SemanticFrame:
    """Build a deterministic filesystem semantic frame.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A semantic frame for filesystem list/count tasks.

    Used by:
        deterministic_semantic_frame.
    """
    extensions = _extract_extension_targets(text)
    target_mapping: dict[str, SemanticTargetSet] = {}
    if extensions:
        target_mapping["extension"] = SemanticTargetSet(dimension="extension", values=extensions, mode="explicit")
    path_filter = SemanticFilter(field="path", value=_extract_path_target(text) or ".")
    intent: SemanticIntent = "count" if COUNT_GOAL_RE.search(text) else "list"
    return SemanticFrame(
        domain="filesystem",
        intent=intent,
        entity="files",
        metric=SemanticMetric(name="count") if intent == "count" else None,
        dimensions=["extension"] if len(extensions) > 1 else [],
        targets=target_mapping,
        filters=[path_filter],
        output=SemanticOutputContract(kind="table" if len(extensions) > 1 or intent == "list" else "scalar"),
        confidence=0.85,
        source="deterministic",
        rationale="Detected filesystem list/count request.",
    )


def _deterministic_shell_frame(text: str) -> SemanticFrame:
    """Build a deterministic shell/system inspection frame.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A semantic frame for read-only system inspection.

    Used by:
        deterministic_semantic_frame.
    """
    intent: SemanticIntent = "count" if COUNT_GOAL_RE.search(text) else "list" if re.search(r"\b(?:list|show)\b", text, re.IGNORECASE) else "status"
    return SemanticFrame(
        domain="shell",
        intent=intent,
        entity=_detect_shell_entity(text),
        metric=SemanticMetric(name="count") if intent == "count" else None,
        output=SemanticOutputContract(kind="scalar" if intent == "count" else "table"),
        confidence=0.82,
        source="deterministic",
        rationale="Detected shell/system inspection request.",
    )


def _deterministic_sql_metadata_frame(text: str) -> SemanticFrame:
    """Build a deterministic SQL metadata or validation frame.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A SQL semantic frame that does not execute user data queries.

    Used by:
        deterministic_semantic_frame.
    """
    database = _extract_database_target(text)
    filters = [SemanticFilter(field="database", value=database)] if database else []
    if SQL_VALIDATE_RE.search(text):
        query = _extract_sql_text(text) or _generated_sql_for_goal(text)
        if query:
            filters.append(SemanticFilter(field="query", value=query))
            return SemanticFrame(
                domain="sql",
                intent="validate_or_explain",
                entity="query",
                filters=filters,
                output=SemanticOutputContract(kind="text"),
                confidence=0.75,
                source="deterministic",
                rationale="Detected SQL validate/explain request.",
            )
    return SemanticFrame(
        domain="sql",
        intent="schema_answer",
        entity="schema",
        filters=filters,
        output=SemanticOutputContract(kind="table"),
        confidence=0.78,
        source="deterministic",
        rationale="Detected SQL schema-answer request.",
    )


def _deterministic_sql_frame(text: str) -> SemanticFrame:
    """Build a SQL semantic frame for fallback action planning.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A frame describing SQL intent without generating executable SQL.

    Used by:
        deterministic_semantic_frame.
    """
    intent: SemanticIntent = "aggregate_metric" if re.search(r"\b(?:average|avg|min|max)\b", text, re.IGNORECASE) else "count"
    database = _extract_database_target(text)
    filters = [SemanticFilter(field="database", value=database)] if database else []
    count_entities = _extract_dicom_count_entities(text)
    if len(count_entities) > 1:
        return SemanticFrame(
            domain="sql",
            intent="count",
            entity="dicom_counts",
            metric=SemanticMetric(name="count"),
            dimensions=[],
            targets={"entity": SemanticTargetSet(dimension="entity", values=count_entities, mode="explicit")},
            filters=filters,
            output=SemanticOutputContract(
                kind="table",
                cardinality="multi_scalar",
                render_style="metric_table",
                result_entities=count_entities,
                source="deterministic_backup",
                reason="Multiple independent DICOM counts requested.",
            ),
            confidence=0.82,
            source="deterministic",
            rationale=text,
        )
    modalities = _extract_modality_targets(text)
    concept_terms = _extract_dicom_concept_terms(text)
    filters.extend(SemanticFilter(field="concept_term", value=term) for term in concept_terms)
    age_filter = _extract_age_filter(text)
    if age_filter is not None:
        operator, age = age_filter
        filters.append(SemanticFilter(field="age", operator=operator, value=age))
    if concept_terms and _all_terms_policy(text):
        filters.append(SemanticFilter(field="match_policy", value="all_terms"))
    targets = {"modality": SemanticTargetSet(dimension="modality", values=modalities, mode="explicit")} if modalities else {}
    grouped = _is_grouped_sql_count_goal(text)
    return SemanticFrame(
        domain="sql",
        intent=intent,
        entity="dicom",
        metric=SemanticMetric(name="count" if intent == "count" else "average"),
        dimensions=["modality"] if grouped and modalities else [],
        targets=targets,
        filters=filters,
        output=SemanticOutputContract(
            kind="table" if grouped or intent == "aggregate_metric" else "scalar",
            cardinality="grouped" if grouped else "single",
            render_style="metric_table" if grouped else "scalar",
            result_entities=["patients"] if "patient" in text.lower() or "patieint" in text.lower() else [],
            group_by=["modality"] if grouped and modalities else [],
            source="deterministic_backup",
        ),
        confidence=0.7,
        source="deterministic",
        rationale=text,
    )


def _deterministic_diagnostic_frame(text: str) -> SemanticFrame:
    """Build a deterministic diagnostic parent frame with bounded child sections.

    Inputs:
        Receives the original user prompt text.

    Returns:
        A diagnostic semantic frame.

    Used by:
        deterministic_semantic_frame.
    """
    return SemanticFrame(
        domain="diagnostic",
        intent="diagnostic",
        entity="runtime",
        composition="diagnostic_sections",
        output=SemanticOutputContract(kind="text"),
        confidence=0.9,
        source="deterministic",
        rationale=text,
    )


def semantic_frame_mode(settings: Settings) -> Literal["off", "shadow", "enforce"]:
    """Read and normalize the semantic-frame mode setting.

    Inputs:
        Receives runtime settings.

    Returns:
        One of off, shadow, or enforce.

    Used by:
        SemanticFramePlanner and tests.
    """
    mode = str(getattr(settings, "semantic_frame_mode", "enforce") or "enforce").strip().lower()
    if mode not in {"off", "shadow", "enforce"}:
        return "enforce"
    return mode  # type: ignore[return-value]


def validate_semantic_coverage(frame: SemanticFrame, plan: ExecutionPlan) -> SemanticCoverageResult:
    """Validate a plan against a semantic frame.

    Inputs:
        Receives a semantic frame and candidate execution plan.

    Returns:
        Coverage validation result.

    Used by:
        TaskPlanner and focused tests.
    """
    return SemanticCoverageValidator().validate(frame, plan)


def project_semantic_result(
    action: str,
    args: dict[str, Any],
    result: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project structured tool results according to semantic-frame metadata.

    Inputs:
        Receives the executed action, sanitized step arguments, optional metadata, and tool result.

    Returns:
        A projected result with requested target groups retained.

    Used by:
        PlanExecutor immediately after tool invocation.
    """
    projection = None
    if isinstance(metadata, dict):
        projection = metadata.get("semantic_projection")
    if projection is None and isinstance(args, dict):
        projection = args.get("__semantic_projection")
    if not isinstance(projection, dict) or not isinstance(result, dict):
        return result
    if action == "slurm.accounting_aggregate" and str(projection.get("field") or "") == "partition":
        return _project_slurm_accounting_groups(result, _target_values(projection.get("values")))
    return result


def normalize_metric_name(value: Any) -> str:
    """Normalize natural metric text to a runtime metric identifier.

    Inputs:
        Receives free-form metric text or an existing identifier.

    Returns:
        A canonical metric name.

    Used by:
        SemanticMetric validation and deterministic extraction.
    """
    text = str(value or "").strip().lower().replace("-", "_")
    if re.search(r"\b(?:min|minimum|shortest)\b", text):
        return "min_elapsed"
    if re.search(r"\b(?:max|maximum|longest)\b", text):
        return "max_elapsed"
    if re.search(r"\b(?:sum|total)\b", text):
        return "sum_elapsed"
    if COUNT_GOAL_RE.search(text):
        return "count"
    return "average_elapsed"


def _looks_like_slurm_count(goal: str) -> bool:
    """Detect SLURM queue count requests.

    Inputs:
        Receives the user goal.

    Returns:
        True when the goal asks for a count of SLURM jobs.

    Used by:
        deterministic_semantic_frame.
    """
    return bool(SLURM_DOMAIN_RE.search(goal) and COUNT_GOAL_RE.search(goal) and re.search(r"\bjobs?\b", goal, re.IGNORECASE))


def _looks_like_slurm_status_or_list(goal: str) -> bool:
    """Detect SLURM status or list requests.

    Inputs:
        Receives the user goal.

    Returns:
        True when the prompt should inspect SLURM without aggregation.

    Used by:
        deterministic_semantic_frame.
    """
    return bool(SLURM_DOMAIN_RE.search(goal) and re.search(r"\b(?:status|health|summary|overview|list|show|queue|nodes?|partitions?)\b", goal, re.IGNORECASE))


def _looks_like_filesystem_frame(goal: str) -> bool:
    """Detect filesystem prompts suitable for semantic compilation.

    Inputs:
        Receives the user goal.

    Returns:
        True for list/count file prompts.

    Used by:
        deterministic_semantic_frame.
    """
    return bool(FILESYSTEM_DOMAIN_RE.search(goal) and re.search(r"\b(?:count|list|show|find)\b", goal, re.IGNORECASE))


def _looks_like_shell_frame(goal: str) -> bool:
    """Detect safe system-inspection prompts.

    Inputs:
        Receives the user goal.

    Returns:
        True for supported shell/system inspection categories.

    Used by:
        deterministic_semantic_frame.
    """
    return bool(SHELL_DOMAIN_RE.search(goal) and re.search(r"\b(?:count|how many|list|show|status|summary)\b", goal, re.IGNORECASE))


def _looks_like_sql_metadata_frame(goal: str) -> bool:
    """Detect SQL schema or validation prompts that do not need query execution.

    Inputs:
        Receives the user goal.

    Returns:
        True for schema-answer or SQL validate/explain requests.

    Used by:
        deterministic_semantic_frame.
    """
    text = str(goal or "")
    return bool(
        SQL_VALIDATE_RE.search(text)
        or re.search(r"\b(?:inspect|schema|data dictionary|join path|columns?|tables?)\b", text, re.IGNORECASE)
        and re.search(r"\b(?:sql|database|dicom)\b", text, re.IGNORECASE)
    )


def _looks_like_sql_frame(goal: str) -> bool:
    """Detect SQL data requests that should carry semantic coverage metadata.

    Inputs:
        Receives the user goal.

    Returns:
        True for DICOM/SQL count or aggregate requests.

    Used by:
        deterministic_semantic_frame.
    """
    text = str(goal or "")
    return bool(
        re.search(r"\b(?:sql|database|dicom|patient|patients|study|studies|series|modality|modalities)\b", text, re.IGNORECASE)
        and (COUNT_GOAL_RE.search(text) or re.search(r"\b(?:average|avg|min|max|top)\b", text, re.IGNORECASE))
    )


def _looks_like_slurm_aggregate(goal: str) -> bool:
    """Detect SLURM aggregate requests that semantic-frame v1 can compile.

    Inputs:
        Receives the user goal.

    Returns:
        True when the goal appears to ask for a SLURM job runtime aggregate.

    Used by:
        deterministic_semantic_frame and LLM attempt gating.
    """
    return bool(SLURM_DOMAIN_RE.search(goal) and SLURM_AGGREGATE_RE.search(goal))


def _should_attempt_llm_frame(goal: str) -> bool:
    """Decide whether an LLM semantic-frame attempt is worthwhile.

    Inputs:
        Receives the user goal.

    Returns:
        True for compound/aggregate patterns that benefit from semantic frames.

    Used by:
        SemanticFramePlanner.
    """
    text = str(goal or "")
    if re.search(r"\b(?:using|use|with)\s+(?:shell|sql|python|filesystem|fs)(?:\.[a-z_]+)?\b", text, re.IGNORECASE):
        return False
    if not (
        SLURM_DOMAIN_RE.search(text)
        or FILESYSTEM_DOMAIN_RE.search(text)
        or SHELL_DOMAIN_RE.search(text)
        or DIAGNOSTIC_DOMAIN_RE.search(text)
        or re.search(r"\b(?:sql|database|dicom)\b", text, re.IGNORECASE)
    ):
        return False
    return bool(COMPOUND_SPLIT_RE.search(text) or EXPLICIT_MULTI_TARGET_RE.search(text) or _looks_like_slurm_aggregate(text))


def _detect_domain(goal: str) -> SemanticDomain:
    """Infer a broad semantic domain from the user goal.

    Inputs:
        Receives the user goal.

    Returns:
        The inferred semantic domain.

    Used by:
        SemanticFrameCanonicalizer.
    """
    text = str(goal or "")
    if SLURM_DOMAIN_RE.search(text):
        return "slurm"
    if re.search(r"\b(?:sql|database|table|dicom|patient|study|series)\b", text, re.IGNORECASE):
        return "sql"
    if re.search(r"\b(?:file|folder|directory|csv|json|md|markdown)\b", text, re.IGNORECASE):
        return "filesystem"
    if re.search(r"\b(?:process|port|disk|memory|cpu|hostname|uptime)\b", text, re.IGNORECASE):
        return "shell"
    if DIAGNOSTIC_DOMAIN_RE.search(text):
        return "diagnostic"
    return "unknown"


def _detect_intent(goal: str) -> SemanticIntent:
    """Infer a semantic intent from the user goal.

    Inputs:
        Receives the user goal.

    Returns:
        The inferred semantic intent.

    Used by:
        SemanticFrameCanonicalizer.
    """
    if _looks_like_slurm_aggregate(goal):
        return "aggregate_metric"
    if COUNT_GOAL_RE.search(str(goal or "")):
        return "count"
    if re.search(r"\b(?:status|health|summary|overview)\b", str(goal or ""), re.IGNORECASE):
        return "status"
    if re.search(r"\b(?:list|show|find)\b", str(goal or ""), re.IGNORECASE):
        return "list"
    return "unknown"


def _normalize_entity(entity: str | None, goal: str, *, domain: str = "unknown") -> str | None:
    """Normalize the semantic entity name.

    Inputs:
        Receives an optional entity, user goal, and semantic domain.

    Returns:
        A canonical entity string where one is obvious.

    Used by:
        SemanticFrameCanonicalizer.
    """
    if entity:
        canonical = normalize_semantic_entity(domain, entity)
        return canonical or _normalize_token(entity)
    if re.search(r"\bjobs?\b", goal, re.IGNORECASE):
        return "jobs"
    if re.search(r"\bfiles?\b", goal, re.IGNORECASE):
        return "files"
    if re.search(r"\bprocess(?:es)?\b", goal, re.IGNORECASE):
        return "processes"
    return None


def _canonicalize_frame_ontology(frame: SemanticFrame, *, goal: str) -> SemanticFrame:
    """Apply shared ontology normalization and explicit prompt grounding.

    Inputs:
        Receives a semantic frame and original user goal.

    Returns:
        The frame with canonical entities, dimensions, filters, targets, and output metadata.

    Used by:
        SemanticFrameCanonicalizer.canonicalize before matrix selection and lowering.
    """
    domain = frame.domain
    grounded_entities = [match.canonical for match in detect_prompt_entities(domain, goal)] if frame.intent in {"count", "list", "status", "aggregate_metric"} else []
    primary_entities = _primary_grounded_entities(frame, grounded_entities)
    if grounded_entities:
        if len(primary_entities) == 1:
            frame.entity = primary_entities[0]
        elif frame.intent == "count" and _frame_requests_multi_entity_output(frame):
            frame.entity = "dicom_counts" if domain == "sql" else frame.entity
            existing_targets = _target_set_values(frame.targets.get("entity"))
            values = [*existing_targets, *[value for value in grounded_entities if value not in existing_targets]]
            frame.targets["entity"] = SemanticTargetSet(dimension="entity", values=values, mode="explicit")
    frame.entity = _normalize_entity(frame.entity, "", domain=domain)
    frame.dimensions = list(dict.fromkeys(normalize_semantic_dimension(value) for value in frame.dimensions if normalize_semantic_dimension(value)))
    frame.targets = _canonicalize_target_mapping(frame.targets)
    frame.filters = [
        SemanticFilter(field=normalize_semantic_filter_field(filter_.field), operator=filter_.operator, value=filter_.value)
        for filter_ in frame.filters
    ]
    if domain == "filesystem":
        frame = _canonicalize_filesystem_prompt_grounding(frame, goal=goal)
    if domain == "sql":
        frame = _canonicalize_sql_prompt_grounding(frame, goal=goal)
    frame.output.result_entities = _canonical_output_entities(domain, frame.output.result_entities, primary_entities)
    frame.output.group_by = list(dict.fromkeys(normalize_semantic_dimension(value) for value in frame.output.group_by if normalize_semantic_dimension(value)))
    if frame.output.cardinality in {"single", "unknown"} and len(frame.output.result_entities) > 1:
        frame.output.cardinality = "multi_scalar"
        if frame.output.kind == "scalar":
            frame.output.kind = "table"
        if frame.output.render_style in {"scalar", "unknown"}:
            frame.output.render_style = "metric_table"
        if "ontology_multiple_entities_force_multi_scalar" not in frame.output.corrections:
            frame.output.corrections.append("ontology_multiple_entities_force_multi_scalar")
    return frame


def _canonicalize_sql_prompt_grounding(frame: SemanticFrame, *, goal: str) -> SemanticFrame:
    """Fill missing SQL/DICOM semantic facts from explicit schema and ranking wording.

    Inputs:
        Receives a SQL semantic frame and the original prompt.

    Returns:
        Frame with schema-answer entities and top-N related-count filters normalized.

    Used by:
        _canonicalize_frame_ontology before semantic action matrix selection.
    """
    text = str(goal or "").lower()
    filters = list(frame.filters or [])
    fields = {filter_.field for filter_ in filters}
    if "database" not in fields and "dicom" in text:
        filters.append(SemanticFilter(field="database", value="dicom"))
        fields.add("database")
    age_filter = _extract_age_filter(goal)
    if age_filter and not fields.intersection({"age", "patient_age", "patient_age_years"}):
        operator, age = age_filter
        filters.append(SemanticFilter(field="age", operator=operator, value=age))
        fields.add("age")
    modalities = _extract_modality_targets(goal)
    if modalities and ("modality" not in frame.targets or not _target_set_values(frame.targets.get("modality"))):
        frame.targets["modality"] = SemanticTargetSet(dimension="modality", values=modalities, mode="explicit")
    if re.search(r"\bage\s+distribution\b|\bgrouped\s+into\s+decades\b|\bby\s+decade\b", text):
        frame.entity = frame.entity or "patients"
        if "decade" not in frame.dimensions:
            frame.dimensions.append("decade")
        if "decade" not in frame.output.group_by:
            frame.output.group_by.append("decade")
        if frame.intent == "aggregate_metric":
            frame.intent = "count"
        frame.output.kind = "table"
        frame.output.cardinality = "grouped"
        frame.output.render_style = "metric_table"
    if frame.intent == "list" and frame.entity in {"studies", "study"} and modalities:
        if re.search(r"\b(?:both|all\s+of)\b", text) and "match_policy" not in fields:
            filters.append(SemanticFilter(field="match_policy", value="all_targets"))
            fields.add("match_policy")
        if re.search(r"\b(?:but\s+no|without|with\s+no|no)\s+(?:rtplan|rtdose|rtstruct|ct|mr|pt)\b", text):
            excluded = _extract_excluded_modality_targets(goal)
            included = [value for value in modalities if value not in excluded]
            if included and "include_modality" not in fields:
                filters.append(SemanticFilter(field="include_modality", operator="in", value=included))
                fields.add("include_modality")
            if excluded and "exclude_modality" not in fields:
                filters.append(SemanticFilter(field="exclude_modality", operator="in", value=excluded))
                fields.add("exclude_modality")
    if frame.intent == "schema_answer":
        if re.search(r"\b(?:relationship|relationships|join|joins|foreign\s+keys?|link|links)\b", text):
            frame.entity = "relationships"
        elif re.search(r"\b(?:tables?|schema)\b", text):
            frame.entity = frame.entity or "tables"
        elif re.search(r"\b(?:columns?|mrn|patient\s*id|study\s*uid|series\s*uid|modality|uid)\b", text):
            frame.entity = frame.entity or "schema"
        else:
            frame.entity = frame.entity or "schema"
    if (
        frame.intent in {"list", "count"}
        and re.search(r"\b(?:top|most|highest|largest|greatest)\b", text)
    ):
        top_entity = _extract_top_result_entity(goal)
        related_entity = _extract_related_count_entity(goal)
        if top_entity:
            frame.entity = top_entity
        if frame.intent == "count" and related_entity:
            frame.intent = "list"
        if "related_count" not in fields:
            filters.append(SemanticFilter(field="related_count", value=related_entity or "studies"))
            fields.add("related_count")
        if "limit" not in fields:
            filters.append(SemanticFilter(field="limit", value=_extract_limit_value(goal) or 10))
        if frame.output.kind in {"scalar", "unknown"}:
            frame.output.kind = "table"
        if frame.output.cardinality in {"single", "unknown"}:
            frame.output.cardinality = "collection"
        if frame.output.render_style in {"scalar", "unknown"}:
            frame.output.render_style = "record_table"
    frame.filters = filters
    return frame


def _canonicalize_filesystem_prompt_grounding(frame: SemanticFrame, *, goal: str) -> SemanticFrame:
    """Fill missing filesystem frame facts from explicit prompt references.

    Inputs:
        Receives a filesystem semantic frame and original user prompt.

    Returns:
        Frame with file/path/search filters needed by filesystem lowerers.

    Used by:
        _canonicalize_frame_ontology for LLM frames that omit safe filesystem filter fields.
    """
    filters = list(frame.filters or [])
    existing_fields = {filter_.field for filter_ in filters}
    file_ref = _extract_file_reference(goal)
    search_needle = _extract_search_needle(goal)
    if file_ref and "file" not in existing_fields and _looks_like_read_file_goal(goal):
        filters.append(SemanticFilter(field="file", value=file_ref))
        frame.entity = "file_content"
    if search_needle and not existing_fields.intersection({"needle", "query", "term"}):
        filters.append(SemanticFilter(field="needle", value=search_needle))
        frame.entity = frame.entity or "content"
    if "path" not in {filter_.field for filter_ in filters}:
        path = _extract_path_target(goal)
        if path or _looks_like_repo_scope(goal) or search_needle:
            filters.append(SemanticFilter(field="path", value=path or "."))
    frame.filters = filters
    if not frame.entity:
        if search_needle:
            frame.entity = "content"
        elif file_ref:
            frame.entity = "file_content" if _looks_like_read_file_goal(goal) else "files"
    return frame


def _looks_like_read_file_goal(goal: str) -> bool:
    """Return whether the prompt asks to read or summarize a specific file.

    Inputs:
        Receives the user prompt.

    Returns:
        True for read/show/summarize requests with an explicit file reference.

    Used by:
        _canonicalize_filesystem_prompt_grounding.
    """
    return bool(re.search(r"\b(?:read|summarize|show|open|inspect)\b", str(goal or ""), re.IGNORECASE))


def _looks_like_repo_scope(goal: str) -> bool:
    """Return whether the prompt explicitly scopes work to the repository.

    Inputs:
        Receives the user prompt.

    Returns:
        True when the current repo/folder is the intended filesystem root.

    Used by:
        _canonicalize_filesystem_prompt_grounding.
    """
    return bool(re.search(r"\b(?:repo|repository|current folder|current directory|this folder|this directory)\b", str(goal or ""), re.IGNORECASE))


def _extract_file_reference(goal: str) -> str | None:
    """Extract a literal file path from a filesystem prompt.

    Inputs:
        Receives the user prompt.

    Returns:
        File-like path or None when absent.

    Used by:
        _canonicalize_filesystem_prompt_grounding.
    """
    match = re.search(r"\b([~./A-Za-z0-9_-]+\.(?:md|markdown|txt|py|yaml|yml|json|csv|toml|ini|cfg))\b", str(goal or ""), re.IGNORECASE)
    return match.group(1) if match else None


def _extract_search_needle(goal: str) -> str | None:
    """Extract a safe content-search needle from an explicit search prompt.

    Inputs:
        Receives the user prompt.

    Returns:
        Search needle text or None when absent.

    Used by:
        _canonicalize_filesystem_prompt_grounding.
    """
    text = str(goal or "")
    patterns = [
        r"\breferences?\s+to\s+([A-Za-z0-9_.:-]+)",
        r"\bsearch\b.+?\bfor\s+([A-Za-z0-9_.:-]+)",
        r"\bcontaining\s+['\"]?([^'\"\s,;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_limit_value(goal: str) -> int | None:
    """Extract an explicit top/limit count from a user prompt.

    Inputs:
        Receives the user prompt.

    Returns:
        Positive integer limit or None when no bounded count is stated.

    Used by:
        SQL top-N semantic grounding.
    """
    text = str(goal or "")
    patterns = [
        r"\btop\s+(?P<limit>\d{1,3})\b",
        r"\bfirst\s+(?P<limit>\d{1,3})\b",
        r"\blimit\s+(?P<limit>\d{1,3})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = int(match.group("limit"))
            return value if value > 0 else None
    return None


def _extract_top_result_entity(goal: str) -> str | None:
    """Extract the row entity for a top-N DICOM prompt.

    Inputs:
        Receives the user prompt.

    Returns:
        Canonical DICOM result entity such as patients, studies, series, or modalities.

    Used by:
        SQL prompt grounding for generic top-N related-count frames.
    """
    text = str(goal or "").lower()
    if re.search(r"\bmodalit(?:y|ies)\b", text):
        return "modalities"
    if re.search(r"\bper\s+patient\b|\bpatients?\b", text):
        return "patients"
    if re.search(r"\bper\s+study\b|\bstudies\b", text):
        return "studies"
    if re.search(r"\bper\s+series\b|\bseries\b", text):
        return "series"
    return None


def _extract_related_count_entity(goal: str) -> str | None:
    """Extract the related entity being counted in a top-N DICOM prompt.

    Inputs:
        Receives the user prompt.

    Returns:
        Related count entity such as studies, series, instances, or modality_records.

    Used by:
        SQL prompt grounding and top-N SQL builders.
    """
    text = str(goal or "").lower()
    if re.search(r"\brtplans?\b|\brtdoses?\b|\brtstructs?\b", text):
        return "modality_records"
    if re.search(r"\b(?:most|number\s+of|count\s+of|counts?\s+of)\s+instances?\b|\binstances?\s+per\s+series\b", text):
        return "instances"
    if re.search(r"\b(?:most|number\s+of|count\s+of|counts?\s+of)\s+series\b|\bseries\s+per\s+study\b", text):
        return "series"
    if re.search(r"\b(?:most|number\s+of|count\s+of|counts?\s+of)\s+stud(?:y|ies)\b|\bstud(?:y|ies)\s+per\s+patient\b", text):
        return "studies"
    if re.search(r"\bseries\b", text):
        return "series"
    if re.search(r"\binstances?\b|\bimages?\b", text):
        return "instances"
    return None


def _primary_grounded_entities(frame: SemanticFrame, grounded_entities: list[str]) -> list[str]:
    """Choose prompt-grounded entities that describe the requested result unit.

    Inputs:
        Receives the semantic frame and ordered ontology matches from the prompt.

    Returns:
        One primary entity for scalar/grouped requests, or all entities for explicit multi-scalar output.

    Used by:
        _canonicalize_frame_ontology.
    """
    if not grounded_entities:
        return []
    if _frame_requests_multi_entity_output(frame):
        return grounded_entities
    existing = normalize_semantic_entity(frame.domain, frame.entity)
    if existing and existing in grounded_entities:
        return [existing]
    output_entities = [normalize_semantic_entity(frame.domain, value) for value in frame.output.result_entities]
    for entity in output_entities:
        if entity in grounded_entities:
            return [entity]
    return grounded_entities[:1]


def _frame_requests_multi_entity_output(frame: SemanticFrame) -> bool:
    """Return whether the frame explicitly asks for several independent entity metrics.

    Inputs:
        Receives a semantic frame.

    Returns:
        True for multi-scalar output or an explicit entity target set with multiple values.

    Used by:
        _canonicalize_frame_ontology to avoid mistaking related nouns for output entities.
    """
    entity_targets = _target_set_values(frame.targets.get("entity"))
    return frame.output.cardinality == "multi_scalar" or len(entity_targets) > 1


def _canonicalize_target_mapping(targets: dict[str, SemanticTargetSet]) -> dict[str, SemanticTargetSet]:
    """Normalize target mapping keys and dimension names.

    Inputs:
        Receives target sets from a semantic frame.

    Returns:
        Target sets keyed by canonical dimensions.

    Used by:
        _canonicalize_frame_ontology.
    """
    canonical: dict[str, SemanticTargetSet] = {}
    for key, target in (targets or {}).items():
        dimension = normalize_semantic_dimension(key)
        if not dimension:
            continue
        values = _target_set_values(target)
        canonical[dimension] = SemanticTargetSet(dimension=dimension, values=values, mode=target.mode)
    return canonical


def _canonical_output_entities(domain: str, result_entities: list[str], grounded_entities: list[str]) -> list[str]:
    """Normalize output result entities and merge explicit prompt entities.

    Inputs:
        Receives semantic domain, LLM-proposed output entities, and prompt-grounded entities.

    Returns:
        Ordered canonical result entities.

    Used by:
        _canonicalize_frame_ontology.
    """
    values: list[str] = []
    for entity in [*list(result_entities or []), *grounded_entities]:
        canonical = normalize_semantic_entity(domain, entity)
        if canonical and canonical not in values:
            values.append(canonical)
    return values


def _canonical_targets_for_domain(frame: SemanticFrame, goal: str) -> dict[str, SemanticTargetSet]:
    """Add deterministic target backups for all supported domains.

    Inputs:
        Receives a semantic frame and user goal.

    Returns:
        Target mapping with domain-specific backups applied.

    Used by:
        SemanticFrameCanonicalizer.
    """
    targets = dict(frame.targets or {})
    if frame.domain == "filesystem":
        extensions = _extract_extension_targets(goal)
        if extensions and ("extension" not in targets or not _target_set_values(targets["extension"])):
            targets["extension"] = SemanticTargetSet(dimension="extension", values=extensions, mode="explicit")
    if frame.domain == "sql":
        modalities = _extract_modality_targets(goal)
        if modalities and ("modality" not in targets or not _target_set_values(targets["modality"])):
            targets["modality"] = SemanticTargetSet(dimension="modality", values=modalities, mode="explicit")
    return targets


def _canonical_composition(frame: SemanticFrame, goal: str) -> SemanticComposition:
    """Infer composition for parent and compound frames.

    Inputs:
        Receives a semantic frame and original goal.

    Returns:
        Canonical composition value.

    Used by:
        SemanticFrameCanonicalizer.
    """
    if frame.children and frame.composition == "single":
        return "sequence" if COMPOUND_SPLIT_RE.search(goal) else "dashboard"
    if frame.domain == "diagnostic":
        return "diagnostic_sections"
    return frame.composition


def _prepare_child_frame(child: SemanticFrame, *, parent: SemanticFrame, index: int) -> SemanticFrame:
    """Attach recursive metadata to a child frame.

    Inputs:
        Receives a child frame, parent frame, and child index.

    Returns:
        A child frame with parent/depth/frame identifiers filled.

    Used by:
        SemanticFrameCanonicalizer.
    """
    child_data = child.model_dump()
    child_data["parent_id"] = parent.frame_id or child.parent_id or "root"
    child_data["depth"] = parent.depth + 1
    child_data["frame_id"] = child.frame_id or f"{child_data['parent_id']}.{index + 1}"
    return SemanticFrame.model_validate(child_data)


def _validate_frame_runtime_limits(frame: SemanticFrame, settings: Settings) -> None:
    """Enforce configured recursion depth and child-count limits.

    Inputs:
        Receives a canonical frame and runtime settings.

    Returns:
        Raises ValueError if limits are exceeded.

    Used by:
        SemanticFrameCanonicalizer.
    """
    max_depth = int(getattr(settings, "semantic_frame_max_depth", 3) or 3)
    max_children = int(getattr(settings, "semantic_frame_max_children", 8) or 8)
    if frame.depth > max_depth:
        raise ValueError(f"Semantic frame depth {frame.depth} exceeds configured maximum {max_depth}.")
    if len(frame.children) > max_children:
        raise ValueError(f"Semantic frame has {len(frame.children)} children; maximum is {max_children}.")


def _canonical_metric(metric: SemanticMetric | None, goal: str) -> SemanticMetric | None:
    """Return a canonical metric, filling from the goal if needed.

    Inputs:
        Receives an optional metric and user goal.

    Returns:
        A canonical SemanticMetric or None.

    Used by:
        SemanticFrameCanonicalizer.
    """
    if metric is not None:
        return SemanticMetric(name=metric.name, unit=metric.unit)
    if _looks_like_slurm_aggregate(goal):
        return SemanticMetric(name=normalize_metric_name(goal), unit="seconds")
    return None


def _extract_common_filters(goal: str) -> list[SemanticFilter]:
    """Extract common SLURM/user-state filters from text.

    Inputs:
        Receives the user goal.

    Returns:
        List of semantic filters.

    Used by:
        Deterministic frame extraction.
    """
    filters: list[SemanticFilter] = []
    if _slurm_all_states_phrase(goal):
        filters.append(SemanticFilter(field="include_all_states", value=True))
    elif state := _extract_state_filter(goal):
        filters.append(SemanticFilter(field="state", value=state))
    user = _extract_user_filter(goal)
    if user:
        filters.append(SemanticFilter(field="user", value=user))
    partition_targets = _extract_partition_targets(goal)
    if len(partition_targets) == 1 and not EXPLICIT_MULTI_TARGET_RE.search(goal):
        filters.append(SemanticFilter(field="partition", value=partition_targets[0]))
    return filters


def _canonical_filters_for_domain(frame: SemanticFrame, goal: str) -> list[SemanticFilter]:
    """Canonicalize domain-specific semantic filters after domain detection.

    Inputs:
        Receives a frame and original user goal.

    Returns:
        Filters with deterministic domain policy repairs applied.

    Used by:
        SemanticFrameCanonicalizer.
    """
    if frame.domain == "slurm" and frame.intent == "aggregate_metric":
        return _canonical_slurm_accounting_filters(frame.filters, goal)
    return frame.filters


def _promote_list_filters_to_targets(
    targets: dict[str, SemanticTargetSet],
    filters: list[SemanticFilter],
    domain: str,
) -> tuple[dict[str, SemanticTargetSet], list[SemanticFilter]]:
    """Promote typed list filters into domain target sets when supported.

    Inputs:
        Receives target mapping, filters, and semantic domain.

    Returns:
        Updated targets plus remaining filters.

    Used by:
        SemanticFrameCanonicalizer to keep multi-target meaning in one typed place.
    """
    spec = SEMANTIC_CAPABILITIES.get(domain)
    target_dimensions = set(spec.dimensions if spec else ())
    promoted_targets = dict(targets or {})
    kept_filters: list[SemanticFilter] = []
    for filter_ in filters:
        if filter_.operator == "in" and isinstance(filter_.value, list) and filter_.field in target_dimensions:
            values = _target_values(filter_.value)
            if values:
                existing = _target_set_values(promoted_targets.get(filter_.field))
                promoted_targets[filter_.field] = SemanticTargetSet(
                    dimension=filter_.field,
                    values=[*existing, *[value for value in values if value not in existing]],
                    mode="explicit",
                )
                continue
        kept_filters.append(filter_)
    return promoted_targets, kept_filters


def _canonical_slurm_accounting_filters(filters: list[SemanticFilter], goal: str) -> list[SemanticFilter]:
    """Canonicalize SLURM accounting state filters.

    Inputs:
        Receives semantic filters and the original goal.

    Returns:
        Filters where all-state intent overrides completed-only defaults.

    Used by:
        _canonical_filters_for_domain and semantic-frame tests.
    """
    all_states_requested = _slurm_all_states_phrase(goal) or _filter_bool(filters, "include_all_states")
    if not all_states_requested:
        return filters
    kept = [item for item in filters if item.field not in {"state", "include_all_states"}]
    kept.append(SemanticFilter(field="include_all_states", value=True))
    return kept


def _slurm_all_states_phrase(goal: str) -> str | None:
    """Return the all-state phrase if a SLURM accounting goal requests all jobs.

    Inputs:
        Receives the original user goal.

    Returns:
        The matched phrase or None.

    Used by:
        Deterministic semantic extraction and SLURM state-policy canonicalization.
    """
    match = SLURM_ALL_STATES_RE.search(str(goal or ""))
    return match.group(0) if match else None


def _extract_group_dimension(goal: str, allowed: set[str]) -> str | None:
    """Extract requested grouping dimension from text.

    Inputs:
        Receives the user goal and allowed dimension names.

    Returns:
        Group dimension or None.

    Used by:
        Deterministic frame extraction and SLURM compiler.
    """
    text = str(goal or "").lower()
    aliases = {
        "partition": ("partition", "partitions"),
        "state": ("state", "states", "status"),
        "user": ("user", "users"),
        "job_name": ("job name", "job_name", "job"),
        "extension": ("extension", "extensions", "type", "types"),
    }
    for dimension, words in aliases.items():
        if dimension not in allowed:
            continue
        for word in words:
            if re.search(rf"\b(?:by|per|each|for each|in each)\s+{re.escape(word)}s?\b", text):
                return dimension
            if re.search(rf"\b{re.escape(word)}s?\s+(?:separately|breakdown|break down)\b", text):
                return dimension
    return None


def _first_group_dimension(frame: SemanticFrame, allowed: set[str]) -> str | None:
    """Return the first frame dimension that can be used for grouping.

    Inputs:
        Receives a frame and allowed dimension names.

    Returns:
        Matching dimension or None.

    Used by:
        SemanticFrameCompiler.
    """
    for dimension in frame.dimensions:
        if dimension in allowed:
            return dimension
    for dimension, target in frame.targets.items():
        if dimension in allowed and (target.mode == "all" or len(_target_set_values(target)) > 1):
            return dimension
    return None


def _extract_state_filter(goal: str) -> str | None:
    """Extract a SLURM job state filter from text.

    Inputs:
        Receives the user goal.

    Returns:
        Canonical SLURM state or None.

    Used by:
        _extract_common_filters.
    """
    text = str(goal or "").lower()
    if re.search(r"\bcompleted|complete|successful|success\b", text):
        return "COMPLETED"
    if re.search(r"\bfailed|failure|errored\b", text):
        return "FAILED"
    if re.search(r"\brunning\b", text):
        return "RUNNING"
    if re.search(r"\bpending|queued\b", text):
        return "PENDING"
    return None


def _extract_user_filter(goal: str) -> str | None:
    """Extract a user filter from simple phrases.

    Inputs:
        Receives the user goal.

    Returns:
        User value or None.

    Used by:
        _extract_common_filters.
    """
    match = re.search(r"\b(?:for|by|user)\s+user\s+([A-Za-z0-9._-]+)\b|\buser\s*=\s*([A-Za-z0-9._-]+)\b", goal, re.IGNORECASE)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _canonical_time_window(time_window: SemanticTimeWindow | None, goal: str, settings: Settings) -> SemanticTimeWindow | None:
    """Normalize a semantic time window using runtime-local time.

    Inputs:
        Receives an optional time window, user goal, and settings.

    Returns:
        A normalized time window when one is present.

    Used by:
        SemanticFrameCanonicalizer and compiler.
    """
    candidate = time_window or SemanticTimeWindow(phrase=_extract_time_phrase(goal))
    phrase = candidate.phrase or _extract_time_phrase(goal)
    now = current_local_datetime(settings)
    resolved: TemporalRange | None = None
    if phrase:
        resolved = parse_temporal_range(phrase, now=now)
    if resolved is None and (candidate.start or candidate.end):
        resolved = TemporalRange(start=candidate.start, end=candidate.end, time_window_label=candidate.label)
    if resolved is None:
        return candidate if any([candidate.phrase, candidate.start, candidate.end, candidate.label]) else None
    return SemanticTimeWindow(
        phrase=phrase or resolved.original_text,
        start=resolved.start,
        end=resolved.end,
        label=resolved.time_window_label or candidate.label,
    )


def _canonical_output(frame: SemanticFrame, *, allow_goal_backup: bool = True) -> SemanticOutputContract:
    """Resolve the LLM-informed output contract implied by a semantic frame.

    Inputs:
        Receives a semantic frame and whether prompt-derived backup rules are allowed.

    Returns:
        A canonical output contract.

    Used by:
        SemanticFrameCanonicalizer.
    """
    resolved = resolve_output_intent(
        frame.rationale or "",
        semantic_output=frame.output,
        planner_expected_shape=frame.output.kind,
        output_format=frame.output.format,
        allow_goal_backup=allow_goal_backup,
    )
    kind = resolved.kind
    cardinality = resolved.cardinality
    render_style = resolved.render_style
    corrections = list(resolved.corrections)
    if (_has_multi_targets(frame) or frame.dimensions) and kind == "scalar":
        kind = "table"
        corrections.append("semantic_dimensions_force_table")
    if frame.dimensions and cardinality not in {"grouped", "sectioned"}:
        cardinality = "grouped"
        corrections.append("semantic_dimensions_force_grouped")
    if _has_multi_targets(frame) and cardinality == "single":
        cardinality = "multi_scalar"
        corrections.append("semantic_targets_force_multi_scalar")
    if frame.intent in {"count", "aggregate_metric"} and kind == "unknown":
        kind = "scalar"
        cardinality = "single"
        render_style = "scalar"
    if frame.intent == "status" and kind == "unknown":
        kind = "status"
        cardinality = "single"
        render_style = "key_value"
    if render_style == "unknown":
        if cardinality in {"multi_scalar", "grouped"}:
            render_style = "metric_table"
        elif kind == "table":
            render_style = "record_table"
        elif kind == "scalar":
            render_style = "scalar"
        elif kind == "status":
            render_style = "key_value"
    group_by = list(resolved.group_by or tuple(frame.dimensions))
    if frame.dimensions:
        group_by = list(dict.fromkeys([*group_by, *frame.dimensions]))
    return SemanticOutputContract(
        kind=kind,
        format=resolved.format,
        cardinality=cardinality,
        render_style=render_style,
        result_entities=list(resolved.result_entities),
        group_by=group_by,
        source=resolved.source,
        corrections=corrections,
        reason=resolved.reason or frame.output.reason,
    )


def _has_multi_targets(frame: SemanticFrame) -> bool:
    """Check whether a frame has an explicit multi-target set.

    Inputs:
        Receives a semantic frame.

    Returns:
        True when any target dimension has multiple values or all mode.

    Used by:
        SemanticFrameCanonicalizer and compiler.
    """
    return any(target.mode == "all" or len(_target_set_values(target)) > 1 for target in frame.targets.values())


def _merge_target_backups(targets: dict[str, SemanticTargetSet], goal: str) -> dict[str, SemanticTargetSet]:
    """Merge deterministic target extraction into LLM-provided targets.

    Inputs:
        Receives existing targets and user goal.

    Returns:
        Target mapping with deterministic partition backups applied.

    Used by:
        SemanticFrameCanonicalizer.
    """
    merged = dict(targets or {})
    extracted = _extract_partition_targets(goal)
    if extracted and ("partition" not in merged or not _target_set_values(merged["partition"])):
        merged["partition"] = SemanticTargetSet(dimension="partition", values=extracted, mode="explicit")
    return merged


def _frame_targets(frame: SemanticFrame) -> dict[str, list[str]]:
    """Serialize semantic target sets for metadata.

    Inputs:
        Receives a semantic frame.

    Returns:
        Mapping of target dimension to requested values.

    Used by:
        Semantic compilation metadata.
    """
    return {dimension: _target_set_values(target) for dimension, target in frame.targets.items() if _target_set_values(target)}


def _slurm_filter_args(frame: SemanticFrame) -> dict[str, Any]:
    """Convert semantic filters/targets to SLURM tool args.

    Inputs:
        Receives a canonical SLURM frame.

    Returns:
        Safe SLURM filter arguments.

    Used by:
        SemanticFrameCompiler.
    """
    args: dict[str, Any] = {}
    for field in ("partition", "state", "user"):
        value = _filter_value(frame, field)
        if value:
            args[field] = value
    partition_targets = _target_set_values(frame.targets.get("partition"))
    if len(partition_targets) == 1 and "partition" not in args:
        args["partition"] = partition_targets[0]
    return args


def _slurm_partition_args(frame: SemanticFrame) -> dict[str, Any]:
    """Convert partition-target semantic facts to slurm.partitions args.

    Inputs:
        Receives a canonical SLURM frame.

    Returns:
        Tool arguments accepted by slurm.partitions.

    Used by:
        SemanticFrameCompiler._compile_slurm_inspection for the semantic action matrix.
    """
    partition = _filter_value(frame, "partition")
    partition_targets = _target_set_values(frame.targets.get("partition"))
    if partition_targets:
        partition = partition_targets[0]
    return {"partition": partition} if partition else {}


def _slurm_node_args(frame: SemanticFrame) -> dict[str, Any]:
    """Convert node-target semantic facts to slurm.nodes args.

    Inputs:
        Receives a canonical SLURM frame.

    Returns:
        Tool arguments accepted by slurm.nodes.

    Used by:
        SemanticFrameCompiler._compile_slurm_inspection for node list/unique-node actions.
    """
    args: dict[str, Any] = {}
    for field in ("node", "partition", "state", "state_group"):
        value = _filter_value(frame, field)
        if value:
            args[field] = value
    partition_targets = _target_set_values(frame.targets.get("partition"))
    if len(partition_targets) == 1 and "partition" not in args:
        args["partition"] = partition_targets[0]
    return args


def _slurm_metric_group(frame: SemanticFrame) -> str:
    """Choose SLURM metric group for status frames.

    Inputs:
        Receives a SLURM semantic frame.

    Returns:
        Metric group accepted by slurm.metrics.

    Used by:
        SemanticFrameCompiler.
    """
    entity = str(frame.entity or "").lower()
    if "queue" in entity or "job" in entity:
        return "queue_summary"
    if "node" in entity:
        return "node_summary"
    if "partition" in entity:
        return "partition_summary"
    return "cluster_summary"


def _filesystem_pattern(frame: SemanticFrame, extensions: list[str]) -> str:
    """Choose a filesystem glob pattern from semantic targets.

    Inputs:
        Receives a filesystem frame and optional extension targets.

    Returns:
        Glob pattern for fs.glob/fs.aggregate.

    Used by:
        SemanticFrameCompiler.
    """
    pattern = _filter_value(frame, "pattern")
    if pattern:
        return str(pattern)
    if extensions:
        return _extension_pattern(extensions[0])
    return "*"


def _extension_pattern(extension: str) -> str:
    """Convert an extension target to a glob pattern.

    Inputs:
        Receives extension text such as csv or .json.

    Returns:
        A glob pattern.

    Used by:
        Filesystem semantic compiler.
    """
    value = _display_extension(extension).lstrip(".")
    return f"*.{value}" if value else "*"


def _display_extension(extension: str) -> str:
    """Normalize extension display text.

    Inputs:
        Receives extension text.

    Returns:
        Dotless lower-case extension.

    Used by:
        Filesystem semantic compiler.
    """
    value = str(extension or "").strip().lower().lstrip(".")
    return {"markdown": "md", "python": "py"}.get(value, value)


def _extract_extension_targets(goal: str) -> list[str]:
    """Extract file-extension target sets from a prompt.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered extension targets.

    Used by:
        Deterministic filesystem extraction and canonicalization.
    """
    values: list[str] = []
    seen: set[str] = set()
    for match in EXTENSION_RE.finditer(goal):
        value = _display_extension(match.group(0))
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _extract_path_target(goal: str) -> str | None:
    """Extract a simple folder/path target from a prompt.

    Inputs:
        Receives the user goal.

    Returns:
        Path text or None for workspace root.

    Used by:
        Deterministic filesystem extraction.
    """
    if re.search(r"\b(?:under|in|from)\s+(?:the\s+)?(?:current|this)\s+(?:folder|directory|repo|repository)\b", goal, re.IGNORECASE):
        return "."
    match = re.search(r"\b(?:under|in|from)\s+([~./A-Za-z0-9_-][^\s,;]*)", goal)
    if not match:
        return None
    candidate = match.group(1).strip()
    if candidate.lower() in {"this", "current", "folder", "directory", "repo", "repository"}:
        return "."
    return candidate


def _detect_shell_entity(goal: str) -> str:
    """Infer shell/system entity for safe command selection.

    Inputs:
        Receives the user goal.

    Returns:
        Entity identifier.

    Used by:
        Deterministic shell extraction.
    """
    text = str(goal or "").lower()
    if "port" in text:
        return "ports"
    if "disk" in text or "filesystem" in text or "mount" in text:
        return "disk"
    if "memory" in text or "cpu" in text or "resource" in text:
        return "resources"
    if "hostname" in text or "uptime" in text or "user" in text:
        return "system"
    return "processes"


def _shell_command_for_frame(frame: SemanticFrame) -> str | None:
    """Return a read-only shell command for supported system frames.

    Inputs:
        Receives a shell semantic frame.

    Returns:
        Command string or None when unsupported.

    Used by:
        SemanticFrameCompiler.
    """
    entity = str(frame.entity or "processes").lower()
    if entity in {"current_user", "user"}:
        return "whoami"
    if entity == "hostname":
        return "hostname"
    if entity in {"pwd", "working_directory"}:
        return "pwd"
    if entity == "uptime":
        return "uptime"
    if entity == "processes" and frame.intent == "count":
        return "ps -eo pid= | wc -l"
    if entity in {"process", "processes"}:
        return "ps -eo pid,ppid,user,stat,pcpu,pmem,comm --sort=-pcpu | head -n 21"
    if entity in {"port", "ports"}:
        return "ss -ltnp"
    if entity in {"disk", "filesystem", "filesystems"}:
        return "df -h"
    if entity in {"mount", "mounts"}:
        return "findmnt"
    if entity == "memory":
        return "free -h"
    if entity == "cpu":
        return "lscpu"
    if entity == "resources":
        return "free -h"
    if entity in {"docker", "container", "containers"}:
        return "docker ps"
    if entity == "system":
        return "uptime"
    return None


def _extract_database_target(goal: str) -> str | None:
    """Extract a configured database name mention from a prompt.

    Inputs:
        Receives the user goal.

    Returns:
        Database name or None.

    Used by:
        SQL semantic extraction.
    """
    if re.search(r"\bdicom\b", goal, re.IGNORECASE):
        return "dicom"
    match = re.search(r"\bdatabase\s+([A-Za-z0-9_]+)\b", goal, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_sql_text(goal: str) -> str | None:
    """Extract embedded SQL text for validate/explain requests.

    Inputs:
        Receives the user goal.

    Returns:
        SQL text or None.

    Used by:
        SQL semantic extraction.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", goal, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    inline = re.search(r"(\bSELECT\b.+)$", goal, re.IGNORECASE | re.DOTALL)
    return inline.group(1).strip() if inline else None


def _generated_sql_for_goal(goal: str) -> str | None:
    """Generate conservative read-only SQL templates for validate-only prompts.

    Inputs:
        Receives the user goal.

    Returns:
        A schema-grounded DICOM SQL template string, or None.

    Used by:
        SQL semantic metadata extraction.
    """
    text = str(goal or "").lower()
    if "studies per patient" in text or "study per patient" in text:
        return 'SELECT p."PatientID", COUNT(DISTINCT s."StudyInstanceUID") AS study_count FROM flathr."Patient" p JOIN flathr."Study" s ON p."PatientID" = s."PatientID" GROUP BY p."PatientID"'
    if "series per study" in text:
        return 'SELECT s."StudyInstanceUID", COUNT(DISTINCT se."SeriesInstanceUID") AS series_count FROM flathr."Study" s JOIN flathr."Series" se ON s."StudyInstanceUID" = se."StudyInstanceUID" GROUP BY s."StudyInstanceUID"'
    if "instances per series" in text or "instance per series" in text:
        return 'SELECT se."SeriesInstanceUID", COUNT(DISTINCT i."SOPInstanceUID") AS instance_count FROM flathr."Series" se JOIN flathr."Instance" i ON se."SeriesInstanceUID" = i."SeriesInstanceUID" GROUP BY se."SeriesInstanceUID"'
    if "rtstruct" in text and "rtdose" in text:
        return 'SELECT DISTINCT st."StudyInstanceUID" FROM flathr."Study" st WHERE EXISTS (SELECT 1 FROM flathr."Series" se JOIN flathr."Instance" i ON se."SeriesInstanceUID" = i."SeriesInstanceUID" JOIN flathr."RTSTRUCT" rs ON i."SOPInstanceUID" = rs."SOPInstanceUID" WHERE se."StudyInstanceUID" = st."StudyInstanceUID") AND EXISTS (SELECT 1 FROM flathr."Series" se JOIN flathr."Instance" i ON se."SeriesInstanceUID" = i."SeriesInstanceUID" JOIN flathr."RTDOSE" rd ON i."SOPInstanceUID" = rd."SOPInstanceUID" WHERE se."StudyInstanceUID" = st."StudyInstanceUID")'
    return None


def _sql_query_for_frame(frame: SemanticFrame) -> str | None:
    """Build read-only SQL for supported semantic SQL data frames.

    Inputs:
        Receives a canonical SQL frame.

    Returns:
        SQL text for narrow DICOM patterns or None.

    Used by:
        SemanticFrameCompiler._compile_sql_query.
    """
    text = str(frame.rationale or "").lower()
    if frame.intent == "list":
        study_set_query = _dicom_study_modality_set_query(frame)
        if study_set_query:
            return study_set_query
        top_query = _dicom_top_related_count_query(frame)
        if top_query:
            return top_query
    count_entities = _target_set_values(frame.targets.get("entity")) or [str(value) for value in frame.output.result_entities if str(value)]
    if frame.intent == "count" and (frame.entity in {"dicom_counts", "dicom"} or frame.domain == "sql") and len(count_entities) > 1:
        return _dicom_multi_entity_count_query(count_entities)
    if frame.intent == "count" and _frame_groups_by(frame, "decade") and _frame_targets_patients(frame, text):
        return _dicom_patient_age_decade_distribution_query()
    if frame.intent == "count" and _frame_groups_by(frame, "modality"):
        grouped_query = _dicom_grouped_modality_count_query(_frame_count_entity(frame, text))
        if grouped_query:
            return grouped_query
    modalities = _target_set_values(frame.targets.get("modality"))
    concept_terms = _filter_values(frame, "concept_term")
    age_filter = _semantic_age_filter(frame)
    if frame.intent == "count" and _frame_targets_patients(frame, text) and (age_filter or concept_terms or (modalities and frame.output.kind == "scalar")):
        return _dicom_patient_concept_count_query(
            concept_terms,
            modalities=modalities,
            match_policy=str(_filter_value(frame, "match_policy") or "any_term"),
            age_filter=age_filter,
        )
    if frame.intent == "count":
        single_entity_query = _dicom_single_entity_count_query(_frame_count_entity(frame, text))
        if single_entity_query:
            return single_entity_query
    if frame.intent == "count" and modalities and _mentions_patient_entity(text):
        quoted = ", ".join(f"'{value}'" for value in modalities)
        return f'SELECT se."Modality" AS modality, COUNT(DISTINCT st."PatientID") AS patient_count FROM flathr."Study" st JOIN flathr."Series" se ON st."StudyInstanceUID" = se."StudyInstanceUID" WHERE se."Modality" IN ({quoted}) GROUP BY se."Modality" ORDER BY patient_count DESC'
    if frame.intent == "aggregate_metric" and "studies per patient" in text:
        return 'WITH patient_study_counts AS (SELECT st."PatientID", COUNT(DISTINCT st."StudyInstanceUID") AS study_count FROM flathr."Study" st GROUP BY st."PatientID") SELECT MIN(study_count) AS min_studies, MAX(study_count) AS max_studies, AVG(study_count) AS average_studies FROM patient_study_counts'
    if "top" in text and "stud" in text and modalities:
        quoted = ", ".join(f"'{value}'" for value in modalities)
        return f'SELECT st."StudyInstanceUID", se."Modality", COUNT(DISTINCT i."SOPInstanceUID") AS instance_count FROM flathr."Study" st JOIN flathr."Series" se ON st."StudyInstanceUID" = se."StudyInstanceUID" LEFT JOIN flathr."Instance" i ON se."SeriesInstanceUID" = i."SeriesInstanceUID" WHERE se."Modality" IN ({quoted}) GROUP BY st."StudyInstanceUID", se."Modality" ORDER BY instance_count DESC LIMIT 10'
    return None


def _frame_groups_by(frame: SemanticFrame, dimension: str) -> bool:
    """Return whether a frame requests grouping by a dimension.

    Inputs:
        Receives a semantic frame and canonical dimension token.

    Returns:
        True when dimensions or output group_by contains the dimension.

    Used by:
        SQL grouped-count lowerers.
    """
    normalized = normalize_semantic_dimension(dimension)
    values = [*list(frame.dimensions or []), *list(frame.output.group_by or [])]
    return normalized in {normalize_semantic_dimension(value) for value in values}


def _dicom_grouped_modality_count_query(entity: str | None) -> str | None:
    """Build grouped DICOM count SQL keyed by series modality.

    Inputs:
        Receives the result entity being counted.

    Returns:
        Read-only SQL grouped by modality, or None for unsupported entities.

    Used by:
        _sql_query_for_frame for sql.count_grouped frames.
    """
    normalized = _normalize_dicom_count_entity(entity)
    if normalized == "studies":
        return (
            'SELECT se."Modality" AS modality, COUNT(DISTINCT st."StudyInstanceUID") AS study_count\n'
            'FROM flathr."Study" st\n'
            'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"\n'
            'GROUP BY se."Modality"\n'
            'ORDER BY study_count DESC'
        )
    if normalized == "series":
        return (
            'SELECT se."Modality" AS modality, COUNT(DISTINCT se."SeriesInstanceUID") AS series_count\n'
            'FROM flathr."Series" se\n'
            'GROUP BY se."Modality"\n'
            'ORDER BY series_count DESC'
        )
    if normalized == "instances":
        return (
            'SELECT se."Modality" AS modality, COUNT(DISTINCT i."SOPInstanceUID") AS instance_count\n'
            'FROM flathr."Series" se\n'
            'JOIN flathr."Instance" i ON i."SeriesInstanceUID" = se."SeriesInstanceUID"\n'
            'GROUP BY se."Modality"\n'
            'ORDER BY instance_count DESC'
        )
    if normalized == "patients":
        return (
            'SELECT se."Modality" AS modality, COUNT(DISTINCT st."PatientID") AS patient_count\n'
            'FROM flathr."Study" st\n'
            'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"\n'
            'GROUP BY se."Modality"\n'
            'ORDER BY patient_count DESC'
        )
    return None


def _dicom_patient_age_decade_distribution_query() -> str:
    """Build a DICOM patient-count distribution grouped into age decades.

    Inputs:
        Uses the standard DICOM PatientBirthDate column.

    Returns:
        Read-only SQL that groups patients by decade buckets.

    Used by:
        _sql_query_for_frame for grouped age-distribution frames.
    """
    return (
        "WITH patient_ages AS (\n"
        "  SELECT EXTRACT(YEAR FROM AGE(CURRENT_DATE, p.\"PatientBirthDate\"))::int AS age_years\n"
        "  FROM flathr.\"Patient\" p\n"
        "  WHERE p.\"PatientBirthDate\" IS NOT NULL\n"
        ")\n"
        "SELECT ((age_years / 10) * 10)::int AS age_decade, COUNT(*) AS patient_count\n"
        "FROM patient_ages\n"
        "WHERE age_years >= 0 AND age_years <= 130\n"
        "GROUP BY age_decade\n"
        "ORDER BY age_decade"
    )


def _dicom_study_modality_set_query(frame: SemanticFrame) -> str | None:
    """Build DICOM study-list SQL for include/exclude modality set requests.

    Inputs:
        Receives a semantic frame with study entity and modality targets or filters.

    Returns:
        Read-only SQL listing study identifiers satisfying modality set logic.

    Used by:
        _sql_query_for_frame for co-occurrence and absence list frames.
    """
    if _normalize_dicom_count_entity(frame.entity) != "studies":
        return None
    include_values = [value.upper() for value in _dedupe_strings(_filter_values(frame, "include_modality"))]
    exclude_values = [value.upper() for value in _dedupe_strings(_filter_values(frame, "exclude_modality"))]
    target_values = [value.upper() for value in _dedupe_strings(_target_set_values(frame.targets.get("modality")))]
    match_policy = _normalize_token(_filter_value(frame, "match_policy"))
    if not include_values and target_values and match_policy == "all_targets":
        include_values = target_values
    if not include_values and len(target_values) == 1:
        include_values = target_values
    if not include_values and not exclude_values:
        return None
    predicates: list[str] = []
    for index, modality in enumerate(include_values, start=1):
        predicates.append(_dicom_study_modality_exists_clause(modality, alias=f"se_in_{index}"))
    for index, modality in enumerate(exclude_values, start=1):
        predicates.append(f"NOT {_dicom_study_modality_exists_clause(modality, alias=f'se_out_{index}')}")
    where = "\n  AND ".join(predicates) if predicates else "TRUE"
    return (
        'SELECT st."StudyInstanceUID"\n'
        'FROM flathr."Study" st\n'
        f"WHERE {where}\n"
        'ORDER BY st."StudyInstanceUID"\n'
        "LIMIT 100"
    )


def _dicom_study_modality_exists_clause(modality: str, *, alias: str) -> str:
    """Build an EXISTS clause for one modality related to a study.

    Inputs:
        Receives a modality token and SQL alias for Series.

    Returns:
        SQL EXISTS clause correlated to study alias st.

    Used by:
        _dicom_study_modality_set_query.
    """
    escaped = str(modality).replace("'", "''").upper()
    return (
        "EXISTS (\n"
        "    SELECT 1\n"
        f'    FROM flathr."Series" {alias}\n'
        f'    WHERE {alias}."StudyInstanceUID" = st."StudyInstanceUID"\n'
        f"      AND {alias}.\"Modality\" = '{escaped}'\n"
        "  )"
    )


def _dicom_top_related_count_query(frame: SemanticFrame) -> str | None:
    """Build schema-grounded top-N queries ordered by related DICOM counts.

    Inputs:
        Receives a semantic frame with entity, related_count, limit, and optional modality/age filters.

    Returns:
        Read-only SQL for supported related-count rankings, or None when unsupported.

    Used by:
        _sql_query_for_frame for sql.top_n/list patients frames.
    """
    entity = _normalize_dicom_count_entity(frame.entity)
    related = _normalize_token(_filter_value(frame, "related_count"))
    limit = _positive_int_filter(frame, "limit", default=10)
    modalities = [value.upper() for value in _dedupe_strings(_target_set_values(frame.targets.get("modality")))]
    age_filter = _semantic_age_filter(frame)
    if entity == "patients" and related == "studies":
        return _dicom_top_patients_by_study_count_query(limit, age_filter=age_filter)
    if entity == "patients" and related == "modality_records" and modalities:
        return _dicom_top_patients_by_modality_count_query(limit, modalities=modalities, age_filter=age_filter)
    if entity == "studies" and related == "series":
        return _dicom_top_studies_by_series_count_query(limit)
    if entity == "series" and related == "instances":
        return _dicom_top_series_by_instance_count_query(limit)
    if entity == "modalities" and related == "studies":
        return _dicom_top_modalities_by_study_count_query(limit)
    if entity == "modalities" and related == "series":
        return _dicom_top_modalities_by_series_count_query(limit)
    return None


def _dicom_top_patients_by_study_count_query(limit: int, *, age_filter: tuple[str, int] | None = None) -> str:
    """Build a top-patient query ordered by distinct study count.

    Inputs:
        Receives a positive row limit and optional patient-age filter.

    Returns:
        Read-only SQL listing patient identifiers and distinct study counts.

    Used by:
        _dicom_top_related_count_query.
    """
    safe_limit = max(1, min(int(limit or 10), 100))
    where = f"\nWHERE {_dicom_age_predicate(age_filter, alias='p')}" if age_filter else ""
    return (
        'SELECT p."PatientID", COUNT(DISTINCT st."StudyInstanceUID") AS study_count\n'
        'FROM flathr."Patient" p\n'
        'JOIN flathr."Study" st ON st."PatientID" = p."PatientID"'
        f"{where}\n"
        'GROUP BY p."PatientID"\n'
        "ORDER BY study_count DESC\n"
        f"LIMIT {safe_limit}"
    )


def _dicom_top_patients_by_modality_count_query(
    limit: int,
    *,
    modalities: list[str],
    age_filter: tuple[str, int] | None = None,
) -> str:
    """Build a top-patient query ordered by modality-specific series count.

    Inputs:
        Receives a row limit, modality targets, and optional age filter.

    Returns:
        Read-only SQL grouped by patient for requested DICOM modalities.

    Used by:
        _dicom_top_related_count_query.
    """
    safe_limit = max(1, min(int(limit or 10), 100))
    quoted = ", ".join(f"'{value}'" for value in modalities)
    predicates = [f'se."Modality" IN ({quoted})']
    if age_predicate := _dicom_age_predicate(age_filter, alias="p"):
        predicates.insert(0, age_predicate)
    where = "\nWHERE " + "\n  AND ".join(predicates)
    return (
        'SELECT p."PatientID", COUNT(DISTINCT se."SeriesInstanceUID") AS modality_record_count\n'
        'FROM flathr."Patient" p\n'
        'JOIN flathr."Study" st ON st."PatientID" = p."PatientID"\n'
        'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"'
        f"{where}\n"
        'GROUP BY p."PatientID"\n'
        'ORDER BY modality_record_count DESC\n'
        f"LIMIT {safe_limit}"
    )


def _dicom_top_studies_by_series_count_query(limit: int) -> str:
    """Build a top-study query ordered by distinct series count.

    Inputs:
        Receives a positive row limit.

    Returns:
        Read-only SQL listing study identifiers and series counts.

    Used by:
        _dicom_top_related_count_query.
    """
    safe_limit = max(1, min(int(limit or 10), 100))
    return (
        'SELECT st."StudyInstanceUID", COUNT(DISTINCT se."SeriesInstanceUID") AS series_count\n'
        'FROM flathr."Study" st\n'
        'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"\n'
        'GROUP BY st."StudyInstanceUID"\n'
        'ORDER BY series_count DESC\n'
        f"LIMIT {safe_limit}"
    )


def _dicom_top_series_by_instance_count_query(limit: int) -> str:
    """Build a top-series query ordered by distinct instance count.

    Inputs:
        Receives a positive row limit.

    Returns:
        Read-only SQL listing series identifiers and instance counts.

    Used by:
        _dicom_top_related_count_query.
    """
    safe_limit = max(1, min(int(limit or 10), 100))
    return (
        'SELECT se."SeriesInstanceUID", COUNT(DISTINCT i."SOPInstanceUID") AS instance_count\n'
        'FROM flathr."Series" se\n'
        'JOIN flathr."Instance" i ON i."SeriesInstanceUID" = se."SeriesInstanceUID"\n'
        'GROUP BY se."SeriesInstanceUID"\n'
        'ORDER BY instance_count DESC\n'
        f"LIMIT {safe_limit}"
    )


def _dicom_top_modalities_by_series_count_query(limit: int) -> str:
    """Build a top-modality query ordered by series count.

    Inputs:
        Receives a positive row limit.

    Returns:
        Read-only SQL listing modalities and distinct series counts.

    Used by:
        _dicom_top_related_count_query.
    """
    safe_limit = max(1, min(int(limit or 10), 100))
    return (
        'SELECT se."Modality", COUNT(DISTINCT se."SeriesInstanceUID") AS series_count\n'
        'FROM flathr."Series" se\n'
        'GROUP BY se."Modality"\n'
        'ORDER BY series_count DESC\n'
        f"LIMIT {safe_limit}"
    )


def _dicom_top_modalities_by_study_count_query(limit: int) -> str:
    """Build a top-modality query ordered by distinct study count.

    Inputs:
        Receives a positive row limit.

    Returns:
        Read-only SQL listing modalities and distinct study counts.

    Used by:
        _dicom_top_related_count_query.
    """
    safe_limit = max(1, min(int(limit or 10), 100))
    return (
        'SELECT se."Modality", COUNT(DISTINCT st."StudyInstanceUID") AS study_count\n'
        'FROM flathr."Study" st\n'
        'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"\n'
        'GROUP BY se."Modality"\n'
        'ORDER BY study_count DESC\n'
        f"LIMIT {safe_limit}"
    )


def _positive_int_filter(frame: SemanticFrame, field: str, *, default: int) -> int:
    """Read a positive integer semantic filter with a bounded default.

    Inputs:
        Receives a semantic frame, filter field, and default value.

    Returns:
        A positive integer suitable for safe LIMIT clauses and bounded fan-out.

    Used by:
        SQL semantic lowerers that need an explicit result limit.
    """
    value = _filter_value(frame, field)
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _dicom_multi_entity_count_query(entities: list[str]) -> str | None:
    """Build one SQL row containing several DICOM count aggregates.

    Inputs:
        Receives normalized DICOM entity labels requested by the user.

    Returns:
        A read-only SQL query with one aggregate column per requested entity.

    Used by:
        _sql_query_for_frame for multi-scalar DICOM count prompts.
    """
    columns: list[str] = []
    for entity in _dedupe_strings(entities):
        normalized = _normalize_dicom_count_entity(entity)
        if normalized == "patients":
            columns.append('(SELECT COUNT(*) FROM flathr."Patient") AS patient_count')
        elif normalized == "studies":
            columns.append('(SELECT COUNT(*) FROM flathr."Study") AS study_count')
        elif normalized == "series":
            columns.append('(SELECT COUNT(*) FROM flathr."Series") AS series_count')
        elif normalized == "instances":
            columns.append('(SELECT COUNT(*) FROM flathr."Instance") AS instance_count')
        elif normalized in {"rtplan", "rtdose", "rtstruct", "ct", "mr", "pt"}:
            columns.append(f'(SELECT COUNT(*) FROM flathr."Series" se WHERE se."Modality" = \'{normalized.upper()}\') AS {normalized}_count')
    if len(columns) < 2:
        return None
    return "SELECT\n  " + ",\n  ".join(columns)


def _dicom_single_entity_count_query(entity: str | None) -> str | None:
    """Build a scalar DICOM count query for one known entity.

    Inputs:
        Receives a normalized DICOM entity label such as patients, studies, or RTPLAN.

    Returns:
        A one-row, one-column read-only SQL count query.

    Used by:
        _sql_query_for_frame for scalar DICOM count compilation.
    """
    normalized = _normalize_dicom_count_entity(entity)
    if normalized == "patients":
        return 'SELECT COUNT(*) AS patient_count FROM flathr."Patient"'
    if normalized == "studies":
        return 'SELECT COUNT(*) AS study_count FROM flathr."Study"'
    if normalized == "series":
        return 'SELECT COUNT(*) AS series_count FROM flathr."Series"'
    if normalized == "instances":
        return 'SELECT COUNT(*) AS instance_count FROM flathr."Instance"'
    if normalized in {"rtplan", "rtdose", "rtstruct", "ct", "mr", "pt"}:
        return f'SELECT COUNT(*) AS {normalized}_count FROM flathr."Series" se WHERE se."Modality" = \'{normalized.upper()}\''
    if normalized == "modalities":
        return 'SELECT COUNT(DISTINCT se."Modality") AS modality_count FROM flathr."Series" se'
    return None


def _frame_count_entity(frame: SemanticFrame, rationale: str) -> str | None:
    """Resolve the requested scalar count entity from typed frame metadata.

    Inputs:
        Receives a semantic SQL frame and optional rationale text.

    Returns:
        A DICOM entity label, or None when no supported scalar entity is present.

    Used by:
        _sql_query_for_frame.
    """
    entity_targets = _target_set_values(frame.targets.get("entity"))
    if len(entity_targets) == 1:
        return entity_targets[0]
    if len(frame.output.result_entities) == 1:
        return str(frame.output.result_entities[0])
    entity = _normalize_dicom_count_entity(frame.entity)
    if entity:
        return entity
    extracted = _extract_dicom_count_entities(rationale)
    if len(extracted) == 1:
        return extracted[0]
    return None


def _normalize_dicom_count_entity(entity: Any) -> str | None:
    """Normalize DICOM count entity aliases.

    Inputs:
        Receives raw entity text from a semantic frame.

    Returns:
        Canonical plural entity or modality label.

    Used by:
        DICOM count query builders.
    """
    normalized = _normalize_token(entity)
    aliases = {
        "patient": "patients",
        "patients": "patients",
        "study": "studies",
        "studies": "studies",
        "series": "series",
        "instance": "instances",
        "instances": "instances",
        "rtplans": "rtplan",
        "rtplan": "rtplan",
        "rtdoses": "rtdose",
        "rtdose": "rtdose",
        "rtstructs": "rtstruct",
        "rtstruct": "rtstruct",
        "ct": "ct",
        "mr": "mr",
        "pt": "pt",
        "modality": "modalities",
        "modalities": "modalities",
    }
    return aliases.get(normalized)


def _dicom_patient_concept_count_query(
    concept_terms: list[str],
    *,
    modalities: list[str],
    age_filter: tuple[str, int] | None = None,
    match_policy: str = "any_term",
) -> str | None:
    """Build a scalar DICOM patient count query for text concepts and modalities.

    Inputs:
        Receives normalized text concepts, optional modality targets, optional patient age filter, and match policy.

    Returns:
        A single aggregate SQL query, or None when no semantic filter exists.

    Used by:
        _sql_query_for_frame for deterministic DICOM scalar-count compilation.
    """
    terms = [term for term in _dedupe_strings(concept_terms) if term]
    normalized_modalities = [value.upper() for value in _dedupe_strings(modalities)]
    age_predicate = _dicom_age_predicate(age_filter, alias="p")
    if not terms and not normalized_modalities and not age_predicate:
        return None
    if match_policy == "all_terms" and len(terms) > 1:
        clauses = [
            _dicom_patient_exists_clause(
                term=term,
                modalities=normalized_modalities if normalized_modalities else [],
                alias=f"st_{index}",
                series_alias=f"se_{index}",
            )
            for index, term in enumerate(terms, start=1)
        ]
        if age_predicate:
            clauses.insert(0, age_predicate)
        where_clause = "\n  AND ".join(clauses)
        return f'SELECT COUNT(DISTINCT p."PatientID") AS patient_count\nFROM flathr."Patient" p\nWHERE {where_clause}'
    predicates: list[str] = []
    if age_predicate:
        predicates.append(age_predicate)
    if normalized_modalities:
        quoted = ", ".join(f"'{value}'" for value in normalized_modalities)
        predicates.append(f'se."Modality" IN ({quoted})')
    if terms:
        predicates.append("(" + " OR ".join(_dicom_text_predicate(term, study_alias="st", series_alias="se") for term in terms) + ")")
    where_clause = "\nWHERE " + "\n  AND ".join(predicates) if predicates else ""
    if normalized_modalities or terms:
        return (
            'SELECT COUNT(DISTINCT p."PatientID") AS patient_count\n'
            'FROM flathr."Patient" p\n'
            'JOIN flathr."Study" st ON st."PatientID" = p."PatientID"\n'
            'JOIN flathr."Series" se ON se."StudyInstanceUID" = st."StudyInstanceUID"'
            f"{where_clause}"
        )
    return (
        'SELECT COUNT(DISTINCT p."PatientID") AS patient_count\n'
        'FROM flathr."Patient" p'
        f"{where_clause}"
    )


def _semantic_age_filter(frame: SemanticFrame) -> tuple[str, int] | None:
    """Return a typed age comparison from semantic SQL filters.

    Inputs:
        Receives a semantic SQL frame with filters such as age gt 45.

    Returns:
        Operator and integer age when a supported patient-age comparison exists.

    Used by:
        _sql_query_for_frame for schema-grounded DICOM patient count compilation.
    """
    age_fields = {"age", "patient_age", "patient_age_years"}
    for filter_ in frame.filters:
        if filter_.field not in age_fields or isinstance(filter_.value, list):
            continue
        try:
            age = int(float(str(filter_.value)))
        except (TypeError, ValueError):
            continue
        if age < 0 or age > 130:
            continue
        if filter_.operator in {"gt", "gte", "lt", "lte", "eq"}:
            return filter_.operator, age
    return None


def _dicom_age_predicate(age_filter: tuple[str, int] | None, *, alias: str) -> str | None:
    """Lower a typed patient-age comparison to a PatientBirthDate predicate.

    Inputs:
        Receives an age operator/value pair and SQL alias for flathr.Patient.

    Returns:
        SQL predicate using PatientBirthDate, or None when unsupported.

    Used by:
        DICOM patient-count SQL builders.
    """
    if age_filter is None:
        return None
    operator, age = age_filter
    if operator == "gt":
        return f'{alias}."PatientBirthDate" < CURRENT_DATE - INTERVAL \'{age} years\''
    if operator == "gte":
        return f'{alias}."PatientBirthDate" <= CURRENT_DATE - INTERVAL \'{age} years\''
    if operator == "lt":
        return f'{alias}."PatientBirthDate" > CURRENT_DATE - INTERVAL \'{age} years\''
    if operator == "lte":
        return f'{alias}."PatientBirthDate" >= CURRENT_DATE - INTERVAL \'{age} years\''
    if operator == "eq":
        return (
            f'{alias}."PatientBirthDate" <= CURRENT_DATE - INTERVAL \'{age} years\' '
            f'AND {alias}."PatientBirthDate" > CURRENT_DATE - INTERVAL \'{age + 1} years\''
        )
    return None


def _frame_targets_patients(frame: SemanticFrame, rationale: str) -> bool:
    """Return whether a SQL semantic frame asks about patients.

    Inputs:
        Receives a semantic frame and optional rationale text.

    Returns:
        True when entity or output metadata targets patients.

    Used by:
        SQL semantic query builders.
    """
    entity = _normalize_token(frame.entity)
    if entity in {"patient", "patients"}:
        return True
    if any(_normalize_token(value) in {"patient", "patients"} for value in frame.output.result_entities):
        return True
    return _mentions_patient_entity(rationale)


def _dicom_patient_exists_clause(*, term: str, modalities: list[str], alias: str, series_alias: str) -> str:
    """Build one correlated EXISTS clause for a DICOM patient text concept.

    Inputs:
        Receives the concept term, modality targets, and SQL aliases.

    Returns:
        SQL EXISTS clause correlated to patient alias p.

    Used by:
        _dicom_patient_concept_count_query for all-terms patient counts.
    """
    predicates = [_dicom_text_predicate(term, study_alias=alias, series_alias=series_alias)]
    if modalities:
        quoted = ", ".join(f"'{value}'" for value in modalities)
        predicates.append(f'{series_alias}."Modality" IN ({quoted})')
    where = "\n      AND ".join(predicates)
    return (
        "EXISTS (\n"
        "    SELECT 1\n"
        f'    FROM flathr."Study" {alias}\n'
        f'    JOIN flathr."Series" {series_alias} ON {series_alias}."StudyInstanceUID" = {alias}."StudyInstanceUID"\n'
        f'    WHERE {alias}."PatientID" = p."PatientID"\n'
        f"      AND {where}\n"
        "  )"
    )


def _dicom_text_predicate(term: str, *, study_alias: str, series_alias: str) -> str:
    """Build a schema-backed DICOM description predicate.

    Inputs:
        Receives one concept term and the study/series aliases to reference.

    Returns:
        SQL predicate over StudyDescription and SeriesDescription.

    Used by:
        DICOM patient scalar-count query builders.
    """
    escaped = str(term).replace("'", "''").lower()
    return (
        f"(LOWER({study_alias}.\"StudyDescription\") LIKE '%{escaped}%' "
        f"OR LOWER({series_alias}.\"SeriesDescription\") LIKE '%{escaped}%')"
    )


def _extract_modality_targets(goal: str) -> list[str]:
    """Extract DICOM modality target values.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered modality names.

    Used by:
        SQL semantic target canonicalization.
    """
    modalities = []
    for value in ("CT", "MR", "RTSTRUCT", "RTPLAN", "RTDOSE", "PT"):
        if re.search(rf"\b{re.escape(value)}\b", goal, re.IGNORECASE):
            modalities.append(value)
    return modalities


def _extract_excluded_modality_targets(goal: str) -> list[str]:
    """Extract DICOM modality targets that are explicitly negated.

    Inputs:
        Receives the user prompt.

    Returns:
        Ordered modality names that appear after without/no exclusion wording.

    Used by:
        SQL semantic grounding for include/exclude modality list prompts.
    """
    text = str(goal or "")
    excluded: list[str] = []
    for value in ("CT", "MR", "RTSTRUCT", "RTPLAN", "RTDOSE", "PT"):
        pattern = rf"\b(?:but\s+no|without|with\s+no|no)\s+{re.escape(value)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            excluded.append(value)
    return excluded


def _extract_dicom_concept_terms(goal: str) -> list[str]:
    """Extract simple DICOM text concept terms from patient/study prompts.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered concept terms such as brain and breast.

    Used by:
        SQL semantic extraction and canonicalization backups.
    """
    terms: list[str] = []
    for term in ("brain", "breast"):
        if re.search(rf"\b{re.escape(term)}\b", goal, re.IGNORECASE):
            terms.append(term)
    return terms


def _extract_dicom_count_entities(goal: str) -> list[str]:
    """Extract independently requested DICOM count entities.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered labels for entity-count dashboard queries.

    Used by:
        _deterministic_sql_frame.
    """
    text = str(goal or "").lower()
    if not COUNT_GOAL_RE.search(text) or "dicom" not in text:
        return []
    if re.search(r"\b(?:by|group(?:ed)?\s+by|per|each|separately|that\s+have|with|having|where)\b", text):
        return []
    entities: list[str] = []
    patterns = [
        ("patients", r"\b(?:patients?|patieints?)\b"),
        ("studies", r"\bstudies\b"),
        ("series", r"\bseries\b"),
        ("instances", r"\binstances?\b"),
        ("rtplan", r"\brtplans?\b"),
        ("rtdose", r"\brtdoses?\b"),
        ("rtstruct", r"\brtstructs?\b"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            entities.append(label)
    if len(entities) < 2:
        return []
    return entities


def _extract_age_filter(goal: str) -> tuple[str, int] | None:
    """Extract a patient-age comparison for deterministic SQL fallback frames.

    Inputs:
        Receives a user prompt.

    Returns:
        Operator and age when age wording is present.

    Used by:
        _deterministic_sql_frame test/fallback support; active LLM frames use typed filters directly.
    """
    text = str(goal or "").lower()
    patterns = [
        (r"\b(?:over|above|greater\s+than|older\s+than)\s+(?:age\s*)?(?P<age>\d{1,3})\b", "gt"),
        (r"\b(?:age\s*)?(?P<age>\d{1,3})\s+(?:and\s+older|or\s+older|and\s+above|or\s+above)\b", "gte"),
        (r"\b(?:under|below|less\s+than|younger\s+than)\s+(?:age\s*)?(?P<age>\d{1,3})\b", "lt"),
        (r"\b(?:age\s*)?(?P<age>\d{1,3})\s+(?:and\s+younger|or\s+younger|and\s+below|or\s+below)\b", "lte"),
    ]
    for pattern, operator in patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group("age"))
            if 0 <= age <= 130:
                return operator, age
    return None


def _mentions_patient_entity(goal: str) -> bool:
    """Detect patient wording with a small typo tolerance.

    Inputs:
        Receives the user goal.

    Returns:
        True when the goal mentions patients or common misspellings.

    Used by:
        SQL semantic query builders.
    """
    return bool(re.search(r"\b(?:patient|patients|patieint|patieints)\b", str(goal or ""), re.IGNORECASE))


def _all_terms_policy(goal: str) -> bool:
    """Detect whether all extracted DICOM concept terms are required.

    Inputs:
        Receives the user goal.

    Returns:
        True when terms are joined by all/both-style wording.

    Used by:
        SQL semantic extraction.
    """
    text = str(goal or "").lower()
    return bool(re.search(r"\b(?:both|all)\b", text) or re.search(r"\band\b", text))


def _is_grouped_sql_count_goal(goal: str) -> bool:
    """Detect whether a SQL count prompt asks for grouped table output.

    Inputs:
        Receives the user goal.

    Returns:
        True for by/each/separately-style count requests.

    Used by:
        _deterministic_sql_frame.
    """
    return bool(re.search(r"\b(?:by|group(?:ed)?\s+by|per|each|separately)\b", str(goal or ""), re.IGNORECASE))


def _filter_values(frame: SemanticFrame, field: str) -> list[str]:
    """Read all semantic filter values for a field from a frame.

    Inputs:
        Receives a semantic frame and filter field name.

    Returns:
        Ordered string values.

    Used by:
        SQL semantic query builders.
    """
    values: list[str] = []
    for filter_ in frame.filters:
        if filter_.field != field:
            continue
        raw_values = filter_.value if isinstance(filter_.value, list) else [filter_.value]
        for raw_value in raw_values:
            text = str(raw_value or "").strip().lower()
            if text:
                values.append(text)
    return values


def _dedupe_strings(values: list[str]) -> list[str]:
    """Dedupe strings while preserving order.

    Inputs:
        Receives candidate string values.

    Returns:
        Ordered unique non-empty strings.

    Used by:
        SQL semantic query builders.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _rewrite_refs(value: Any, alias_map: dict[str, str]) -> Any:
    """Rewrite nested step references when composing child plans.

    Inputs:
        Receives arbitrary args and an output-alias mapping.

    Returns:
        Args with child-local refs rewritten to global aliases.

    Used by:
        SemanticFrameCompiler._compile_compound.
    """
    if isinstance(value, dict):
        if "$ref" in value:
            rewritten = dict(value)
            ref = str(rewritten.get("$ref") or "")
            if ref in alias_map:
                rewritten["$ref"] = alias_map[ref]
            return {key: _rewrite_refs(nested, alias_map) for key, nested in rewritten.items()}
        return {key: _rewrite_refs(nested, alias_map) for key, nested in value.items()}
    if isinstance(value, list):
        return [_rewrite_refs(item, alias_map) for item in value]
    return value


def _target_set_values(target: SemanticTargetSet | None) -> list[str]:
    """Read SemanticTargetSet values without colliding with BaseModel.values.

    Inputs:
        Receives an optional target set.

    Returns:
        The target values field as a list of strings.

    Used by:
        Semantic frame canonicalization, compilation, and coverage checks.
    """
    if target is None:
        return []
    raw = target.model_dump().get("values")
    return [str(item) for item in raw or [] if str(item)]


def _extract_partition_targets(goal: str) -> list[str]:
    """Extract one or more partition targets from a prompt.

    Inputs:
        Receives the user goal.

    Returns:
        Ordered partition names, excluding generic words.

    Used by:
        Deterministic frame extraction and canonicalization backup.
    """
    text = str(goal or "")
    match = PARTITION_TARGETS_RE.search(text)
    if match:
        return _target_values(match.group(1))
    multi = re.search(
        r"\b(?:partition\s+)?([A-Za-z0-9._-]+)\s*(?:,|\+|/|\band\b)\s*(?:partition\s+)?([A-Za-z0-9._-]+)\s+partitions?\b",
        text,
        re.IGNORECASE,
    )
    if multi:
        return _target_values([multi.group(1), multi.group(2)])
    repeated = re.findall(r"\bpartition\s+([A-Za-z0-9._-]+)\b", text, re.IGNORECASE)
    if len(repeated) > 1:
        return _target_values(repeated)
    single = SINGLE_PARTITION_RE.search(text)
    if single:
        candidate = single.group(1) or single.group(2)
        values = _target_values(candidate)
        if values and values[0].lower() not in {"all", "each", "every", "slurm"}:
            return values
    return []


def _extract_time_phrase(goal: str) -> str | None:
    """Extract a temporal phrase from a user goal.

    Inputs:
        Receives the user goal.

    Returns:
        Natural time phrase or None.

    Used by:
        Deterministic semantic extraction.
    """
    text = str(goal or "")
    patterns = [
        r"\b(?:last|past)\s+\d+\s+(?:hours?|days?)\b",
        r"\bsince\s+yesterday\b",
        r"\byesterday\b",
        r"\btoday\b",
        r"\bthis\s+week\b",
        r"\blast\s+week\b",
        r"\bthis\s+month\b",
        r"\bfrom\s+\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _target_values(value: Any) -> list[str]:
    """Normalize scalar/list target values.

    Inputs:
        Receives raw target values.

    Returns:
        Ordered unique target strings.

    Used by:
        SemanticTargetSet validation, extraction, and projection.
    """
    raw_items = value if isinstance(value, list) else str(value or "").split(",")
    values: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = str(item or "").strip().strip("`'\"")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        values.append(cleaned)
        seen.add(key)
    return values


def _filter_value(frame: SemanticFrame, field: str) -> Any | None:
    """Return the first matching filter value from a semantic frame.

    Inputs:
        Receives a frame and filter field.

    Returns:
        Matching filter value or None.

    Used by:
        SemanticFrameCompiler.
    """
    normalized = _normalize_token(field)
    for item in frame.filters:
        if item.field == normalized:
            return item.value
    return None


def _filter_bool(filters: list[SemanticFilter], field: str) -> bool:
    """Return whether a boolean semantic filter is truthy.

    Inputs:
        Receives filters and a field name.

    Returns:
        True when a matching filter carries a truthy boolean-like value.

    Used by:
        SLURM accounting state-policy canonicalization.
    """
    normalized = _normalize_token(field)
    for item in filters:
        if item.field != normalized:
            continue
        value = item.value
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
    return False


def _slurm_accounting_state_policy(frame: SemanticFrame) -> SlurmAccountingStatePolicy:
    """Resolve the state policy for a SLURM accounting aggregate frame.

    Inputs:
        Receives a canonical SLURM aggregate frame.

    Returns:
        Explicit all-state, explicit state, or default-completed policy.

    Used by:
        SemanticFrameCompiler._compile_slurm_aggregate.
    """
    if frame.state_policy is not None:
        if frame.state_policy.include_all_states:
            return SlurmAccountingStatePolicy(include_all_states=True)
        if frame.state_policy.state:
            return SlurmAccountingStatePolicy(
                state=str(frame.state_policy.state),
                default_state_applied=bool(frame.state_policy.default_state_applied),
            )
    if _filter_value(frame, "include_all_states"):
        return SlurmAccountingStatePolicy(include_all_states=True)
    state_filter = _filter_value(frame, "state")
    if state_filter:
        return SlurmAccountingStatePolicy(state=str(state_filter))
    return SlurmAccountingStatePolicy(state="COMPLETED", default_state_applied=True)


def _target_mode(frame: SemanticFrame, dimension: str) -> str:
    """Return the mode for a target dimension.

    Inputs:
        Receives a frame and target dimension.

    Returns:
        Target mode string.

    Used by:
        SemanticFrameCompiler.
    """
    target = frame.targets.get(_normalize_token(dimension))
    return target.mode if target else "none"


def _project_slurm_accounting_groups(result: dict[str, Any], target_values: list[str]) -> dict[str, Any]:
    """Filter SLURM aggregate groups to requested targets and recompute summary fields.

    Inputs:
        Receives a slurm.accounting_aggregate result and target partition values.

    Returns:
        A projected result with only requested group rows.

    Used by:
        project_semantic_result.
    """
    if not target_values or not isinstance(result.get("groups"), list):
        return result
    order = {value.lower(): index for index, value in enumerate(target_values)}
    groups = [dict(group) for group in result.get("groups") or [] if str(group.get("key") or "").lower() in order]
    groups.sort(key=lambda group: order.get(str(group.get("key") or "").lower(), 9999))
    projected = dict(result)
    projected["groups"] = groups
    projected["semantic_projection"] = {"field": "partition", "values": target_values, "matched_groups": len(groups)}
    _preserve_projection_source_counts(projected, result)
    projected["warnings"] = [str(warning) for warning in result.get("warnings") or [] if str(warning)]
    if bool(projected.get("source_truncated")) and not any("returned accounting rows only" in warning for warning in projected["warnings"]):
        projected["warnings"].append("Projected aggregate is over returned accounting rows only because the source result was truncated.")
    missing = [value for value in target_values if value.lower() not in {str(group.get("key") or "").lower() for group in groups}]
    if missing:
        projected["warnings"].append(f"No matching accounting groups were found for: {', '.join(missing)}.")
    _recompute_projected_aggregate(projected, groups)
    return projected


def _preserve_projection_source_counts(projected: dict[str, Any], source: dict[str, Any]) -> None:
    """Preserve pre-projection fetch counts separately from projected result counts.

    Inputs:
        Receives a mutable projected result plus the original aggregate payload.

    Returns:
        Mutates projected with source_* metadata used for warnings and validation.

    Used by:
        _project_slurm_accounting_groups before recomputing projected top-level counts.
    """
    projected["source_total_count"] = source.get("source_total_count", source.get("total_count", source.get("job_count")))
    projected["source_returned_count"] = source.get("source_returned_count", source.get("returned_count", source.get("job_count")))
    projected["source_limit"] = source.get("source_limit", source.get("limit"))
    projected["source_truncated"] = bool(source.get("source_truncated", source.get("truncated")))


def _recompute_projected_aggregate(result: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    """Recompute top-level aggregate fields from projected group rows.

    Inputs:
        Receives a mutable result dictionary and projected groups.

    Returns:
        Mutates the result in place with aggregate summary fields.

    Used by:
        _project_slurm_accounting_groups.
    """
    counts = [int(group.get("job_count") or 0) for group in groups]
    total_jobs = sum(counts)
    total_elapsed = sum(float(group.get("sum_elapsed_seconds") or 0) for group in groups)
    minimums = [float(group.get("min_elapsed_seconds")) for group in groups if group.get("min_elapsed_seconds") is not None]
    maximums = [float(group.get("max_elapsed_seconds")) for group in groups if group.get("max_elapsed_seconds") is not None]
    average = (total_elapsed / total_jobs) if total_jobs else None
    minimum = min(minimums) if minimums else None
    maximum = max(maximums) if maximums else None
    result["job_count"] = total_jobs
    result["total_count"] = total_jobs
    result["returned_count"] = total_jobs
    result["limit"] = None
    result["truncated"] = False
    result["average_elapsed_seconds"] = average
    result["average_elapsed_human"] = format_elapsed_seconds(average)
    result["min_elapsed_seconds"] = minimum
    result["min_elapsed_human"] = format_elapsed_seconds(minimum)
    result["max_elapsed_seconds"] = maximum
    result["max_elapsed_human"] = format_elapsed_seconds(maximum)
    result["sum_elapsed_seconds"] = total_elapsed
    result["sum_elapsed_human"] = format_elapsed_seconds(total_elapsed)
    metric = str(result.get("metric") or "average_elapsed")
    if metric == "min_elapsed":
        result["value_seconds"] = minimum
        result["value_human"] = format_elapsed_seconds(minimum)
    elif metric == "max_elapsed":
        result["value_seconds"] = maximum
        result["value_human"] = format_elapsed_seconds(maximum)
    elif metric == "sum_elapsed":
        result["value_seconds"] = total_elapsed
        result["value_human"] = format_elapsed_seconds(total_elapsed)
    elif metric == "count":
        result["count"] = total_jobs
        result["value_seconds"] = total_jobs
        result["value_human"] = str(total_jobs)
    else:
        result["value_seconds"] = average
        result["value_human"] = format_elapsed_seconds(average)


def _contains_executable_payload(value: Any) -> bool:
    """Detect executable or tool-like payloads in a semantic frame.

    Inputs:
        Receives arbitrary decoded LLM JSON.

    Returns:
        True when the payload crosses the semantic/executable boundary.

    Used by:
        LLMSemanticFrameExtractor.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in DISALLOWED_FRAME_KEYS:
                return True
            if _contains_executable_payload(nested):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_executable_payload(item) for item in value)
    if isinstance(value, str):
        return bool(DISALLOWED_FRAME_TEXT_RE.search(value))
    return False


def _semantic_validation_facts(error: ValidationError) -> list[dict[str, Any]]:
    """Convert Pydantic validation details into compact safe repair facts.

    Inputs:
        Receives a Pydantic validation error from SemanticFrame parsing.

    Returns:
        A list of field/type facts without raw operational payloads.

    Used by:
        LLMSemanticFrameExtractor repair prompts and safe failure metadata.
    """
    facts: list[dict[str, Any]] = []
    for item in error.errors(include_input=False):
        location = ".".join(str(part) for part in item.get("loc", ()))
        facts.append(
            {
                "field": location,
                "error_type": item.get("type"),
                "message": str(item.get("msg") or "")[:160],
            }
        )
    return facts[:20]


def _sanitize_semantic_payload_for_repair(value: Any) -> Any:
    """Strip executable-looking keys and long strings from an invalid frame payload.

    Inputs:
        Receives arbitrary decoded LLM JSON.

    Returns:
        A JSON-compatible object safe to include in a typed repair prompt.

    Used by:
        build_semantic_frame_repair_prompt.
    """
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in DISALLOWED_FRAME_KEYS:
                continue
            sanitized[str(key)] = _sanitize_semantic_payload_for_repair(nested)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_semantic_payload_for_repair(item) for item in value[:20]]
    if isinstance(value, str):
        text = value.strip()
        if DISALLOWED_FRAME_TEXT_RE.search(text):
            return "[removed]"
        return text[:500]
    return value


def _normalize_token(value: Any) -> str:
    """Normalize a token for semantic matching.

    Inputs:
        Receives arbitrary token-like values.

    Returns:
        A lower-case underscore-separated token.

    Used by:
        Semantic frame validators and helpers.
    """
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
