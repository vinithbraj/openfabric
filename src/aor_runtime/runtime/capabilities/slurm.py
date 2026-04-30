"""OpenFABRIC Runtime Module: aor_runtime.runtime.capabilities.slurm

Purpose:
    Provide compatibility capability-pack helpers and fixtures for domain-specific tests and utilities.

Responsibilities:
    Classify or compile typed intents when called directly by tests or compatibility surfaces.

Data flow / Interfaces:
    Consumes compile contexts, allowed tools, and typed intents; returns execution-plan fragments or eval metadata.

Boundaries:
    These modules are not the active top-level natural-language planner; user prompts route through LLMActionPlanner.
"""

from __future__ import annotations

import getpass
import re
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.capabilities.base import CapabilityPack, ClassificationContext, CompileContext, CompiledIntentPlan
from aor_runtime.runtime.llm_intent_extractor import LLMIntentExtractor
from aor_runtime.runtime.intents import IntentResult
from aor_runtime.runtime.output_contract import build_output_contract
from aor_runtime.runtime.slurm_coverage import SlurmCoverageResult, validate_slurm_coverage
from aor_runtime.runtime.slurm_safety import (
    normalize_group_by,
    normalize_metric_group,
    normalize_node_state_group,
    validate_slurm_intent_safety,
)
from aor_runtime.runtime.slurm_semantics import SlurmRequest, SlurmSemanticFrame, extract_slurm_semantic_frame
from aor_runtime.tools.slurm import _validate_job_id, _validate_node_name, _validate_safe_token, _validate_time_value


SLURM_KEYWORD_RE = re.compile(r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|accounting)\b", re.IGNORECASE)
SLURM_MUTATION_RE = re.compile(
    r"\b(?:sbatch|submit(?:\s+a)?\s+job|run\s+this\s+job|scancel|cancel(?:\s+(?:all|my|the|pending|running))*\s+jobs?|drain\s+node|resume\s+node|requeue\s+job|update\s+node|change\s+partition|kill(?:\s+my)?\s+job)\b",
    re.IGNORECASE,
)
OUTPUT_JSON_RE = re.compile(r"\b(?:json|json only|as json)\b", re.IGNORECASE)
OUTPUT_CSV_RE = re.compile(r"\b(?:csv|csv only|as csv|comma separated)\b", re.IGNORECASE)
OUTPUT_COUNT_RE = re.compile(r"\b(?:count only|number only|how many|count)\b", re.IGNORECASE)
USER_FOR_RE = re.compile(r"\bfor\s+user\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
USER_RE = re.compile(r"\buser\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
PARTITION_RE = re.compile(r"\b(?:partition|in partition|on partition)\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
JOB_ID_RE = re.compile(r"\bjob\s+(\d+(?:[._][A-Za-z0-9_-]+)*)\b", re.IGNORECASE)
NODE_DETAIL_RE = re.compile(r"\bnode\s+details?\s+for\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
NODE_STATUS_FOR_RE = re.compile(r"\bnode\s+status\s+for\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
NODE_NAME_RE = re.compile(r"\bshow\s+(?:slurm\s+)?node\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
CLUSTER_ROUTE_RE = re.compile(r"\b(?:on|via|through)\s+cluster\s+([A-Za-z0-9._-]+)\b", re.IGNORECASE)
Bare_ROUTE_RE = re.compile(r"\bon\s+([A-Za-z0-9._-]+)\s*$", re.IGNORECASE)
SLURM_LLM_DOMAIN_RE = re.compile(
    r"\b(?:slurm|squeue|sacct|sinfo|scontrol|slurmdbd|scheduler|accounting|partition|partitions|node|nodes|queue|cluster|gpu|gres)\b",
    re.IGNORECASE,
)
SLURM_LLM_FUZZY_RE = re.compile(
    r"\b(?:my jobs|stuck jobs|jobs waiting|cluster health|busy cluster|gpu availability|failed recently|jobs yesterday|scheduler health|anything unhealthy|what is wrong with the cluster|queue pressure|gpus free|jobs stuck)\b",
    re.IGNORECASE,
)
SUSPICIOUS_VALUE_RE = re.compile(r"[;|&$`><\n/]")
ALLOWED_JOB_STATES = {"RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
ALLOWED_NODE_STATES = {"idle", "allocated", "mixed", "down", "drained"}
ALLOWED_NODE_STATE_GROUPS = {"idle", "allocated", "mixed", "down", "drained", "problematic", "all"}
ALLOWED_GROUP_BY = {"state", "user", "partition", "node", "job_name"}
ALLOWED_METRIC_GROUPS = {
    "cluster_summary",
    "queue_summary",
    "node_summary",
    "problematic_nodes",
    "partition_summary",
    "gpu_summary",
    "accounting_summary",
    "slurmdbd_health",
    "scheduler_health",
    "accounting_health",
}


class SlurmQueueIntent(BaseModel):
    """Represent slurm queue intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmQueueIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmQueueIntent and related tests.
    """
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    limit: int | None = 100
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmJobDetailIntent(BaseModel):
    """Represent slurm job detail intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmJobDetailIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmJobDetailIntent and related tests.
    """
    job_id: str | None = None
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    limit: int | None = 100
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmAccountingIntent(BaseModel):
    """Represent slurm accounting intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmAccountingIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmAccountingIntent and related tests.
    """
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    start: str | None = None
    end: str | None = None
    min_elapsed_seconds: int | None = None
    max_elapsed_seconds: int | None = None
    limit: int | None = 100
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmAccountingAggregateIntent(BaseModel):
    """Represent slurm accounting aggregate intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmAccountingAggregateIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmAccountingAggregateIntent and related tests.
    """
    user: str | None = None
    state: str | None = None
    include_all_states: bool = False
    excluded_states: list[str] = Field(default_factory=list)
    default_state_applied: bool = False
    partition: str | None = None
    start: str | None = None
    end: str | None = None
    metric: Literal[
        "average_elapsed",
        "min_elapsed",
        "max_elapsed",
        "sum_elapsed",
        "count",
        "count_longer_than",
        "runtime_summary",
    ] = "average_elapsed"
    group_by: Literal["partition", "user", "state", "job_name"] | None = None
    threshold_seconds: int | None = None
    limit: int | None = 1000
    gateway_node: str | None = None
    time_window_label: str | None = None
    output_mode: Literal["text", "json", "csv"] = "text"


class SlurmJobCountIntent(BaseModel):
    """Represent slurm job count intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmJobCountIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmJobCountIntent and related tests.
    """
    user: str | None = None
    state: str | None = None
    partition: str | None = None
    source: Literal["squeue", "sacct"] = "squeue"
    start: str | None = None
    end: str | None = None
    group_by: Literal["state", "user", "partition", "node"] | None = None
    gateway_node: str | None = None
    output_mode: Literal["count", "json"] = "count"


class SlurmNodeStatusIntent(BaseModel):
    """Represent slurm node status intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmNodeStatusIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmNodeStatusIntent and related tests.
    """
    node: str | None = None
    partition: str | None = None
    state: str | None = None
    state_group: Literal["idle", "allocated", "mixed", "down", "drained", "problematic", "all"] | None = None
    gpu_only: bool = False
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmNodeDetailIntent(BaseModel):
    """Represent slurm node detail intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmNodeDetailIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmNodeDetailIntent and related tests.
    """
    node: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmPartitionSummaryIntent(BaseModel):
    """Represent slurm partition summary intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmPartitionSummaryIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmPartitionSummaryIntent and related tests.
    """
    partition: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SlurmMetricsIntent(BaseModel):
    """Represent slurm metrics intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmMetricsIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmMetricsIntent and related tests.
    """
    metric_group: Literal[
        "cluster_summary",
        "queue_summary",
        "node_summary",
        "problematic_nodes",
        "partition_summary",
        "gpu_summary",
        "accounting_summary",
        "slurmdbd_health",
        "scheduler_health",
        "accounting_health",
    ] = "cluster_summary"
    start: str | None = None
    end: str | None = None
    gateway_node: str | None = None
    output_mode: Literal["text", "json"] = "json"


class SlurmDBDHealthIntent(BaseModel):
    """Represent slurm d b d health intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmDBDHealthIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmDBDHealthIntent and related tests.
    """
    gateway_node: str | None = None
    output_mode: Literal["text", "json"] = "json"


class SlurmUnsupportedMutationIntent(BaseModel):
    """Represent slurm unsupported mutation intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmUnsupportedMutationIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmUnsupportedMutationIntent and related tests.
    """
    operation: str
    reason: str


class SlurmCompoundIntent(BaseModel):
    """Represent slurm compound intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmCompoundIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmCompoundIntent and related tests.
    """
    intents: list[Any] = Field(default_factory=list)
    output_mode: Literal["text", "json"] = "json"
    return_policy: Literal["combined_summary", "all_results"] = "combined_summary"
    request_ids: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)


class SlurmFailureIntent(BaseModel):
    """Represent slurm failure intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmFailureIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmFailureIntent and related tests.
    """
    message: str
    error_type: str = "slurm_request_uncovered"
    suggestions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlurmCapabilityPack(CapabilityPack):
    """Represent slurm capability pack within the OpenFABRIC runtime. It extends CapabilityPack.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SlurmCapabilityPack.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.capabilities.slurm.SlurmCapabilityPack and related tests.
    """
    name = "slurm"
    supports_llm_intent_extraction = True
    intent_types = (
        SlurmQueueIntent,
        SlurmJobDetailIntent,
        SlurmAccountingIntent,
        SlurmAccountingAggregateIntent,
        SlurmJobCountIntent,
        SlurmNodeStatusIntent,
        SlurmNodeDetailIntent,
        SlurmPartitionSummaryIntent,
        SlurmMetricsIntent,
        SlurmDBDHealthIntent,
        SlurmUnsupportedMutationIntent,
        SlurmCompoundIntent,
        SlurmFailureIntent,
    )

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        """Classify for SlurmCapabilityPack instances.

        Inputs:
            Receives goal, context for this SlurmCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmCapabilityPack.classify calls and related tests.
        """
        prompt = str(goal or "").strip()
        if not prompt:
            return IntentResult(matched=False, reason="slurm_no_match")

        routed_prompt, gateway_node = _extract_gateway_node(prompt, context.settings.available_nodes)
        semantic_frame = extract_slurm_semantic_frame(routed_prompt)
        if semantic_frame.query_type == "unsupported_mutation" or SLURM_MUTATION_RE.search(prompt):
            operation = SLURM_MUTATION_RE.search(prompt)
            return IntentResult(
                matched=True,
                intent=SlurmUnsupportedMutationIntent(
                    operation=str(operation.group(0) if operation is not None else "mutation"),
                    reason="This runtime supports read-only SLURM inspection and metrics only.",
                ),
                metadata=_semantic_metadata(
                    semantic_frame,
                    generation_mode="unsupported",
                    coverage=None,
                    extra={
                        "planning_mode": "deterministic_intent",
                        "capability": self.name,
                        "slurm_generation_mode": "unsupported",
                    },
                ),
            )

        if _is_unsupported_runtime_metric(routed_prompt):
            metadata = _semantic_metadata(
                semantic_frame,
                generation_mode="unsupported",
                coverage=None,
                extra={"planning_mode": "deterministic_intent", "capability": self.name},
            )
            return IntentResult(
                matched=True,
                intent=SlurmFailureIntent(
                    message="Median runtime is not currently supported. Supported runtime metrics are average, min, max, total, count, count longer than a duration, and runtime summary.",
                    error_type="slurm_request_unresolved",
                    suggestions=[
                        "Show average runtime of completed jobs in the totalseg partition.",
                        "Show min and max runtime by partition for the last 7 days.",
                        "Count completed jobs longer than 2 hours on totalseg.",
                    ],
                    metadata=metadata,
                ),
                metadata=metadata,
            )

        if semantic_frame.requests:
            try:
                semantic_intent = _resolve_slurm_semantic_frame(semantic_frame, routed_prompt, gateway_node=gateway_node)
                coverage = validate_slurm_coverage(semantic_frame, semantic_intent)
                metadata = _semantic_metadata(
                    semantic_frame,
                    generation_mode="deterministic",
                    coverage=coverage,
                    intent=semantic_intent,
                    extra={
                        "planning_mode": "deterministic_intent",
                        "capability": self.name,
                        "llm_calls": 0,
                        "llm_intent_calls": 0,
                        "raw_planner_llm_calls": 0,
                    },
                )
                safety = validate_slurm_intent_safety(semantic_intent)
                if coverage.passed and safety.valid:
                    semantic_intent = _attach_compound_coverage(semantic_intent, semantic_frame, coverage)
                    return IntentResult(matched=True, intent=semantic_intent, metadata=metadata)
                if not safety.valid:
                    metadata["slurm_safety_failure"] = safety.reason
                return IntentResult(
                    matched=True,
                    intent=SlurmFailureIntent(
                        message="That SLURM request could not be covered safely by read-only SLURM intents.",
                        error_type="slurm_request_uncovered",
                        suggestions=_slurm_failure_suggestions(),
                        metadata=metadata,
                    ),
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                metadata = _semantic_metadata(
                    semantic_frame,
                    generation_mode="deterministic",
                    coverage=None,
                    extra={
                        "planning_mode": "deterministic_intent",
                        "capability": self.name,
                        "slurm_resolution_failure": str(exc),
                    },
                )
                return IntentResult(
                    matched=True,
                    intent=SlurmFailureIntent(
                        message="That SLURM request could not be resolved into safe read-only SLURM intents.",
                        error_type="slurm_request_unresolved",
                        suggestions=_slurm_failure_suggestions(),
                        metadata=metadata,
                    ),
                    metadata=metadata,
                )

        if not SLURM_KEYWORD_RE.search(prompt):
            return IntentResult(matched=False, reason="slurm_no_match")
        output_mode = _detect_output_mode(routed_prompt)
        user = _detect_user(routed_prompt)
        partition = _detect_partition(routed_prompt)
        state = _detect_job_state(routed_prompt)
        start, end = _detect_time_range(routed_prompt)

        if re.search(r"\b(?:slurmdbd\s+health|accounting\s+health)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmDBDHealthIntent(
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\b(?:cluster\s+metrics|cluster\s+summary|dashboard)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="cluster_summary",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\b(?:gpu|gres)\s+(?:availability|summary)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="gpu_summary",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\bsummar(?:ize|y)\s+slurm\s+partition\b|\bpartition\s+cpu\s+allocation\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmPartitionSummaryIntent(partition=partition, gateway_node=gateway_node, output_mode=output_mode),
            )

        if re.search(r"\bsummar(?:ize|y)\s+slurm\s+nodes?\b|\bnode\s+summary\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group="node_summary",
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        if re.search(r"\bsummar(?:ize|y)\s+(?:jobs|queue)\s+by\s+(?:state|user|partition)\b", routed_prompt, re.IGNORECASE):
            metric_group = "accounting_summary" if _looks_like_accounting_prompt(routed_prompt) else "queue_summary"
            return IntentResult(
                matched=True,
                intent=SlurmMetricsIntent(
                    metric_group=metric_group,
                    start=start,
                    end=end,
                    gateway_node=gateway_node,
                    output_mode=_detect_metrics_output_mode(routed_prompt),
                ),
            )

        node_name = _detect_node_name(routed_prompt)
        if node_name and re.search(r"\bdetails?\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmNodeDetailIntent(node=node_name, gateway_node=gateway_node, output_mode=output_mode),
            )

        job_id_match = JOB_ID_RE.search(routed_prompt)
        if job_id_match and re.search(r"\b(?:details?|show)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmJobDetailIntent(job_id=job_id_match.group(1), gateway_node=gateway_node, output_mode=output_mode),
            )

        if _looks_like_accounting_prompt(routed_prompt):
            if _detect_count_mode(routed_prompt):
                return IntentResult(
                    matched=True,
                    intent=SlurmJobCountIntent(
                        user=user,
                        state=state,
                        partition=partition,
                        source="sacct",
                        start=start,
                        end=end,
                        gateway_node=gateway_node,
                        output_mode="json" if output_mode == "json" else "count",
                    ),
                )
            return IntentResult(
                matched=True,
                intent=SlurmAccountingIntent(
                    user=user,
                    state=state,
                    partition=partition,
                    start=start,
                    end=end,
                    gateway_node=gateway_node,
                    output_mode=output_mode,
                ),
            )

        if re.search(r"\bpartitions?\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmPartitionSummaryIntent(partition=partition, gateway_node=gateway_node, output_mode=output_mode),
            )

        if re.search(r"\bnodes?\b", routed_prompt, re.IGNORECASE):
            if node_name:
                return IntentResult(
                    matched=True,
                    intent=SlurmNodeStatusIntent(
                        node=node_name,
                        partition=partition,
                        state=_detect_node_state(routed_prompt),
                        gateway_node=gateway_node,
                        output_mode=output_mode,
                    ),
                )
            return IntentResult(
                matched=True,
                intent=SlurmNodeStatusIntent(
                    partition=partition,
                    state=_detect_node_state(routed_prompt),
                    gateway_node=gateway_node,
                    output_mode=output_mode,
                ),
            )

        if _detect_count_mode(routed_prompt):
            return IntentResult(
                matched=True,
                intent=SlurmJobCountIntent(
                    user=user,
                    state=state,
                    partition=partition,
                    source="squeue",
                    gateway_node=gateway_node,
                    output_mode="json" if output_mode == "json" else "count",
                ),
            )

        if job_id_match:
            return IntentResult(
                matched=True,
                intent=SlurmJobDetailIntent(job_id=job_id_match.group(1), gateway_node=gateway_node, output_mode=output_mode),
            )

        if re.search(r"\b(?:queue|jobs?)\b", routed_prompt, re.IGNORECASE):
            return IntentResult(
                matched=True,
                intent=SlurmQueueIntent(
                    user=user,
                    state=state,
                    partition=partition,
                    gateway_node=gateway_node,
                    output_mode=output_mode,
                ),
            )

        return IntentResult(matched=False, reason="slurm_no_match")

    def is_llm_intent_domain(self, goal: str, context: ClassificationContext) -> bool:
        """Is llm intent domain for SlurmCapabilityPack instances.

        Inputs:
            Receives goal, context for this SlurmCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmCapabilityPack.is_llm_intent_domain calls and related tests.
        """
        if not context.settings.enable_llm_intent_extraction:
            return False
        prompt = str(goal or "").strip()
        if not prompt:
            return False
        if SLURM_MUTATION_RE.search(prompt):
            return True
        lower = prompt.lower()
        if SLURM_LLM_FUZZY_RE.search(prompt):
            return True
        if re.search(r"\bare\s+gpus?\s+available\b", lower):
            return True
        if re.search(r"\bwhat\s+failed\s+recently\b", lower):
            return True
        if re.search(r"\bhow\s+did\s+jobs?\s+do\s+yesterday\b", lower):
            return True
        if re.search(r"\bdo\s+the\s+gpu\s+nodes?\s+look\s+available\b", lower):
            return True
        if "scheduler" in lower:
            return True
        if "cluster" in lower and any(token in lower for token in ("busy", "health", "healthy", "unhealthy", "status")):
            return True
        if "jobs" in lower and any(token in lower for token in ("stuck", "waiting", "yesterday", "recently", "failed", "pending")):
            return True
        return SLURM_LLM_DOMAIN_RE.search(prompt) is not None and any(
            token in lower for token in ("queue", "jobs", "nodes", "partitions", "gpu", "accounting", "health", "busy")
        )

    def try_llm_extract(self, goal: str, context: ClassificationContext, extractor: LLMIntentExtractor) -> IntentResult:
        """Try llm extract for SlurmCapabilityPack instances.

        Inputs:
            Receives goal, context, extractor for this SlurmCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmCapabilityPack.try_llm_extract calls and related tests.
        """
        if not context.settings.enable_llm_intent_extraction:
            return IntentResult(matched=False, reason="slurm_llm_intent_disabled")
        prompt = str(goal or "").strip()
        if not prompt:
            return IntentResult(matched=False, reason="slurm_llm_empty_goal")

        routed_prompt, gateway_node = _extract_gateway_node(prompt, context.settings.available_nodes)
        semantic_frame = extract_slurm_semantic_frame(routed_prompt)
        mutation_match = SLURM_MUTATION_RE.search(routed_prompt)
        if mutation_match is not None:
            return IntentResult(
                matched=True,
                intent=SlurmUnsupportedMutationIntent(
                    operation=mutation_match.group(0),
                    reason="This runtime supports read-only SLURM inspection and metrics only.",
                ),
                metadata={
                    "planning_mode": "llm_intent_extractor",
                    "capability": self.name,
                    "llm_calls": 0,
                    "llm_intent_calls": 0,
                    "raw_planner_llm_calls": 0,
                    "llm_intent_reason": "Rejected mutating or administrative SLURM request before LLM extraction.",
                },
            )

        extracted = extractor.extract_intent(
            routed_prompt,
            self.name,
            [
                SlurmQueueIntent,
                SlurmJobDetailIntent,
                SlurmAccountingIntent,
                SlurmAccountingAggregateIntent,
                SlurmJobCountIntent,
                SlurmNodeStatusIntent,
                SlurmNodeDetailIntent,
                SlurmPartitionSummaryIntent,
                SlurmMetricsIntent,
                SlurmDBDHealthIntent,
                SlurmCompoundIntent,
            ],
            context={
                "system_prompt": _slurm_llm_intent_system_prompt(),
                "current_user": getpass.getuser(),
                "semantic_frame": semantic_frame.to_dict(),
                "confidence_threshold": 0.70,
                "temperature": 0.0,
            },
        )
        if not extracted.matched or extracted.intent is None:
            return IntentResult(matched=False, reason=extracted.reason or "slurm_llm_no_match")

        try:
            safe_intent = _finalize_llm_slurm_intent(
                extracted.intent,
                routed_prompt,
                gateway_node=gateway_node,
                available_nodes=context.settings.available_nodes,
            )
        except Exception as exc:  # noqa: BLE001
            return IntentResult(matched=False, reason=str(exc))

        if semantic_frame.requests:
            coverage = validate_slurm_coverage(semantic_frame, safe_intent)
            safety = validate_slurm_intent_safety(safe_intent)
            if not coverage.passed:
                return IntentResult(matched=False, reason=f"slurm_llm_intent_rejected:{coverage.reason}")
            if not safety.valid:
                return IntentResult(matched=False, reason=f"slurm_llm_intent_rejected:{safety.reason}")
            safe_intent = _attach_compound_coverage(safe_intent, semantic_frame, coverage)
        else:
            coverage = None

        return IntentResult(
            matched=True,
            intent=safe_intent,
            metadata=_semantic_metadata(
                semantic_frame,
                generation_mode="llm_intent",
                coverage=coverage,
                intent=safe_intent,
                extra={
                "planning_mode": "llm_intent_extractor",
                "capability": self.name,
                "llm_calls": 1,
                "llm_intent_calls": 1,
                "raw_planner_llm_calls": 0,
                "llm_intent_type": type(safe_intent).__name__,
                "llm_intent_confidence": extracted.confidence,
                "llm_intent_reason": extracted.reason,
                },
            ),
        )

    def compile(self, intent: Any, context: CompileContext) -> CompiledIntentPlan | None:
        """Compile for SlurmCapabilityPack instances.

        Inputs:
            Receives intent, context for this SlurmCapabilityPack method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SlurmCapabilityPack.compile calls and related tests.
        """
        if not isinstance(intent, self.intent_types):
            return None

        if isinstance(intent, SlurmUnsupportedMutationIntent):
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {
                                "id": 1,
                                "action": "runtime.return",
                                "args": {
                                    "value": f"{intent.reason} Unsupported operation: {intent.operation}.",
                                    "mode": "text",
                                    "output_contract": build_output_contract(mode="text"),
                                },
                            }
                        ]
                    }
                ),
                metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
            )

        if isinstance(intent, SlurmFailureIntent):
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate(
                    {
                        "steps": [
                            {
                                "id": 1,
                                "action": "runtime.return",
                                "args": {
                                    "value": _failure_message(intent),
                                    "mode": "text",
                                    "output_contract": build_output_contract(mode="text"),
                                },
                            }
                        ]
                    }
                ),
                metadata={
                    "capability_pack": self.name,
                    "intent_type": type(intent).__name__,
                    "slurm_error_type": intent.error_type,
                    **intent.metadata,
                },
            )

        if isinstance(intent, SlurmCompoundIntent):
            steps = _compile_slurm_compound_intent(intent, context.allowed_tools)
            return CompiledIntentPlan(
                plan=ExecutionPlan.model_validate({"steps": steps}),
                metadata={
                    "capability_pack": self.name,
                    "intent_type": type(intent).__name__,
                    **intent.coverage,
                },
            )

        steps = _compile_slurm_intent(intent, context.allowed_tools)
        return CompiledIntentPlan(
            plan=ExecutionPlan.model_validate({"steps": steps}),
            metadata={"capability_pack": self.name, "intent_type": type(intent).__name__},
        )


def _compile_slurm_intent(intent: Any, allowed_tools: list[str]) -> list[dict[str, Any]]:
    """Handle the internal compile slurm intent helper path for this module.

    Inputs:
        Receives intent, allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._compile_slurm_intent.
    """
    if isinstance(intent, SlurmQueueIntent):
        _require_tools(allowed_tools, "slurm.queue")
        return [
            {
                "id": 1,
                "action": "slurm.queue",
                "args": {
                    "user": intent.user,
                    "state": intent.state,
                    "partition": intent.partition,
                    "limit": intent.limit,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_queue_jobs",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_queue_jobs",
                collection_path="jobs",
                mode=intent.output_mode,
                wrapper_key="jobs",
            ),
        ]

    if isinstance(intent, SlurmAccountingIntent):
        _require_tools(allowed_tools, "slurm.accounting")
        return [
            {
                "id": 1,
                "action": "slurm.accounting",
                "args": {
                    "user": intent.user,
                    "state": intent.state,
                    "partition": intent.partition,
                    "start": intent.start,
                    "end": intent.end,
                    "min_elapsed_seconds": intent.min_elapsed_seconds,
                    "max_elapsed_seconds": intent.max_elapsed_seconds,
                    "limit": intent.limit,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_accounting_jobs",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_accounting_jobs",
                collection_path="jobs",
                mode=intent.output_mode,
                wrapper_key="jobs",
            ),
        ]

    if isinstance(intent, SlurmAccountingAggregateIntent):
        _require_tools(allowed_tools, "slurm.accounting_aggregate")
        mode = "json" if intent.output_mode == "json" else "text"
        return [
            {
                "id": 1,
                "action": "slurm.accounting_aggregate",
                "args": {
                    "user": intent.user,
                    "state": intent.state,
                    "include_all_states": intent.include_all_states,
                    "excluded_states": intent.excluded_states,
                    "default_state_applied": intent.default_state_applied,
                    "partition": intent.partition,
                    "start": intent.start,
                    "end": intent.end,
                    "metric": intent.metric,
                    "group_by": intent.group_by,
                    "threshold_seconds": intent.threshold_seconds,
                    "limit": intent.limit,
                    "gateway_node": intent.gateway_node,
                    "time_window_label": intent.time_window_label,
                },
                "output": "slurm_accounting_aggregate",
            },
            {
                "id": 2,
                "action": "runtime.return",
                "input": ["slurm_accounting_aggregate"],
                "args": {
                    "value": {"$ref": "slurm_accounting_aggregate"},
                    "mode": mode,
                    "output_contract": build_output_contract(mode=mode),
                },
            },
        ]

    if isinstance(intent, SlurmJobCountIntent):
        if intent.group_by:
            action = "slurm.accounting" if intent.source == "sacct" else "slurm.queue"
            _require_tools(allowed_tools, action)
            args: dict[str, Any] = {
                "user": intent.user,
                "state": intent.state,
                "partition": intent.partition,
                "group_by": intent.group_by,
                "limit": None,
                "gateway_node": intent.gateway_node,
            }
            if intent.source == "sacct":
                args.update(
                    {
                        "start": intent.start,
                        "end": intent.end,
                        "min_elapsed_seconds": None,
                        "max_elapsed_seconds": None,
                    }
                )
            return [
                {
                    "id": 1,
                    "action": action,
                    "args": args,
                    "output": "slurm_job_count_grouped",
                },
                {
                    "id": 2,
                    "action": "runtime.return",
                    "input": ["slurm_job_count_grouped"],
                    "args": {
                        "value": {"$ref": "slurm_job_count_grouped", "path": "grouped"},
                        "mode": "json",
                        "output_contract": build_output_contract(mode="json"),
                    },
                },
            ]
        action = "slurm.accounting" if intent.source == "sacct" else "slurm.queue"
        _require_tools(allowed_tools, action)
        args = {
            "user": intent.user,
            "state": intent.state,
            "partition": intent.partition,
            "limit": None,
            "gateway_node": intent.gateway_node,
        }
        if intent.source == "sacct":
            args["start"] = intent.start
            args["end"] = intent.end
            args["min_elapsed_seconds"] = None
            args["max_elapsed_seconds"] = None
        return [
            {"id": 1, "action": action, "args": args, "output": "slurm_job_count"},
            {
                "id": 2,
                "action": "runtime.return",
                "input": ["slurm_job_count"],
                "args": {
                    "value": {"$ref": "slurm_job_count", "path": "count"},
                    "mode": intent.output_mode,
                    "output_contract": build_output_contract(
                        mode=intent.output_mode,
                        json_shape="count" if intent.output_mode == "json" else None,
                    ),
                },
            },
        ]

    if isinstance(intent, SlurmJobDetailIntent):
        if not intent.job_id:
            raise ValueError("Deterministic SLURM job detail requires a job_id.")
        _require_tools(allowed_tools, "slurm.job_detail")
        if intent.output_mode == "text":
            value = {"$ref": "slurm_job_detail", "path": "raw"}
            contract = build_output_contract(mode="text")
        elif intent.output_mode == "json":
            value = {"$ref": "slurm_job_detail"}
            contract = build_output_contract(mode="json")
        else:
            value = {"$ref": "slurm_job_detail", "path": "field_rows"}
            contract = build_output_contract(mode="csv", json_shape="rows")
        return [
            {
                "id": 1,
                "action": "slurm.job_detail",
                "args": {"job_id": intent.job_id, "gateway_node": intent.gateway_node},
                "output": "slurm_job_detail",
            },
            {"id": 2, "action": "runtime.return", "input": ["slurm_job_detail"], "args": {"value": value, "mode": intent.output_mode, "output_contract": contract}},
        ]

    if isinstance(intent, SlurmNodeStatusIntent):
        _require_tools(allowed_tools, "slurm.nodes")
        return [
            {
                "id": 1,
                "action": "slurm.nodes",
                "args": {
                    "node": intent.node,
                    "partition": intent.partition,
                    "state": intent.state,
                    "state_group": intent.state_group,
                    "gpu_only": intent.gpu_only,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_nodes",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_nodes",
                collection_path="nodes",
                mode=intent.output_mode,
                wrapper_key="nodes",
            ),
        ]

    if isinstance(intent, SlurmNodeDetailIntent):
        if not intent.node:
            raise ValueError("Deterministic SLURM node detail requires a node name.")
        _require_tools(allowed_tools, "slurm.node_detail")
        if intent.output_mode == "text":
            value = {"$ref": "slurm_node_detail", "path": "raw"}
            contract = build_output_contract(mode="text")
        elif intent.output_mode == "json":
            value = {"$ref": "slurm_node_detail"}
            contract = build_output_contract(mode="json")
        else:
            value = {"$ref": "slurm_node_detail", "path": "field_rows"}
            contract = build_output_contract(mode="csv", json_shape="rows")
        return [
            {
                "id": 1,
                "action": "slurm.node_detail",
                "args": {"node": intent.node, "gateway_node": intent.gateway_node},
                "output": "slurm_node_detail",
            },
            {"id": 2, "action": "runtime.return", "input": ["slurm_node_detail"], "args": {"value": value, "mode": intent.output_mode, "output_contract": contract}},
        ]

    if isinstance(intent, SlurmPartitionSummaryIntent):
        _require_tools(allowed_tools, "slurm.partitions")
        return [
            {
                "id": 1,
                "action": "slurm.partitions",
                "args": {"partition": intent.partition, "gateway_node": intent.gateway_node},
                "output": "slurm_partitions",
            },
            _rows_return_step(
                step_id=2,
                alias="slurm_partitions",
                collection_path="partitions",
                mode=intent.output_mode,
                wrapper_key="partitions",
            ),
        ]

    if isinstance(intent, SlurmDBDHealthIntent):
        _require_tools(allowed_tools, "slurm.slurmdbd_health")
        value = {"$ref": "slurm_health"} if intent.output_mode == "json" else {"$ref": "slurm_health", "path": "text_lines"}
        return [
            {"id": 1, "action": "slurm.slurmdbd_health", "args": {"gateway_node": intent.gateway_node}, "output": "slurm_health"},
            {
                "id": 2,
                "action": "runtime.return",
                "input": ["slurm_health"],
                "args": {
                    "value": value,
                    "mode": intent.output_mode,
                    "output_contract": build_output_contract(mode=intent.output_mode),
                },
            },
        ]

    if isinstance(intent, SlurmMetricsIntent):
        if intent.metric_group in {"slurmdbd_health", "accounting_health"}:
            _require_tools(allowed_tools, "slurm.slurmdbd_health")
            value = {"$ref": "slurm_health"} if intent.output_mode == "json" else {"$ref": "slurm_health", "path": "text_lines"}
            return [
                {"id": 1, "action": "slurm.slurmdbd_health", "args": {"gateway_node": intent.gateway_node}, "output": "slurm_health"},
                {
                    "id": 2,
                    "action": "runtime.return",
                    "input": ["slurm_health"],
                    "args": {
                        "value": value,
                        "mode": intent.output_mode,
                        "output_contract": build_output_contract(mode=intent.output_mode),
                    },
                },
            ]
        _require_tools(allowed_tools, "slurm.metrics")
        value = {"$ref": "slurm_metrics", "path": "payload"} if intent.output_mode == "json" else {"$ref": "slurm_metrics", "path": "text_lines"}
        return [
            {
                "id": 1,
                "action": "slurm.metrics",
                "args": {
                    "metric_group": intent.metric_group,
                    "start": intent.start,
                    "end": intent.end,
                    "gateway_node": intent.gateway_node,
                },
                "output": "slurm_metrics",
            },
            {
                "id": 2,
                "action": "runtime.return",
                "input": ["slurm_metrics"],
                "args": {
                    "value": value,
                    "mode": intent.output_mode,
                    "output_contract": build_output_contract(mode=intent.output_mode),
                },
            },
        ]

    raise ValueError(f"Unsupported SLURM intent: {type(intent).__name__}")


def _compile_slurm_compound_intent(intent: SlurmCompoundIntent, allowed_tools: list[str]) -> list[dict[str, Any]]:
    """Handle the internal compile slurm compound intent helper path for this module.

    Inputs:
        Receives intent, allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._compile_slurm_compound_intent.
    """
    steps: list[dict[str, Any]] = []
    inputs: list[str] = []
    result_refs: dict[str, Any] = {}
    used_labels: set[str] = set()
    for index, child in enumerate(intent.intents, start=1):
        alias = _compound_alias(child, index, used_labels)
        used_labels.add(alias)
        steps.append(_compile_slurm_tool_step(child, step_id=index, alias=alias, allowed_tools=allowed_tools))
        inputs.append(alias)
        result_refs[alias] = {"$ref": alias}

    steps.append(
        {
            "id": len(steps) + 1,
            "action": "runtime.return",
            "input": inputs,
            "args": {
                "value": {
                    "summary": {
                        "request_count": len(intent.request_ids) or len(intent.intents),
                        "tool_count": len(intent.intents),
                    },
                    "results": result_refs,
                    "coverage": intent.coverage,
                },
                "mode": intent.output_mode,
                "output_contract": build_output_contract(mode=intent.output_mode),
            },
        }
    )
    return steps


def _compile_slurm_tool_step(intent: Any, *, step_id: int, alias: str, allowed_tools: list[str]) -> dict[str, Any]:
    """Handle the internal compile slurm tool step helper path for this module.

    Inputs:
        Receives intent, step_id, alias, allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._compile_slurm_tool_step.
    """
    if isinstance(intent, SlurmQueueIntent):
        _require_tools(allowed_tools, "slurm.queue")
        return {
            "id": step_id,
            "action": "slurm.queue",
            "args": {
                "user": intent.user,
                "state": intent.state,
                "partition": intent.partition,
                "limit": intent.limit,
                "gateway_node": intent.gateway_node,
            },
            "output": alias,
        }
    if isinstance(intent, SlurmAccountingIntent):
        _require_tools(allowed_tools, "slurm.accounting")
        return {
            "id": step_id,
            "action": "slurm.accounting",
            "args": {
                "user": intent.user,
                "state": intent.state,
                "partition": intent.partition,
                "start": intent.start,
                "end": intent.end,
                "min_elapsed_seconds": intent.min_elapsed_seconds,
                "max_elapsed_seconds": intent.max_elapsed_seconds,
                "limit": intent.limit,
                "gateway_node": intent.gateway_node,
            },
            "output": alias,
        }
    if isinstance(intent, SlurmAccountingAggregateIntent):
        _require_tools(allowed_tools, "slurm.accounting_aggregate")
        return {
            "id": step_id,
            "action": "slurm.accounting_aggregate",
            "args": {
                "user": intent.user,
                "state": intent.state,
                "include_all_states": intent.include_all_states,
                "excluded_states": intent.excluded_states,
                "default_state_applied": intent.default_state_applied,
                "partition": intent.partition,
                "start": intent.start,
                "end": intent.end,
                "metric": intent.metric,
                "group_by": intent.group_by,
                "threshold_seconds": intent.threshold_seconds,
                "limit": intent.limit,
                "gateway_node": intent.gateway_node,
                "time_window_label": intent.time_window_label,
            },
            "output": alias,
        }
    if isinstance(intent, SlurmJobCountIntent):
        if intent.group_by:
            action = "slurm.accounting" if intent.source == "sacct" else "slurm.queue"
            _require_tools(allowed_tools, action)
            args: dict[str, Any] = {
                "user": intent.user,
                "state": intent.state,
                "partition": intent.partition,
                "group_by": intent.group_by,
                "limit": None,
                "gateway_node": intent.gateway_node,
            }
            if intent.source == "sacct":
                args.update({"start": intent.start, "end": intent.end, "min_elapsed_seconds": None, "max_elapsed_seconds": None})
            return {
                "id": step_id,
                "action": action,
                "args": args,
                "output": alias,
            }
        action = "slurm.accounting" if intent.source == "sacct" else "slurm.queue"
        _require_tools(allowed_tools, action)
        args: dict[str, Any] = {
            "user": intent.user,
            "state": intent.state,
            "partition": intent.partition,
            "limit": None,
            "gateway_node": intent.gateway_node,
        }
        if intent.source == "sacct":
            args.update({"start": intent.start, "end": intent.end, "min_elapsed_seconds": None, "max_elapsed_seconds": None})
        return {"id": step_id, "action": action, "args": args, "output": alias}
    if isinstance(intent, SlurmJobDetailIntent):
        if not intent.job_id:
            raise ValueError("SLURM job detail requires a job_id.")
        _require_tools(allowed_tools, "slurm.job_detail")
        return {"id": step_id, "action": "slurm.job_detail", "args": {"job_id": intent.job_id, "gateway_node": intent.gateway_node}, "output": alias}
    if isinstance(intent, SlurmNodeStatusIntent):
        _require_tools(allowed_tools, "slurm.nodes")
        return {
            "id": step_id,
            "action": "slurm.nodes",
            "args": {
                "node": intent.node,
                "partition": intent.partition,
                "state": intent.state,
                "state_group": intent.state_group,
                "gpu_only": intent.gpu_only,
                "gateway_node": intent.gateway_node,
            },
            "output": alias,
        }
    if isinstance(intent, SlurmNodeDetailIntent):
        if not intent.node:
            raise ValueError("SLURM node detail requires a node name.")
        _require_tools(allowed_tools, "slurm.node_detail")
        return {"id": step_id, "action": "slurm.node_detail", "args": {"node": intent.node, "gateway_node": intent.gateway_node}, "output": alias}
    if isinstance(intent, SlurmPartitionSummaryIntent):
        _require_tools(allowed_tools, "slurm.partitions")
        return {"id": step_id, "action": "slurm.partitions", "args": {"partition": intent.partition, "gateway_node": intent.gateway_node}, "output": alias}
    if isinstance(intent, SlurmDBDHealthIntent):
        _require_tools(allowed_tools, "slurm.slurmdbd_health")
        return {"id": step_id, "action": "slurm.slurmdbd_health", "args": {"gateway_node": intent.gateway_node}, "output": alias}
    if isinstance(intent, SlurmMetricsIntent):
        if intent.metric_group in {"slurmdbd_health", "accounting_health"}:
            _require_tools(allowed_tools, "slurm.slurmdbd_health")
            return {"id": step_id, "action": "slurm.slurmdbd_health", "args": {"gateway_node": intent.gateway_node}, "output": alias}
        _require_tools(allowed_tools, "slurm.metrics")
        return {
            "id": step_id,
            "action": "slurm.metrics",
            "args": {
                "metric_group": intent.metric_group,
                "start": intent.start,
                "end": intent.end,
                "gateway_node": intent.gateway_node,
            },
            "output": alias,
        }
    raise ValueError(f"Unsupported SLURM compound child intent: {type(intent).__name__}")


def _compound_alias(intent: Any, index: int, used_labels: set[str]) -> str:
    """Handle the internal compound alias helper path for this module.

    Inputs:
        Receives intent, index, used_labels for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._compound_alias.
    """
    base = "slurm_result"
    if isinstance(intent, SlurmJobCountIntent):
        state = str(intent.state or "jobs").lower()
        base = f"{state}_job_count" if not intent.group_by else f"jobs_by_{intent.group_by}"
    elif isinstance(intent, SlurmQueueIntent):
        state = str(intent.state or "queue").lower()
        base = f"{state}_jobs"
    elif isinstance(intent, SlurmNodeStatusIntent):
        base = "problematic_nodes" if intent.state_group == "problematic" else "nodes"
    elif isinstance(intent, SlurmPartitionSummaryIntent):
        base = "partitions"
    elif isinstance(intent, SlurmMetricsIntent):
        base = str(intent.metric_group)
    elif isinstance(intent, SlurmDBDHealthIntent):
        base = "slurmdbd_health"
    elif isinstance(intent, SlurmAccountingAggregateIntent):
        base = "accounting_runtime"
    elif isinstance(intent, SlurmAccountingIntent):
        base = "accounting_jobs"
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or f"slurm_result_{index}"
    if candidate not in used_labels:
        return candidate
    return f"{candidate}_{index}"


def _rows_return_step(*, step_id: int, alias: str, collection_path: str, mode: str, wrapper_key: str) -> dict[str, Any]:
    """Handle the internal rows return step helper path for this module.

    Inputs:
        Receives step_id, alias, collection_path, mode, wrapper_key for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._rows_return_step.
    """
    if mode == "json":
        value = {wrapper_key: {"$ref": alias, "path": collection_path}}
        contract = build_output_contract(mode="json")
    else:
        value = {"$ref": alias, "path": collection_path}
        contract = build_output_contract(mode=mode, json_shape="rows")
    return {
        "id": step_id,
        "action": "runtime.return",
        "input": [alias],
        "args": {
            "value": value,
            "mode": mode,
            "output_contract": contract,
        },
    }


def _resolve_slurm_semantic_frame(frame: SlurmSemanticFrame, prompt: str, *, gateway_node: str | None) -> Any:
    """Handle the internal resolve slurm semantic frame helper path for this module.

    Inputs:
        Receives frame, prompt, gateway_node for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._resolve_slurm_semantic_frame.
    """
    intents: list[Any] = []
    for request in frame.requests:
        intents.extend(_intents_for_slurm_request(request, prompt, gateway_node=gateway_node, frame_output_mode=frame.output_mode))
    intents = _dedupe_slurm_intents(intents)
    if not intents:
        raise ValueError("No supported read-only SLURM intent covered the semantic frame.")
    if len(intents) == 1 and len(frame.requests) == 1 and frame.query_type != "compound":
        return intents[0]
    compound_output_mode: Literal["text", "json"] = "json" if frame.output_mode != "text" or len(frame.requests) > 1 or len(intents) > 1 else "text"
    return SlurmCompoundIntent(
        intents=intents,
        output_mode=compound_output_mode,
        return_policy="combined_summary",
        request_ids=[request.id for request in frame.requests],
    )


def _intents_for_slurm_request(
    request: SlurmRequest,
    prompt: str,
    *,
    gateway_node: str | None,
    frame_output_mode: str,
) -> list[Any]:
    """Handle the internal intents for slurm request helper path for this module.

    Inputs:
        Receives request, prompt, gateway_node, frame_output_mode for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._intents_for_slurm_request.
    """
    filters = dict(request.filters or {})
    output_mode = _output_mode_for_request(request, frame_output_mode)
    metrics_output = "json" if output_mode == "json" else "text"

    if request.kind == "cluster_health":
        return [
            SlurmMetricsIntent(metric_group="cluster_summary", gateway_node=gateway_node, output_mode="json"),
            SlurmMetricsIntent(metric_group="partition_summary", gateway_node=gateway_node, output_mode="json"),
            SlurmMetricsIntent(metric_group="gpu_summary", gateway_node=gateway_node, output_mode="json"),
            SlurmDBDHealthIntent(gateway_node=gateway_node, output_mode="json"),
        ]
    if request.kind == "queue_status":
        return [SlurmMetricsIntent(metric_group="queue_summary", gateway_node=gateway_node, output_mode=metrics_output)]
    if request.kind == "scheduler_health":
        return [SlurmMetricsIntent(metric_group="scheduler_health", gateway_node=gateway_node, output_mode=metrics_output)]
    if request.kind == "gpu_availability":
        return [SlurmMetricsIntent(metric_group="gpu_summary", gateway_node=gateway_node, output_mode="json")]
    if request.kind == "partition_status":
        return [SlurmPartitionSummaryIntent(partition=filters.get("partition"), gateway_node=gateway_node, output_mode=output_mode if output_mode in {"text", "csv", "json"} else "json")]
    if request.kind == "problematic_nodes":
        return [SlurmNodeStatusIntent(state_group="problematic", gateway_node=gateway_node, output_mode=output_mode if output_mode in {"text", "csv", "json"} else "json")]
    if request.kind == "node_status":
        state_group = filters.get("state_group")
        state = filters.get("state")
        if state_group in {"idle", "allocated", "mixed", "down", "drained"}:
            state = state_group
            state_group = None
        return [
            SlurmNodeStatusIntent(
                node=filters.get("node"),
                partition=filters.get("partition"),
                state=state,
                state_group=state_group,
                gateway_node=gateway_node,
                output_mode=output_mode if output_mode in {"text", "csv", "json"} else "json",
            )
        ]
    if request.kind == "job_count":
        source: Literal["squeue", "sacct"] = "sacct" if any(filters.get(key) for key in ("start", "end", "min_elapsed_seconds", "max_elapsed_seconds")) else "squeue"
        return [
            SlurmJobCountIntent(
                user=filters.get("user"),
                state=filters.get("state"),
                partition=filters.get("partition"),
                source=source,
                start=filters.get("start"),
                end=filters.get("end"),
                group_by=filters.get("group_by"),
                gateway_node=gateway_node,
                output_mode="json" if output_mode == "json" or filters.get("group_by") else "count",
            )
        ]
    if request.kind == "job_listing":
        return [
            SlurmQueueIntent(
                user=filters.get("user"),
                state=filters.get("state"),
                partition=filters.get("partition"),
                limit=filters.get("limit", 100),
                gateway_node=gateway_node,
                output_mode=output_mode if output_mode in {"text", "csv", "json"} else "json",
            )
        ]
    if request.kind == "accounting_jobs":
        if request.output == "count":
            return [
                SlurmJobCountIntent(
                    user=filters.get("user"),
                    state=filters.get("state"),
                    partition=filters.get("partition"),
                    source="sacct",
                    start=filters.get("start"),
                    end=filters.get("end"),
                    group_by=filters.get("group_by"),
                    gateway_node=gateway_node,
                    output_mode="json" if output_mode == "json" or filters.get("group_by") else "count",
                )
            ]
        return [
            SlurmAccountingIntent(
                user=filters.get("user"),
                state=filters.get("state"),
                partition=filters.get("partition"),
                start=filters.get("start"),
                end=filters.get("end"),
                min_elapsed_seconds=filters.get("min_elapsed_seconds"),
                max_elapsed_seconds=filters.get("max_elapsed_seconds"),
                limit=filters.get("limit", 100),
                gateway_node=gateway_node,
                output_mode=output_mode if output_mode in {"text", "csv", "json"} else "json",
            )
        ]
    if request.kind == "accounting_aggregate":
        include_all_states = bool(filters.get("include_all_states") or filters.get("all_states"))
        aggregate_state = None if include_all_states else filters.get("state")
        return [
            SlurmAccountingAggregateIntent(
                user=filters.get("user"),
                state=aggregate_state,
                include_all_states=include_all_states,
                excluded_states=list(filters.get("excluded_states") or []),
                default_state_applied=bool(filters.get("default_state_applied") or False),
                partition=filters.get("partition"),
                start=filters.get("start"),
                end=filters.get("end"),
                metric=filters.get("metric", "average_elapsed"),
                group_by=filters.get("group_by"),
                threshold_seconds=filters.get("threshold_seconds"),
                limit=filters.get("limit", 1000),
                gateway_node=gateway_node,
                time_window_label=filters.get("time_window_label"),
                output_mode=output_mode if output_mode in {"text", "json", "csv"} else "text",
            )
        ]
    if request.kind == "job_detail":
        return [SlurmJobDetailIntent(job_id=filters.get("job_id"), gateway_node=gateway_node, output_mode=output_mode if output_mode in {"text", "csv", "json"} else "json")]
    if request.kind in {"slurmdbd_health", "accounting_health"}:
        return [SlurmDBDHealthIntent(gateway_node=gateway_node, output_mode="json" if output_mode == "json" else "text")]
    if request.kind == "resource_summary":
        return [SlurmMetricsIntent(metric_group="cluster_summary", gateway_node=gateway_node, output_mode=metrics_output)]
    raise ValueError(f"Unsupported SLURM request kind: {request.kind}")


def _output_mode_for_request(request: SlurmRequest, frame_output_mode: str) -> str:
    """Handle the internal output mode for request helper path for this module.

    Inputs:
        Receives request, frame_output_mode for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._output_mode_for_request.
    """
    if frame_output_mode in {"json", "csv", "count"}:
        return frame_output_mode
    if request.output == "json":
        return "json"
    if request.output == "count":
        return "count"
    return "text"


def _dedupe_slurm_intents(intents: list[Any]) -> list[Any]:
    """Handle the internal dedupe slurm intents helper path for this module.

    Inputs:
        Receives intents for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._dedupe_slurm_intents.
    """
    deduped: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for intent in intents:
        payload = intent.model_dump(exclude_none=True) if hasattr(intent, "model_dump") else {}
        key = (type(intent).__name__, str(sorted(payload.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(intent)
    return deduped


def _attach_compound_coverage(intent: Any, frame: SlurmSemanticFrame, coverage: SlurmCoverageResult) -> Any:
    """Handle the internal attach compound coverage helper path for this module.

    Inputs:
        Receives intent, frame, coverage for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._attach_compound_coverage.
    """
    coverage_payload = _semantic_metadata(frame, generation_mode="deterministic", coverage=coverage, intent=intent)
    if isinstance(intent, SlurmCompoundIntent):
        return intent.model_copy(update={"coverage": coverage_payload})
    return intent


def _semantic_metadata(
    frame: SlurmSemanticFrame,
    *,
    generation_mode: str,
    coverage: SlurmCoverageResult | None,
    intent: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Handle the internal semantic metadata helper path for this module.

    Inputs:
        Receives frame, generation_mode, coverage, intent, extra for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._semantic_metadata.
    """
    covered_requests = list(coverage.covered_requests) if coverage else []
    missing_requests = [request.id for request in coverage.missing_requests] if coverage else []
    covered_constraints = list(coverage.covered_constraints) if coverage else []
    missing_constraints = [constraint.id for constraint in coverage.missing_constraints] if coverage else []
    metadata = {
        "slurm_semantic_frame": frame.to_dict(),
        "slurm_requests_extracted": [request.to_dict() for request in frame.requests],
        "slurm_requests_covered": covered_requests,
        "slurm_requests_missing": missing_requests,
        "slurm_constraints_extracted": [constraint.to_dict() for constraint in frame.constraints],
        "slurm_constraints_covered": covered_constraints,
        "slurm_constraints_missing": missing_constraints,
        "slurm_coverage_passed": bool(coverage.passed) if coverage else False,
        "slurm_coverage_reason": coverage.reason if coverage else "",
        "slurm_generation_mode": generation_mode,
        "slurm_tools_used": _slurm_tools_for_intent(intent) if intent is not None else [],
        "slurm_compound_children": len(getattr(intent, "intents", []) or []) if intent is not None else 0,
        "slurm_metric_groups": _slurm_metric_groups_for_intent(intent) if intent is not None else [],
        "raw_planner_llm_calls": 0,
    }
    metadata.update(extra or {})
    return metadata


def _slurm_tools_for_intent(intent: Any | None) -> list[str]:
    """Handle the internal slurm tools for intent helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._slurm_tools_for_intent.
    """
    if intent is None:
        return []
    if isinstance(intent, SlurmCompoundIntent):
        tools: list[str] = []
        for child in intent.intents:
            tools.extend(_slurm_tools_for_intent(child))
        return sorted(set(tools))
    if isinstance(intent, SlurmQueueIntent):
        return ["slurm.queue"]
    if isinstance(intent, SlurmAccountingIntent):
        return ["slurm.accounting"]
    if isinstance(intent, SlurmAccountingAggregateIntent):
        return ["slurm.accounting_aggregate"]
    if isinstance(intent, SlurmJobCountIntent):
        return ["slurm.accounting"] if intent.source == "sacct" else ["slurm.queue"]
    if isinstance(intent, SlurmJobDetailIntent):
        return ["slurm.job_detail"]
    if isinstance(intent, SlurmNodeStatusIntent):
        return ["slurm.nodes"]
    if isinstance(intent, SlurmNodeDetailIntent):
        return ["slurm.node_detail"]
    if isinstance(intent, SlurmPartitionSummaryIntent):
        return ["slurm.partitions"]
    if isinstance(intent, SlurmDBDHealthIntent):
        return ["slurm.slurmdbd_health"]
    if isinstance(intent, SlurmMetricsIntent):
        return ["slurm.slurmdbd_health"] if intent.metric_group in {"slurmdbd_health", "accounting_health"} else ["slurm.metrics"]
    return []


def _slurm_metric_groups_for_intent(intent: Any | None) -> list[str]:
    """Handle the internal slurm metric groups for intent helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._slurm_metric_groups_for_intent.
    """
    if intent is None:
        return []
    if isinstance(intent, SlurmCompoundIntent):
        groups: list[str] = []
        for child in intent.intents:
            groups.extend(_slurm_metric_groups_for_intent(child))
        return sorted(set(groups))
    if isinstance(intent, SlurmMetricsIntent):
        return [intent.metric_group]
    return []


def _failure_message(intent: SlurmFailureIntent) -> str:
    """Handle the internal failure message helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._failure_message.
    """
    lines = [intent.message]
    if intent.suggestions:
        lines.append("")
        lines.append("Suggested prompts:")
        for index, suggestion in enumerate(intent.suggestions, start=1):
            lines.append(f"{index}. {suggestion}")
    return "\n".join(lines)


def _slurm_failure_suggestions() -> list[str]:
    """Handle the internal slurm failure suggestions helper path for this module.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._slurm_failure_suggestions.
    """
    return [
        "Show SLURM queue as JSON.",
        "Count running and pending SLURM jobs.",
        "Show problematic SLURM nodes.",
        "Summarize queue, node, and GPU status.",
        "Check SLURMDBD health.",
    ]


def _slurm_llm_intent_system_prompt() -> str:
    """Handle the internal slurm llm intent system prompt helper path for this module.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._slurm_llm_intent_system_prompt.
    """
    return """You convert user requests into one typed SLURM read-only intent.
You must output JSON only.
You may only choose one of the allowed intent types.
You must not create shell commands.
You must not create slurm commands.
You must not create tool calls.
You must not create execution plans.
You must not use python.
Only read-only inspection and metrics are supported.
Mutation/admin operations are unsupported:
- sbatch
- scancel
- scontrol update
- drain
- resume
- requeue
- kill job
- submit job
- change partition
If the user asks for mutation/admin action, return matched=false or an unsupported reason.
If the request is ambiguous, choose a safe inspection intent or return matched=false.
Prefer SlurmMetricsIntent for broad health, busy, scheduler, availability, and summary questions.
Use SlurmCompoundIntent when the user asks for multiple SLURM facts in one request.
SlurmCompoundIntent children must be typed intent objects with intent_type and arguments, never commands.
Use the exact JSON keys matched, intent_type, confidence, arguments, reason."""


def _finalize_llm_slurm_intent(
    intent: Any,
    prompt: str,
    *,
    gateway_node: str | None,
    available_nodes: list[str],
) -> Any:
    """Handle the internal finalize llm slurm intent helper path for this module.

    Inputs:
        Receives intent, prompt, gateway_node, available_nodes for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._finalize_llm_slurm_intent.
    """
    if isinstance(intent, SlurmCompoundIntent):
        finalized_children = [
            _finalize_llm_slurm_intent(
                _coerce_slurm_child_intent(child),
                prompt,
                gateway_node=gateway_node,
                available_nodes=available_nodes,
            )
            for child in intent.intents
        ]
        updated_compound = intent.model_copy(update={"intents": finalized_children})
        return _validate_llm_slurm_intent(updated_compound)

    updated: dict[str, Any] = {}
    prompt_user = _detect_user(prompt)
    prompt_start, prompt_end = _detect_time_range(prompt)

    if hasattr(intent, "gateway_node") and getattr(intent, "gateway_node", None) is None and gateway_node is not None:
        updated["gateway_node"] = gateway_node
    if hasattr(intent, "user") and getattr(intent, "user", None) is None and prompt_user is not None:
        updated["user"] = prompt_user
    if hasattr(intent, "start") and getattr(intent, "start", None) is None and prompt_start is not None:
        updated["start"] = prompt_start
    if hasattr(intent, "end") and getattr(intent, "end", None) is None and prompt_end is not None:
        updated["end"] = prompt_end
    if isinstance(intent, SlurmQueueIntent) and getattr(intent, "state", None) is None and re.search(r"\bstuck\b|\bwaiting\b", prompt, re.IGNORECASE):
        updated["state"] = "PENDING"
    if isinstance(intent, SlurmAccountingIntent) and getattr(intent, "state", None) is None and re.search(r"\bfailed\b", prompt, re.IGNORECASE):
        updated["state"] = "FAILED"
    if updated:
        intent = intent.model_copy(update=updated)
    validated_intent = _validate_llm_slurm_intent(intent)
    intent_gateway_node = getattr(validated_intent, "gateway_node", None)
    if intent_gateway_node is not None and intent_gateway_node not in set(available_nodes):
        raise ValueError(f"SLURM gateway_node is not available: {intent_gateway_node}")
    return validated_intent


def _validate_llm_slurm_intent(intent: Any) -> Any:
    """Handle the internal validate llm slurm intent helper path for this module.

    Inputs:
        Receives intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._validate_llm_slurm_intent.
    """
    _reject_suspicious_strings(intent.model_dump())

    if isinstance(intent, SlurmCompoundIntent):
        children = [_validate_llm_slurm_intent(_coerce_slurm_child_intent(child)) for child in intent.intents]
        return intent.model_copy(update={"intents": children})

    if isinstance(intent, SlurmQueueIntent):
        state = _validate_job_state(intent.state)
        partition = _validate_safe_token(intent.partition, field_name="partition")
        user = _validate_safe_token(intent.user, field_name="user")
        gateway_node = _validate_safe_token(intent.gateway_node, field_name="gateway_node")
        return intent.model_copy(update={"state": state, "partition": partition, "user": user, "gateway_node": gateway_node})

    if isinstance(intent, SlurmJobDetailIntent):
        if intent.job_id is None:
            raise ValueError("SLURM job detail intent requires a job_id.")
        return intent.model_copy(
            update={
                "job_id": _validate_job_id(intent.job_id),
                "user": _validate_safe_token(intent.user, field_name="user"),
                "state": _validate_job_state(intent.state),
                "partition": _validate_safe_token(intent.partition, field_name="partition"),
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmAccountingIntent):
        return intent.model_copy(
            update={
                "user": _validate_safe_token(intent.user, field_name="user"),
                "state": _validate_job_state(intent.state),
                "partition": _validate_safe_token(intent.partition, field_name="partition"),
                "start": _validate_time_value(intent.start, field_name="start"),
                "end": _validate_time_value(intent.end, field_name="end"),
                "min_elapsed_seconds": intent.min_elapsed_seconds,
                "max_elapsed_seconds": intent.max_elapsed_seconds,
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmAccountingAggregateIntent):
        normalized_include_all = bool(intent.include_all_states)
        return intent.model_copy(
            update={
                "user": _validate_safe_token(intent.user, field_name="user"),
                "state": None if normalized_include_all else _validate_job_state(intent.state),
                "include_all_states": normalized_include_all,
                "excluded_states": [_validate_job_state(state) for state in list(intent.excluded_states or [])],
                "default_state_applied": bool(intent.default_state_applied and not normalized_include_all),
                "partition": _validate_safe_token(intent.partition, field_name="partition"),
                "start": _validate_time_value(intent.start, field_name="start"),
                "end": _validate_time_value(intent.end, field_name="end"),
                "group_by": normalize_group_by(intent.group_by),
                "threshold_seconds": intent.threshold_seconds,
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmJobCountIntent):
        return intent.model_copy(
            update={
                "user": _validate_safe_token(intent.user, field_name="user"),
                "state": _validate_job_state(intent.state),
                "partition": _validate_safe_token(intent.partition, field_name="partition"),
                "start": _validate_time_value(intent.start, field_name="start"),
                "end": _validate_time_value(intent.end, field_name="end"),
                "group_by": normalize_group_by(intent.group_by),
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmNodeStatusIntent):
        return intent.model_copy(
            update={
                "node": _validate_node_name(intent.node) if intent.node is not None else None,
                "partition": _validate_safe_token(intent.partition, field_name="partition"),
                "state": _validate_node_state(intent.state),
                "state_group": normalize_node_state_group(intent.state_group),
                "gpu_only": bool(intent.gpu_only),
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmNodeDetailIntent):
        if intent.node is None:
            raise ValueError("SLURM node detail intent requires a node name.")
        return intent.model_copy(
            update={
                "node": _validate_node_name(intent.node),
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmPartitionSummaryIntent):
        return intent.model_copy(
            update={
                "partition": _validate_safe_token(intent.partition, field_name="partition"),
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmMetricsIntent):
        metric_group = normalize_metric_group(intent.metric_group)
        return intent.model_copy(
            update={
                "metric_group": metric_group,
                "start": _validate_time_value(intent.start, field_name="start"),
                "end": _validate_time_value(intent.end, field_name="end"),
                "gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node"),
            }
        )

    if isinstance(intent, SlurmDBDHealthIntent):
        return intent.model_copy(update={"gateway_node": _validate_safe_token(intent.gateway_node, field_name="gateway_node")})

    raise ValueError(f"Unsupported SLURM LLM intent: {type(intent).__name__}")


def _coerce_slurm_child_intent(child: Any) -> Any:
    """Handle the internal coerce slurm child intent helper path for this module.

    Inputs:
        Receives child for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._coerce_slurm_child_intent.
    """
    if isinstance(child, self_intent_types()):
        return child
    if not isinstance(child, dict):
        raise ValueError("SLURM compound children must be typed intent objects.")
    intent_type = str(child.get("intent_type") or child.get("type") or "").strip()
    arguments = child.get("arguments") if isinstance(child.get("arguments"), dict) else {
        key: value for key, value in child.items() if key not in {"intent_type", "type"}
    }
    model_lookup = {model.__name__: model for model in self_intent_types() if model is not SlurmCompoundIntent and model is not SlurmFailureIntent}
    if intent_type not in model_lookup:
        raise ValueError(f"Unsupported SLURM compound child intent type: {intent_type}")
    return model_lookup[intent_type].model_validate(arguments)


def self_intent_types() -> tuple[type, ...]:
    """Self intent types for the surrounding runtime workflow.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm.self_intent_types.
    """
    return (
        SlurmQueueIntent,
        SlurmJobDetailIntent,
        SlurmAccountingIntent,
        SlurmAccountingAggregateIntent,
        SlurmJobCountIntent,
        SlurmNodeStatusIntent,
        SlurmNodeDetailIntent,
        SlurmPartitionSummaryIntent,
        SlurmMetricsIntent,
        SlurmDBDHealthIntent,
        SlurmCompoundIntent,
        SlurmUnsupportedMutationIntent,
        SlurmFailureIntent,
    )


def _reject_suspicious_strings(value: Any) -> None:
    """Handle the internal reject suspicious strings helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._reject_suspicious_strings.
    """
    if isinstance(value, dict):
        for item in value.values():
            _reject_suspicious_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_suspicious_strings(item)
        return
    if isinstance(value, str) and value.strip():
        if SUSPICIOUS_VALUE_RE.search(value):
            raise ValueError("SLURM intent values may not contain shell metacharacters or path separators.")


def _validate_job_state(state: str | None) -> str | None:
    """Handle the internal validate job state helper path for this module.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._validate_job_state.
    """
    if state is None:
        return None
    normalized = str(state).strip().upper()
    if normalized not in ALLOWED_JOB_STATES:
        raise ValueError(f"Unsupported SLURM job state: {state}")
    return normalized


def _validate_node_state(state: str | None) -> str | None:
    """Handle the internal validate node state helper path for this module.

    Inputs:
        Receives state for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._validate_node_state.
    """
    if state is None:
        return None
    normalized = str(state).strip().lower()
    if normalized not in ALLOWED_NODE_STATES:
        raise ValueError(f"Unsupported SLURM node state: {state}")
    return normalized


def _require_tools(allowed_tools: list[str], *required: str) -> None:
    """Handle the internal require tools helper path for this module.

    Inputs:
        Receives allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._require_tools.
    """
    missing = [tool for tool in required if tool not in allowed_tools]
    if missing:
        raise ValueError(f"Deterministic intent requires unavailable tools: {', '.join(missing)}")


def _detect_output_mode(prompt: str) -> Literal["text", "csv", "json"]:
    """Handle the internal detect output mode helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_output_mode.
    """
    if OUTPUT_JSON_RE.search(prompt):
        return "json"
    if OUTPUT_CSV_RE.search(prompt):
        return "csv"
    return "text"


def _detect_metrics_output_mode(prompt: str) -> Literal["text", "json"]:
    """Handle the internal detect metrics output mode helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_metrics_output_mode.
    """
    return "json" if OUTPUT_JSON_RE.search(prompt) else "text"


def _detect_count_mode(prompt: str) -> bool:
    """Handle the internal detect count mode helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_count_mode.
    """
    return OUTPUT_COUNT_RE.search(prompt) is not None


def _detect_user(prompt: str) -> str | None:
    """Handle the internal detect user helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_user.
    """
    if re.search(r"\bmy\b", prompt, re.IGNORECASE):
        return getpass.getuser()
    for pattern in (USER_FOR_RE, USER_RE):
        match = pattern.search(prompt)
        if match is not None:
            return match.group(1)
    return None


def _detect_partition(prompt: str) -> str | None:
    """Handle the internal detect partition helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_partition.
    """
    match = PARTITION_RE.search(prompt)
    if match is None:
        return None
    candidate = match.group(1)
    if candidate.lower() in {"status", "summary", "cpu", "allocation", "as", "json", "csv", "only"}:
        return None
    return candidate


def _extract_gateway_node(prompt: str, available_nodes: list[str]) -> tuple[str, str | None]:
    """Handle the internal extract gateway node helper path for this module.

    Inputs:
        Receives prompt, available_nodes for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._extract_gateway_node.
    """
    explicit_match = CLUSTER_ROUTE_RE.search(prompt)
    if explicit_match is not None:
        gateway_node = explicit_match.group(1)
        cleaned = CLUSTER_ROUTE_RE.sub("", prompt, count=1)
        return _collapse_spaces(cleaned), gateway_node

    trailing_match = Bare_ROUTE_RE.search(prompt)
    if trailing_match is not None:
        gateway_node = trailing_match.group(1)
        if gateway_node in set(available_nodes):
            cleaned = Bare_ROUTE_RE.sub("", prompt, count=1)
            return _collapse_spaces(cleaned), gateway_node

    return prompt, None


def _detect_node_name(prompt: str) -> str | None:
    """Handle the internal detect node name helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_node_name.
    """
    for pattern in (NODE_DETAIL_RE, NODE_STATUS_FOR_RE, NODE_NAME_RE):
        match = pattern.search(prompt)
        if match is not None:
            candidate = match.group(1)
            if candidate.lower() in {"status", "details", "detail", "node", "nodes"}:
                continue
            return candidate
    return None


def _detect_job_state(prompt: str) -> str | None:
    """Handle the internal detect job state helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_job_state.
    """
    lower = prompt.lower()
    if "running" in lower:
        return "RUNNING"
    if "pending" in lower:
        return "PENDING"
    if "completed" in lower:
        return "COMPLETED"
    if "failed" in lower:
        return "FAILED"
    if "cancelled" in lower or "canceled" in lower:
        return "CANCELLED"
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT"
    return None


def _detect_node_state(prompt: str) -> str | None:
    """Handle the internal detect node state helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_node_state.
    """
    lower = prompt.lower()
    if "idle" in lower:
        return "idle"
    if "allocated" in lower or re.search(r"\balloc\b", lower):
        return "allocated"
    if "mixed" in lower:
        return "mixed"
    if "drained" in lower or "drain" in lower:
        return "drained"
    if "down" in lower:
        return "down"
    return None


def _looks_like_accounting_prompt(prompt: str) -> bool:
    """Handle the internal looks like accounting prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._looks_like_accounting_prompt.
    """
    lower = prompt.lower()
    return _looks_like_runtime_aggregate_prompt(prompt) or any(
        token in lower
        for token in (
            "sacct",
            "accounting",
            "completed",
            "failed",
            "cancelled",
            "canceled",
            "recent",
            "since yesterday",
            "last 24 hours",
            "last 7 days",
            "this week",
        )
    )


def _looks_like_runtime_aggregate_prompt(prompt: str) -> bool:
    """Handle the internal looks like runtime aggregate prompt helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._looks_like_runtime_aggregate_prompt.
    """
    lower = prompt.lower()
    return re.search(
        r"\b(?:how\s+long|average\s+(?:run\s*time|runtime|elapsed|duration)|avg\s+elapsed|mean\s+runtime|"
        r"min(?:imum)?\s+(?:run\s*time|runtime|elapsed|duration)|max(?:imum)?\s+(?:run\s*time|runtime|elapsed|duration)|"
        r"longest\s+(?:run\s*time|runtime|elapsed|duration)|total\s+(?:run\s*time|runtime|elapsed|duration)|"
        r"runtime\s+summary|elapsed\s+time|took\s+to\s+run|ran\s+longer\s+than|jobs?\s+longer\s+than)\b",
        lower,
    ) is not None


def _is_unsupported_runtime_metric(prompt: str) -> bool:
    """Handle the internal is unsupported runtime metric helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._is_unsupported_runtime_metric.
    """
    lower = prompt.lower()
    return "median" in lower and re.search(r"\b(?:runtime|run\s*time|elapsed|duration|jobs?)\b", lower) is not None


def _detect_time_range(prompt: str) -> tuple[str | None, str | None]:
    """Handle the internal detect time range helper path for this module.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._detect_time_range.
    """
    lower = prompt.lower()
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    if "since yesterday" in lower:
        return yesterday_start.strftime("%Y-%m-%d %H:%M:%S"), None
    if re.search(r"\byesterday\b", lower):
        return yesterday_start.strftime("%Y-%m-%d %H:%M:%S"), today_start.strftime("%Y-%m-%d %H:%M:%S")
    if re.search(r"\btoday\b", lower):
        return today_start.strftime("%Y-%m-%d %H:%M:%S"), None
    if "last 24 hours" in lower:
        return (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"), None
    if "last 7 days" in lower:
        return (today_start - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"), None
    if "this week" in lower:
        start_of_week = today_start - timedelta(days=today_start.weekday())
        return start_of_week.strftime("%Y-%m-%d %H:%M:%S"), None
    return None, None


def _collapse_spaces(value: str) -> str:
    """Handle the internal collapse spaces helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.capabilities.slurm._collapse_spaces.
    """
    return re.sub(r"\s+", " ", value).strip()
