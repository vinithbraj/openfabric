"""OpenFABRIC Runtime Module: aor_runtime.runtime.insights

Purpose:
    Generate compact deterministic insight summaries over runtime outputs.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.facts import validate_facts_for_llm


InsightSeverity = Literal["info", "warning", "critical"]


@dataclass
class Insight:
    """Represent insight within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by Insight.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.insights.Insight and related tests.
    """
    title: str
    message: str
    severity: InsightSeverity = "info"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """To dict for Insight instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through Insight.to_dict calls and related tests.
        """
        return asdict(self)


@dataclass
class InsightResult:
    """Represent insight result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by InsightResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.insights.InsightResult and related tests.
    """
    domain: str
    summary: str
    insights: list[Insight] = field(default_factory=list)
    facts_used: dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """To dict for InsightResult instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through InsightResult.to_dict calls and related tests.
        """
        return {
            "domain": self.domain,
            "summary": self.summary,
            "insights": [insight.to_dict() for insight in self.insights],
            "facts_used": self.facts_used,
            "llm_used": self.llm_used,
            "warnings": list(self.warnings),
        }


@dataclass
class InsightContext:
    """Represent insight context within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by InsightContext.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.insights.InsightContext and related tests.
    """
    enable_llm: bool = False
    max_facts: int = 50
    max_input_chars: int = 4000
    max_output_chars: int = 1500
    mode: Literal["user", "debug", "raw"] = "user"


def generate_insights(facts: dict[str, Any], context: InsightContext | None = None) -> InsightResult:
    """Generate insights for the surrounding runtime workflow.

    Inputs:
        Receives facts, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights.generate_insights.
    """
    ctx = context or InsightContext()
    domain = str(facts.get("domain") or "generic")
    if domain == "slurm":
        return generate_slurm_insights(facts)
    if domain == "sql":
        return generate_sql_insights(facts)
    if domain == "filesystem":
        return generate_filesystem_insights(facts)
    return generate_generic_insights(facts)


def generate_slurm_insights(facts: dict[str, Any]) -> InsightResult:
    """Generate slurm insights for the surrounding runtime workflow.

    Inputs:
        Receives facts for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights.generate_slurm_insights.
    """
    queue = dict(facts.get("queue") or {})
    nodes = dict(facts.get("nodes") or {})
    gpu = dict(facts.get("gpu") or {})
    accounting = dict(facts.get("accounting") or {})
    warnings = list(facts.get("warnings") or [])
    insights: list[Insight] = []

    running = _int(queue.get("running_jobs"))
    pending = _int(queue.get("pending_jobs"))
    total_jobs = _int(queue.get("total_jobs"))
    problematic = _int(nodes.get("problematic_nodes"))
    drained = _int(nodes.get("drained_nodes"))
    down = _int(nodes.get("down_nodes"))
    failed_recent = _int(accounting.get("failed_jobs_recent"))

    if running == 0 and pending and pending > 0:
        insights.append(
            Insight(
                title="Queue is not making progress",
                message=f"There are {pending:,} pending jobs and no running jobs in the sanitized queue facts.",
                severity="warning",
                evidence={"running_jobs": running, "pending_jobs": pending},
            )
        )
    elif pending and running is not None and pending >= max(10, running * 5):
        insights.append(
            Insight(
                title="Queue pressure is high",
                message=f"Pending jobs ({pending:,}) are much higher than running jobs ({running:,}).",
                severity="warning",
                evidence={"running_jobs": running, "pending_jobs": pending},
            )
        )

    if problematic and problematic > 0:
        insights.append(
            Insight(
                title="Node health needs attention",
                message=f"{problematic:,} problematic nodes were reported in the sanitized node summary.",
                severity="warning",
                evidence={"problematic_nodes": problematic, "affected_partition_rows": nodes.get("affected_partition_rows")},
            )
        )
    if drained and drained > 0:
        insights.append(
            Insight(
                title="Drained nodes may reduce capacity",
                message=f"{drained:,} drained nodes may reduce available scheduling capacity.",
                severity="warning",
                evidence={"drained_nodes": drained},
            )
        )
    if down and down > 0:
        insights.append(
            Insight(
                title="Down nodes detected",
                message=f"{down:,} nodes are down according to the sanitized node summary.",
                severity="critical" if down > 1 else "warning",
                evidence={"down_nodes": down},
            )
        )

    if gpu.get("available") is False:
        insights.append(
            Insight(
                title="GPU capacity may be unavailable",
                message="The sanitized GPU facts report no available GPU capacity.",
                severity="warning",
                evidence={"gpu_available": False},
            )
        )
    elif gpu.get("available") is True and pending and pending >= 10:
        insights.append(
            Insight(
                title="Pending jobs may be constrained by scheduling policy",
                message="GPU resources appear available, so a large pending queue may be caused by partition, CPU, memory, priority, or scheduling policy constraints.",
                severity="info",
                evidence={"gpu_available": True, "pending_jobs": pending},
            )
        )

    slurmdbd_status = str(accounting.get("slurmdbd_status") or "").lower()
    if slurmdbd_status and slurmdbd_status != "ok":
        insights.append(
            Insight(
                title="Accounting health warning",
                message=f"SLURM accounting status is `{accounting.get('slurmdbd_status')}`.",
                severity="warning",
                evidence={"slurmdbd_status": accounting.get("slurmdbd_status")},
            )
        )
    if accounting.get("available") is False:
        insights.append(
            Insight(
                title="Accounting unavailable",
                message="SLURM accounting appears unavailable in the sanitized facts.",
                severity="warning",
                evidence={"accounting_available": False},
            )
        )
    if failed_recent and failed_recent > 0:
        insights.append(
            Insight(
                title="Recent job failures detected",
                message=f"{failed_recent:,} recent failed jobs were reported.",
                severity="warning",
                evidence={"failed_jobs_recent": failed_recent},
            )
        )

    if insights:
        summary = _slurm_problem_summary(total_jobs, running, pending, problematic, gpu.get("available"))
    else:
        summary = "The SLURM cluster looks healthy from the sanitized facts available."
    return InsightResult(domain="slurm", summary=summary, insights=insights, facts_used=facts, warnings=warnings)


def generate_sql_insights(facts: dict[str, Any]) -> InsightResult:
    """Generate sql insights for the surrounding runtime workflow.

    Inputs:
        Receives facts for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights.generate_sql_insights.
    """
    insights: list[Insight] = []
    result = dict(facts.get("result") or {})
    count = _int(facts.get("result_count", result.get("count")))
    constraints = list(facts.get("constraints_applied") or [])
    projections = list(facts.get("projections_used") or [])

    if count == 0:
        insights.append(
            Insight(
                title="No matching records",
                message="The query returned a count of 0 for the requested criteria.",
                severity="info",
                evidence={"count": 0},
            )
        )
    if constraints:
        insights.append(
            Insight(
                title="Constraints were applied",
                message="The final SQL used the extracted semantic constraints.",
                severity="info",
                evidence={"constraints_applied": constraints[:10]},
            )
        )
    if projections:
        insights.append(
            Insight(
                title="Projection was resolved",
                message="The response used schema-resolved projected columns.",
                severity="info",
                evidence={"projections_used": projections[:10]},
            )
        )
    if facts.get("truncated"):
        insights.append(
            Insight(
                title="Result was truncated",
                message="Only a capped subset of SQL rows was returned.",
                severity="warning",
                evidence={"row_count": facts.get("row_count")},
            )
        )

    if count is not None:
        summary = f"The SQL query returned a count of {count:,}."
    elif facts.get("row_count") is not None:
        summary = f"The SQL query returned {int(facts.get('row_count') or 0):,} rows."
    else:
        summary = "The SQL result was reviewed using sanitized query facts."
    return InsightResult(domain="sql", summary=summary, insights=insights, facts_used=facts, warnings=list(facts.get("warnings") or []))


def generate_filesystem_insights(facts: dict[str, Any]) -> InsightResult:
    """Generate filesystem insights for the surrounding runtime workflow.

    Inputs:
        Receives facts for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights.generate_filesystem_insights.
    """
    insights: list[Insight] = []
    count = _int(facts.get("file_count"))
    total_size = _int(facts.get("total_size_bytes"))
    if count == 0:
        insights.append(
            Insight(
                title="No matching files",
                message="No files matched the requested filesystem criteria.",
                severity="info",
                evidence={"file_count": 0},
            )
        )
    if total_size is not None and total_size >= 10 * 1024**3:
        insights.append(
            Insight(
                title="Large storage footprint",
                message=f"Matching files total {_human_size(total_size)}, which may be worth reviewing for storage impact.",
                severity="warning",
                evidence={"total_size_bytes": total_size},
            )
        )
    elif total_size is not None and total_size >= 1024**3:
        insights.append(
            Insight(
                title="Notable storage usage",
                message=f"Matching files total {_human_size(total_size)}.",
                severity="info",
                evidence={"total_size_bytes": total_size},
            )
        )
    if facts.get("recursive") is True:
        insights.append(
            Insight(
                title="Recursive scope",
                message="The filesystem operation searched recursively under the requested path.",
                severity="info",
                evidence={"recursive": True},
            )
        )
    if facts.get("truncated"):
        insights.append(
            Insight(
                title="Results were truncated",
                message="Only a capped subset of filesystem results was returned.",
                severity="warning",
                evidence={"truncated": True},
            )
        )

    if count is not None and total_size is not None:
        summary = f"Found {count:,} matching files totaling {_human_size(total_size)}."
    elif count is not None:
        summary = f"Found {count:,} matching files."
    else:
        summary = "The filesystem result was reviewed using sanitized facts."
    return InsightResult(domain="filesystem", summary=summary, insights=insights, facts_used=facts, warnings=list(facts.get("warnings") or []))


def generate_generic_insights(facts: dict[str, Any]) -> InsightResult:
    """Generate generic insights for the surrounding runtime workflow.

    Inputs:
        Receives facts for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights.generate_generic_insights.
    """
    domain = str(facts.get("domain") or "generic")
    return InsightResult(
        domain=domain,
        summary="The result was reviewed using sanitized facts.",
        insights=[],
        facts_used=facts,
        warnings=list(facts.get("warnings") or []),
    )


def summarize_insights_with_llm(
    facts: dict[str, Any],
    deterministic_insights: InsightResult,
    context: InsightContext,
    settings: Any,
) -> str | None:
    """Summarize insights with llm for the surrounding runtime workflow.

    Inputs:
        Receives facts, deterministic_insights, context, settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights.summarize_insights_with_llm.
    """
    if not context.enable_llm:
        return None
    compact_payload = {
        "facts": facts,
        "deterministic_summary": deterministic_insights.summary,
        "deterministic_insights": [
            {"title": insight.title, "message": insight.message, "severity": insight.severity}
            for insight in deterministic_insights.insights[: context.max_facts]
        ],
    }
    encoded = json.dumps(compact_payload, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) > context.max_input_chars:
        return None
    if not validate_facts_for_llm(compact_payload, max_items=context.max_facts, max_string_length=context.max_input_chars):
        return None
    try:
        from aor_runtime.llm.client import LLMClient

        summary = LLMClient(settings).complete(
            system_prompt=(
                "You are summarizing sanitized operational facts for a user. Use only the facts provided. "
                "Do not invent values. Do not change numbers. Do not mention hidden data, internal telemetry, "
                "raw rows, command output, or raw JSON. Produce concise useful Markdown."
            ),
            user_prompt=encoded,
            temperature=0.0,
        ).strip()
    except Exception:
        return None
    if not summary or summary.lstrip().startswith(("{", "[")):
        return None
    if any(word in summary.lower() for word in ("raw_output", "semantic_frame", "coverage", "token", "password")):
        return None
    return summary[: context.max_output_chars].strip()


def _slurm_problem_summary(total: int | None, running: int | None, pending: int | None, problematic: int | None, gpu_available: Any) -> str:
    """Handle the internal slurm problem summary helper path for this module.

    Inputs:
        Receives total, running, pending, problematic, gpu_available for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights._slurm_problem_summary.
    """
    parts: list[str] = []
    if total is not None:
        parts.append(f"{total:,} jobs total")
    if running is not None:
        parts.append(f"{running:,} running")
    if pending is not None:
        parts.append(f"{pending:,} pending")
    if problematic is not None:
        parts.append(f"{problematic:,} problematic nodes")
    if gpu_available is not None:
        parts.append("GPU available" if bool(gpu_available) else "GPU unavailable")
    if not parts:
        return "The SLURM result was reviewed using sanitized operational facts."
    return "SLURM status: " + ", ".join(parts) + "."


def _int(value: Any) -> int | None:
    """Handle the internal int helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights._int.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _human_size(value: int) -> str:
    """Handle the internal human size helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.insights._human_size.
    """
    units = ["bytes", "KB", "MB", "GB", "TB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    if unit == "bytes":
        return f"{int(amount):,} bytes"
    return f"{amount:.1f} {unit}"
