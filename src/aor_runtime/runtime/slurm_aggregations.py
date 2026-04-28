from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal


SlurmAccountingAggregateMetric = Literal[
    "average_elapsed",
    "min_elapsed",
    "max_elapsed",
    "sum_elapsed",
    "count",
    "count_longer_than",
    "runtime_summary",
]


def aggregate_slurm_accounting_jobs(jobs: list[dict[str, Any]], intent: Any) -> dict[str, Any]:
    metric = _metric(getattr(intent, "metric", None))
    group_by = _group_by(getattr(intent, "group_by", None))
    threshold_seconds = _optional_int(getattr(intent, "threshold_seconds", None))
    excluded_states = {str(state).upper() for state in list(getattr(intent, "excluded_states", []) or [])}
    warnings: list[str] = []

    prepared: list[dict[str, Any]] = []
    ignored_elapsed = 0
    for job in jobs:
        row = dict(job)
        if excluded_states and str(row.get("state") or "").upper() in excluded_states:
            continue
        elapsed = _optional_int(row.get("elapsed_seconds"))
        if elapsed is None:
            ignored_elapsed += 1
            continue
        row["elapsed_seconds"] = elapsed
        prepared.append(row)

    if ignored_elapsed:
        warnings.append(f"Ignored {ignored_elapsed} accounting rows with unparseable elapsed time.")

    if metric == "count_longer_than":
        if threshold_seconds is None:
            warnings.append("No duration threshold was available for the count-longer-than metric.")
            prepared = []
        else:
            prepared = [job for job in prepared if int(job["elapsed_seconds"]) > threshold_seconds]

    if group_by:
        groups = [
            {"key": key, **_aggregate_elapsed(group_jobs, metric, threshold_seconds)}
            for key, group_jobs in sorted(_group_jobs(prepared, group_by).items(), key=lambda item: str(item[0]))
        ]
    else:
        groups = []

    aggregate = _aggregate_elapsed(prepared, metric, threshold_seconds)
    result: dict[str, Any] = {
        "result_kind": "accounting_aggregate",
        "metric": metric,
        "partition": getattr(intent, "partition", None),
        "user": getattr(intent, "user", None),
        "state": getattr(intent, "state", None),
        "include_all_states": bool(getattr(intent, "include_all_states", False)),
        "excluded_states": sorted(excluded_states),
        "default_state_applied": bool(getattr(intent, "default_state_applied", False)),
        "start": getattr(intent, "start", None),
        "end": getattr(intent, "end", None),
        "time_window_label": getattr(intent, "time_window_label", None),
        "group_by": group_by,
        "threshold_seconds": threshold_seconds,
        "job_count": aggregate["job_count"],
        "warnings": warnings,
        "groups": groups,
    }
    result.update(aggregate)
    if not prepared:
        result["warnings"] = [*warnings, "No matching accounting jobs were found."]
    return result


def format_elapsed_seconds(seconds: int | float | None) -> str:
    if seconds is None:
        return "Unknown"
    total = int(round(float(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _aggregate_elapsed(
    jobs: list[dict[str, Any]],
    metric: SlurmAccountingAggregateMetric,
    threshold_seconds: int | None,
) -> dict[str, Any]:
    elapsed_values = [int(job["elapsed_seconds"]) for job in jobs if _optional_int(job.get("elapsed_seconds")) is not None]
    count = len(elapsed_values)
    total = sum(elapsed_values)
    average = (total / count) if count else None
    minimum = min(elapsed_values) if elapsed_values else None
    maximum = max(elapsed_values) if elapsed_values else None
    result: dict[str, Any] = {
        "job_count": count,
        "average_elapsed_seconds": average,
        "average_elapsed_human": format_elapsed_seconds(average),
        "min_elapsed_seconds": minimum,
        "min_elapsed_human": format_elapsed_seconds(minimum),
        "max_elapsed_seconds": maximum,
        "max_elapsed_human": format_elapsed_seconds(maximum),
        "sum_elapsed_seconds": total,
        "sum_elapsed_human": format_elapsed_seconds(total),
    }
    if metric == "count_longer_than":
        result["count_longer_than"] = count
        result["threshold_seconds"] = threshold_seconds
        result["threshold_human"] = format_elapsed_seconds(threshold_seconds)
    elif metric == "count":
        result["count"] = count
    elif metric == "min_elapsed":
        result["value_seconds"] = minimum
        result["value_human"] = format_elapsed_seconds(minimum)
    elif metric == "max_elapsed":
        result["value_seconds"] = maximum
        result["value_human"] = format_elapsed_seconds(maximum)
    elif metric == "sum_elapsed":
        result["value_seconds"] = total
        result["value_human"] = format_elapsed_seconds(total)
    else:
        result["value_seconds"] = average
        result["value_human"] = format_elapsed_seconds(average)
    return result


def _group_jobs(jobs: list[dict[str, Any]], group_by: str) -> dict[str, list[dict[str, Any]]]:
    field = "name" if group_by == "job_name" else group_by
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[str(job.get(field) or "unknown")].append(job)
    return dict(grouped)


def _metric(value: Any) -> SlurmAccountingAggregateMetric:
    text = str(value or "average_elapsed").strip()
    allowed = {
        "average_elapsed",
        "min_elapsed",
        "max_elapsed",
        "sum_elapsed",
        "count",
        "count_longer_than",
        "runtime_summary",
    }
    if text not in allowed:
        raise ValueError(f"Unsupported SLURM accounting aggregate metric: {value}")
    return text  # type: ignore[return-value]


def _group_by(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None
