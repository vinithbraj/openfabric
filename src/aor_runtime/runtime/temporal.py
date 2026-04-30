"""OpenFABRIC Runtime Module: aor_runtime.runtime.temporal

Purpose:
    Canonicalize natural-language time ranges into tool-safe date arguments.

Responsibilities:
    Parse relative and absolute windows such as today, yesterday, last N days, and from/to ranges.

Data flow / Interfaces:
    Feeds action canonicalization with ISO-like start/end values and runtime metadata labels.

Boundaries:
    Natural-language temporal phrases must not reach strict tool schemas unvalidated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aor_runtime.config import Settings
from aor_runtime.core.utils import extract_json_object


ISO_LIKE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
TEMPORAL_TOOLS = {"slurm.accounting", "slurm.accounting_aggregate", "slurm.metrics"}


class TemporalNormalizationError(ValueError):
    """Represent temporal normalization error within the OpenFABRIC runtime. It extends ValueError.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TemporalNormalizationError.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.temporal.TemporalNormalizationError and related tests.
    """
    pass


@dataclass(frozen=True)
class TemporalRange:
    """Represent temporal range within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TemporalRange.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.temporal.TemporalRange and related tests.
    """
    start: str | None = None
    end: str | None = None
    time_window_label: str | None = None
    original_text: str | None = None


@dataclass
class TemporalCanonicalizationResult:
    """Represent temporal canonicalization result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TemporalCanonicalizationResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.temporal.TemporalCanonicalizationResult and related tests.
    """
    actions: list[dict[str, Any]]
    repairs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_calls: int = 0


class TemporalArgumentCanonicalizer:
    """Represent temporal argument canonicalizer within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TemporalArgumentCanonicalizer.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.temporal.TemporalArgumentCanonicalizer and related tests.
    """
    def __init__(
        self,
        *,
        goal: str,
        settings: Settings,
        llm: Any | None = None,
        now: datetime | None = None,
    ) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives goal, settings, llm, now for this TemporalArgumentCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through TemporalArgumentCanonicalizer.__init__ calls and related tests.
        """
        self.goal = str(goal or "")
        self.settings = settings
        self.llm = llm
        self.now = now or current_local_datetime(settings)

    def canonicalize(self, actions: list[dict[str, Any]]) -> TemporalCanonicalizationResult:
        """Canonicalize for TemporalArgumentCanonicalizer instances.

        Inputs:
            Receives actions for this TemporalArgumentCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TemporalArgumentCanonicalizer.canonicalize calls and related tests.
        """
        repairs: list[str] = []
        metadata: dict[str, Any] = {}
        llm_calls = 0
        goal_relative_range = parse_relative_temporal_range(self.goal, now=self.now)
        goal_range = parse_temporal_range(self.goal, now=self.now)
        for action in actions:
            if str(action.get("tool") or "") not in TEMPORAL_TOOLS:
                continue
            inputs = action.setdefault("inputs", {})
            original_start = inputs.get("start")
            original_end = inputs.get("end")
            requested = _temporal_phrase_from_inputs(inputs)
            reason = ""
            if requested:
                resolved, used_llm = self._resolve_or_fail(requested)
                llm_calls += used_llm
                reason = "normalized_planner_temporal_phrase"
            elif goal_relative_range is not None:
                resolved = goal_relative_range
                reason = (
                    "goal_temporal_phrase_overrode_planner_bounds"
                    if _has_temporal_bounds(inputs)
                    else "filled_from_goal_relative_time"
                )
            elif goal_range is not None and not _has_temporal_bounds(inputs):
                resolved = goal_range
                reason = "filled_from_goal_time_range"
            else:
                resolved = _range_from_valid_bounds(inputs)
                if resolved is None:
                    continue
                reason = "validated_planner_time_range"
            changed = _apply_temporal_range(inputs, resolved)
            if changed:
                phrase = resolved.original_text or requested or resolved.time_window_label or "time range"
                if reason == "goal_temporal_phrase_overrode_planner_bounds":
                    repairs.append(
                        f"Overrode {action.get('tool')} planner time bounds with user-requested time range {phrase!r}."
                    )
                else:
                    repairs.append(f"Normalized {action.get('tool')} time range {phrase!r} to ISO-like start/end.")
                metadata = {
                    "original_time_phrase": phrase,
                    "reason": reason,
                    "original_planner_start": original_start,
                    "original_planner_end": original_end,
                    "time_window_label": resolved.time_window_label,
                    "start": resolved.start,
                    "end": resolved.end,
                }
        return TemporalCanonicalizationResult(actions=actions, repairs=repairs, metadata=metadata, llm_calls=llm_calls)

    def _resolve_or_fail(self, phrase: str) -> tuple[TemporalRange, int]:
        """Handle the internal resolve or fail helper path for this module.

        Inputs:
            Receives phrase for this TemporalArgumentCanonicalizer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TemporalArgumentCanonicalizer._resolve_or_fail calls and related tests.
        """
        resolved = parse_temporal_range(phrase, now=self.now)
        if resolved is not None:
            return resolved, 0
        if self.llm is not None:
            llm_resolved = _resolve_with_llm(self.llm, phrase=phrase, goal=self.goal, now=self.now)
            if llm_resolved is not None:
                return llm_resolved, 1
        raise TemporalNormalizationError(f"Could not resolve the requested time range: {phrase}")


def current_local_datetime(settings: Settings | None = None) -> datetime:
    """Current local datetime for the surrounding runtime workflow.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal.current_local_datetime.
    """
    return _current_aware_datetime(settings).replace(tzinfo=None)


def runtime_date_context(settings: Settings | None = None, *, now: datetime | None = None) -> dict[str, str]:
    """Runtime date context for the surrounding runtime workflow.

    Inputs:
        Receives settings, now for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal.runtime_date_context.
    """
    if now is None:
        aware_now = _current_aware_datetime(settings)
        local_now = aware_now.replace(tzinfo=None)
    else:
        local_now = now.replace(microsecond=0)
        aware_now = _current_aware_datetime(settings)
    timezone_name = str(getattr(settings, "runtime_timezone", "") or "").strip() or str(aware_now.tzname() or "local")
    return {
        "current_local_date": local_now.strftime("%Y-%m-%d"),
        "current_local_datetime": _format_dt(local_now),
        "timezone": timezone_name,
        "utc_offset": aware_now.strftime("%z"),
    }


def _current_aware_datetime(settings: Settings | None = None) -> datetime:
    """Handle the internal current aware datetime helper path for this module.

    Inputs:
        Receives settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._current_aware_datetime.
    """
    timezone_name = str(getattr(settings, "runtime_timezone", "") or "").strip()
    if timezone_name:
        try:
            return datetime.now(ZoneInfo(timezone_name)).replace(microsecond=0)
        except ZoneInfoNotFoundError:
            pass
    return datetime.now().astimezone().replace(microsecond=0)


def is_iso_like_temporal(value: Any) -> bool:
    """Is iso like temporal for the surrounding runtime workflow.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal.is_iso_like_temporal.
    """
    text = str(value or "").strip()
    return bool(text and ISO_LIKE_RE.match(text))


def parse_temporal_range(value: Any, *, now: datetime | None = None) -> TemporalRange | None:
    """Parse temporal range for the surrounding runtime workflow.

    Inputs:
        Receives value, now for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal.parse_temporal_range.
    """
    text = _clean_time_phrase(value)
    if not text:
        return None
    clock = (now or datetime.now()).replace(microsecond=0)
    lower = text.lower()

    explicit = _parse_explicit_range(text)
    if explicit is not None:
        return explicit
    since = re.search(r"\bsince\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\b", text, re.IGNORECASE)
    if since:
        return TemporalRange(start=_normalize_iso_value(since.group(1)), original_text=text)

    relative = parse_relative_temporal_range(text, now=clock)
    if relative is not None:
        return relative

    if is_iso_like_temporal(text):
        return TemporalRange(start=_normalize_iso_value(text), original_text=text)
    return None


def parse_relative_temporal_range(value: Any, *, now: datetime | None = None) -> TemporalRange | None:
    """Parse relative temporal range for the surrounding runtime workflow.

    Inputs:
        Receives value, now for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal.parse_relative_temporal_range.
    """
    text = _clean_time_phrase(value)
    if not text:
        return None
    clock = (now or datetime.now()).replace(microsecond=0)
    lower = text.lower()
    today_start = clock.replace(hour=0, minute=0, second=0, microsecond=0)

    if "since yesterday" in lower:
        return TemporalRange(start=_format_dt(today_start - timedelta(days=1)), time_window_label="Since yesterday", original_text=text)
    if re.search(r"\byesterday\b", lower):
        return TemporalRange(
            start=_format_dt(today_start - timedelta(days=1)),
            end=_format_dt(today_start),
            time_window_label="Yesterday",
            original_text=text,
        )
    if re.search(r"\btoday\b", lower):
        return TemporalRange(start=_format_dt(today_start), time_window_label="Today", original_text=text)

    hours = re.search(r"\b(?:last|past)\s+(\d+)\s+hours?\b", lower)
    if hours:
        count = int(hours.group(1))
        return TemporalRange(start=_format_dt(clock - timedelta(hours=count)), time_window_label=f"Last {count} hours", original_text=text)

    days = re.search(r"\b(?:last|past)\s+(\d+)\s+days?\b", lower)
    if days:
        count = int(days.group(1))
        return TemporalRange(start=_format_dt(today_start - timedelta(days=count)), time_window_label=f"Last {count} days", original_text=text)

    if "this week" in lower:
        start = today_start - timedelta(days=today_start.weekday())
        return TemporalRange(start=_format_dt(start), time_window_label="This week", original_text=text)
    if "last week" in lower:
        this_week = today_start - timedelta(days=today_start.weekday())
        return TemporalRange(start=_format_dt(this_week - timedelta(days=7)), end=_format_dt(this_week), time_window_label="Last week", original_text=text)
    if "this month" in lower:
        start = today_start.replace(day=1)
        return TemporalRange(start=_format_dt(start), time_window_label="This month", original_text=text)
    return None


def _resolve_with_llm(llm: Any, *, phrase: str, goal: str, now: datetime) -> TemporalRange | None:
    """Handle the internal resolve with llm helper path for this module.

    Inputs:
        Receives llm, phrase, goal, now for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._resolve_with_llm.
    """
    prompt = {
        "task": "Resolve the user time phrase to ISO-like local start/end values.",
        "current_local_datetime": _format_dt(now),
        "constraints": {
            "format": "YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
            "unknown_values": None,
            "return_json_only": True,
        },
        "user_goal": goal,
        "time_phrase": phrase,
        "output_schema": {"start": "string|null", "end": "string|null", "time_window_label": "string|null"},
    }
    try:
        raw = llm.complete(
            system_prompt="Return JSON only. Resolve time phrases into safe ISO-like local datetimes.",
            user_prompt=json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            temperature=0.0,
        )
        payload = extract_json_object(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    start = _normalize_iso_value(payload.get("start"))
    end = _normalize_iso_value(payload.get("end"))
    if payload.get("start") and start is None:
        return None
    if payload.get("end") and end is None:
        return None
    if not start and not end:
        return None
    label = str(payload.get("time_window_label") or "").strip() or None
    return TemporalRange(start=start, end=end, time_window_label=label, original_text=phrase)


def _temporal_phrase_from_inputs(inputs: dict[str, Any]) -> str | None:
    """Handle the internal temporal phrase from inputs helper path for this module.

    Inputs:
        Receives inputs for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._temporal_phrase_from_inputs.
    """
    time_range = inputs.pop("time_range", None)
    if isinstance(time_range, dict):
        start = time_range.get("start")
        end = time_range.get("end")
        if _invalid_temporal(start):
            return str(start)
        if _invalid_temporal(end):
            return str(end)
        if start and not inputs.get("start"):
            inputs["start"] = start
        if end and not inputs.get("end"):
            inputs["end"] = end
        label = time_range.get("label") or time_range.get("time_window_label")
        if label and not inputs.get("time_window_label"):
            inputs["time_window_label"] = label
    elif _invalid_temporal(time_range):
        return str(time_range)

    for source_key, target_key in (("since", "start"), ("until", "end"), ("date", "start")):
        raw = inputs.pop(source_key, None)
        if raw in (None, ""):
            continue
        if _invalid_temporal(raw):
            return str(raw)
        if not inputs.get(target_key):
            inputs[target_key] = raw

    for key in ("start", "end"):
        raw = inputs.get(key)
        if _invalid_temporal(raw):
            return str(raw)
    return None


def _has_temporal_bounds(inputs: dict[str, Any]) -> bool:
    """Handle the internal has temporal bounds helper path for this module.

    Inputs:
        Receives inputs for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._has_temporal_bounds.
    """
    return any(str(inputs.get(key) or "").strip() for key in ("start", "end", "time_range", "since", "until", "date"))


def _range_from_valid_bounds(inputs: dict[str, Any]) -> TemporalRange | None:
    """Handle the internal range from valid bounds helper path for this module.

    Inputs:
        Receives inputs for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._range_from_valid_bounds.
    """
    start = _normalize_iso_value(inputs.get("start"))
    end = _normalize_iso_value(inputs.get("end"))
    if inputs.get("start") and start is None:
        raise TemporalNormalizationError(f"Could not resolve the requested time range: {inputs.get('start')}")
    if inputs.get("end") and end is None:
        raise TemporalNormalizationError(f"Could not resolve the requested time range: {inputs.get('end')}")
    if not start and not end:
        return None
    label = str(inputs.get("time_window_label") or "").strip() or None
    return TemporalRange(start=start, end=end, time_window_label=label)


def _apply_temporal_range(inputs: dict[str, Any], temporal_range: TemporalRange) -> bool:
    """Handle the internal apply temporal range helper path for this module.

    Inputs:
        Receives inputs, temporal_range for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._apply_temporal_range.
    """
    changed = False
    if temporal_range.start and inputs.get("start") != temporal_range.start:
        inputs["start"] = temporal_range.start
        changed = True
    if temporal_range.end and inputs.get("end") != temporal_range.end:
        inputs["end"] = temporal_range.end
        changed = True
    if temporal_range.end is None and "end" in inputs:
        if str(inputs.get("end") or "").strip():
            changed = True
        inputs.pop("end", None)
    if temporal_range.time_window_label and inputs.get("time_window_label") != temporal_range.time_window_label:
        inputs["time_window_label"] = temporal_range.time_window_label
        changed = True
    return changed


def _invalid_temporal(value: Any) -> bool:
    """Handle the internal invalid temporal helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._invalid_temporal.
    """
    if value in (None, ""):
        return False
    if isinstance(value, (int, float, bool)):
        return True
    text = str(value).strip()
    return bool(text and not is_iso_like_temporal(text))


def _parse_explicit_range(text: str) -> TemporalRange | None:
    """Handle the internal parse explicit range helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._parse_explicit_range.
    """
    match = re.search(
        r"\b(?:from|between)\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\s+(?:to|and)\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return TemporalRange(start=_normalize_iso_value(match.group(1)), end=_normalize_iso_value(match.group(2)), original_text=text)


def _normalize_iso_value(value: Any) -> str | None:
    """Handle the internal normalize iso value helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._normalize_iso_value.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not ISO_LIKE_RE.match(text):
        return None
    return text.replace("T", " ")


def _clean_time_phrase(value: Any) -> str:
    """Handle the internal clean time phrase helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._clean_time_phrase.
    """
    text = str(value or "").strip().strip("'\"")
    text = re.sub(r"\s+", " ", text)
    return text


def _format_dt(value: datetime) -> str:
    """Handle the internal format dt helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.temporal._format_dt.
    """
    return value.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
