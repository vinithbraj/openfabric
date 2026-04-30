from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import PlannerConfig
from aor_runtime.runtime.action_planner import LLMActionPlanner
from aor_runtime.runtime.temporal import TemporalArgumentCanonicalizer, TemporalNormalizationError, parse_temporal_range
from aor_runtime.tools.factory import build_tool_registry

from tests.runtime.test_action_planner import FakeLLM


NOW = datetime(2026, 4, 29, 10, 30, 0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")


def test_parse_last_7_days_uses_calendar_midnight_window() -> None:
    resolved = parse_temporal_range("last 7 days", now=NOW)

    assert resolved is not None
    assert resolved.start == "2026-04-22 00:00:00"
    assert resolved.end is None
    assert resolved.time_window_label == "Last 7 days"


@pytest.mark.parametrize(
    ("phrase", "start", "end", "label"),
    [
        ("today", "2026-04-29 00:00:00", None, "Today"),
        ("yesterday", "2026-04-28 00:00:00", "2026-04-29 00:00:00", "Yesterday"),
        ("last 24 hours", "2026-04-28 10:30:00", None, "Last 24 hours"),
        ("this week", "2026-04-27 00:00:00", None, "This week"),
        ("from 2026-04-01 to 2026-04-10", "2026-04-01", "2026-04-10", None),
    ],
)
def test_parse_common_time_phrases(phrase: str, start: str, end: str | None, label: str | None) -> None:
    resolved = parse_temporal_range(phrase, now=NOW)

    assert resolved is not None
    assert resolved.start == start
    assert resolved.end == end
    assert resolved.time_window_label == label


def test_temporal_canonicalizer_rewrites_start_phrase(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {"partition": "totalseg", "metric": "average_elapsed", "start": "last 7 days"},
        }
    ]

    result = TemporalArgumentCanonicalizer(
        goal="average time for jobs in totalseg",
        settings=_settings(tmp_path),
        now=NOW,
    ).canonicalize(actions)

    assert result.actions[0]["inputs"]["start"] == "2026-04-22 00:00:00"
    assert result.actions[0]["inputs"]["time_window_label"] == "Last 7 days"
    assert result.repairs


def test_temporal_canonicalizer_fills_missing_bounds_from_goal(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {"partition": "totalseg", "metric": "average_elapsed"},
        }
    ]

    result = TemporalArgumentCanonicalizer(
        goal="average time taken for job in totalseg partition for the last 7 days",
        settings=_settings(tmp_path),
        now=NOW,
    ).canonicalize(actions)

    assert result.actions[0]["inputs"]["start"] == "2026-04-22 00:00:00"
    assert result.actions[0]["inputs"]["time_window_label"] == "Last 7 days"


def test_temporal_canonicalizer_overrides_stale_planner_dates_for_relative_goal(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {
                "partition": "totalseg",
                "metric": "average_elapsed",
                "start": "2024-05-13 00:00:00",
                "end": "2024-05-13 00:00:00",
                "time_window_label": "2024-05-13",
            },
        }
    ]

    result = TemporalArgumentCanonicalizer(
        goal="calculate average job time in totalseg. Only consider jobs in the past 7 days",
        settings=_settings(tmp_path),
        now=NOW,
    ).canonicalize(actions)

    inputs = result.actions[0]["inputs"]
    assert inputs["start"] == "2026-04-22 00:00:00"
    assert "end" not in inputs
    assert inputs["time_window_label"] == "Last 7 days"
    assert result.metadata["reason"] == "goal_temporal_phrase_overrode_planner_bounds"
    assert result.metadata["original_planner_start"] == "2024-05-13 00:00:00"
    assert result.metadata["original_planner_end"] == "2024-05-13 00:00:00"


def test_temporal_canonicalizer_overrides_stale_planner_dates_for_last_24_hours(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {"start": "2024-05-13 00:00:00", "end": "2024-05-13 01:00:00"},
        }
    ]

    result = TemporalArgumentCanonicalizer(
        goal="average job time in totalseg for the last 24 hours",
        settings=_settings(tmp_path),
        now=NOW,
    ).canonicalize(actions)

    assert result.actions[0]["inputs"]["start"] == "2026-04-28 10:30:00"
    assert "end" not in result.actions[0]["inputs"]
    assert result.actions[0]["inputs"]["time_window_label"] == "Last 24 hours"


def test_temporal_canonicalizer_overrides_stale_planner_dates_for_yesterday(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {"start": "2024-05-13 00:00:00", "end": "2024-05-14 00:00:00"},
        }
    ]

    result = TemporalArgumentCanonicalizer(
        goal="average job time in totalseg yesterday",
        settings=_settings(tmp_path),
        now=NOW,
    ).canonicalize(actions)

    assert result.actions[0]["inputs"]["start"] == "2026-04-28 00:00:00"
    assert result.actions[0]["inputs"]["end"] == "2026-04-29 00:00:00"
    assert result.actions[0]["inputs"]["time_window_label"] == "Yesterday"


def test_explicit_absolute_planner_dates_pass_unchanged(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {"start": "2024-05-01", "end": "2024-05-13"},
        }
    ]

    result = TemporalArgumentCanonicalizer(
        goal="average job time in totalseg from 2024-05-01 to 2024-05-13",
        settings=_settings(tmp_path),
        now=NOW,
    ).canonicalize(actions)

    assert result.actions[0]["inputs"]["start"] == "2024-05-01"
    assert result.actions[0]["inputs"]["end"] == "2024-05-13"
    assert result.repairs == []


def test_valid_iso_time_values_pass_unchanged(tmp_path: Path) -> None:
    actions = [
        {
            "id": "aggregate_runtime",
            "tool": "slurm.accounting_aggregate",
            "inputs": {"start": "2026-04-01", "end": "2026-04-10 12:30:00"},
        }
    ]

    result = TemporalArgumentCanonicalizer(goal="average runtime", settings=_settings(tmp_path), now=NOW).canonicalize(actions)

    assert result.actions[0]["inputs"]["start"] == "2026-04-01"
    assert result.actions[0]["inputs"]["end"] == "2026-04-10 12:30:00"
    assert result.repairs == []


def test_unsupported_time_phrase_fails_cleanly(tmp_path: Path) -> None:
    actions = [{"id": "aggregate_runtime", "tool": "slurm.accounting_aggregate", "inputs": {"start": "after the moonrise"}}]

    with pytest.raises(TemporalNormalizationError, match="Could not resolve the requested time range: after the moonrise"):
        TemporalArgumentCanonicalizer(goal="average runtime", settings=_settings(tmp_path), now=NOW).canonicalize(actions)


def test_action_planner_normalizes_slurm_time_before_validation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aor_runtime.runtime.temporal.current_local_datetime",
        lambda settings=None: NOW,
    )
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "average time taken for job in totalseg partition for the last 7 days",
                    "actions": [
                        {
                            "id": "aggregate_runtime",
                            "tool": "slurm.accounting_aggregate",
                            "inputs": {"partition": "totalseg", "metric": "average_elapsed", "start": "last 7 days"},
                            "output_binding": "runtime_stats",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(_settings(tmp_path)),
        settings=_settings(tmp_path),
    )

    plan = planner.build_plan(
        goal="average time taken for job in totalseg partition for the last 7 days",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate"],
        input_payload={},
    )

    assert plan.steps[0].action == "slurm.accounting_aggregate"
    assert plan.steps[0].args["start"] == "2026-04-22 00:00:00"
    assert plan.steps[0].args["time_window_label"] == "Last 7 days"
    assert "last 7 days" not in str(plan.steps[0].args["start"]).lower()
    assert planner.last_temporal_normalization_repairs


def test_action_planner_strips_temporal_label_from_slurm_accounting_args(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aor_runtime.runtime.temporal.current_local_datetime",
        lambda settings=None: NOW,
    )
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "Show recent SLURM jobs for the last 24 hours.",
                    "actions": [
                        {
                            "id": "recent_jobs",
                            "tool": "slurm.accounting",
                            "inputs": {"start": "last 24 hours", "time_window_label": "Last 24 hours"},
                            "output_binding": "recent_jobs",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(_settings(tmp_path)),
        settings=_settings(tmp_path),
    )

    plan = planner.build_plan(
        goal="Show recent SLURM jobs for the last 24 hours.",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting"],
        input_payload={},
    )

    assert plan.steps[0].action == "slurm.accounting"
    assert plan.steps[0].args["start"] == "2026-04-28 10:30:00"
    assert "time_window_label" not in plan.steps[0].args
    assert any("time_window_label" in repair for repair in planner.last_canonicalization_repairs)
    scrubbed = planner.last_tool_argument_canonicalization_metadata["scrubbed_arguments"]
    assert scrubbed == [{"action_id": "recent_jobs", "tool": "slurm.accounting", "keys": ["time_window_label"]}]


def test_action_planner_keeps_temporal_label_for_slurm_accounting_aggregate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aor_runtime.runtime.temporal.current_local_datetime",
        lambda settings=None: NOW,
    )
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "Summarize average runtime for the last 24 hours.",
                    "actions": [
                        {
                            "id": "runtime_summary",
                            "tool": "slurm.accounting_aggregate",
                            "inputs": {"metric": "average_elapsed", "start": "last 24 hours"},
                            "output_binding": "runtime_summary",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(_settings(tmp_path)),
        settings=_settings(tmp_path),
    )

    plan = planner.build_plan(
        goal="Summarize average runtime for the last 24 hours.",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate"],
        input_payload={},
    )

    assert plan.steps[0].action == "slurm.accounting_aggregate"
    assert plan.steps[0].args["time_window_label"] == "Last 24 hours"


def test_action_planner_prompt_includes_runtime_date_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aor_runtime.runtime.temporal.current_local_datetime",
        lambda settings=None: NOW,
    )
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "average time taken for job in totalseg partition",
                    "actions": [
                        {
                            "id": "aggregate_runtime",
                            "tool": "slurm.accounting_aggregate",
                            "inputs": {"partition": "totalseg", "metric": "average_elapsed"},
                            "output_binding": "runtime_stats",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(_settings(tmp_path)),
        settings=_settings(tmp_path),
    )

    planner.build_plan(
        goal="average time taken for job in totalseg partition",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate"],
        input_payload={},
    )

    assert planner.last_prompt is not None
    assert planner.last_prompt["runtime_date"]["current_local_date"] == "2026-04-29"
    assert planner.last_prompt["runtime_date"]["current_local_datetime"] == "2026-04-29 10:30:00"
    assert "relative phrases" in planner.last_prompt["runtime_rules"]["temporal_args"]


def test_action_planner_overrides_stale_llm_dates_for_relative_goal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aor_runtime.runtime.temporal.current_local_datetime",
        lambda settings=None: NOW,
    )
    planner = LLMActionPlanner(
        llm=FakeLLM(
            [
                {
                    "goal": "calculate average job time in totalseg. Only consider jobs in the past 7 days",
                    "actions": [
                        {
                            "id": "aggregate_runtime",
                            "tool": "slurm.accounting_aggregate",
                            "inputs": {
                                "partition": "totalseg",
                                "metric": "average_elapsed",
                                "start": "2024-05-13 00:00:00",
                                "end": "2024-05-13 00:00:00",
                                "time_window_label": "2024-05-13",
                            },
                            "output_binding": "runtime_stats",
                            "expected_result_shape": {"kind": "table"},
                        }
                    ],
                    "expected_final_shape": {"kind": "table"},
                }
            ]
        ),
        tools=build_tool_registry(_settings(tmp_path)),
        settings=_settings(tmp_path),
    )

    plan = planner.build_plan(
        goal="calculate average job time in totalseg. Only consider jobs in the past 7 days",
        planner=PlannerConfig(),
        allowed_tools=["slurm.accounting_aggregate"],
        input_payload={},
    )

    assert plan.steps[0].action == "slurm.accounting_aggregate"
    assert plan.steps[0].args["start"] == "2026-04-22 00:00:00"
    assert "end" not in plan.steps[0].args
    assert plan.steps[0].args["time_window_label"] == "Last 7 days"
    assert planner.last_temporal_normalization_metadata["reason"] == "goal_temporal_phrase_overrode_planner_bounds"
