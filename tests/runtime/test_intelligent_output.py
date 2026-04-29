from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.response_renderer import ResponseRenderContext, render_agent_response


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def _multi_metric_payload() -> dict:
    return {
        "min": {
            "result_kind": "accounting_aggregate",
            "metric": "min_elapsed",
            "partition": "totalseg",
            "job_count": 364,
            "average_elapsed_human": "10m 53s",
            "min_elapsed_human": "0s",
            "max_elapsed_human": "1h 11m 3s",
            "sum_elapsed_human": "2d 18h 3m 6s",
            "value_human": "0s",
            "source": "sacct",
            "time_window_label": "Last 7 days",
        },
        "max": {
            "result_kind": "accounting_aggregate",
            "metric": "max_elapsed",
            "partition": "totalseg",
            "job_count": 364,
            "average_elapsed_human": "10m 53s",
            "min_elapsed_human": "0s",
            "max_elapsed_human": "1h 11m 3s",
            "sum_elapsed_human": "2d 18h 3m 6s",
            "value_human": "1h 11m 3s",
            "source": "sacct",
            "time_window_label": "Last 7 days",
        },
        "avg": {
            "result_kind": "accounting_aggregate",
            "metric": "average_elapsed",
            "partition": "totalseg",
            "job_count": 364,
            "average_elapsed_human": "10m 53s",
            "min_elapsed_human": "0s",
            "max_elapsed_human": "1h 11m 3s",
            "sum_elapsed_human": "2d 18h 3m 10s",
            "value_human": "10m 53s",
            "source": "sacct",
            "time_window_label": "Last 7 days",
        },
    }


def _history(payload: dict) -> list[StepLog]:
    return [
        StepLog(
            step=ExecutionStep(
                id=1,
                action="slurm.accounting_aggregate",
                args={"partition": "totalseg", "metric": "average_elapsed", "start": "2026-04-22 00:00:00"},
                output="runtime_stats",
            ),
            result=payload["avg"],
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="text.format", args={"source": {"$ref": "runtime_stats"}, "format": "markdown"}, output="formatted_output"),
            result={"content": "| field | value |"},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=3, action="runtime.return", args={"value": {"$ref": "formatted_output", "path": "content"}}),
            result={"value": payload, "output": str(payload)},
            success=True,
        ),
    ]


def test_multi_metric_slurm_result_prefers_domain_renderer_when_wrapped_by_text_format() -> None:
    payload = _multi_metric_payload()

    rendered = render_agent_response(
        payload,
        execution_events=_history(payload),
        context=ResponseRenderContext(source_action="text.format", goal="min max average runtime for totalseg"),
    )

    assert "## SLURM Job Runtime Summary" in rendered.markdown
    assert "| Metric | Value | Jobs | Average | Min | Max | Total |" in rendered.markdown
    assert "Minimum runtime" in rendered.markdown
    assert "| `min` |" not in rendered.markdown


def test_intelligent_output_compare_uses_field_catalog_without_values(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete_json(self, **kwargs):
            prompts.append(kwargs["user_prompt"])
            return {
                "title": "Requested runtime metrics",
                "render_style": "table",
                "selected_fields": ["metric", "value", "jobs", "time_window_label"],
                "rationale": "The user asked for min, max, and average runtime over a time window.",
                "confidence": 0.95,
            }

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    payload = _multi_metric_payload()

    rendered = render_agent_response(
        payload,
        execution_events=_history(payload),
        context=ResponseRenderContext(
            source_action="text.format",
            goal="calculate min max and average times for totalseg over the past 7 days",
            intelligent_output_mode="compare",
            llm_settings=_settings(tmp_path, intelligent_output_mode="compare"),
        ),
    )

    assert "## Intelligent Output" in rendered.markdown
    assert "Requested runtime metrics" in rendered.markdown
    assert "| Metric | Value | Jobs | Time Window |" in rendered.markdown
    assert "| `Minimum runtime` | `0s` | `364` | `Last 7 days` |" in rendered.markdown
    assert prompts
    assert "available_fields" in prompts[0]
    assert "selected_fields" not in prompts[0]
    assert "10m 53s" not in prompts[0]
    assert "1h 11m 3s" not in prompts[0]
    assert "value_human" not in prompts[0]


def test_invalid_intelligent_output_selection_falls_back_to_deterministic_output(tmp_path: Path, monkeypatch) -> None:
    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete_json(self, **kwargs):
            return {"selected_fields": ["raw_payload", "does_not_exist"], "render_style": "table"}

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    payload = _multi_metric_payload()

    rendered = render_agent_response(
        payload,
        execution_events=_history(payload),
        context=ResponseRenderContext(
            source_action="text.format",
            goal="calculate min max and average times for totalseg over the past 7 days",
            intelligent_output_mode="compare",
            llm_settings=_settings(tmp_path, intelligent_output_mode="compare"),
        ),
    )

    assert "## SLURM Job Runtime Summary" in rendered.markdown
    assert "## Intelligent Output" not in rendered.markdown
