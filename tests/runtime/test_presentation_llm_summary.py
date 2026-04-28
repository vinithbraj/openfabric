from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.presentation import (
    PresentationContext,
    build_sanitized_presentation_facts,
    summarize_presented_facts_with_llm,
    validate_presentation_facts_for_llm,
)
from aor_runtime.runtime.response_renderer import ResponseRenderContext, render_agent_response


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def test_llm_summary_is_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            calls.append(kwargs["user_prompt"])
            return "Summary text"

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    summary = summarize_presented_facts_with_llm(
        {"domain": "sql", "result": {"count": 2}},
        PresentationContext(enable_llm_summary=False),
        _settings(tmp_path),
    )

    assert summary is None
    assert calls == []


def test_llm_summary_receives_sanitized_facts_only(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            prompts.append(kwargs["user_prompt"])
            return "The query returned **2** rows."

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    result = {
        "database": "dicom",
        "row_count": 2,
        "rows": [{"PatientName": "Alice"}, {"PatientName": "Bob"}],
        "coverage": {"internal": True},
        "sql_semantic_frame": {"internal": True},
    }
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"database": "dicom", "query": "SELECT PatientName FROM patients"}),
            result=result,
            success=True,
        )
    ]

    rendered = render_agent_response(
        result,
        execution_events=history,
        context=ResponseRenderContext(
            source_action="sql.query",
            enable_insight_layer=False,
            enable_llm_summary=True,
            llm_settings=_settings(tmp_path, enable_presentation_llm_summary=True),
        ),
    )

    assert "## Summary" in rendered.markdown
    assert prompts
    payload = prompts[0]
    assert "Alice" not in payload
    assert "Bob" not in payload
    assert "coverage" not in payload
    assert "semantic_frame" not in payload
    assert "rows" not in payload


def test_sanitized_facts_builder_drops_raw_slurm_internals() -> None:
    result = {
        "results": {
            "cluster_summary": {
                "metric_group": "cluster_summary",
                "payload": {"queue_count": 4, "running_jobs": 1, "pending_jobs": 3, "problematic_nodes": 2},
            }
        },
        "stdout": "raw",
        "coverage": {"covered_requests": ["r1"]},
        "slurm_semantic_frame": {"requests": ["internal"]},
    }

    facts = build_sanitized_presentation_facts(result, [], PresentationContext(source_action="slurm.metrics"))
    rendered = json.dumps(facts, sort_keys=True)

    assert facts["domain"] == "slurm"
    assert facts["queue"]["running_jobs"] == 1
    assert "stdout" not in rendered
    assert "coverage" not in rendered
    assert "semantic_frame" not in rendered


def test_fact_validation_rejects_unsafe_fields() -> None:
    assert not validate_presentation_facts_for_llm({"stdout": "raw"})
    assert not validate_presentation_facts_for_llm({"stderr": "raw"})
    assert not validate_presentation_facts_for_llm({"coverage": {"x": 1}})
    assert not validate_presentation_facts_for_llm({"semantic_frame": {"x": 1}})
    assert not validate_presentation_facts_for_llm({"api_token": "secret"})
    assert not validate_presentation_facts_for_llm({"message": "x" * 5000}, max_string_length=100)


def test_llm_failure_falls_back_to_deterministic_markdown(tmp_path: Path, monkeypatch) -> None:
    class BrokenLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            raise RuntimeError("llm unavailable")

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", BrokenLLM)

    rendered = render_agent_response(
        2,
        execution_events=[
            StepLog(
                step=ExecutionStep(id=1, action="sql.query", args={"database": "dicom", "query": "SELECT COUNT(*) FROM patients"}),
                result={"database": "dicom", "row_count": 1, "rows": [{"count": 2}]},
                success=True,
            )
        ],
        context=ResponseRenderContext(
            source_action="sql.query",
            output_mode="count",
            enable_insight_layer=False,
            enable_llm_summary=True,
            llm_settings=_settings(tmp_path, enable_presentation_llm_summary=True),
        ),
    )

    assert "## Summary" not in rendered.markdown
    assert "Count: 2" in rendered.markdown
    assert "## Query Used" in rendered.markdown
