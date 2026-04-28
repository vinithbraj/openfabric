from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.insights import InsightContext, generate_sql_insights, summarize_insights_with_llm


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def test_llm_insights_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            calls.append(kwargs["user_prompt"])
            return "Summary"

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    facts = {"domain": "sql", "result": {"count": 2}}
    deterministic = generate_sql_insights(facts)

    summary = summarize_insights_with_llm(facts, deterministic, InsightContext(enable_llm=False), _settings(tmp_path))

    assert summary is None
    assert calls == []


def test_llm_insights_receive_sanitized_facts_only(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            prompts.append(kwargs["user_prompt"])
            return "The SQL result count is **2**."

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    facts = {
        "domain": "sql",
        "result": {"count": 2},
        "row_count": 1,
        "constraints_applied": ["age > 70"],
    }
    deterministic = generate_sql_insights(facts)

    summary = summarize_insights_with_llm(
        facts,
        deterministic,
        InsightContext(enable_llm=True, max_input_chars=4000),
        _settings(tmp_path, enable_llm_insights=True),
    )

    assert summary == "The SQL result count is **2**."
    assert prompts
    assert "coverage" not in prompts[0]
    assert "semantic_frame" not in prompts[0]
    assert "rows" not in prompts[0]


def test_llm_insights_reject_raw_or_secret_facts(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            calls.append(kwargs["user_prompt"])
            return "Unsafe"

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", FakeLLM)
    unsafe_facts = {"domain": "sql", "rows": [{"PatientName": "Alice"}], "api_token": "secret"}
    deterministic = generate_sql_insights({"domain": "sql"})

    summary = summarize_insights_with_llm(
        unsafe_facts,
        deterministic,
        InsightContext(enable_llm=True),
        _settings(tmp_path, enable_llm_insights=True),
    )

    assert summary is None
    assert calls == []


def test_llm_insight_failure_falls_back_to_none(tmp_path: Path, monkeypatch) -> None:
    class BrokenLLM:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, **kwargs):
            raise RuntimeError("nope")

    monkeypatch.setattr("aor_runtime.llm.client.LLMClient", BrokenLLM)
    facts = {"domain": "sql", "result": {"count": 2}}
    deterministic = generate_sql_insights(facts)

    assert summarize_insights_with_llm(facts, deterministic, InsightContext(enable_llm=True), _settings(tmp_path)) is None
