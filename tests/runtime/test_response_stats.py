from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.response_stats import append_response_stats


def test_response_stats_include_backend_tools_metrics_and_dag_steps() -> None:
    state = {
        "plan": {
            "steps": [
                {"id": 1, "action": "slurm.accounting_aggregate", "output": "aggregate_result"},
                {"id": 2, "action": "text.format", "output": "formatted_output"},
                {"id": 3, "action": "runtime.return"},
            ]
        }
    }
    metrics = {
        "latency_ms": 1530,
        "llm_calls": 2,
        "llm_prompt_tokens": 1200,
        "llm_completion_tokens": 340,
    }

    rendered = append_response_stats("## Result\n\nDone", state=state, metrics=metrics, status="completed")

    assert "## Stats" in rendered
    assert "| `Backend` | `SLURM gateway` |" in rendered
    assert "| `Tools` | `slurm.accounting_aggregate` |" in rendered
    assert "| `Status` | `Completed` |" in rendered
    assert "| `Time Taken` | `1.53 s` |" in rendered
    assert "| `LLM Passes` | `2` |" in rendered
    assert "| `Tokens In` | `1,200` |" in rendered
    assert "| `Tokens Out` | `340` |" in rendered
    assert "| `Steps` | `3` |" in rendered
    assert "## DAG Steps" in rendered
    assert "1. `slurm.accounting_aggregate` -> `aggregate_result`" in rendered
    assert "2. `text.format` -> `formatted_output`" in rendered
    assert "3. `runtime.return`" in rendered


def test_response_stats_report_unavailable_tokens_when_provider_does_not_return_usage() -> None:
    rendered = append_response_stats(
        "## Result\n\nDone",
        state={"plan": {"steps": [{"id": 1, "action": "sql.query", "output": "rows"}]}},
        metrics={"latency_ms": 20, "llm_calls": 1},
        status="completed",
    )

    assert "| `Tokens In` | `Unavailable` |" in rendered
    assert "| `Tokens Out` | `Unavailable` |" in rendered


def test_response_stats_are_not_added_twice() -> None:
    content = append_response_stats(
        "## Result\n\nDone",
        state={"plan": {"steps": [{"id": 1, "action": "fs.list", "output": "entries"}]}},
        metrics={},
        status="completed",
    )

    rendered = append_response_stats(content, state={}, metrics={}, status="completed")

    assert rendered.count("## Stats") == 1


def test_engine_final_decoration_appends_response_stats(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", show_response_stats=True)
    engine = ExecutionEngine(settings)
    final_output = {"content": "## Result\n\nDone", "artifacts": [], "metadata": {}}
    state = {
        "goal": "show slurm aggregate",
        "planning_metadata": {},
        "plan": {
            "steps": [
                {"id": 1, "action": "slurm.accounting_aggregate", "output": "groups"},
                {"id": 2, "action": "runtime.return"},
            ]
        },
        "attempt_history": [],
    }

    decorated = engine._decorate_final_output(
        state, final_output, status="completed", metrics={"latency_ms": 100, "llm_calls": 0, "steps_executed": 2}
    )

    assert "## Stats" in decorated["content"]
    assert "| `Backend` | `SLURM gateway` |" in decorated["content"]
    assert "1. `slurm.accounting_aggregate` -> `groups`" in decorated["content"]
