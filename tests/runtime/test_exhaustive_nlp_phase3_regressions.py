from __future__ import annotations

import json
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.intent_classifier import classify_intent
from tests.runtime.exhaustive_nlp_phase3_support import WORKSPACE_NAME, CaseSpec, load_cases, rebuild_workspace


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"
REGRESSION_CASE_IDS = {
    "count_top_02",
    "count_top_04",
    "count_top_08",
    "list_csv_04",
    "list_csv_09",
    "json_03",
    "json_05",
    "bonus_02",
    "bonus_04",
}


def _settings(workspace: Path, sql_payload: dict[str, object]) -> Settings:
    return Settings(
        workspace_root=workspace,
        run_store_path=workspace / "runtime.db",
        sql_databases=dict(sql_payload["sql_databases"]),
        sql_default_database=str(sql_payload["sql_default_database"]),
        response_render_mode="raw",
    )


def _run_case(settings: Settings, prompt: str) -> dict:
    engine = ExecutionEngine(settings)
    return engine.run_spec(str(SPEC_PATH), {"task": prompt})


def _content_matches(case: CaseSpec, content: str) -> bool:
    if case.mode == "json":
        return json.loads(content) == case.expected
    return content == str(case.expected)


@pytest.fixture(scope="module")
def phase3_settings() -> Settings:
    workspace = REPO_ROOT / "artifacts" / WORKSPACE_NAME
    sql_payload = rebuild_workspace(workspace)
    return _settings(workspace, sql_payload)


@pytest.mark.parametrize("case", load_cases(REGRESSION_CASE_IDS), ids=lambda case: case.case_id)
def test_phase3_regression_prompts_classify_deterministically(case: CaseSpec) -> None:
    result = classify_intent(case.prompt)
    assert result.matched is True


@pytest.mark.parametrize("case", load_cases(REGRESSION_CASE_IDS), ids=lambda case: case.case_id)
@pytest.mark.skip(reason="LLM-exclusive runtime no longer supports zero-LLM natural-language execution")
def test_phase3_regression_prompts_run_without_llm(case: CaseSpec, phase3_settings: Settings) -> None:
    state = _run_case(phase3_settings, case.prompt)
    assert state["metrics"]["llm_calls"] == 0
    assert _content_matches(case, state["final_output"]["content"])
