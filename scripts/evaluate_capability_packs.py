from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.eval import (
    CapabilityEvalCase,
    CapabilityEvalPack,
    CapabilityEvalResult,
    ensure_unique_case_ids,
    load_capability_eval_packs,
)
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.eval_fixtures import rebuild_eval_workspace, render_case_prompt, render_template
from aor_runtime.runtime.failure_classifier import classify_failure, generate_prompt_suggestions
from aor_runtime.runtime.llm_intent_extractor import LLM_INTENT_FIXTURE_PATH_ENV
from aor_runtime.runtime.prompt_suggestions import append_prompt_suggestions
from aor_runtime.tools.slurm import SLURM_FIXTURE_DIR_ENV
from aor_runtime.core.contracts import ExecutionPlan


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"
PACKS_DIR = REPO_ROOT / "evals" / "capabilities"
WORKSPACE = REPO_ROOT / "artifacts" / "capability_eval"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "capability_eval_report.json"
PROMPT_TIMEOUT_SECONDS = 20


def _settings(settings_payload: dict[str, Any]) -> Settings:
    return Settings(
        workspace_root=Path(settings_payload["workspace_root"]),
        run_store_path=Path(settings_payload["run_store_path"]),
        sql_databases=dict(settings_payload.get("sql_databases", {})),
        sql_default_database=settings_payload.get("sql_default_database"),
    )


def _run_case(queue: mp.Queue, settings_payload: dict[str, Any], prompt: str) -> None:
    try:
        engine = ExecutionEngine(_settings(settings_payload))
        queue.put({"ok": True, "state": engine.run_spec(str(SPEC_PATH), {"task": prompt})})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc)})


def _validate_case_setup(case: CapabilityEvalCase) -> None:
    setup = case.setup or {}
    fixture_set = str(setup.get("fixture_set", "core")).strip() or "core"
    if fixture_set != "core":
        raise ValueError(f"unsupported fixture_set for {case.id}: {fixture_set}")


def _matches_expected(case: CapabilityEvalCase, content: str, expected: Any, expected_contains: list[str] | None, expected_regex: str | None) -> bool:
    if expected is not None and not _matches_exact_expected(content, expected):
        return False
    if expected_contains is not None and not all(fragment in content for fragment in expected_contains):
        return False
    if expected_regex is not None and re.search(expected_regex, content, re.DOTALL) is None:
        return False
    return True


def _matches_exact_expected(content: str, expected: Any) -> bool:
    if isinstance(expected, (dict, list)):
        try:
            return json.loads(content) == expected
        except json.JSONDecodeError:
            return False
    return content == str(expected)


def _evaluate_prompt_suggestion_case(
    case: CapabilityEvalCase,
    render_values: dict[str, Any],
) -> dict[str, Any]:
    setup = render_values["render_expected"](case.setup or {})
    prompt = str(render_values["render_prompt"](case.prompt))
    metadata = dict(setup.get("metadata") or {})
    plan_payload = setup.get("plan")
    plan = ExecutionPlan.model_validate(plan_payload) if isinstance(plan_payload, dict) else None
    error_detail = str(setup.get("error_detail") or "").strip()
    error = RuntimeError(error_detail) if error_detail else None
    failure_type = str(setup.get("force_error_type") or classify_failure(prompt, error=error, plan=plan, metadata=metadata))
    suggestion_result = generate_prompt_suggestions(
        prompt,
        failure_type,
        context=dict(setup.get("context") or {}),
    )
    base_message = str(setup.get("base_message") or suggestion_result.message)
    content = append_prompt_suggestions(base_message, suggestion_result)
    return {
        "content": content,
        "llm_calls": 0,
        "failure_type": failure_type,
        "suggestion_count": len(suggestion_result.suggestions),
    }


def _check_setup_expectations(
    setup: dict[str, Any],
    *,
    failure_type: str | None,
    suggestion_count: int | None,
) -> tuple[bool, str | None]:
    expected_failure_type = str(setup.get("expected_failure_type") or "").strip()
    if expected_failure_type and expected_failure_type != str(failure_type or ""):
        return False, "failure_type_mismatch"

    minimum_suggestions = int(setup.get("min_suggestion_count") or 0)
    if minimum_suggestions and int(suggestion_count or 0) < minimum_suggestions:
        return False, "suggestion_count_mismatch"

    return True, None


def _evaluate_pack(pack: CapabilityEvalPack, settings_payload: dict[str, Any], render_values: dict[str, Any]) -> tuple[CapabilityEvalResult, list[dict[str, Any]]]:
    strict_pass = 0
    semantic_pass = 0
    llm_fallbacks = 0
    llm_intent_calls = 0
    raw_planner_llm_calls = 0
    deterministic_calls = 0
    failures: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []

    for case in pack.cases:
        _validate_case_setup(case)
        prompt = str(render_values["render_prompt"](case.prompt))
        expected = render_values["render_expected"](case.expected) if case.expected is not None else None
        expected_contains = render_values["render_expected"](case.expected_contains) if case.expected_contains is not None else None
        expected_regex = render_values["render_expected"](case.expected_regex) if case.expected_regex is not None else None
        setup = render_values["render_expected"](case.setup or {})

        if str(setup.get("evaluation_mode") or "") == "suggestion_only":
            started = time.monotonic()
            synthetic_result = _evaluate_prompt_suggestion_case(case, render_values)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            content = str(synthetic_result["content"])
            llm_calls = int(synthetic_result["llm_calls"])
            semantic = _matches_expected(case, content, expected, expected_contains, expected_regex)
            metadata_ok, metadata_reason = _check_setup_expectations(
                setup,
                failure_type=str(synthetic_result.get("failure_type") or ""),
                suggestion_count=int(synthetic_result.get("suggestion_count") or 0),
            )
            strict = semantic and llm_calls == int(case.expect_llm_calls or 0) and metadata_ok
            if semantic:
                semantic_pass += 1
            if strict:
                strict_pass += 1

            failure_category: str | None = None
            if not semantic:
                failure_category = "incorrect_output"
            elif llm_calls != int(case.expect_llm_calls or 0):
                failure_category = "llm_calls_mismatch"
            elif not metadata_ok:
                failure_category = str(metadata_reason)

            if failure_category is not None:
                failures.append(
                    {
                        "capability": pack.capability,
                        "case_id": case.id,
                        "category": case.category,
                        "reason": failure_category,
                        "prompt": prompt,
                        "expected": expected,
                        "expected_contains": expected_contains,
                        "expected_regex": expected_regex,
                        "content": content,
                        "llm_calls": llm_calls,
                        "latency_ms": elapsed_ms,
                    }
                )

            case_results.append(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "prompt": prompt,
                    "strict_pass": strict,
                    "semantic_pass": semantic,
                    "failure_category": failure_category,
                    "llm_calls": llm_calls,
                    "latency_ms": elapsed_ms,
                    "content": content,
                    "failure_type": synthetic_result.get("failure_type"),
                    "suggestion_count": synthetic_result.get("suggestion_count"),
                }
            )
            continue

        queue: mp.Queue = mp.Queue()
        started = time.monotonic()
        process = mp.Process(target=_run_case, args=(queue, settings_payload, prompt))
        process.start()
        process.join(PROMPT_TIMEOUT_SECONDS)
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)

        if process.is_alive():
            process.terminate()
            process.join()
            failure = {
                "capability": pack.capability,
                "case_id": case.id,
                "category": case.category,
                "reason": "timeout",
                "prompt": prompt,
                "latency_ms": elapsed_ms,
            }
            failures.append(failure)
            case_results.append(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "prompt": prompt,
                    "strict_pass": False,
                    "semantic_pass": False,
                    "failure_category": "timeout",
                    "llm_calls": None,
                    "latency_ms": elapsed_ms,
                }
            )
            continue

        payload = queue.get() if not queue.empty() else {"ok": False, "error": "no_result"}
        if not payload.get("ok"):
            failure = {
                "capability": pack.capability,
                "case_id": case.id,
                "category": case.category,
                "reason": "runtime_error",
                "prompt": prompt,
                "error": payload.get("error"),
                "latency_ms": elapsed_ms,
            }
            failures.append(failure)
            case_results.append(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "prompt": prompt,
                    "strict_pass": False,
                    "semantic_pass": False,
                    "failure_category": "runtime_error",
                    "error": payload.get("error"),
                    "llm_calls": None,
                    "latency_ms": elapsed_ms,
                }
            )
            continue

        state = dict(payload["state"])
        content = str(dict(state.get("final_output", {})).get("content", ""))
        metrics = dict(state.get("metrics", {}))
        llm_calls = int(metrics.get("llm_calls", 0))
        case_llm_intent_calls = int(metrics.get("llm_intent_calls", 0))
        case_raw_planner_llm_calls = int(metrics.get("raw_planner_llm_calls", 0))
        final_output_metadata = dict(dict(state.get("final_output", {})).get("metadata", {}))
        planning_mode = str(final_output_metadata.get("planning_mode") or dict(state.get("planning_metadata") or {}).get("planning_mode") or "")
        metadata_ok, metadata_reason = _check_setup_expectations(
            setup,
            failure_type=str(final_output_metadata.get("failure_type") or ""),
            suggestion_count=int(final_output_metadata.get("suggestion_count") or 0),
        )
        if llm_calls > 0:
            llm_fallbacks += 1
        llm_intent_calls += case_llm_intent_calls
        raw_planner_llm_calls += case_raw_planner_llm_calls
        if planning_mode == "deterministic_intent":
            deterministic_calls += 1

        semantic = _matches_expected(case, content, expected, expected_contains, expected_regex)
        strict = (
            semantic
            and llm_calls == int(case.expect_llm_calls or 0)
            and metadata_ok
            and int(setup.get("expected_raw_planner_llm_calls") or 0) == case_raw_planner_llm_calls
        )

        if semantic:
            semantic_pass += 1
        if strict:
            strict_pass += 1

        failure_category: str | None = None
        if not semantic:
            failure_category = "incorrect_output"
        elif llm_calls != int(case.expect_llm_calls or 0):
            failure_category = "llm_calls_mismatch"
        elif int(setup.get("expected_raw_planner_llm_calls") or 0) != case_raw_planner_llm_calls:
            failure_category = "raw_planner_llm_calls_mismatch"
        elif not metadata_ok:
            failure_category = str(metadata_reason)

        if failure_category is not None:
            failures.append(
                {
                    "capability": pack.capability,
                    "case_id": case.id,
                    "category": case.category,
                    "reason": failure_category,
                    "prompt": prompt,
                    "expected": expected,
                    "expected_contains": expected_contains,
                    "expected_regex": expected_regex,
                    "content": content,
                    "llm_calls": llm_calls,
                    "latency_ms": elapsed_ms,
                }
            )

        case_results.append(
            {
                "case_id": case.id,
                "category": case.category,
                "prompt": prompt,
                "strict_pass": strict,
                "semantic_pass": semantic,
                "failure_category": failure_category,
                "llm_calls": llm_calls,
                "llm_intent_calls": case_llm_intent_calls,
                "raw_planner_llm_calls": case_raw_planner_llm_calls,
                "planning_mode": planning_mode,
                "latency_ms": elapsed_ms,
                "content": content,
                "failure_type": final_output_metadata.get("failure_type"),
                "suggestion_count": final_output_metadata.get("suggestion_count"),
            }
        )

    return (
        CapabilityEvalResult(
            capability=pack.capability,
            total=len(pack.cases),
            strict_pass=strict_pass,
            semantic_pass=semantic_pass,
            llm_fallbacks=llm_fallbacks,
            llm_intent_calls=llm_intent_calls,
            raw_planner_llm_calls=raw_planner_llm_calls,
            deterministic_calls=deterministic_calls,
            failures=failures,
        ),
        case_results,
    )


def main() -> None:
    packs = load_capability_eval_packs(PACKS_DIR)
    ensure_unique_case_ids(packs)
    fixtures = rebuild_eval_workspace(WORKSPACE)
    os.environ["AOR_ENABLE_LLM_INTENT_EXTRACTION"] = "1"
    os.environ[SLURM_FIXTURE_DIR_ENV] = str(fixtures.variables.get("slurm_fixture_dir", ""))
    os.environ[LLM_INTENT_FIXTURE_PATH_ENV] = str(
        fixtures.variables.get("llm_intent_fixture_path", fixtures.variables.get("slurm_llm_intent_fixture_path", ""))
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings_payload = fixtures.settings_payload()
    render_values = {
        "render_prompt": lambda value: render_case_prompt(value, fixtures),
        "render_expected": lambda value: render_template(value, fixtures),
    }

    pack_reports: list[dict[str, Any]] = []
    total_cases = 0
    total_strict = 0
    total_semantic = 0
    total_fallbacks = 0
    total_llm_intent_calls = 0
    total_raw_planner_llm_calls = 0
    total_deterministic_calls = 0
    threshold_failures: list[dict[str, Any]] = []

    for pack in packs:
        result, case_results = _evaluate_pack(pack, settings_payload, render_values)
        strict_ratio = result.strict_pass / result.total if result.total else 0.0
        semantic_ratio = result.semantic_pass / result.total if result.total else 0.0
        thresholds_passed = (
            strict_ratio >= pack.strict_threshold
            and semantic_ratio >= pack.semantic_threshold
            and result.llm_fallbacks <= pack.max_llm_fallbacks
        )
        if not thresholds_passed:
            threshold_failures.append(
                {
                    "capability": pack.capability,
                    "strict_ratio": strict_ratio,
                    "semantic_ratio": semantic_ratio,
                    "llm_fallbacks": result.llm_fallbacks,
                    "strict_threshold": pack.strict_threshold,
                    "semantic_threshold": pack.semantic_threshold,
                    "max_llm_fallbacks": pack.max_llm_fallbacks,
                }
            )

        total_cases += result.total
        total_strict += result.strict_pass
        total_semantic += result.semantic_pass
        total_fallbacks += result.llm_fallbacks
        total_llm_intent_calls += result.llm_intent_calls
        total_raw_planner_llm_calls += result.raw_planner_llm_calls
        total_deterministic_calls += result.deterministic_calls
        pack_reports.append(
            {
                "capability": pack.capability,
                "strict_threshold": pack.strict_threshold,
                "semantic_threshold": pack.semantic_threshold,
                "max_llm_fallbacks": pack.max_llm_fallbacks,
                "strict_ratio": strict_ratio,
                "semantic_ratio": semantic_ratio,
                "thresholds_passed": thresholds_passed,
                "summary": result.model_dump(),
                "case_results": case_results,
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(WORKSPACE),
        "spec_path": str(SPEC_PATH),
        "totals": {
            "total": total_cases,
            "strict_pass": total_strict,
            "semantic_pass": total_semantic,
            "llm_fallbacks": total_fallbacks,
            "llm_intent_calls": total_llm_intent_calls,
            "raw_planner_llm_calls": total_raw_planner_llm_calls,
            "deterministic_calls": total_deterministic_calls,
        },
        "packs": pack_reports,
        "threshold_failures": threshold_failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for pack_report in pack_reports:
        summary = pack_report["summary"]
        print(
            f"{pack_report['capability']}: "
            f"strict {summary['strict_pass']}/{summary['total']} "
            f"semantic {summary['semantic_pass']}/{summary['total']} "
            f"fallbacks {summary['llm_fallbacks']} "
            f"llm_intent_calls {summary['llm_intent_calls']} "
            f"raw_planner_llm_calls {summary['raw_planner_llm_calls']}"
        )

    print(
        f"total: strict {total_strict}/{total_cases} "
        f"semantic {total_semantic}/{total_cases} "
        f"fallbacks {total_fallbacks} "
        f"llm_intent_calls {total_llm_intent_calls} "
        f"raw_planner_llm_calls {total_raw_planner_llm_calls}"
    )

    print("failures:")
    any_failures = False
    for pack_report in pack_reports:
        for failure in pack_report["summary"]["failures"]:
            any_failures = True
            print(f"- {failure['capability']} {failure['case_id']}: {failure['reason']}")
    if not any_failures:
        print("- none")

    print(f"gate_status: {'passed' if not threshold_failures else 'failed'}")

    if threshold_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
