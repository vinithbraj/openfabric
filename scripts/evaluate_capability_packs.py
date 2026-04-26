from __future__ import annotations

import json
import multiprocessing as mp
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


def _evaluate_pack(pack: CapabilityEvalPack, settings_payload: dict[str, Any], render_values: dict[str, Any]) -> tuple[CapabilityEvalResult, list[dict[str, Any]]]:
    strict_pass = 0
    semantic_pass = 0
    llm_fallbacks = 0
    failures: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []

    for case in pack.cases:
        _validate_case_setup(case)
        prompt = str(render_values["render_prompt"](case.prompt))
        expected = render_values["render_expected"](case.expected) if case.expected is not None else None
        expected_contains = render_values["render_expected"](case.expected_contains) if case.expected_contains is not None else None
        expected_regex = render_values["render_expected"](case.expected_regex) if case.expected_regex is not None else None

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
        llm_calls = int(dict(state.get("metrics", {})).get("llm_calls", 0))
        if llm_calls > 0:
            llm_fallbacks += 1

        semantic = _matches_expected(case, content, expected, expected_contains, expected_regex)
        strict = semantic and llm_calls == int(case.expect_llm_calls or 0)

        if semantic:
            semantic_pass += 1
        if strict:
            strict_pass += 1

        failure_category: str | None = None
        if not semantic:
            failure_category = "incorrect_output"
        elif llm_calls != int(case.expect_llm_calls or 0):
            failure_category = "llm_calls_mismatch"

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
            }
        )

    return (
        CapabilityEvalResult(
            capability=pack.capability,
            total=len(pack.cases),
            strict_pass=strict_pass,
            semantic_pass=semantic_pass,
            llm_fallbacks=llm_fallbacks,
            failures=failures,
        ),
        case_results,
    )


def main() -> None:
    packs = load_capability_eval_packs(PACKS_DIR)
    ensure_unique_case_ids(packs)
    fixtures = rebuild_eval_workspace(WORKSPACE)
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
            f"fallbacks {summary['llm_fallbacks']}"
        )

    print(
        f"total: strict {total_strict}/{total_cases} "
        f"semantic {total_semantic}/{total_cases} "
        f"fallbacks {total_fallbacks}"
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
