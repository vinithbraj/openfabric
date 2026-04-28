from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.runtime.exhaustive_nlp_phase3_support import (  # noqa: E402
    WORKSPACE_NAME,
    CaseSpec,
    load_cases,
    rebuild_workspace,
)


SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"
WORKSPACE = REPO_ROOT / "artifacts" / WORKSPACE_NAME
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "exhaustive_nlp_phase3_100.json"
PROMPT_TIMEOUT_SECONDS = 20
STRICT_MIN = 90
SEMANTIC_MIN = 90
FALLBACK_MAX = 10


def _settings(settings_payload: dict[str, Any]) -> Settings:
    return Settings(
        workspace_root=Path(settings_payload["workspace_root"]),
        run_store_path=Path(settings_payload["run_store_path"]),
        sql_databases=dict(settings_payload.get("sql_databases", {})),
        sql_default_database=settings_payload.get("sql_default_database"),
        response_render_mode="raw",
    )


def _run_case(queue: mp.Queue, settings_payload: dict[str, Any], prompt: str) -> None:
    try:
        engine = ExecutionEngine(_settings(settings_payload))
        queue.put({"ok": True, "state": engine.run_spec(str(SPEC_PATH), {"task": prompt})})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc)})


def _strict_pass(case: CaseSpec, content: str, llm_calls: int) -> bool:
    if case.deterministic_expected and llm_calls != 0:
        return False
    return _semantic_pass(case, content)


def _semantic_pass(case: CaseSpec, content: str) -> bool:
    if case.mode == "json":
        try:
            return json.loads(content) == case.expected
        except json.JSONDecodeError:
            return False
    if case.mode == "csv_set":
        return set(_csv_items(content)) == set(case.expected.split(","))
    return content == str(case.expected)


def _csv_items(content: str) -> list[str]:
    return [item.strip() for item in content.split(",") if item.strip()]


def main() -> None:
    sql_payload = rebuild_workspace(WORKSPACE)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings_payload = {
        "workspace_root": str(WORKSPACE),
        "run_store_path": str(WORKSPACE / "runtime.db"),
        "sql_databases": sql_payload["sql_databases"],
        "sql_default_database": sql_payload["sql_default_database"],
    }

    results: list[dict[str, Any]] = []
    strict_pass_count = 0
    semantic_pass_count = 0
    failure_categories: Counter[str] = Counter()
    fallback_prompts: list[str] = []
    failing_case_ids: list[str] = []

    for case in load_cases():
        queue: mp.Queue = mp.Queue()
        started = time.monotonic()
        process = mp.Process(target=_run_case, args=(queue, settings_payload, case.prompt))
        process.start()
        process.join(PROMPT_TIMEOUT_SECONDS)
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)

        if process.is_alive():
            process.terminate()
            process.join()
            failure_categories["timeout"] += 1
            failing_case_ids.append(case.case_id)
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "prompt": case.prompt,
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
            failure_categories["runtime_error"] += 1
            failing_case_ids.append(case.case_id)
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "prompt": case.prompt,
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
        semantic_pass = _semantic_pass(case, content)
        strict_pass = _strict_pass(case, content, llm_calls)

        if strict_pass:
            strict_pass_count += 1
        if semantic_pass:
            semantic_pass_count += 1

        failure_category: str | None = None
        if case.deterministic_expected and llm_calls != 0:
            fallback_prompts.append(case.prompt)
            failure_category = "llm_called"
        if not semantic_pass:
            failure_category = "incorrect_output"
        if failure_category:
            failure_categories[failure_category] += 1
            failing_case_ids.append(case.case_id)

        results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "prompt": case.prompt,
                "strict_pass": strict_pass,
                "semantic_pass": semantic_pass,
                "failure_category": failure_category,
                "llm_calls": llm_calls,
                "latency_ms": elapsed_ms,
                "content": content,
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(WORKSPACE),
        "prompt_count": len(results),
        "strict_pass_count": strict_pass_count,
        "semantic_pass_count": semantic_pass_count,
        "fallback_prompts": fallback_prompts,
        "failure_categories": dict(failure_categories),
        "failing_case_ids": failing_case_ids,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "strict_pass_count": strict_pass_count,
                "semantic_pass_count": semantic_pass_count,
                "fallback_count": len(fallback_prompts),
                "failure_categories": dict(failure_categories),
                "failing_case_ids": failing_case_ids,
                "report_path": str(REPORT_PATH),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    if strict_pass_count < STRICT_MIN or semantic_pass_count < SEMANTIC_MIN or len(fallback_prompts) > FALLBACK_MAX:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
