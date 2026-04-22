from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aor_runtime.runtime.engine import ExecutionEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT / "artifacts" / "eval_100_cases"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "runtime_eval_100.json"
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


@dataclass
class Case:
    case_id: str
    category: str
    prompt: str
    setup: Callable[[Path], dict[str, Any]]
    validate: Callable[[Path, dict[str, Any], dict[str, Any]], tuple[bool, str]]


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _last_answer(state: dict[str, Any]) -> str:
    final_output = state.get("final_output")
    if isinstance(final_output, dict):
        content = final_output.get("content")
        if isinstance(content, str):
            return content
    error = state.get("error")
    if isinstance(error, str):
        return error
    return ""


def setup_empty(case_dir: Path) -> dict[str, Any]:
    _clean_dir(case_dir)
    return {}


def setup_read_fixture(case_dir: Path, lines: list[str]) -> dict[str, Any]:
    _clean_dir(case_dir)
    target = case_dir / "input.txt"
    target.write_text("\n".join(lines) + "\n")
    return {"target": str(target)}


def setup_count_fixture(case_dir: Path, file_count: int) -> dict[str, Any]:
    _clean_dir(case_dir)
    for index in range(file_count):
        (case_dir / f"item_{index:02d}.txt").write_text(f"sample {index}\n")
    (case_dir / "ignore.md").write_text("ignore\n")
    return {"expected_count": file_count}


def setup_overwrite_fixture(case_dir: Path, existing_text: str) -> dict[str, Any]:
    _clean_dir(case_dir)
    target = case_dir / "existing.txt"
    target.write_text(existing_text)
    return {"target": str(target)}


def validate_write(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    target = case_dir / "note.txt"
    expected = context["expected_text"]
    actual = _read_text(target)
    if actual == expected:
        return True, "file created with exact content"
    return False, f"expected {expected!r}, found {actual!r}"


def validate_copy(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    source = case_dir / "source.txt"
    copy = case_dir / "copy.txt"
    expected = context["expected_text"]
    if not source.exists() or not copy.exists():
        return False, "source or copy missing"
    if source.read_text() != expected or copy.read_text() != expected:
        return False, "source/copy content mismatch"
    return True, "source and copy matched expected content"


def validate_nested(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    target = case_dir / "nested" / "deep" / "result.txt"
    expected = context["expected_text"]
    if not target.exists():
        return False, "nested target missing"
    if target.read_text() != expected:
        return False, "nested target content mismatch"
    return True, "nested file created correctly"


def validate_read_phrase(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    answer = _last_answer(state).lower()
    expected_phrase = context["expected_phrase"].lower()
    if expected_phrase in answer:
        return True, "answer contained expected phrase"
    return False, f"expected phrase {context['expected_phrase']!r} missing from answer {answer!r}"


def validate_count(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    answer = _last_answer(state)
    expected = str(context["expected_count"])
    if expected in answer:
        return True, "answer contained expected count"
    return False, f"expected count {expected!r} missing from answer {answer!r}"


def validate_overwrite(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    target = case_dir / "existing.txt"
    expected = context["expected_text"]
    actual = _read_text(target)
    if actual == expected:
        return True, "file overwritten correctly"
    return False, f"expected overwrite {expected!r}, found {actual!r}"


def build_cases() -> list[Case]:
    cases: list[Case] = []

    for index in range(1, 21):
        expected = f"hello world write {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_empty(case_dir)
            return {"expected_text": expected_text}

        prompt = f'create a file named artifacts/eval_100_cases/case_{index:03d}/note.txt and put exactly "{expected.strip()}" in it'
        cases.append(Case(f"case_{index:03d}", "write_file", prompt, setup, validate_write))

    for index in range(21, 41):
        expected = f"copy payload {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_empty(case_dir)
            return {"expected_text": expected_text}

        prompt = (
            f'create a file artifacts/eval_100_cases/case_{index:03d}/source.txt with exactly "{expected.strip()}", '
            f"then create a copy of that file named artifacts/eval_100_cases/case_{index:03d}/copy.txt"
        )
        cases.append(Case(f"case_{index:03d}", "copy_file", prompt, setup, validate_copy))

    for index in range(41, 56):
        expected = f"nested result {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_empty(case_dir)
            return {"expected_text": expected_text}

        prompt = (
            f'create the directory artifacts/eval_100_cases/case_{index:03d}/nested/deep '
            f'and then create a file there named result.txt with exactly "{expected.strip()}"'
        )
        cases.append(Case(f"case_{index:03d}", "nested_create", prompt, setup, validate_nested))

    for index in range(56, 76):
        phrase = f"token-{index:03d}-phrase"
        lines = [f"line one {index}", phrase, f"tail {index}"]

        def setup(case_dir: Path, *, lines_value: list[str] = lines, expected_phrase: str = phrase) -> dict[str, Any]:
            setup_read_fixture(case_dir, lines_value)
            return {"expected_phrase": expected_phrase}

        prompt = (
            f"read the file artifacts/eval_100_cases/case_{index:03d}/input.txt and tell me the exact phrase on line 2"
        )
        cases.append(Case(f"case_{index:03d}", "read_phrase", prompt, setup, validate_read_phrase))

    for index in range(76, 91):
        expected_count = (index % 5) + 2

        def setup(case_dir: Path, *, expected_value: int = expected_count) -> dict[str, Any]:
            return setup_count_fixture(case_dir, expected_value)

        prompt = f"how many txt files are in artifacts/eval_100_cases/case_{index:03d}"
        cases.append(Case(f"case_{index:03d}", "count_files", prompt, setup, validate_count))

    for index in range(91, 101):
        expected = f"overwrite target {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_overwrite_fixture(case_dir, "stale content\n")
            return {"expected_text": expected_text}

        prompt = (
            f'overwrite the file artifacts/eval_100_cases/case_{index:03d}/existing.txt so that it contains exactly "{expected.strip()}"'
        )
        cases.append(Case(f"case_{index:03d}", "overwrite_file", prompt, setup, validate_overwrite))

    return cases


def extract_route(state: dict[str, Any]) -> str:
    plan = state.get("plan", {})
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            first = steps[0]
            if isinstance(first, dict):
                return str(first.get("action") or "")
    return "planner"


def extract_tool_names(state: dict[str, Any]) -> list[str]:
    names: list[str] = []
    history = state.get("history", [])
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                step = item.get("step")
                if isinstance(step, dict) and isinstance(step.get("action"), str):
                    names.append(step["action"])
    return names


def tool_usage_score(category: str, tool_names: list[str]) -> float:
    required: dict[str, set[str]] = {
        "write_file": {"fs.write"},
        "copy_file": {"fs.copy"},
        "nested_create": {"fs.mkdir", "fs.write"},
        "read_phrase": {"python.exec"},
        "count_files": {"python.exec"},
        "overwrite_file": {"fs.write"},
    }
    expected = required.get(category, set())
    if not expected:
        return 1.0
    used = set(tool_names)
    if expected.issubset(used):
        return 1.0
    if used & expected:
        return 0.5
    return 0.0


def classify_failure(state: dict[str, Any], passed: bool, tool_names: list[str]) -> str | None:
    if passed:
        return None
    status = str(state.get("status", ""))
    history = state.get("history", [])
    error = str(state.get("error") or "")
    validation = state.get("validation") or {}
    failure_context = state.get("failure_context") or {}

    if isinstance(failure_context, dict) and failure_context.get("reason") == "tool_execution_failed":
        return "tool_failure"
    if isinstance(failure_context, dict) and failure_context.get("reason") == "validation_failed":
        return "validation_failure"

    if status == "failed" and not history:
        return "planner_failure"
    if any(isinstance(item, dict) and item.get("success") is False for item in history):
        return "tool_failure"
    if validation and not bool(validation.get("success", True)):
        return "validation_failure"
    if not tool_names:
        return "missing_step"
    if "disallowed tool" in error.lower():
        return "wrong_tool"
    return "incorrect_output"


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = ExecutionEngine()
    cases = build_cases()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_dir = WORKSPACE / case.case_id
        context = case.setup(case_dir)
        started = time.perf_counter()
        state = engine.run_spec(str(SPEC_PATH), {"task": case.prompt})
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        passed, detail = case.validate(case_dir, context, state)
        tool_names = extract_tool_names(state)
        metrics = state.get("metrics", {}) if isinstance(state.get("metrics"), dict) else {}
        results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "prompt": case.prompt,
                "passed": passed,
                "detail": detail,
                "duration_ms": duration_ms,
                "correctness": 1.0 if passed else 0.0,
                "tool_usage": tool_usage_score(case.category, tool_names),
                "latency": round(duration_ms / 1000, 4),
                "llm_calls": int(metrics.get("llm_calls", 0)),
                "steps_executed": int(metrics.get("steps_executed", 0)),
                "failure_classification": classify_failure(state, passed, tool_names),
                "run_id": state.get("run_id"),
                "status": state.get("status"),
                "route": extract_route(state),
                "history": state.get("history"),
                "answer": _last_answer(state),
                "tool_names": tool_names,
                "metrics": metrics,
            }
        )
        print(f"[{index:03d}/100] {case.case_id} {case.category}: {'PASS' if passed else 'FAIL'} ({duration_ms} ms) - {detail}")

    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        category = item["category"]
        bucket = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if item["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    summary = {
        "total_cases": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "pass_rate": round(sum(1 for item in results if item["passed"]) / len(results), 4),
        "avg_latency_ms": round(sum(item["duration_ms"] for item in results) / len(results), 2),
        "avg_llm_calls": round(sum(item["llm_calls"] for item in results) / len(results), 4),
        "avg_tool_usage": round(sum(item["tool_usage"] for item in results) / len(results), 4),
        "by_category": by_category,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
