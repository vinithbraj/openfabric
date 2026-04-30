from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = REPO_ROOT / "evals" / "runtime" / "live_100.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "reports" / "live_100_runtime_report.json"

FAILURE_PREFIX_RE = re.compile(r"(^|\n)\s*(Failed:|Validation failed|Traceback)", re.IGNORECASE)
REFERENCE_RE = re.compile(r"Reference path not found|Unknown step output reference|unresolved reference", re.IGNORECASE)
ARCHITECTURE_RE = re.compile(
    r"unknown action|Planner produced|validation error for|Field required|literal_error|Pydantic|extra_forbidden",
    re.IGNORECASE,
)
SQL_SCHEMA_RE = re.compile(
    r"unknown column|undefinedcolumn|column .* does not exist|table .* does not exist|"
    r"SQL references unknown|relation .* does not exist",
    re.IGNORECASE,
)
SQL_TIMEOUT_RE = re.compile(r"SQL query timed out|statement timeout|canceling statement due to statement timeout", re.IGNORECASE)
SLURM_TEMPORAL_RE = re.compile(r"time_window_label|slurm start must be|slurm end must be", re.IGNORECASE)
RAW_JSON_RE = re.compile(r"^\s*[\[{]")
TOOL_DOMAIN_RE = re.compile(r"wrong tool|domain mismatch|SLURM.*process|process.*SLURM", re.IGNORECASE | re.DOTALL)
DATA_UNAVAILABLE_RE = re.compile(
    r"No matching|0 rows|no rows|not available|unavailable|not found|does not appear|not present",
    re.IGNORECASE,
)
SUGGESTIONS_RE = re.compile(r"Suggested prompts:", re.IGNORECASE)


@dataclass(frozen=True)
class EvalCase:
    id: int
    domain: str
    prompt: str
    expected_outcome: str = "pass"
    expected_issue_class: str | None = None
    timeout_seconds: int = 90
    sensitive_output: bool = True


def main() -> int:
    args = _parse_args()
    cases = _load_cases(args.cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.id in wanted]
    if args.domain:
        wanted_domains = set(args.domain)
        cases = [case for case in cases if case.domain in wanted_domains]
    if not cases:
        raise SystemExit("No eval cases matched the requested filters.")

    model = args.model
    if model == "auto":
        model = _discover_model(args.base_url)

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for case in cases:
        timeout = int(args.timeout or case.timeout_seconds)
        record = _run_case(args.base_url, model, case, timeout=timeout)
        records.append(record)
        issue_text = ",".join(record["issues"]) if record["issues"] else "-"
        print(
            f"{case.id:03d} {record['status'].upper():4s} "
            f"{record['elapsed_seconds']:6.2f}s {case.domain:15s} {issue_text}",
            flush=True,
        )

    report = _build_report(records, model=model, base_url=args.base_url, elapsed_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report written to {args.output}")
    print(json.dumps(report["summary"], sort_keys=True))
    return 1 if _has_unexpected_failures(records) else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live 100-case OpenFABRIC runtime eval.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8310")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-id", type=int, action="append", default=[])
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _load_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in payload.get("cases", []):
        cases.append(
            EvalCase(
                id=int(item["id"]),
                domain=str(item["domain"]),
                prompt=str(item["prompt"]),
                expected_outcome=str(item.get("expected_outcome") or "pass"),
                expected_issue_class=item.get("expected_issue_class"),
                timeout_seconds=int(item.get("timeout_seconds") or 90),
                sensitive_output=bool(item.get("sensitive_output", True)),
            )
        )
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Eval case ids must be unique.")
    return sorted(cases, key=lambda case: case.id)


def _discover_model(base_url: str) -> str:
    response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=10)
    response.raise_for_status()
    data = response.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError("No models returned by /v1/models.")
    return str(models[0]["id"])


def _run_case(base_url: str, model: str, case: EvalCase, *, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    content = ""
    http_status = 0
    transport_error = ""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": case.prompt}], "stream": False},
            timeout=timeout,
        )
        http_status = response.status_code
        if response.status_code >= 400:
            content = response.text
        else:
            payload = response.json()
            content = str(payload["choices"][0]["message"].get("content") or "")
    except Exception as exc:  # noqa: BLE001
        transport_error = str(exc)

    elapsed = round(time.monotonic() - started, 2)
    status, issues = _classify(content, http_status=http_status, transport_error=transport_error)
    return {
        "id": case.id,
        "domain": case.domain,
        "prompt": case.prompt,
        "expected_outcome": case.expected_outcome,
        "expected_issue_class": case.expected_issue_class,
        "status": status,
        "issues": issues,
        "elapsed_seconds": elapsed,
        "http_status": http_status,
        "output_sha256_12": hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:12],
        "output_chars": len(content),
        "artifacted": bool(re.search(r"Rows written:\s*\d+", content) and "Output file:" in content),
        "tools": _extract_stats_value(content, "Tools"),
        "llm_passes": _extract_int_stats_value(content, "LLM Passes"),
        "steps": _extract_int_stats_value(content, "Steps"),
        "failure_excerpt": _safe_failure_excerpt(content, transport_error=transport_error) if status == "fail" else "",
    }


def _classify(content: str, *, http_status: int, transport_error: str) -> tuple[str, list[str]]:
    if transport_error:
        issue = "transport_timeout" if "timed out" in transport_error.lower() else "transport_error"
        return "fail", [issue]
    if http_status >= 400:
        return "fail", ["api_error"]
    text = str(content or "")
    stripped = text.strip()
    if not stripped:
        return "fail", ["architecture_boundary"]

    issues: list[str] = []
    hard_failure = bool(FAILURE_PREFIX_RE.search(text))
    if REFERENCE_RE.search(text):
        issues.append("dataflow_reference")
    if ARCHITECTURE_RE.search(text):
        issues.append("architecture_boundary")
    if SQL_SCHEMA_RE.search(text):
        issues.append("sql_schema_relationship")
    if SQL_TIMEOUT_RE.search(text):
        issues.append("sql_cost_timeout")
    if SLURM_TEMPORAL_RE.search(text) and ("extra_forbidden" in text or "Invalid inputs for slurm" in text):
        issues.append("slurm_temporal_schema")
    if TOOL_DOMAIN_RE.search(text):
        issues.append("tool_domain")
    if RAW_JSON_RE.search(stripped):
        issues.append("formatting_presentation")
        hard_failure = True
    if SUGGESTIONS_RE.search(text):
        issues.append("formatting_presentation")
    if DATA_UNAVAILABLE_RE.search(text):
        issues.append("data_unavailable")
    if len(text) > 30000 and "Output file:" not in text:
        issues.append("formatting_presentation")

    failure_issues = {
        "architecture_boundary",
        "dataflow_reference",
        "sql_schema_relationship",
        "sql_cost_timeout",
        "slurm_temporal_schema",
        "formatting_presentation",
    }
    if hard_failure or any(issue in failure_issues for issue in issues):
        return "fail", sorted(set(issues))
    if issues:
        return "warn", sorted(set(issues))
    return "pass", []


def _safe_failure_excerpt(content: str, *, transport_error: str) -> str:
    text = transport_error or content
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    return text[:700]


def _extract_stats_value(content: str, field: str) -> str | None:
    pattern = re.compile(rf"`?{re.escape(field)}`?\s*\|\s*`?([^`\n|]+)`?", re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    block_pattern = re.compile(rf"{re.escape(field)}\s*\n\s*([^\n]+)", re.IGNORECASE)
    match = block_pattern.search(content)
    return match.group(1).strip() if match else None


def _extract_int_stats_value(content: str, field: str) -> int | None:
    value = _extract_stats_value(content, field)
    if value is None:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _build_report(records: list[dict[str, Any]], *, model: str, base_url: str, elapsed_seconds: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(records),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "by_status": {},
        "by_issue": {},
        "by_domain": {},
    }
    for record in records:
        status = str(record["status"])
        domain = str(record["domain"])
        summary["by_status"][status] = int(summary["by_status"].get(status, 0)) + 1
        summary["by_domain"].setdefault(domain, {})
        summary["by_domain"][domain][status] = int(summary["by_domain"][domain].get(status, 0)) + 1
        for issue in record["issues"]:
            summary["by_issue"][issue] = int(summary["by_issue"].get(issue, 0)) + 1
    return {
        "name": "live_100_runtime",
        "model": model,
        "base_url": base_url,
        "summary": summary,
        "cases": records,
    }


def _has_unexpected_failures(records: list[dict[str, Any]]) -> bool:
    for record in records:
        expected = str(record.get("expected_outcome") or "pass")
        if expected != "fail_allowed" and record.get("status") == "fail":
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
