#!/usr/bin/env python3
"""Evaluate semantic-frame coverage over safe compound prompts.

This runner exercises semantic extraction, canonicalization, compilation, and
coverage validation only. It does not execute tools and does not store raw data
payloads, SQL rows, shell output, file contents, or SLURM job details.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.runtime.semantic_frame import (
    SemanticCoverageValidator,
    SemanticFrameCompiler,
    deterministic_semantic_frame,
)


ALLOWED_TOOLS = [
    "fs.aggregate",
    "fs.glob",
    "fs.find",
    "fs.list",
    "fs.read",
    "fs.search_content",
    "shell.exec",
    "slurm.accounting",
    "slurm.accounting_aggregate",
    "slurm.metrics",
    "slurm.nodes",
    "slurm.partitions",
    "slurm.queue",
    "sql.query",
    "sql.schema",
    "sql.validate",
    "text.format",
    "runtime.return",
]


@dataclass(frozen=True)
class Case:
    """Represent one safe planner-level semantic-frame eval case."""

    id: int
    domain: str
    prompt: str


def _cases() -> list[Case]:
    """Build the 100 prompt semantic-frame eval set."""

    prompts: list[tuple[str, str]] = []
    partitions = ["slicer", "totalseg", "hpc", "vllm"]
    metrics = ["average", "minimum", "maximum", "total"]
    for metric in metrics:
        prompts.append(("slurm", f"Give me the {metric} job times in slicer, totalseg partitions for the last 7 days"))
        prompts.append(("slurm", f"Show {metric} runtime by partition for completed jobs in the past 3 days"))
    for partition in partitions:
        prompts.append(("slurm", f"Count jobs in {partition} partition and return only the count"))
        prompts.append(("slurm", f"List jobs in {partition} partition"))
    prompts.extend(
        [
            ("slurm", "Count jobs in each slurm partition separately"),
            ("slurm", "Count jobs by state in the slurm queue"),
            ("slurm", "Show slurm cluster status and summarize queue health"),
            ("slurm", "Show slurm node status and partition status"),
            ("slurm", "Average runtime for partition slicer and partition totalseg over last 24 hours"),
            ("slurm", "Average job elapsed time for slicer + totalseg partitions in the past 7 days"),
        ]
    )

    extensions = ["CSV", "JSON", "TXT", "Markdown", "Python"]
    for first, second in zip(extensions, extensions[1:]):
        prompts.append(("filesystem", f"Count {first} and {second} files under the current folder separately"))
        prompts.append(("filesystem", f"List {first} files and {second} files in this repository"))
    prompts.extend(
        [
            ("filesystem", "Find all CSV files under the current folder and summarize path counts"),
            ("filesystem", "List all files in this folder and show a table"),
            ("filesystem", "Count Python and Markdown files in the repository"),
            ("filesystem", "Show JSON and CSV file counts under docs separately"),
            ("filesystem", "Find txt and md files under the current directory"),
            ("filesystem", "Count files by extension for CSV and JSON targets"),
            ("filesystem", "List Python files and summarize modules by folder"),
            ("filesystem", "Count Markdown files under docs and README style files separately"),
        ]
    )

    prompts.extend(
        [
            ("shell", "How many processes are running on this machine?"),
            ("shell", "List running processes sorted by CPU usage"),
            ("shell", "Show listening TCP ports and summarize them"),
            ("shell", "Show disk usage for the current filesystem"),
            ("shell", "Show memory usage and CPU information"),
            ("shell", "Show hostname and uptime"),
            ("shell", "Count processes and list top CPU processes"),
            ("shell", "Show mounted filesystems and disk usage"),
            ("shell", "List ports and show process summary"),
            ("shell", "Show current system resource status"),
        ]
    )

    prompts.extend(
        [
            ("sql", "Inspect the dicom schema and identify patient, study, series, and instance tables"),
            ("sql", "Generate and validate SQL for counting studies per patient in dicom"),
            ("sql", "Validate this SQL against dicom: SELECT COUNT(*) FROM flathr.\"Patient\""),
            ("sql", "Explain this SQL without executing it: SELECT COUNT(*) FROM flathr.\"Study\""),
            ("sql", "Inspect dicom schema and infer joins from study to series"),
            ("sql", "Show a data dictionary for patient and study tables in dicom"),
            ("sql", "Validate SQL for counting instances per series in dicom"),
            ("sql", "Inspect dicom schema for modality and body part columns"),
            ("sql", "Explain a read-only SQL query for series per study in dicom"),
            ("sql", "Inspect dicom database tables and group them by schema"),
            ("sql", "Count patients by CT and MR modality in dicom"),
            ("sql", "Show min max average studies per patient in dicom"),
            ("sql", "Show top studies by CT and MR modality counts"),
            ("sql", "Explain SQL join path for patient to instance in dicom"),
            ("sql", "Validate SQL for RTSTRUCT and RTDOSE co-occurrence in dicom"),
        ]
    )

    prompts.extend(
        [
            ("diagnostic", "Run an end-to-end diagnostic of workspace files, config flags, SQL capabilities, filesystem capabilities, and shell capabilities"),
            ("diagnostic", "Summarize repository files, AOR config flags, SQL tools, filesystem tools, and shell inspection tools"),
            ("diagnostic", "Show available OpenFABRIC runtime capabilities across workspace, config, SQL, filesystem, and shell"),
            ("diagnostic", "Diagnostic summary for workspace, config flags, SQL database, filesystem safety, and shell safety"),
            ("diagnostic", "Summarize current workspace files, config flags, and available SQL filesystem shell capabilities"),
        ]
    )

    compound_pairs = [
        "Count CSV files then count running processes",
        "List JSON files then show disk usage",
        "Show slurm status then count CSV and JSON files",
        "Inspect dicom schema then show slurm queue status",
        "Count processes then show slurm node status",
        "Count jobs in each partition then count Python and Markdown files",
        "Show memory status then inspect dicom tables",
        "List CSV files then validate SQL against dicom: SELECT COUNT(*) FROM flathr.\"Patient\"",
        "Show slurm partition status then list Python files",
        "Count JSON files then show listening ports",
    ]
    prompts.extend(("compound", prompt) for prompt in compound_pairs)

    while len(prompts) < 100:
        prompts.append(("filesystem", f"Count CSV and JSON files separately under current folder case {len(prompts) + 1}"))
    return [Case(id=index, domain=domain, prompt=prompt) for index, (domain, prompt) in enumerate(prompts[:100], start=1)]


def evaluate(settings: Settings) -> dict[str, Any]:
    """Evaluate all cases without executing tools."""

    compiler = SemanticFrameCompiler(settings=settings, allowed_tools=ALLOWED_TOOLS)
    validator = SemanticCoverageValidator()
    results: list[dict[str, Any]] = []
    summary = {"total": 0, "frame_matched": 0, "compiled": 0, "coverage_ok": 0, "no_frame": 0, "not_compiled": 0}
    by_domain: dict[str, dict[str, int]] = {}
    for case in _cases():
        summary["total"] += 1
        domain_summary = by_domain.setdefault(case.domain, {"total": 0, "frame_matched": 0, "compiled": 0, "coverage_ok": 0})
        domain_summary["total"] += 1
        extraction = deterministic_semantic_frame(case.prompt, settings)
        record: dict[str, Any] = {
            "id": case.id,
            "domain": case.domain,
            "prompt": case.prompt,
            "frame_matched": extraction.matched,
            "compiled": False,
            "coverage_ok": False,
            "issue": None,
        }
        if not extraction.matched or extraction.frame is None:
            summary["no_frame"] += 1
            record["issue"] = extraction.reason or "no_frame"
            results.append(record)
            continue
        summary["frame_matched"] += 1
        domain_summary["frame_matched"] += 1
        record["frame"] = {
            "domain": extraction.frame.domain,
            "intent": extraction.frame.intent,
            "composition": extraction.frame.composition,
            "output": extraction.frame.output.kind,
            "targets": {key: target.model_dump().get("values") for key, target in extraction.frame.targets.items()},
            "children": len(extraction.frame.children),
        }
        compiled = compiler.compile(extraction.frame)
        if compiled is None:
            summary["not_compiled"] += 1
            record["issue"] = "not_compiled"
            results.append(record)
            continue
        summary["compiled"] += 1
        domain_summary["compiled"] += 1
        record["compiled"] = True
        record["strategy"] = compiled.strategy
        record["actions"] = [step.action for step in compiled.plan.steps]
        coverage = validator.validate(extraction.frame, compiled.plan)
        record["coverage_ok"] = coverage.covered
        if coverage.covered:
            summary["coverage_ok"] += 1
            domain_summary["coverage_ok"] += 1
        else:
            record["issue"] = "; ".join(coverage.errors)
        results.append(record)
    return {"summary": summary, "by_domain": by_domain, "results": results}


def main() -> int:
    """Run the semantic-frame compound eval from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/reports/semantic_frame_compound_eval_100.json")
    args = parser.parse_args()
    settings = Settings(workspace_root=Path.cwd(), run_store_path=Path.cwd() / "artifacts" / "runtime.db")
    report = evaluate(settings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
