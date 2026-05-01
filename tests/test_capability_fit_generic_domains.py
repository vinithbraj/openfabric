from __future__ import annotations

from typing import Any

import pytest

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.types import CapabilityRef, TaskFrame
from agent_runtime.input_pipeline.capability_fit import assess_capability_fit
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult


class FitLLMClient:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        for marker, payload in self.payloads.items():
            if marker in prompt:
                return dict(payload)
        raise AssertionError(f"unexpected prompt: {prompt}")


def _task(task_id: str, description: str, semantic_verb: str, object_type: str) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb=semantic_verb,
        object_type=object_type,
        intent_confidence=0.95,
        constraints={},
        raw_evidence=description,
    )


def _selection(task_id: str, capability_id: str, operation_id: str) -> CapabilitySelectionResult:
    selected = CapabilityRef(
        capability_id=capability_id,
        operation_id=operation_id,
        confidence=0.98,
        reason="Selected by capability selection.",
    )
    return CapabilitySelectionResult(task_id=task_id, candidates=[selected], selected=selected, unresolved_reason=None)


def _context(prompt: str, likely_domains: list[str]) -> dict[str, Any]:
    return {
        "original_prompt": prompt,
        "prompt_type": "simple_tool_task",
        "likely_domains": likely_domains,
        "risk_level": "low",
    }


def test_runtime_capabilities_fit_runtime_describe_capabilities() -> None:
    task = _task("task_1", "what are my capabilities?", "read", "runtime.capabilities")
    selection = _selection("task_1", "runtime.describe_capabilities", "describe_capabilities")
    llm = FitLLMClient(
        {
            "what are my capabilities?": {
                "task_id": "task_1",
                "candidate_capability_id": "runtime.describe_capabilities",
                "candidate_operation_id": "describe_capabilities",
                "proposed_status": "fit",
                "confidence": 0.97,
                "semantic_reason": "The task asks about registered capabilities.",
                "domain_reason": "capabilities and tools map to the runtime domain.",
                "object_type_reason": "runtime.capabilities matches capability registry metadata.",
                "argument_reason": "No required arguments.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "runtime",
                "suggested_object_type": "runtime.capabilities",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, gaps = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["capabilities", "runtime"]), llm)
    assert decisions[0].is_fit
    assert gaps == []


def test_runtime_capabilities_reject_filesystem_list_directory() -> None:
    task = _task("task_1", "what are my capabilities?", "read", "runtime.capabilities")
    selection = _selection("task_1", "filesystem.list_directory", "list_directory")
    llm = FitLLMClient(
        {
            "what are my capabilities?": {
                "task_id": "task_1",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "Listing files does not answer what capabilities exist.",
                "domain_reason": "runtime introspection is not filesystem browsing.",
                "object_type_reason": "runtime.capabilities is incompatible with filesystem.directory.",
                "argument_reason": "A path would not solve the mismatch.",
                "risk_reason": "Low risk but wrong domain.",
                "better_capability_id": "runtime.describe_capabilities",
                "missing_capability_description": None,
                "suggested_domain": "runtime",
                "suggested_object_type": "runtime.capabilities",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["runtime"]), llm)
    assert not decisions[0].is_fit


def test_git_repository_fits_shell_git_status() -> None:
    task = _task("task_1", "show git status for this repository", "read", "git.repository")
    selection = _selection("task_1", "shell.git_status", "git_status")
    llm = FitLLMClient(
        {
            "show git status for this repository": {
                "task_id": "task_1",
                "candidate_capability_id": "shell.git_status",
                "candidate_operation_id": "git_status",
                "proposed_status": "fit",
                "confidence": 0.96,
                "semantic_reason": "The task explicitly asks for git status.",
                "domain_reason": "repo and repository map to git semantics here.",
                "object_type_reason": "git.repository matches repository/workspace objects.",
                "argument_reason": "Path is optional.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "git",
                "suggested_object_type": "git.repository",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, gaps = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["repo", "git"]), llm)
    assert decisions[0].is_fit
    assert gaps == []


def test_git_repository_rejects_filesystem_list_directory() -> None:
    task = _task("task_1", "show git status for this repository", "read", "git.repository")
    selection = _selection("task_1", "filesystem.list_directory", "list_directory")
    llm = FitLLMClient(
        {
            "show git status for this repository": {
                "task_id": "task_1",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "This is a git status task, not generic file listing.",
                "domain_reason": "git.repository should not be satisfied by filesystem.directory alone.",
                "object_type_reason": "repository is incompatible with filesystem.directory for this intent.",
                "argument_reason": "A path would not yield git status.",
                "risk_reason": "Low risk but wrong capability.",
                "better_capability_id": "shell.git_status",
                "missing_capability_description": None,
                "suggested_domain": "git",
                "suggested_object_type": "git.repository",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["git"]), llm)
    assert not decisions[0].is_fit


def test_csv_task_fit_if_capability_exists() -> None:
    registry = build_default_registry()
    if "table.read_csv" not in {manifest.capability_id for manifest in registry.list_manifests()}:
        pytest.skip("table.read_csv does not exist in this runtime yet")


def test_csv_task_rejects_shell_list_processes() -> None:
    task = _task("task_1", "read this csv", "read", "csv")
    selection = _selection("task_1", "shell.list_processes", "list_processes")
    llm = FitLLMClient(
        {
            "read this csv": {
                "task_id": "task_1",
                "candidate_capability_id": "shell.list_processes",
                "candidate_operation_id": "list_processes",
                "proposed_status": "domain_mismatch",
                "confidence": 0.94,
                "semantic_reason": "A CSV is tabular data, not a process listing.",
                "domain_reason": "table and shell.process are different domains.",
                "object_type_reason": "csv is incompatible with system.process.",
                "argument_reason": "A process pattern would not solve it.",
                "risk_reason": "Low risk but wrong domain.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "table",
                "suggested_object_type": "table.csv",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["csv", "table"]), llm)
    assert not decisions[0].is_fit


def test_database_query_fits_sql_read_query() -> None:
    task = _task("task_1", "query patient count", "read", "database")
    selection = _selection("task_1", "sql.read_query", "read_query")
    llm = FitLLMClient(
        {
            "query patient count": {
                "task_id": "task_1",
                "candidate_capability_id": "sql.read_query",
                "candidate_operation_id": "read_query",
                "proposed_status": "fit",
                "confidence": 0.95,
                "semantic_reason": "The task is a database read/query task.",
                "domain_reason": "database maps to sql.",
                "object_type_reason": "database is compatible with sql query intent.",
                "argument_reason": "The capability expects structured query input later.",
                "risk_reason": "Read-only planning only.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "sql",
                "suggested_object_type": "sql.database",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, gaps = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["database", "sql"]), llm)
    assert decisions[0].is_fit
    assert gaps == []


def test_database_patient_rejects_filesystem_read_file() -> None:
    task = _task("task_1", "show top 10 patients by study count", "analyze", "patient")
    selection = _selection("task_1", "filesystem.read_file", "read_file")
    llm = FitLLMClient(
        {
            "show top 10 patients by study count": {
                "task_id": "task_1",
                "candidate_capability_id": "filesystem.read_file",
                "candidate_operation_id": "read_file",
                "proposed_status": "domain_mismatch",
                "confidence": 0.96,
                "semantic_reason": "The task is about database patient records.",
                "domain_reason": "patient/study analysis is not local file reading unless a file path is explicit.",
                "object_type_reason": "sql.patient is incompatible with filesystem.file.",
                "argument_reason": "A file path would change the task meaning.",
                "risk_reason": "Low risk but wrong capability.",
                "better_capability_id": "sql.read_query",
                "missing_capability_description": None,
                "suggested_domain": "sql",
                "suggested_object_type": "sql.patient",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["database", "sql"]), llm)
    assert not decisions[0].is_fit
