from __future__ import annotations

from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.types import CapabilityRef, TaskFrame, UserRequest
from agent_runtime.input_pipeline.argument_extraction import extract_arguments
from agent_runtime.input_pipeline.capability_fit import (
    assess_capability_fit,
)
from agent_runtime.input_pipeline.dag_builder import build_action_dag
from agent_runtime.input_pipeline.decomposition import DecompositionResult
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult
from agent_runtime.llm.reproducibility import PlanningTrace


class FitLLMClient:
    """Fake LLM client for capability-fit and argument extraction tests."""

    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        self.prompts.append(prompt)
        for marker, payload in self.payloads.items():
            if marker in prompt:
                return dict(payload)
        raise AssertionError(f"no fake payload configured for prompt: {prompt}")


def _task(task_id: str, description: str, semantic_verb: str, object_type: str) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb=semantic_verb,
        object_type=object_type,
        intent_confidence=0.95,
        constraints={},
        dependencies=[],
        raw_evidence=description,
    )


def _selection(task_id: str, capability_id: str | None, operation_id: str | None, confidence: float = 0.95) -> CapabilitySelectionResult:
    selected = (
        CapabilityRef(
            capability_id=capability_id,
            operation_id=operation_id,
            confidence=confidence,
            reason="Candidate selected by capability selection.",
        )
        if capability_id and operation_id
        else None
    )
    return CapabilitySelectionResult(
        task_id=task_id,
        candidates=[selected] if selected is not None else [],
        selected=selected,
        unresolved_reason=None if selected is not None else "No candidate matched this task.",
    )


def _classification_context(prompt: str, likely_domains: list[str]) -> dict[str, Any]:
    return {
        "original_prompt": prompt,
        "prompt_type": "simple_tool_task",
        "likely_domains": likely_domains,
        "risk_level": "low",
    }


def _runtime_request() -> UserRequest:
    registry = build_default_registry()
    return UserRequest(
        raw_prompt="test",
        safety_context={
            "capability_registry": registry,
        },
    )


def test_system_memory_rejects_filesystem_when_llm_says_mismatch() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "filesystem.list_directory", "list_directory")
    client = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "domain_mismatch",
                "confidence": 0.94,
                "semantic_reason": "This is a system resource question, not a directory listing.",
                "domain_reason": "Filesystem metadata does not answer memory usage.",
                "object_type_reason": "The task refers to system memory, not a directory.",
                "argument_reason": "A path argument would not solve the semantic mismatch.",
                "risk_reason": "Low risk but wrong domain.",
                "missing_capability_description": "A runtime capability for system memory inspection is missing.",
                "suggested_domain": "shell",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        client,
    )

    assert decisions[0].status in {"domain_mismatch", "object_type_mismatch"}
    assert gaps
    assert "system memory inspection" in gaps[0].missing_capability_description.lower()


def test_system_memory_rejects_filesystem_even_if_llm_says_fit() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "filesystem.list_directory", "list_directory")
    client = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "fit",
                "confidence": 0.96,
                "semantic_reason": "This might read some system state.",
                "domain_reason": "I think reading a directory could help.",
                "object_type_reason": "It is still a read operation.",
                "argument_reason": "The path could maybe be inferred.",
                "risk_reason": "Low risk.",
                "missing_capability_description": None,
                "suggested_domain": "shell",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        client,
    )

    assert not decisions[0].is_fit
    assert decisions[0].status in {"domain_mismatch", "object_type_mismatch"}
    assert gaps


def test_filesystem_directory_fits_list_directory() -> None:
    registry = build_default_registry()
    task = _task("task-list", "list files in this directory", "read", "filesystem.directory")
    selection = _selection("task-list", "filesystem.list_directory", "list_directory")
    client = FitLLMClient(
        {
            "list files in this directory": {
                "task_id": "task-list",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "fits",
                "confidence": 0.95,
                "semantic_reason": "The task is to read a directory listing.",
                "domain_reason": "Filesystem is the correct domain.",
                "object_type_reason": "The object type is a directory.",
                "argument_reason": "A path argument can be extracted separately.",
                "risk_reason": "Read-only and low risk.",
                "missing_capability_description": None,
                "suggested_domain": "filesystem",
                "suggested_object_type": "directory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["filesystem"]),
        client,
    )

    assert decisions[0].is_fit
    assert decisions[0].candidate_capability_id == "filesystem.list_directory"
    assert gaps == []


def test_git_repository_fits_shell_git_status() -> None:
    registry = build_default_registry()
    task = _task("task-git", "show git status for this repository", "read", "git.repository")
    selection = _selection("task-git", "shell.git_status", "git_status")
    client = FitLLMClient(
        {
            "show git status for this repository": {
                "task_id": "task-git",
                "candidate_capability_id": "shell.git_status",
                "candidate_operation_id": "git_status",
                "proposed_status": "fit",
                "confidence": 0.95,
                "semantic_reason": "The task explicitly asks for git status.",
                "domain_reason": "Repository inspection is handled through shell git status.",
                "object_type_reason": "The object is a repository/workspace.",
                "argument_reason": "Path is optional and can default to the current workspace.",
                "risk_reason": "Read-only and low risk.",
                "missing_capability_description": None,
                "suggested_domain": "shell",
                "suggested_object_type": "repository",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        client,
    )

    assert decisions[0].is_fit
    assert gaps == []


def test_memory_task_with_no_memory_capability_creates_capability_gap() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", None, None)
    client = FitLLMClient({})

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        client,
    )

    assert decisions[0].status == "unsupported_capability_gap"
    assert gaps
    assert "compatible capability" in gaps[0].user_facing_message.lower()


def test_no_generic_read_fallback() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "filesystem.read_file", "read_file")
    client = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "candidate_capability_id": "filesystem.read_file",
                "candidate_operation_id": "read_file",
                "proposed_status": "fit",
                "confidence": 0.91,
                "semantic_reason": "Both are read tasks.",
                "domain_reason": "Maybe a file could contain the answer.",
                "object_type_reason": "Still a read operation.",
                "argument_reason": "A file path might be inferred later.",
                "risk_reason": "Low risk.",
                "missing_capability_description": None,
                "suggested_domain": "shell",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    decisions, _ = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        client,
    )

    assert not decisions[0].is_fit


def test_argument_extraction_skipped_for_non_fit() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "filesystem.list_directory", "list_directory")
    fit_client = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "Wrong domain.",
                "domain_reason": "Memory is not a directory listing.",
                "object_type_reason": "Wrong object type.",
                "argument_reason": "Path would not help.",
                "risk_reason": "Low risk.",
                "missing_capability_description": "A memory capability is missing.",
                "suggested_domain": "shell",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    fit_decisions, _ = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        fit_client,
    )
    extraction_client = FitLLMClient({})

    results = extract_arguments(
        [task],
        [selection],
        registry,
        extraction_client,
        capability_fit_decisions=fit_decisions,
    )

    assert results == []
    assert extraction_client.prompts == []


def test_dag_builder_refuses_executable_node_for_gap() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "filesystem.list_directory", "list_directory")
    fit_client = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "Wrong domain.",
                "domain_reason": "Memory is not a directory listing.",
                "object_type_reason": "Wrong object type.",
                "argument_reason": "Path would not help.",
                "risk_reason": "Low risk.",
                "missing_capability_description": "A memory capability is missing.",
                "suggested_domain": "shell",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    fit_decisions, _ = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        fit_client,
    )
    dag = build_action_dag(
        _runtime_request(),
        DecompositionResult(tasks=[task]),
        [selection],
        [],
        capability_fit_decisions=fit_decisions,
    )

    assert dag.nodes == []


def test_read_the_file_produces_missing_path_only_after_fit() -> None:
    registry = build_default_registry()
    task = _task("task-read-file", "read the file", "read", "file")
    selection = _selection("task-read-file", "filesystem.read_file", "read_file")
    fit_client = FitLLMClient(
        {
            "read the file": {
                "task_id": "task-read-file",
                "candidate_capability_id": "filesystem.read_file",
                "candidate_operation_id": "read_file",
                "proposed_status": "fit",
                "confidence": 0.90,
                "semantic_reason": "The task is clearly asking to read a file.",
                "domain_reason": "Filesystem is the right domain.",
                "object_type_reason": "The object is a file.",
                "argument_reason": "The path still needs extraction.",
                "risk_reason": "Read-only and low risk.",
                "missing_capability_description": None,
                "suggested_domain": "filesystem",
                "suggested_object_type": "file",
                "requires_clarification": False,
                "clarification_question": None,
            },
        }
    )
    fit_decisions, _ = assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["filesystem"]),
        fit_client,
    )
    extraction_client = FitLLMClient(
        {
            "read the file": {
                "task_id": "task-read-file",
                "capability_id": "filesystem.read_file",
                "operation_id": "read_file",
                "arguments": {},
                "missing_required_arguments": ["path"],
                "assumptions": ["The prompt does not name a file path."],
                "confidence": 0.55,
            }
        }
    )

    results = extract_arguments(
        [task],
        [selection],
        registry,
        extraction_client,
        capability_fit_decisions=fit_decisions,
    )

    assert fit_decisions[0].is_fit
    assert results[0].capability_id == "filesystem.read_file"
    assert "path" in results[0].missing_required_arguments


def test_trace_records_rejected_candidates_if_trace_exists() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "filesystem.list_directory", "list_directory")
    trace = PlanningTrace(request_id="req-1")
    client = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "fit",
                "confidence": 0.95,
                "semantic_reason": "It is a read task.",
                "domain_reason": "Maybe close enough.",
                "object_type_reason": "Maybe close enough.",
                "argument_reason": "Path could be extracted.",
                "risk_reason": "Low risk.",
                "missing_capability_description": None,
                "suggested_domain": "shell",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    assess_capability_fit(
        [task],
        [selection],
        registry,
        _classification_context(task.description, ["shell"]),
        client,
        trace=trace,
    )

    fit_entries = [entry for entry in trace.entries if entry.stage == "capability_fit"]
    assert fit_entries
    assert fit_entries[0].rejection_reasons
