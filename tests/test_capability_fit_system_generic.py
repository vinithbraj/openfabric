from __future__ import annotations

from typing import Any

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
    return CapabilitySelectionResult(
        task_id=task_id,
        candidates=[selected],
        selected=selected,
        unresolved_reason=None,
    )


def _context(prompt: str, likely_domains: list[str]) -> dict[str, Any]:
    return {
        "original_prompt": prompt,
        "prompt_type": "simple_tool_task",
        "likely_domains": likely_domains,
        "risk_level": "low",
    }


def test_memory_fits_system_memory_status() -> None:
    task = _task("task_1", "how much free memory do i have on this system?", "read", "memory")
    selection = _selection("task_1", "system.memory_status", "memory_status")
    llm = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task_1",
                "candidate_capability_id": "system.memory_status",
                "candidate_operation_id": "memory_status",
                "proposed_status": "fit",
                "confidence": 0.95,
                "semantic_reason": "The user is asking to inspect memory availability.",
                "domain_reason": "system_administration and operating_system are synonyms for the system domain.",
                "object_type_reason": "The task object 'memory' matches system.memory and memory.",
                "argument_reason": "The capability requires no required arguments and optionally supports human_readable.",
                "risk_reason": "The capability is read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        build_default_registry(),
        _context(task.description, ["system_administration", "operating_system", "system"]),
        llm,
    )

    assert decisions[0].is_fit
    assert decisions[0].normalized_task_domain == "system"
    assert decisions[0].normalized_manifest_domain == "system"
    assert decisions[0].normalized_task_object_type == "system.memory"
    assert gaps == []


def test_disk_fits_system_disk_usage() -> None:
    task = _task("task_1", "how much disk space is free?", "read", "disk_usage")
    selection = _selection("task_1", "system.disk_usage", "disk_usage")
    llm = FitLLMClient(
        {
            "how much disk space is free?": {
                "task_id": "task_1",
                "candidate_capability_id": "system.disk_usage",
                "candidate_operation_id": "disk_usage",
                "proposed_status": "fit",
                "confidence": 0.94,
                "semantic_reason": "The task is about disk capacity and free storage.",
                "domain_reason": "operating_system is a synonym for system.",
                "object_type_reason": "disk_usage maps to system.disk.",
                "argument_reason": "Path is optional.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.disk",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit(
        [task],
        [selection],
        build_default_registry(),
        _context(task.description, ["operating_system", "system"]),
        llm,
    )
    assert decisions[0].is_fit


def test_cpu_fits_system_cpu_load() -> None:
    task = _task("task_1", "what is the CPU load?", "read", "cpu_load")
    selection = _selection("task_1", "system.cpu_load", "cpu_load")
    llm = FitLLMClient(
        {
            "what is the CPU load?": {
                "task_id": "task_1",
                "candidate_capability_id": "system.cpu_load",
                "candidate_operation_id": "cpu_load",
                "proposed_status": "fit",
                "confidence": 0.94,
                "semantic_reason": "The task is about CPU load.",
                "domain_reason": "This is a system resource inspection task.",
                "object_type_reason": "cpu_load maps to system.cpu.",
                "argument_reason": "No required arguments.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.cpu",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["system"]), llm)
    assert decisions[0].is_fit


def test_uptime_fits_system_uptime() -> None:
    task = _task("task_1", "what is the system uptime?", "read", "uptime")
    selection = _selection("task_1", "system.uptime", "uptime")
    llm = FitLLMClient(
        {
            "what is the system uptime?": {
                "task_id": "task_1",
                "candidate_capability_id": "system.uptime",
                "candidate_operation_id": "uptime",
                "proposed_status": "fit",
                "confidence": 0.93,
                "semantic_reason": "The task is to read system uptime.",
                "domain_reason": "This belongs to the system domain.",
                "object_type_reason": "uptime maps to system.uptime.",
                "argument_reason": "No required arguments.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.uptime",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["system"]), llm)
    assert decisions[0].is_fit


def test_environment_fits_system_environment_summary() -> None:
    task = _task("task_1", "show system environment summary", "summarize", "environment")
    selection = _selection("task_1", "system.environment_summary", "environment_summary")
    llm = FitLLMClient(
        {
            "show system environment summary": {
                "task_id": "task_1",
                "candidate_capability_id": "system.environment_summary",
                "candidate_operation_id": "environment_summary",
                "proposed_status": "fit",
                "confidence": 0.92,
                "semantic_reason": "The task asks for a machine summary.",
                "domain_reason": "This belongs to the system domain.",
                "object_type_reason": "environment maps to system.environment.",
                "argument_reason": "No required arguments.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.environment",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["system"]), llm)
    assert decisions[0].is_fit


def test_memory_rejects_filesystem_list_directory() -> None:
    task = _task("task_1", "how much free memory do i have on this system?", "read", "memory")
    selection = _selection("task_1", "filesystem.list_directory", "list_directory")
    llm = FitLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task_1",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "This is not a directory listing task.",
                "domain_reason": "system memory is not a filesystem task.",
                "object_type_reason": "memory is incompatible with filesystem.directory.",
                "argument_reason": "A path would not solve the mismatch.",
                "risk_reason": "Low risk but wrong domain.",
                "better_capability_id": "system.memory_status",
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["system"]), llm)
    assert not decisions[0].is_fit


def test_cpu_rejects_filesystem_read_file() -> None:
    task = _task("task_1", "what is the CPU load?", "read", "cpu_load")
    selection = _selection("task_1", "filesystem.read_file", "read_file")
    llm = FitLLMClient(
        {
            "what is the CPU load?": {
                "task_id": "task_1",
                "candidate_capability_id": "filesystem.read_file",
                "candidate_operation_id": "read_file",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "This is about CPU load, not reading a file.",
                "domain_reason": "system CPU is not a filesystem file task.",
                "object_type_reason": "cpu_load is incompatible with filesystem.file.",
                "argument_reason": "A path would not make it a CPU inspection capability.",
                "risk_reason": "Low risk but wrong domain.",
                "better_capability_id": "system.cpu_load",
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.cpu",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["system"]), llm)
    assert not decisions[0].is_fit


def test_disk_rejects_shell_list_processes() -> None:
    task = _task("task_1", "show disk usage", "read", "disk_usage")
    selection = _selection("task_1", "shell.list_processes", "list_processes")
    llm = FitLLMClient(
        {
            "show disk usage": {
                "task_id": "task_1",
                "candidate_capability_id": "shell.list_processes",
                "candidate_operation_id": "list_processes",
                "proposed_status": "domain_mismatch",
                "confidence": 0.95,
                "semantic_reason": "This is about storage, not processes.",
                "domain_reason": "system disk is not a process inspection task.",
                "object_type_reason": "disk usage is incompatible with process list.",
                "argument_reason": "Process filters would not solve the mismatch.",
                "risk_reason": "Low risk but wrong domain.",
                "better_capability_id": "system.disk_usage",
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.disk",
                "requires_clarification": False,
                "clarification_question": None,
            }
        }
    )
    decisions, _ = assess_capability_fit([task], [selection], build_default_registry(), _context(task.description, ["system"]), llm)
    assert not decisions[0].is_fit

