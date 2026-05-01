from __future__ import annotations

from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.input_pipeline.domain_selection import select_capabilities
from agent_runtime.llm.reproducibility import PlanningTrace
from agent_runtime.core.types import TaskFrame


class ShortlistLLMClient:
    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        self.prompts.append(prompt)
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
        intent_confidence=0.98,
        constraints={},
        raw_evidence=description,
    )


def _context(prompt: str, likely_domains: list[str]) -> dict[str, Any]:
    return {
        "original_prompt": prompt,
        "prompt_type": "simple_tool_task",
        "likely_domains": likely_domains,
        "risk_level": "low",
    }


def test_shortlist_selects_best_binary_fit_for_memory_prompt() -> None:
    registry = build_default_registry()
    task = _task("task_1", "how much free memory do i have on this system?", "read", "memory")
    llm = ShortlistLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task_1",
                "evaluations": [
                    {
                        "capability_id": "system.memory_status",
                        "operation_id": "memory_status",
                        "fits": True,
                        "confidence": 0.97,
                        "reason": "This directly answers free memory.",
                        "domain_reason": "system_administration and system are compatible.",
                        "object_type_reason": "memory maps to system.memory.",
                        "argument_reason": "No required arguments.",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.environment_summary",
                        "operation_id": "environment_summary",
                        "fits": False,
                        "confidence": 0.80,
                        "reason": "This is related but too broad.",
                        "domain_reason": "Correct broad domain, wrong specificity.",
                        "object_type_reason": "environment is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.disk_usage",
                        "operation_id": "disk_usage",
                        "fits": False,
                        "confidence": 0.90,
                        "reason": "Disk space is not memory.",
                        "domain_reason": "Same domain, wrong resource.",
                        "object_type_reason": "disk is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.cpu_load",
                        "operation_id": "cpu_load",
                        "fits": False,
                        "confidence": 0.90,
                        "reason": "CPU load is not memory.",
                        "domain_reason": "Same domain, wrong resource.",
                        "object_type_reason": "cpu is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.uptime",
                        "operation_id": "uptime",
                        "fits": False,
                        "confidence": 0.90,
                        "reason": "Uptime is not memory.",
                        "domain_reason": "Same domain, wrong resource.",
                        "object_type_reason": "uptime is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                ],
            }
        }
    )

    results = select_capabilities(
        [task],
        registry,
        llm,
        classification_context=_context(task.description, ["system_administration", "operating_system", "system"]),
    )

    assert results[0].selected is not None
    assert results[0].selected.capability_id == "system.memory_status"
    candidate_ids = [candidate.capability_id for candidate in results[0].candidates]
    assert "system.memory_status" in candidate_ids


def test_shortlist_rejects_candidate_outside_runtime_shortlist() -> None:
    registry = build_default_registry()
    task = _task("task_1", "how much free memory do i have on this system?", "read", "memory")
    llm = ShortlistLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task_1",
                "evaluations": [
                    {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "fits": True,
                        "confidence": 0.99,
                        "reason": "Wrongly chosen external candidate.",
                        "domain_reason": "Wrong domain.",
                        "object_type_reason": "Wrong object type.",
                        "argument_reason": "",
                        "risk_reason": "Low risk.",
                        "missing_arguments_likely": [],
                    }
                ],
            }
        }
    )

    results = select_capabilities(
        [task],
        registry,
        llm,
        classification_context=_context(task.description, ["system"]),
    )

    assert results[0].selected is None
    assert results[0].unresolved_reason is not None
    assert "not in the deterministic shortlist" in results[0].unresolved_reason


def test_shortlist_trace_records_binary_evaluations() -> None:
    registry = build_default_registry()
    task = _task("task_1", "how much free memory do i have on this system?", "read", "memory")
    trace = PlanningTrace(request_id="req-shortlist")
    llm = ShortlistLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task_1",
                "evaluations": [
                    {
                        "capability_id": "system.memory_status",
                        "operation_id": "memory_status",
                        "fits": True,
                        "confidence": 0.96,
                        "reason": "Direct memory capability.",
                        "domain_reason": "Correct system domain.",
                        "object_type_reason": "Correct memory object type.",
                        "argument_reason": "No required arguments.",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.environment_summary",
                        "operation_id": "environment_summary",
                        "fits": False,
                        "confidence": 0.70,
                        "reason": "Too broad.",
                        "domain_reason": "Correct broad domain.",
                        "object_type_reason": "Wrong specificity.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.disk_usage",
                        "operation_id": "disk_usage",
                        "fits": False,
                        "confidence": 0.70,
                        "reason": "Wrong resource.",
                        "domain_reason": "Correct broad domain.",
                        "object_type_reason": "Disk is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.cpu_load",
                        "operation_id": "cpu_load",
                        "fits": False,
                        "confidence": 0.70,
                        "reason": "Wrong resource.",
                        "domain_reason": "Correct broad domain.",
                        "object_type_reason": "CPU is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                    {
                        "capability_id": "system.uptime",
                        "operation_id": "uptime",
                        "fits": False,
                        "confidence": 0.70,
                        "reason": "Wrong resource.",
                        "domain_reason": "Correct broad domain.",
                        "object_type_reason": "Uptime is not memory.",
                        "argument_reason": "",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    },
                ],
            }
        }
    )

    select_capabilities(
        [task],
        registry,
        llm,
        classification_context=_context(task.description, ["system"]),
        trace=trace,
    )

    assert "task_1" in trace.capability_candidates_by_task
    assert "task_1" in trace.capability_shortlist_evaluations_by_task
    evaluations = trace.capability_shortlist_evaluations_by_task["task_1"]
    assert any(item["capability_id"] == "system.memory_status" and item["fits"] for item in evaluations)
