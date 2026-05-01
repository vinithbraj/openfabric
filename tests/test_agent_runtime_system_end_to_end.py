from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import ExecutionResult
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from gateway_agent.remote_runner import run_remote_operation


class SystemLLMClient:
    """Fake LLM client for end-to-end system capability flows."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        self.prompts.append(prompt)
        prompt_l = prompt.lower()

        if "you are classifying a user prompt" in prompt_l:
            if "what are my capabilities?" in prompt_l:
                return {
                    "prompt_type": "simple_tool_task",
                    "requires_tools": True,
                    "likely_domains": ["runtime", "capabilities"],
                    "risk_level": "low",
                    "needs_clarification": False,
                    "clarification_question": None,
                    "reason": "The prompt asks for runtime introspection.",
                    "confidence": 0.97,
                    "assumptions": [],
                }
            return {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["system_administration", "operating_system", "system"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "The prompt asks for safe system inspection.",
                "confidence": 0.97,
                "assumptions": [],
            }

        if "you are decomposing a user prompt" in prompt_l:
            if "free memory" in prompt_l:
                return {
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": "Query the system for the current amount of free memory available.",
                            "semantic_verb": "read",
                            "object_type": "memory",
                            "intent_confidence": 0.98,
                            "constraints": {"human_readable": True},
                            "dependencies": [],
                            "raw_evidence": "how much free memory do i have on this system?",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {"output_format": "human_readable", "data_source": "system_metrics"},
                    "unresolved_references": [],
                    "assumptions": [],
                }
            if "disk space is free" in prompt_l:
                return {
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": "Query the system for current free disk space.",
                            "semantic_verb": "read",
                            "object_type": "disk_usage",
                            "intent_confidence": 0.98,
                            "constraints": {"human_readable": True},
                            "dependencies": [],
                            "raw_evidence": "how much disk space is free?",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {"output_format": "human_readable"},
                    "unresolved_references": [],
                    "assumptions": [],
                }
            if "cpu load" in prompt_l:
                return {
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": "Query the system for the current CPU load.",
                            "semantic_verb": "read",
                            "object_type": "cpu_load",
                            "intent_confidence": 0.98,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "what is the CPU load?",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {},
                    "unresolved_references": [],
                    "assumptions": [],
                }
            if "system uptime" in prompt_l:
                return {
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": "Query the system for current uptime.",
                            "semantic_verb": "read",
                            "object_type": "uptime",
                            "intent_confidence": 0.98,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "what is the system uptime?",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {},
                    "unresolved_references": [],
                    "assumptions": [],
                }
            if "what are my capabilities?" in prompt_l:
                return {
                    "tasks": [
                        {
                            "id": "task_1",
                            "description": "Describe the runtime capabilities available to the user.",
                            "semantic_verb": "read",
                            "object_type": "runtime.capabilities",
                            "intent_confidence": 0.98,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "what are my capabilities?",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {},
                    "unresolved_references": [],
                    "assumptions": [],
                }

        if "you are assigning semantic verbs" in prompt_l:
            if "runtime capabilities" in prompt_l:
                return {
                    "assignments": [
                        {
                            "task_id": "task_1",
                            "semantic_verb": "read",
                            "object_type": "runtime.capabilities",
                            "intent_confidence": 0.97,
                            "risk_level": "low",
                            "requires_confirmation": False,
                        }
                    ]
                }
            object_type = "memory"
            if "free disk space" in prompt_l or "disk space is free" in prompt_l:
                object_type = "disk_usage"
            elif "cpu load" in prompt_l:
                object_type = "cpu_load"
            elif "uptime" in prompt_l:
                object_type = "uptime"
            return {
                "assignments": [
                    {
                        "task_id": "task_1",
                        "semantic_verb": "read",
                        "object_type": object_type,
                        "intent_confidence": 0.97,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            }

        if "you are selecting capability candidates" in prompt_l:
            if "runtime capabilities" in prompt_l or "what are my capabilities?" in prompt_l:
                capability_id = "runtime.describe_capabilities"
                operation_id = "describe_capabilities"
                reason = "The task asks for runtime capability introspection."
            elif "free memory available" in prompt_l or "how much free memory do i have on this system?" in prompt_l:
                capability_id = "system.memory_status"
                operation_id = "memory_status"
                reason = "The task is about memory availability."
            elif "free disk space" in prompt_l or "disk space is free" in prompt_l:
                capability_id = "system.disk_usage"
                operation_id = "disk_usage"
                reason = "The task is about disk usage."
            elif "system uptime" in prompt_l or "current uptime" in prompt_l:
                capability_id = "system.uptime"
                operation_id = "uptime"
                reason = "The task is about system uptime."
            elif "cpu load" in prompt_l:
                capability_id = "system.cpu_load"
                operation_id = "cpu_load"
                reason = "The task is about CPU load."
            else:
                capability_id = "system.memory_status"
                operation_id = "memory_status"
                reason = "The task is about memory availability."
            return {
                "task_id": "task_1",
                "candidates": [
                    {
                        "capability_id": capability_id,
                        "operation_id": operation_id,
                        "confidence": 0.98,
                        "reason": reason,
                    }
                ],
                "selected": {
                    "capability_id": capability_id,
                    "operation_id": operation_id,
                    "confidence": 0.98,
                    "reason": reason,
                },
                "unresolved_reason": None,
            }

        if "you are assessing whether a selected capability truly fits a task" in prompt_l:
            if "runtime.describe_capabilities" in prompt:
                return {
                    "task_id": "task_1",
                    "candidate_capability_id": "runtime.describe_capabilities",
                    "candidate_operation_id": "describe_capabilities",
                    "proposed_status": "fit",
                    "confidence": 0.97,
                    "semantic_reason": "The task asks about the runtime's capabilities.",
                    "domain_reason": "capabilities and runtime are synonyms here.",
                    "object_type_reason": "runtime.capabilities matches the manifest object types.",
                    "argument_reason": "No required arguments.",
                    "risk_reason": "Read-only and low risk.",
                    "better_capability_id": None,
                    "missing_capability_description": None,
                    "suggested_domain": "runtime",
                    "suggested_object_type": "runtime.capabilities",
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            if "system.disk_usage" in prompt:
                return {
                    "task_id": "task_1",
                    "candidate_capability_id": "system.disk_usage",
                    "candidate_operation_id": "disk_usage",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task asks about free disk space.",
                    "domain_reason": "operating_system and system are compatible.",
                    "object_type_reason": "disk_usage maps to system.disk.",
                    "argument_reason": "Path is optional and human_readable can be extracted.",
                    "risk_reason": "Read-only and low risk.",
                    "better_capability_id": None,
                    "missing_capability_description": None,
                    "suggested_domain": "system",
                    "suggested_object_type": "system.disk",
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            if "system.cpu_load" in prompt:
                return {
                    "task_id": "task_1",
                    "candidate_capability_id": "system.cpu_load",
                    "candidate_operation_id": "cpu_load",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task asks about CPU load.",
                    "domain_reason": "system metrics belong to the system domain.",
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
            if "system.uptime" in prompt:
                return {
                    "task_id": "task_1",
                    "candidate_capability_id": "system.uptime",
                    "candidate_operation_id": "uptime",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task asks about system uptime.",
                    "domain_reason": "system metrics belong to the system domain.",
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
            return {
                "task_id": "task_1",
                "candidate_capability_id": "system.memory_status",
                "candidate_operation_id": "memory_status",
                "proposed_status": "fit",
                "confidence": 0.96,
                "semantic_reason": "The task asks about free memory available on the system.",
                "domain_reason": "system_administration and operating_system are synonyms for the system domain.",
                "object_type_reason": "memory maps to system.memory.",
                "argument_reason": "No required arguments and human_readable is optional.",
                "risk_reason": "Read-only and low risk.",
                "better_capability_id": None,
                "missing_capability_description": None,
                "suggested_domain": "system",
                "suggested_object_type": "system.memory",
                "requires_clarification": False,
                "clarification_question": None,
            }

        if "you are extracting typed arguments" in prompt_l:
            if "runtime.describe_capabilities" in prompt:
                return {
                    "task_id": "task_1",
                    "capability_id": "runtime.describe_capabilities",
                    "operation_id": "describe_capabilities",
                    "arguments": {"include_details": True},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                }
            if "system.disk_usage" in prompt:
                return {
                    "task_id": "task_1",
                    "capability_id": "system.disk_usage",
                    "operation_id": "disk_usage",
                    "arguments": {"path": ".", "human_readable": True},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                }
            if "system.memory_status" in prompt:
                return {
                    "task_id": "task_1",
                    "capability_id": "system.memory_status",
                    "operation_id": "memory_status",
                    "arguments": {"human_readable": True},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                }
            return {
                "task_id": "task_1",
                "capability_id": (
                    "system.cpu_load" if "system.cpu_load" in prompt else "system.uptime"
                ),
                "operation_id": "cpu_load" if "system.cpu_load" in prompt else "uptime",
                "arguments": {},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.95,
            }

        if "you are selecting a safe display plan" in prompt_l:
            return {
                "display_type": "table",
                "title": "System Inspection",
                "sections": [
                    {
                        "title": "System Inspection",
                        "display_type": "table",
                        "source_node_id": "node::task_1",
                    }
                ],
                "constraints": {},
                "redaction_policy": "standard",
            }

        raise AssertionError(f"Unexpected prompt: {prompt}")


class FakeGatewayClient:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def invoke(self, *, node, capability, arguments, execution_context):
        result = run_remote_operation(
            capability.manifest.backend_operation or capability.manifest.capability_id,
            arguments,
            workspace_root=self.workspace_root,
        )
        return ExecutionResult(
            node_id=node.id,
            status="success",
            data_preview=result["data_preview"],
            metadata=result["metadata"],
        )


def _runtime(tmp_path: Path) -> AgentRuntime:
    registry = build_default_registry()
    store = InMemoryResultStore()
    engine = ExecutionEngine(
        registry,
        {
            "workspace_root": str(tmp_path),
            "allow_shell_execution": False,
            "allow_network_operations": False,
            "gateway_url": "http://gateway",
        },
        store,
        gateway_client=FakeGatewayClient(tmp_path),
    )
    return AgentRuntime(SystemLLMClient(), registry, engine, OutputPipelineOrchestrator())


def test_memory_prompt_runs_system_memory_status_end_to_end(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_request("how much free memory do i have on this system?", {"workspace_root": str(tmp_path)})
    assert "memory" in response.lower()
    assert runtime.last_plan_summary is not None
    assert runtime.last_plan_summary["selected_capabilities"][0]["capability_id"] == "system.memory_status"
    trace = runtime.last_planning_trace
    assert trace is not None
    assert trace.capability_fit_decisions_by_task["task_1"]["status"] == "fit"


def test_disk_prompt_runs_system_disk_usage_end_to_end(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_request("how much disk space is free?", {"workspace_root": str(tmp_path)})
    assert "free" in response.lower() or "total" in response.lower()
    assert runtime.last_plan_summary["selected_capabilities"][0]["capability_id"] == "system.disk_usage"
    assert runtime.last_planning_trace.capability_fit_decisions_by_task["task_1"]["status"] == "fit"


def test_cpu_prompt_runs_system_cpu_load_end_to_end(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_request("what is the CPU load?", {"workspace_root": str(tmp_path)})
    assert "load" in response.lower() or "cpu" in response.lower()
    assert runtime.last_plan_summary["selected_capabilities"][0]["capability_id"] == "system.cpu_load"
    assert runtime.last_planning_trace.capability_fit_decisions_by_task["task_1"]["status"] == "fit"


def test_uptime_prompt_runs_system_uptime_end_to_end(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_request("what is the system uptime?", {"workspace_root": str(tmp_path)})
    assert "uptime" in response.lower() or "seconds" in response.lower()
    assert runtime.last_plan_summary["selected_capabilities"][0]["capability_id"] == "system.uptime"
    assert runtime.last_planning_trace.capability_fit_decisions_by_task["task_1"]["status"] == "fit"


def test_runtime_capabilities_prompt_fits_runtime_describe_capabilities(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    response = runtime.handle_request("what are my capabilities?", {"workspace_root": str(tmp_path)})
    assert "direct answering without tools is not implemented" not in response.lower()
    assert runtime.last_plan_summary["selected_capabilities"][0]["capability_id"] == "runtime.describe_capabilities"
    assert runtime.last_planning_trace.capability_fit_decisions_by_task["task_1"]["status"] == "fit"
