from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_runtime.capabilities import CapabilityRegistry, build_default_registry
from agent_runtime.capabilities.system import (
    SystemCpuLoadCapability,
    SystemDiskUsageCapability,
    SystemEnvironmentSummaryCapability,
    SystemMemoryStatusCapability,
    SystemUptimeCapability,
)
from agent_runtime.core.types import ActionDAG, ActionNode, ExecutionResult, ResultBundle, TaskFrame, UserRequest
from agent_runtime.input_pipeline.decomposition import classify_prompt
from agent_runtime.input_pipeline.domain_selection import select_capabilities
from agent_runtime.output_pipeline.orchestrator import compose_output
from gateway_agent.remote_runner import RemoteToolError, run_remote_operation


class FakeLLMClient:
    """Small fake client for classification/selection/render tests."""

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


def _system_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(SystemMemoryStatusCapability())
    registry.register(SystemDiskUsageCapability())
    registry.register(SystemCpuLoadCapability())
    registry.register(SystemUptimeCapability())
    registry.register(SystemEnvironmentSummaryCapability())
    return registry


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


def test_registry_includes_all_system_capabilities() -> None:
    manifests = {manifest.capability_id for manifest in build_default_registry().list_manifests()}

    assert {
        "system.memory_status",
        "system.disk_usage",
        "system.cpu_load",
        "system.uptime",
        "system.environment_summary",
    } <= manifests


def test_memory_prompt_classifies_as_tool_task() -> None:
    llm = FakeLLMClient(
        {
            "You are classifying a user prompt": {
                "prompt_type": "simple_question",
                "requires_tools": False,
                "likely_domains": [],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "This looks like a simple question.",
                "confidence": 0.6,
                "assumptions": [],
            }
        }
    )

    classification = classify_prompt(UserRequest(raw_prompt="how much free memory do i have on this system?"), llm)

    assert classification.prompt_type == "simple_tool_task"
    assert classification.requires_tools is True
    assert "system" in {item.lower() for item in classification.likely_domains}


def test_memory_prompt_selects_system_memory_status_not_filesystem() -> None:
    llm = FakeLLMClient(
        {
            "how much free memory do i have on this system?": {
                "task_id": "task-memory",
                "evaluations": [
                    {
                        "capability_id": "system.memory_status",
                        "operation_id": "memory_status",
                        "fits": True,
                        "confidence": 0.97,
                        "reason": "This directly answers free memory.",
                        "domain_reason": "System resource inspection matches the task.",
                        "object_type_reason": "system.memory exactly matches the memory capability.",
                        "argument_reason": "No required arguments.",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    }
                ],
            }
        }
    )
    task = _task(
        "task-memory",
        "how much free memory do i have on this system?",
        "read",
        "system.memory",
    )

    results = select_capabilities([task], build_default_registry(), llm)

    assert results[0].selected is not None
    assert results[0].selected.capability_id == "system.memory_status"
    assert results[0].selected.capability_id != "filesystem.list_directory"


def test_memory_execution_returns_memory_and_swap_keys() -> None:
    result = run_remote_operation("system.memory_status", {"human_readable": True})

    assert "memory" in result["data_preview"]
    assert "swap" in result["data_preview"]
    assert "rows" in result["data_preview"]


def test_disk_execution_returns_total_used_free(tmp_path: Path) -> None:
    result = run_remote_operation(
        "system.disk_usage",
        {"path": ".", "human_readable": True},
        workspace_root=tmp_path,
    )

    assert {"total", "used", "free"} <= set(result["data_preview"])


def test_cpu_execution_returns_load_and_cpu_count() -> None:
    result = run_remote_operation("system.cpu_load", {})

    assert "cpu_count" in result["data_preview"]
    assert {"load_1m", "load_5m", "load_15m"} <= set(result["data_preview"])


def test_uptime_execution_returns_seconds_and_human() -> None:
    result = run_remote_operation("system.uptime", {})

    assert {"seconds", "human"} <= set(result["data_preview"])


def test_environment_summary_returns_safe_keys_only(tmp_path: Path) -> None:
    result = run_remote_operation("system.environment_summary", {}, workspace_root=tmp_path)

    assert {"os", "python_version", "working_directory", "cpu_count"} <= set(result["data_preview"])
    assert "environment_variables" not in result["data_preview"]


def test_unexpected_command_argument_is_rejected() -> None:
    with pytest.raises(RemoteToolError, match="unexpected arguments"):
        run_remote_operation("system.memory_status", {"command": "free -m"})


def test_system_output_renders_cleanly() -> None:
    bundle = ResultBundle(
        dag_id="dag-system",
        status="success",
        results=[
            ExecutionResult(
                node_id="node-memory",
                status="success",
                data_preview={
                    "memory": {
                        "total_bytes": 1000,
                        "available_bytes": 600,
                        "used_bytes": 400,
                        "used_percent": 40.0,
                    },
                    "swap": {
                        "total_bytes": 500,
                        "free_bytes": 500,
                        "used_bytes": 0,
                        "used_percent": 0.0,
                    },
                    "rows": [
                        {
                            "resource": "memory",
                            "total_bytes": 1000,
                            "used_bytes": 400,
                            "available_bytes": 600,
                            "used_percent": 40.0,
                        },
                        {
                            "resource": "swap",
                            "total_bytes": 500,
                            "used_bytes": 0,
                            "free_bytes": 500,
                            "used_percent": 0.0,
                        },
                    ],
                },
                metadata={"capability_id": "system.memory_status", "operation_id": "memory_status"},
            )
        ],
    )
    dag = ActionDAG(
        dag_id="dag-system",
        nodes=[
            ActionNode(
                id="node-memory",
                task_id="task-memory",
                description="Show memory status",
                semantic_verb="read",
                capability_id="system.memory_status",
                operation_id="memory_status",
                arguments={},
                safety_labels=[],
            )
        ],
    )
    llm = FakeLLMClient(
        {
            "You are selecting a safe display plan": {
                "display_type": "table",
                "title": "Memory Status",
                "sections": [
                    {
                        "title": "Memory Status",
                        "display_type": "table",
                        "source_node_id": "node-memory",
                    }
                ],
                "constraints": {},
                "redaction_policy": "standard",
            }
        }
    )

    output = compose_output(
        UserRequest(raw_prompt="how much free memory do i have on this system?"),
        dag,
        bundle,
        llm,
    )

    assert "## Memory Status" in output
    assert "| resource | total_bytes | used_bytes |" in output
    assert "memory" in output
