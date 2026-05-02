from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import ExecutionResult
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.observability import (
    InMemoryEventSink,
    PipelineEvent,
    build_observability_context,
    format_event_for_openwebui,
    redact_event,
)
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from gateway_agent.remote_runner import run_remote_operation


class FakeLLMClient:
    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if "You are classifying a user prompt" in prompt:
            return {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["filesystem"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "The prompt asks for a filesystem action.",
            }
        if "You are decomposing a user prompt" in prompt:
            return {
                "tasks": [
                    {
                        "id": "task-list-files",
                        "description": "List all files in the current folder.",
                        "semantic_verb": "read",
                        "object_type": "directory",
                        "intent_confidence": 0.98,
                        "constraints": {"path": "."},
                        "dependencies": [],
                        "raw_evidence": "list all files in this folder",
                        "requires_confirmation": False,
                        "risk_level": "low",
                    }
                ],
                "global_constraints": {"path": "."},
                "unresolved_references": [],
                "assumptions": [],
            }
        if "You are assigning semantic verbs" in prompt:
            return {
                "assignments": [
                    {
                        "task_id": "task-list-files",
                        "semantic_verb": "read",
                        "object_type": "directory",
                        "intent_confidence": 0.97,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            }
        if "You are selecting capability candidates" in prompt:
            return {
                "task_id": "task-list-files",
                "candidates": [
                    {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.96,
                        "reason": "This task is a directory listing request for the current folder.",
                    }
                ],
                "selected": {
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "confidence": 0.96,
                    "reason": "This task is a directory listing request for the current folder.",
                },
                "unresolved_reason": None,
            }
        if "You are assessing whether a selected capability truly fits a task" in prompt:
            return {
                "task_id": "task-list-files",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "fit",
                "confidence": 0.96,
                "semantic_reason": "The task is to read a directory listing.",
                "domain_reason": "Filesystem is the correct domain.",
                "object_type_reason": "The object is a directory/folder.",
                "argument_reason": "The path can be extracted separately.",
                "risk_reason": "Read-only and low risk.",
                "missing_capability_description": None,
                "suggested_domain": "filesystem",
                "suggested_object_type": "directory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        if "You are extracting typed arguments" in prompt:
            return {
                "task_id": "task-list-files",
                "capability_id": "filesystem.list_directory",
                "operation_id": "list_directory",
                "arguments": {"path": "."},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.95,
            }
        if "You are selecting a safe display plan" in prompt:
            return {
                "display_type": "table",
                "title": "Workspace Files",
                "sections": [
                    {
                        "title": "Workspace Files",
                        "display_type": "table",
                        "source_node_id": "node::task-list-files",
                    }
                ],
                "constraints": {},
                "redaction_policy": "standard",
            }
        return {}


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


class MixedSystemSaveLLMClient:
    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        prompt_l = prompt.lower()

        if "you are classifying a user prompt" in prompt_l:
            return {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["system_administration", "operating_system", "system"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "The prompt asks for safe system inspection plus a follow-up action.",
                "confidence": 0.97,
                "assumptions": [],
            }
        if "you are decomposing a user prompt" in prompt_l:
            return {
                "tasks": [
                    {
                        "id": "task_1",
                        "description": "Retrieve the amount of free memory available on the system",
                        "semantic_verb": "read",
                        "object_type": "system.memory",
                        "intent_confidence": 0.98,
                        "constraints": {"human_readable": True},
                        "dependencies": [],
                        "raw_evidence": "how much free memory do i have on this system?",
                        "requires_confirmation": False,
                        "risk_level": "low",
                    },
                    {
                        "id": "task_2",
                        "description": "Email the system memory report to ops@example.com",
                        "semantic_verb": "create",
                        "object_type": "message",
                        "intent_confidence": 0.96,
                        "constraints": {"recipient": "ops@example.com"},
                        "dependencies": ["task_1"],
                        "raw_evidence": "email the report to ops@example.com",
                        "requires_confirmation": False,
                        "risk_level": "low",
                    },
                ],
                "global_constraints": {"recipient": "ops@example.com"},
                "unresolved_references": [],
                "assumptions": [],
            }
        if "you are assigning semantic verbs" in prompt_l:
            return {
                "assignments": [
                    {
                        "task_id": "task_1",
                        "semantic_verb": "read",
                        "object_type": "system.memory",
                        "intent_confidence": 0.97,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    },
                    {
                        "task_id": "task_2",
                        "semantic_verb": "create",
                        "object_type": "message",
                        "intent_confidence": 0.96,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    },
                ]
            }
        if "you are selecting capability candidates for one task frame" in prompt_l:
            if "retrieve the amount of free memory available on the system" in prompt_l:
                return {
                    "task_id": "task_1",
                    "evaluations": [
                        {
                            "capability_id": "system.memory_status",
                            "operation_id": "memory_status",
                            "fits": True,
                            "confidence": 0.95,
                            "reason": "This directly retrieves memory availability.",
                            "domain_reason": "System inspection belongs to the system domain.",
                            "object_type_reason": "system.memory matches the task object type.",
                            "argument_reason": "human_readable can be filled later.",
                            "risk_reason": "Read-only and low risk.",
                            "missing_arguments_likely": [],
                        }
                    ],
                    "unresolved_reason": None,
                }
            capability_ids = re.findall(r'"capability_id":\s*"([^"]+)"', prompt)
            operation_ids = re.findall(r'"operation_id":\s*"([^"]+)"', prompt)
            capability_id = capability_ids[0]
            operation_id = operation_ids[0]
            return {
                "task_id": "task_2",
                "evaluations": [
                    {
                        "capability_id": capability_id,
                        "operation_id": operation_id,
                        "fits": False,
                        "confidence": 0.96,
                        "reason": "The runtime does not have a safe messaging or email capability for this task.",
                        "domain_reason": "An email request needs a messaging-capable tool.",
                        "object_type_reason": "Filesystem and system inspection tools are not the same as sending a message.",
                        "argument_reason": "The candidate does not accept recipient or delivery arguments.",
                        "risk_reason": "Rejecting avoids pretending an unrelated tool can send a message.",
                        "missing_arguments_likely": [],
                    }
                ],
                "unresolved_reason": "No shortlisted capability was accepted for task task_2.",
            }
        if "you are assessing whether a selected capability truly fits a task" in prompt_l:
            if '"candidate_capability_id": "system.memory_status"' in prompt:
                return {
                    "task_id": "task_1",
                    "candidate_capability_id": "system.memory_status",
                    "candidate_operation_id": "memory_status",
                    "fits": True,
                    "confidence": 0.95,
                    "semantic_reason": "The task asks to read free memory from the system.",
                    "domain_reason": "system_administration and system are compatible.",
                    "object_type_reason": "system.memory exactly matches the capability.",
                    "argument_reason": "No required arguments are missing.",
                    "risk_reason": "The capability is read-only and low risk.",
                    "better_capability_id": None,
                    "missing_capability_description": None,
                    "suggested_domain": "system",
                    "suggested_object_type": "system.memory",
                    "missing_arguments_likely": [],
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            capability_ids = re.findall(r'"candidate_capability_id":\s*"([^"]+)"', prompt)
            operation_ids = re.findall(r'"candidate_operation_id":\s*"([^"]+)"', prompt)
            capability_id = capability_ids[0]
            operation_id = operation_ids[0]
            return {
                "task_id": "task_2",
                "candidate_capability_id": capability_id,
                "candidate_operation_id": operation_id,
                "fits": False,
                "confidence": 0.96,
                "primary_failure_mode": "semantic_mismatch",
                "semantic_reason": "The task is to create and save a report file, not just inspect or search.",
                "domain_reason": "The request spans system memory data plus file creation, which this candidate cannot satisfy.",
                "object_type_reason": "The task needs a writable report artifact.",
                "argument_reason": "The candidate lacks write-oriented arguments.",
                "risk_reason": "Rejecting avoids falsely treating a read-only capability as a write tool.",
                "better_capability_id": None,
                "missing_capability_description": "A safe file-writing capability is required.",
                "suggested_domain": "system",
                "suggested_object_type": "memory_report",
                "missing_arguments_likely": [],
                "requires_clarification": False,
                "clarification_question": None,
            }
        if "you are extracting typed arguments" in prompt_l:
            return {
                "task_id": "task_1",
                "capability_id": "system.memory_status",
                "operation_id": "memory_status",
                "arguments": {"human_readable": True},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.95,
            }
        if "you are reviewing a sanitized action dag" in prompt_l:
            return {
                "missing_user_intents": [
                    "The user wants the memory report saved to report.txt.",
                ],
                "suspicious_nodes": [],
                "dependency_warnings": [],
                "dataflow_warnings": [],
                "output_expectation_warnings": [
                    "The DAG reads memory data but does not save it to a file.",
                ],
                "recommended_repair": None,
                "confidence": 0.92,
            }
        if "you are selecting a safe display plan" in prompt_l:
            return {
                "display_type": "table",
                "title": "System Memory Report",
                "sections": [
                    {
                        "title": "System Memory Report",
                        "display_type": "table",
                        "source_node_id": "node::task_1",
                    }
                ],
                "constraints": {},
                "redaction_policy": "standard",
            }
        return {}


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
    return AgentRuntime(FakeLLMClient(), registry, engine, OutputPipelineOrchestrator())


def test_redact_event_removes_secret_like_content() -> None:
    event = PipelineEvent(
        request_id="req-1",
        level="info",
        stage="test",
        event_type="test",
        title="Test",
        summary="Summary",
        details={"api_key": "sk-secret-1234567890", "note": "safe"},
    )

    redacted = redact_event(event, "standard")

    assert redacted.details["api_key"] == "[redacted]"
    assert redacted.details["note"] == "safe"


def test_build_observability_context_supports_callback() -> None:
    messages: list[str] = []

    context = build_observability_context(
        "req-1",
        {"event_callback": messages.append},
    )
    context.info("test", "demo", "Hello", "World", {"foo": "bar"})

    assert context.enabled is True
    assert messages
    assert "Hello" in messages[0]


def test_openwebui_formatter_uses_markdown_sections_and_inline_code() -> None:
    event = PipelineEvent(
        request_id="req-1",
        level="info",
        stage="capability_fit",
        event_type="validation.accepted",
        title="Capability fit accepted",
        summary="The selected capability passed semantic and deterministic fit checks.",
        details={
            "task_id": "task_1",
            "capability_id": "system.memory_status",
            "operation_id": "memory_status",
            "status": "fit",
            "llm_fits": True,
            "llm_confidence": 0.95,
            "reasons": [
                "The capability matches the system memory task.",
                "Deterministic compatibility confirmed the selection.",
            ],
            "normalized_task_object_type": "system.memory",
        },
    )

    rendered = format_event_for_openwebui(event)

    assert rendered.startswith("### Capability Fit\n---\n")
    assert "> **Capability fit accepted**" in rendered
    assert "> - Task: `task_1`" in rendered
    assert "> - Capability: `system.memory_status`" in rendered
    assert "> - Operation: `memory_status`" in rendered
    assert "> - LLM Fits: `true`" in rendered
    assert "> **Reasons**" in rendered
    assert "> - The capability matches the system memory task." in rendered


def test_openwebui_formatter_renders_tasks_candidates_and_arguments_cleanly() -> None:
    event = PipelineEvent(
        request_id="req-1",
        level="info",
        stage="decomposition",
        event_type="llm.proposal.received",
        title="Task decomposition received",
        summary="The runtime received a structured task decomposition.",
        details={
            "task_count": 2,
            "tasks": [
                {
                    "task_id": "task_1",
                    "description": "Retrieve free memory available on the system",
                    "depends_on": [],
                },
                {
                    "task_id": "task_2",
                    "description": "Save the report to report.txt",
                    "depends_on": ["task_1"],
                },
            ],
            "candidates": [
                {
                    "capability_id": "system.memory_status",
                    "operation_id": "memory_status",
                    "confidence": 0.96,
                    "reason": "Direct match for a memory inspection task.",
                }
            ],
            "arguments": {"human_readable": True},
        },
    )

    rendered = format_event_for_openwebui(event)

    assert "> **Tasks**" in rendered
    assert "> - `task_1`" in rendered
    assert ">   - Details: `Retrieve free memory available on the system`" in rendered
    assert ">   - Depends On: `task_1`" in rendered
    assert "> **Candidates**" in rendered
    assert "`system.memory_status` via `memory_status`" in rendered
    assert "> **Arguments**" in rendered
    assert "> ```json" in rendered
    assert '"human_readable": true' in rendered


def test_openwebui_formatter_renders_verb_assignment_tasks_without_placeholder_details() -> None:
    event = PipelineEvent(
        request_id="req-1",
        level="info",
        stage="verb_assignment",
        event_type="validation.accepted",
        title="Verb assignments accepted",
        summary="Semantic task annotations passed deterministic validation.",
        details={
            "tasks": [
                {
                    "task_id": "task_1",
                    "description": "Retrieve free memory available on the system",
                    "semantic_verb": "read",
                    "object_type": "system.memory",
                    "risk_level": "low",
                },
                {
                    "task_id": "task_2",
                    "semantic_verb": "create",
                    "object_type": "filesystem.file",
                    "risk_level": "low",
                },
            ]
        },
    )

    rendered = format_event_for_openwebui(event)

    assert "> - `task_1`" in rendered
    assert ">   - Details: `Retrieve free memory available on the system`" in rendered
    assert ">   - Semantic Verb: `read`" in rendered
    assert ">   - Object Type: `system.memory`" in rendered
    assert ">   - Risk Level: `low`" in rendered
    assert "No description provided." not in rendered
    assert "> - `task_2`" in rendered
    assert ">   - Semantic Verb: `create`" in rendered
    assert ">   - Object Type: `filesystem.file`" in rendered


def test_runtime_emits_pipeline_events(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    sink = InMemoryEventSink()
    runtime = _runtime(tmp_path)

    response = runtime.handle_request(
        "list all files in this folder",
        {
            "workspace_root": str(tmp_path),
            "observability": {"enabled": True, "sink": sink},
        },
    )

    assert "README.md" in response
    event_types = {(event.stage, event.event_type) for event in sink.events}
    assert ("prompt_classification", "stage.started") in event_types
    assert ("capability_selection", "capability.selected") in event_types
    assert ("execution", "execution.node.completed") in event_types
    assert ("completed", "stage.completed") in event_types
    selected = next(
        event
        for event in sink.events
        if event.stage == "capability_selection" and event.event_type == "capability.selected"
    )
    assert selected.details["capability_id"] == "filesystem.list_directory"


def test_completed_status_is_partial_for_mixed_supported_and_unsupported_tasks(tmp_path: Path) -> None:
    sink = InMemoryEventSink()
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
    runtime = AgentRuntime(MixedSystemSaveLLMClient(), registry, engine, OutputPipelineOrchestrator())

    response = runtime.handle_request(
        "how much free memory do i have on this system ? and email the report to ops@example.com",
        {
            "workspace_root": str(tmp_path),
            "observability": {"enabled": True, "sink": sink},
        },
    )

    assert "memory" in response.lower()
    completed = next(
        event
        for event in sink.events
        if event.stage == "completed" and event.event_type == "stage.completed"
    )
    assert completed.details["final_status"] == "partial"
    assert completed.details["gap_count"] == 1
