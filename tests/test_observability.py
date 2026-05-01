from __future__ import annotations

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
    assert "raw_llm_response" not in selected.details
