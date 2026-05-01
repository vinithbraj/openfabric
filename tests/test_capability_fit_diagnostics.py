from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import CapabilityRef, ExecutionResult, TaskFrame
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.input_pipeline.capability_fit import _fit_prompt, assess_capability_fit
from agent_runtime.input_pipeline.domain_selection import (
    CapabilitySelectionResult,
    _build_selection_prompt,
)
from agent_runtime.llm.client import LLMClientError
from agent_runtime.llm.structured_call import StructuredCallDiagnostics, StructuredCallError, structured_call
from agent_runtime.observability import InMemoryEventSink
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from gateway_agent.remote_runner import run_remote_operation


class _SchemaModel(StructuredCallDiagnostics):
    pass


class InvalidJSONClient:
    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = prompt, schema
        raise LLMClientError(
            error_kind="invalid_json",
            error_message="LLM response did not contain a valid JSON object.",
            raw_response_preview="not-json-at-all",
            raw_payload_preview='{"choices":[{"message":{"content":"not-json-at-all"}}]}',
        )


class WrongShapeClient:
    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = prompt, schema
        return {"unexpected": True}


class FitFailureLLMClient:
    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        prompt_l = prompt.lower()
        if "you are classifying a user prompt" in prompt_l:
            return {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["filesystem"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "Filesystem request.",
                "confidence": 0.95,
                "assumptions": [],
            }
        if "you are decomposing a user prompt" in prompt_l:
            return {
                "tasks": [
                    {
                        "id": "task-list",
                        "description": "List all files in the current folder.",
                        "semantic_verb": "read",
                        "object_type": "directory",
                        "intent_confidence": 0.98,
                        "constraints": {},
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
        if "you are assigning semantic verbs" in prompt_l:
            return {
                "assignments": [
                    {
                        "task_id": "task-list",
                        "semantic_verb": "read",
                        "object_type": "directory",
                        "intent_confidence": 0.97,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            }
        if "you are selecting capability candidates" in prompt_l:
            return {
                "task_id": "task-list",
                "evaluations": [
                    {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "fits": True,
                        "confidence": 0.97,
                        "reason": "This directly answers a directory listing request.",
                        "domain_reason": "Filesystem is the right domain.",
                        "object_type_reason": "Directory matches filesystem.directory.",
                        "argument_reason": "Path can be extracted separately.",
                        "risk_reason": "Read-only and low risk.",
                        "missing_arguments_likely": [],
                    }
                ],
                "unresolved_reason": None,
            }
        if "you are assessing whether a selected capability truly fits a task" in prompt_l:
            raise LLMClientError(
                error_kind="invalid_json",
                error_message="LLM response did not contain a valid JSON object.",
                raw_response_preview="```json\nnope\n```",
                raw_payload_preview='{"choices":[{"message":{"content":"```json\\nnope\\n```"}}]}',
            )
        if "you are extracting typed arguments" in prompt_l:
            return {
                "task_id": "task-list",
                "capability_id": "filesystem.list_directory",
                "operation_id": "list_directory",
                "arguments": {"path": "."},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.95,
            }
        if "you are selecting a safe display plan" in prompt_l:
            return {
                "display_type": "table",
                "title": "Workspace Files",
                "sections": [
                    {
                        "title": "Workspace Files",
                        "display_type": "table",
                        "source_node_id": "node::task-list",
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
        reason="Selected by test.",
    )
    return CapabilitySelectionResult(task_id=task_id, candidates=[selected], selected=selected, unresolved_reason=None)


def _context(prompt: str, likely_domains: list[str]) -> dict[str, Any]:
    return {
        "original_prompt": prompt,
        "prompt_type": "simple_tool_task",
        "likely_domains": likely_domains,
        "risk_level": "low",
    }


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
    return AgentRuntime(FitFailureLLMClient(), registry, engine, OutputPipelineOrchestrator())


def test_fit_prompt_is_deduped_and_json_serialized() -> None:
    registry = build_default_registry()
    task = _task("task_1", "how much free memory do i have on this system?", "read", "memory")
    selection = _selection("task_1", "system.memory_status", "memory_status").selected
    assert selection is not None
    manifest = registry.get("system.memory_status").manifest

    prompt = _fit_prompt(task, selection, manifest, _context(task.description, ["system"]))

    assert "JSON schema:" not in prompt
    assert '"task_id": "task_1"' in prompt
    assert "'task_id': 'task_1'" not in prompt


def test_shortlist_prompt_is_deduped_and_json_serialized() -> None:
    prompt = _build_selection_prompt(
        _task("task_1", "list files in this folder", "read", "directory"),
        [
            {
                "capability_id": "filesystem.list_directory",
                "operation_id": "list_directory",
                "domain": "filesystem",
                "description": "List files.",
                "semantic_verbs": ["read"],
                "object_types": ["filesystem.directory"],
                "required_arguments": ["path"],
                "optional_arguments": [],
                "risk_level": "low",
                "read_only": True,
                "examples": [],
            }
        ],
    )

    assert "JSON schema:" not in prompt
    assert '"capability_id": "filesystem.list_directory"' in prompt
    assert "'capability_id': 'filesystem.list_directory'" not in prompt


def test_structured_call_wraps_schema_validation_failure() -> None:
    try:
        structured_call(WrongShapeClient(), "test prompt", StructuredCallDiagnostics)
    except StructuredCallError as exc:
        assert exc.diagnostics.error_kind == "schema_validation_error"
        assert exc.diagnostics.schema_name == "StructuredCallDiagnostics"
        assert exc.diagnostics.raw_payload_preview is not None
        assert exc.diagnostics.validation_errors
    else:  # pragma: no cover
        raise AssertionError("structured_call should have raised StructuredCallError")


def test_capability_fit_records_llm_diagnostics_and_accepts_deterministically() -> None:
    registry = build_default_registry()
    task = _task("task-memory", "how much free memory do i have on this system?", "read", "system.memory")
    selection = _selection("task-memory", "system.memory_status", "memory_status")

    decisions, gaps = assess_capability_fit(
        [task],
        [selection],
        registry,
        _context(task.description, ["system_administration", "operating_system", "system"]),
        InvalidJSONClient(),
    )

    assert decisions[0].is_fit
    assert decisions[0].llm_failed_structurally is True
    assert decisions[0].llm_diagnostics is not None
    assert decisions[0].llm_diagnostics.error_kind == "invalid_json"
    assert "structured capability-fit assessment" in decisions[0].reasons[0].lower()
    assert any("invalid_json" in reason for reason in decisions[0].reasons)
    assert gaps == []


def test_capability_fit_observability_shows_error_kind_without_raw_preview_by_default(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    sink = InMemoryEventSink()
    runtime = _runtime(tmp_path)

    runtime.handle_request(
        "list all files in this folder",
        {
            "workspace_root": str(tmp_path),
            "observability": {"enabled": True, "sink": sink},
        },
    )

    fit_event = next(
        event
        for event in sink.events
        if event.stage == "capability_fit" and event.event_type == "validation.accepted"
    )
    assert fit_event.details["llm_error_kind"] == "invalid_json"
    assert fit_event.details["llm_error_message"]
    assert fit_event.details["llm_raw_response_preview"] is None
