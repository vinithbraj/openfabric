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


class FakeLLMClient:
    """Route structured prompts to fixed payloads by stage marker."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        if "You are classifying a user prompt" in prompt:
            if "What is 2+2?" in prompt:
                return {
                    "prompt_type": "simple_question",
                    "requires_tools": False,
                    "likely_domains": [],
                    "risk_level": "low",
                    "needs_clarification": False,
                    "clarification_question": None,
                    "reason": "This is a direct question with no tool need.",
                }
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
    return AgentRuntime(FakeLLMClient(), registry, engine, OutputPipelineOrchestrator())


def test_handle_request_returns_direct_answer_placeholder(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    response = runtime.handle_request("What is 2+2?", {"workspace_root": str(tmp_path)})

    assert "Direct answering without tools is not implemented yet" in response


def test_handle_request_runs_full_tool_pipeline(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    runtime = _runtime(tmp_path)

    response = runtime.handle_request(
        "list all files in this folder",
        {"workspace_root": str(tmp_path)},
    )

    assert "Workspace Files" in response
    assert "README.md" in response
