from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import ActionDAG, ActionNode, ExecutionResult, ResultBundle, UserRequest
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.failure_repair import attempt_failure_repair
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.input_pipeline.planning_review import review_action_dag
from agent_runtime.llm.critique import critique_decomposition_with_llm
from agent_runtime.llm.proposals import TaskDecompositionProposal
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator


class QueueLLMClient:
    """Return queued payloads based on prompt marker substrings."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = {key: list(value) for key, value in responses.items()}
        self.prompts: list[str] = []
        self.model = "critique-repair-test-model"
        self.temperature = 0.0

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        _ = schema
        self.prompts.append(prompt)
        for marker, queue in self.responses.items():
            if marker in prompt:
                if not queue:
                    raise AssertionError(f"no queued response left for marker {marker!r}")
                return dict(queue.pop(0))
        raise AssertionError(f"unexpected prompt: {prompt}")


class FailIfCalledLLM:
    """Fail fast when a test expects no LLM invocation."""

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"LLM should not have been called: {prompt}")


class RepairingGatewayClient:
    """Return an execution error until the repaired argument becomes safe."""

    def invoke(self, *, node, capability, arguments, execution_context):
        _ = execution_context
        if capability.manifest.capability_id != "filesystem.read_file":
            raise AssertionError(f"unexpected capability: {capability.manifest.capability_id}")
        path = str(arguments.get("path") or "")
        if path == "README.md":
            return ExecutionResult(
                node_id=node.id,
                status="success",
                data_preview={
                    "path": "README.md",
                    "content_preview": "hello from repaired read",
                    "truncated": False,
                },
                metadata={"source": "repair-test"},
            )
        return ExecutionResult(
            node_id=node.id,
            status="error",
            error=f"File not found: {path}",
            metadata={"error_class": "FileNotFoundError"},
        )


def _repair_runtime(tmp_path: Path, repair_payload: dict[str, Any]) -> AgentRuntime:
    registry = build_default_registry()
    engine = ExecutionEngine(
        registry,
        {
            "workspace_root": str(tmp_path),
            "allow_shell_execution": False,
            "allow_network_operations": False,
            "gateway_url": "http://gateway",
        },
        InMemoryResultStore(),
        gateway_client=RepairingGatewayClient(),
    )
    repeated_decomposition = {
        "tasks": [
            {
                "id": "task-read",
                "description": "Read the requested file.",
                "semantic_verb": "read",
                "object_type": "file",
                "intent_confidence": 0.97,
                "constraints": {},
                "dependencies": [],
                "raw_evidence": "read README.md",
                "requires_confirmation": False,
                "risk_level": "low",
            }
        ],
        "global_constraints": {},
        "unresolved_references": [],
        "assumptions": [],
        "confidence": 0.95,
    }
    repeated_selection = {
        "task_id": "task-read",
        "candidates": [
            {
                "capability_id": "filesystem.read_file",
                "operation_id": "read_file",
                "confidence": 0.96,
                "reason": "Reading a file matches filesystem.read_file.",
                "assumptions": [],
            }
        ],
        "selected": {
            "capability_id": "filesystem.read_file",
            "operation_id": "read_file",
            "confidence": 0.96,
            "reason": "Reading a file matches filesystem.read_file.",
            "assumptions": [],
        },
        "unresolved_reason": None,
    }
    llm = QueueLLMClient(
        {
            "You are classifying a user prompt": [
                {
                    "prompt_type": "simple_tool_task",
                    "requires_tools": True,
                    "likely_domains": ["filesystem"],
                    "risk_level": "low",
                    "needs_clarification": False,
                    "clarification_question": None,
                    "reason": "This is a filesystem read request.",
                    "confidence": 0.95,
                    "assumptions": [],
                }
            ],
            "You are decomposing a user prompt": [
                repeated_decomposition,
                repeated_decomposition,
                repeated_decomposition,
            ],
            "You are critiquing a proposed task decomposition": [
                {
                    "missing_user_intents": [],
                    "hallucinated_tasks": [],
                    "dependency_warnings": [],
                    "unsafe_task_warnings": [],
                    "unresolved_references": [],
                    "recommended_repair": None,
                    "confidence": 0.0,
                }
            ],
            "You are assigning semantic verbs": [
                {
                    "assignments": [
                        {
                            "task_id": "task-read",
                            "semantic_verb": "read",
                            "object_type": "file",
                            "intent_confidence": 0.96,
                            "risk_level": "low",
                            "requires_confirmation": False,
                        }
                    ]
                }
            ],
            "You are selecting capability candidates": [
                repeated_selection,
                repeated_selection,
                repeated_selection,
            ],
            "You are assessing whether a selected capability truly fits a task": [
                {
                    "task_id": "task-read",
                    "candidate_capability_id": "filesystem.read_file",
                    "candidate_operation_id": "read_file",
                    "proposed_status": "fit",
                    "confidence": 0.95,
                    "semantic_reason": "The task is to read a file.",
                    "domain_reason": "Filesystem matches the request.",
                    "object_type_reason": "The object is a file.",
                    "argument_reason": "A path can be extracted.",
                    "risk_reason": "Read-only capability.",
                    "missing_capability_description": None,
                    "suggested_domain": "filesystem",
                    "suggested_object_type": "file",
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            ],
            "You are proposing typed dataflow": [
                {
                    "refs": [],
                    "derived_tasks": [],
                    "dependency_edges": [],
                    "assumptions": [],
                    "unresolved_dataflows": [],
                    "confidence": 0.0,
                }
            ],
            "You are extracting typed arguments": [
                {
                    "task_id": "task-read",
                    "capability_id": "filesystem.read_file",
                    "operation_id": "read_file",
                    "arguments": {"path": "MISSING.md"},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.9,
                },
                {
                    "task_id": "task-read",
                    "capability_id": "filesystem.read_file",
                    "operation_id": "read_file",
                    "arguments": {"path": "MISSING.md"},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.9,
                },
                {
                    "task_id": "task-read",
                    "capability_id": "filesystem.read_file",
                    "operation_id": "read_file",
                    "arguments": {"path": "MISSING.md"},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.9,
                },
            ],
            "You are reviewing a sanitized action DAG": [
                {
                    "missing_user_intents": [],
                    "suspicious_nodes": [],
                    "dependency_warnings": [],
                    "dataflow_warnings": [],
                    "output_expectation_warnings": [],
                    "recommended_repair": None,
                    "confidence": 0.0,
                }
            ],
            "You are proposing one safe repair": [repair_payload],
            "You are selecting a safe display plan": [
                {
                    "display_type": "code_block",
                    "title": "File Content",
                    "sections": [
                        {
                            "title": "File Content",
                            "display_type": "code_block",
                            "source_node_id": "node::task-read",
                        }
                    ],
                    "constraints": {},
                    "redaction_policy": "standard",
                }
            ],
        }
    )
    return AgentRuntime(llm, registry, engine, OutputPipelineOrchestrator())


def test_critique_detects_missing_aggregate_task() -> None:
    client = QueueLLMClient(
        {
            "You are critiquing a proposed task decomposition": [
                {
                    "missing_user_intents": ["Compute the total file size from the listing."],
                    "hallucinated_tasks": [],
                    "dependency_warnings": [],
                    "unsafe_task_warnings": [],
                    "unresolved_references": [],
                    "recommended_repair": {
                        "add_task": "Compute total regular file size.",
                    },
                    "confidence": 0.92,
                }
            ]
        }
    )
    proposal = TaskDecompositionProposal(
        tasks=[
            {
                "id": "task-list",
                "description": "List files in the current directory.",
            }
        ],
        global_constraints={},
        unresolved_references=[],
        assumptions=[],
    )

    critique = critique_decomposition_with_llm(
        "list files and compute the total file size",
        proposal,
        ["filesystem", "data"],
        client,
    )

    assert "total file size" in critique.missing_user_intents[0].lower()
    assert critique.recommended_repair is not None


def test_dag_review_detects_missing_dependency_warning() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-list",
                description="List files",
                semantic_verb="read",
                capability_id="filesystem.list_directory",
                operation_id="list_directory",
                arguments={"path": "."},
                safety_labels=[],
            ),
            ActionNode(
                id="node-2",
                task_id="task-count",
                description="Count the files",
                semantic_verb="analyze",
                capability_id="data.aggregate",
                operation_id="aggregate",
                arguments={"operation": "count"},
                safety_labels=[],
            ),
        ]
    )
    client = QueueLLMClient(
        {
            "You are reviewing a sanitized action DAG": [
                {
                    "missing_user_intents": [],
                    "suspicious_nodes": ["node-2"],
                    "dependency_warnings": ["node-2 should depend on the directory listing from node-1."],
                    "dataflow_warnings": ["node-2 has no input_ref."],
                    "output_expectation_warnings": [],
                    "recommended_repair": {"node_id": "node-2", "action": "add_dependency"},
                    "confidence": 0.85,
                }
            ]
        }
    )

    review = review_action_dag(UserRequest(raw_prompt="find files and count them"), dag, client)

    assert any("depend on" in warning for warning in review.dependency_warnings)
    assert any("input_ref" in warning for warning in review.dataflow_warnings)


def test_dag_review_suggestion_is_validated_before_use() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-read",
                description="Read file",
                semantic_verb="read",
                capability_id="filesystem.read_file",
                operation_id="read_file",
                arguments={"path": "README.md"},
                safety_labels=[],
            )
        ]
    )
    client = QueueLLMClient(
        {
            "You are reviewing a sanitized action DAG": [
                {
                    "missing_user_intents": [],
                    "suspicious_nodes": ["node-999"],
                    "dependency_warnings": [],
                    "dataflow_warnings": [],
                    "output_expectation_warnings": [],
                    "recommended_repair": {"node_id": "node-999", "action": "remove"},
                    "confidence": 0.9,
                }
            ]
        }
    )

    review = review_action_dag(UserRequest(raw_prompt="read README.md"), dag, client)

    assert review.confidence == 0.0
    assert review.suspicious_nodes == []
    assert review.recommended_repair is None


def test_retry_limit_enforced_without_calling_llm(tmp_path: Path) -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-read",
                description="Read a file",
                semantic_verb="read",
                capability_id="filesystem.read_file",
                operation_id="read_file",
                arguments={"path": "README.md"},
                safety_labels=[],
            )
        ]
    )
    bundle = ResultBundle(
        dag_id=dag.dag_id,
        results=[ExecutionResult(node_id="node-1", status="error", error="failed", metadata={})],
        status="error",
        safe_summary="failed",
        metadata={},
    )

    repaired_dag, metadata = attempt_failure_repair(
        user_request=UserRequest(raw_prompt="read README.md"),
        dag=dag,
        result_bundle=bundle,
        registry=build_default_registry(),
        llm_client=FailIfCalledLLM(),
        safety_config=ExecutionEngine(
            build_default_registry(),
            {"workspace_root": str(tmp_path), "gateway_url": "http://gateway"},
        ).safety_policy.config,
        repair_attempt_count=1,
        max_repair_attempts=1,
    )

    assert repaired_dag is None
    assert metadata["retry_limit_enforced"] is True


def test_repair_attempt_recorded_in_planning_trace(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello from repaired read", encoding="utf-8")
    runtime = _repair_runtime(
        tmp_path,
        {
            "failed_node_id": "node::task-read",
            "proposed_action": "retry_with_arguments",
            "corrected_arguments": {"path": "README.md"},
            "alternate_capability_id": None,
            "user_message": "",
            "confidence": 0.94,
            "reason": "The file path should be README.md instead of MISSING.md.",
        },
    )

    response = runtime.handle_request("read README.md", {"workspace_root": str(tmp_path)})

    assert "hello from repaired read" in response
    trace = runtime.last_planning_trace
    assert trace is not None
    assert any(entry.stage == "failure_repair" for entry in trace.entries)
    assert trace.metadata.get("failure_repairs")


def test_failed_repair_produces_clean_output(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello from repaired read", encoding="utf-8")
    runtime = _repair_runtime(
        tmp_path,
        {
            "failed_node_id": "node::task-read",
            "proposed_action": "retry_with_arguments",
            "corrected_arguments": {"path": "../secret.txt"},
            "alternate_capability_id": None,
            "user_message": "",
            "confidence": 0.9,
            "reason": "Try a parent directory path.",
        },
    )

    response = runtime.handle_request("read README.md", {"workspace_root": str(tmp_path)})

    assert "Execution failed" in response
    assert "Traceback" not in response
    assert "FileNotFoundError:" not in response

