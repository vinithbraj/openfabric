from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_runtime.capabilities import CapabilityRegistry, ReadFileCapability, ShellGitStatusCapability, build_default_registry
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import ActionDAG, ActionNode, CapabilityRef, ExecutionResult, ResultBundle, TaskFrame, UserRequest
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.failure_repair import attempt_failure_repair
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.input_pipeline.argument_extraction import extract_arguments
from agent_runtime.input_pipeline.decomposition import DecompositionResult, PromptClassification, decompose_prompt
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult, select_capabilities
from agent_runtime.input_pipeline.planning_review import review_action_dag
from agent_runtime.llm.critique import DecompositionCritique
from agent_runtime.llm.proposals import TaskDecompositionProposal
from agent_runtime.llm.reproducibility import PlanningTrace, hash_action_dag, load_planning_trace, save_planning_trace
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from gateway_agent.remote_runner import run_remote_operation


class QueueLLMClient:
    """Return queued payloads based on prompt marker substrings."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = {key: list(value) for key, value in responses.items()}
        self.prompts: list[str] = []
        self.model = "fake-model"
        self.temperature = 0.0

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        _ = schema
        for marker, queue in self.responses.items():
            if marker in prompt:
                if not queue:
                    raise AssertionError(f"no queued response left for marker {marker!r}")
                return dict(queue.pop(0))
        raise AssertionError(f"unexpected prompt: {prompt}")


class NoLLMClient:
    """Fail fast if a replay path accidentally invokes the LLM."""

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"LLM should not be called during replay: {prompt}")


class FakeGatewayClient:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def invoke(self, *, node, capability, arguments, execution_context) -> ExecutionResult:
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


def _classification() -> PromptClassification:
    return PromptClassification(
        prompt_type="simple_tool_task",
        requires_tools=True,
        likely_domains=["filesystem"],
        risk_level="low",
        needs_clarification=False,
        clarification_question=None,
        reason="test",
    )


def _task(task_id: str, description: str, semantic_verb: str = "read", object_type: str = "file") -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb=semantic_verb,
        object_type=object_type,
        intent_confidence=0.9,
        constraints={},
        dependencies=[],
        risk_level="low",
    )


def _ready_dag(dag: ActionDAG, allowed: bool = True) -> ActionDAG:
    prepared = dag.model_copy(
        update={
            "execution_ready": allowed,
            "safety_decision": {"allowed": allowed},
        }
    )
    return prepared.model_copy(update={"final_dag_hash": hash_action_dag(prepared)})


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
                    "reason": "Filesystem listing request.",
                    "confidence": 0.95,
                }
            ],
            "You are decomposing a user prompt": [
                {
                    "tasks": [
                        {
                            "id": "task-list",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "filesystem",
                            "intent_confidence": 0.97,
                            "constraints": {"path": "."},
                            "dependencies": [],
                            "raw_evidence": "list files in this folder",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
                {
                    "tasks": [
                        {
                            "id": "task-list",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "filesystem",
                            "intent_confidence": 0.97,
                            "constraints": {"path": "."},
                            "dependencies": [],
                            "raw_evidence": "list files in this folder",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
                {
                    "tasks": [
                        {
                            "id": "task-list",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "filesystem",
                            "intent_confidence": 0.97,
                            "constraints": {"path": "."},
                            "dependencies": [],
                            "raw_evidence": "list files in this folder",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
            ],
            "You are assigning semantic verbs": [
                {
                    "assignments": [
                        {
                            "task_id": "task-list",
                            "semantic_verb": "read",
                            "object_type": "filesystem",
                            "intent_confidence": 0.96,
                            "risk_level": "low",
                            "requires_confirmation": False,
                        }
                    ]
                }
            ],
            "You are selecting capability candidates": [
                {
                    "task_id": "task-list",
                    "candidates": [
                        {
                            "capability_id": "filesystem.list_directory",
                            "operation_id": "list_directory",
                            "confidence": 0.95,
                            "reason": "Directory listing request.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.95,
                        "reason": "Directory listing request.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
                {
                    "task_id": "task-list",
                    "candidates": [
                        {
                            "capability_id": "filesystem.list_directory",
                            "operation_id": "list_directory",
                            "confidence": 0.95,
                            "reason": "Directory listing request.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.95,
                        "reason": "Directory listing request.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
                {
                    "task_id": "task-list",
                    "candidates": [
                        {
                            "capability_id": "filesystem.list_directory",
                            "operation_id": "list_directory",
                            "confidence": 0.95,
                            "reason": "Directory listing request.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.95,
                        "reason": "Directory listing request.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
            ],
            "You are assessing whether a selected capability truly fits a task": [
                {
                    "task_id": "task-list",
                    "candidate_capability_id": "filesystem.list_directory",
                    "candidate_operation_id": "list_directory",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task is to list files from the current folder.",
                    "domain_reason": "Filesystem is the correct domain.",
                    "object_type_reason": "The object is a directory/filesystem path.",
                    "argument_reason": "The path can be extracted separately.",
                    "risk_reason": "Read-only and low risk.",
                    "missing_capability_description": None,
                    "suggested_domain": "filesystem",
                    "suggested_object_type": "filesystem",
                    "requires_clarification": False,
                    "clarification_question": None,
                },
                {
                    "task_id": "task-list",
                    "candidate_capability_id": "filesystem.list_directory",
                    "candidate_operation_id": "list_directory",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task is to list files from the current folder.",
                    "domain_reason": "Filesystem is the correct domain.",
                    "object_type_reason": "The object is a directory/filesystem path.",
                    "argument_reason": "The path can be extracted separately.",
                    "risk_reason": "Read-only and low risk.",
                    "missing_capability_description": None,
                    "suggested_domain": "filesystem",
                    "suggested_object_type": "filesystem",
                    "requires_clarification": False,
                    "clarification_question": None,
                },
                {
                    "task_id": "task-list",
                    "candidate_capability_id": "filesystem.list_directory",
                    "candidate_operation_id": "list_directory",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task is to list files from the current folder.",
                    "domain_reason": "Filesystem is the correct domain.",
                    "object_type_reason": "The object is a directory/filesystem path.",
                    "argument_reason": "The path can be extracted separately.",
                    "risk_reason": "Read-only and low risk.",
                    "missing_capability_description": None,
                    "suggested_domain": "filesystem",
                    "suggested_object_type": "filesystem",
                    "requires_clarification": False,
                    "clarification_question": None,
                },
            ],
            "You are extracting typed arguments": [
                {
                    "task_id": "task-list",
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "arguments": {"path": "."},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.94,
                },
                {
                    "task_id": "task-list",
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "arguments": {"path": "."},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.94,
                },
                {
                    "task_id": "task-list",
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "arguments": {"path": "."},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.94,
                },
            ],
            "You are selecting a safe display plan": [
                {
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
            ],
        }
    )
    return AgentRuntime(llm, registry, engine, OutputPipelineOrchestrator())


def test_executor_rejects_raw_proposal_objects() -> None:
    engine = ExecutionEngine(CapabilityRegistry())

    with pytest.raises(ValidationError, match="trusted ActionDAG"):
        engine.execute(TaskDecompositionProposal(tasks=[]))  # type: ignore[arg-type]


def test_multiple_capability_proposals_choose_valid_registered_candidate() -> None:
    client = QueueLLMClient(
        {
            "You are selecting capability candidates": [
                {
                    "task_id": "task-1",
                    "candidates": [
                        {
                            "capability_id": "missing.capability",
                            "operation_id": "nope",
                            "confidence": 0.99,
                            "reason": "invalid first try",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "missing.capability",
                        "operation_id": "nope",
                        "confidence": 0.99,
                        "reason": "invalid first try",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
                {
                    "task_id": "task-1",
                    "candidates": [
                        {
                            "capability_id": "filesystem.read_file",
                            "operation_id": "read_file",
                            "confidence": 0.80,
                            "reason": "Read a specific file safely from the filesystem.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.read_file",
                        "operation_id": "read_file",
                        "confidence": 0.80,
                        "reason": "Read a specific file safely from the filesystem.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
                {
                    "task_id": "task-1",
                    "candidates": [],
                    "selected": None,
                    "unresolved_reason": "unused third candidate",
                },
            ]
        }
    )
    results = select_capabilities(
        [_task("task-1", "read README.md", semantic_verb="read", object_type="file")],
        build_default_registry(),
        client,
        n_best=3,
    )
    assert results[0].selected is not None
    assert results[0].selected.capability_id == "filesystem.read_file"


def test_llm_critique_can_trigger_one_repair_attempt_but_not_force_acceptance() -> None:
    client = QueueLLMClient(
        {
            "You are decomposing a user prompt": [
                {
                    "tasks": [
                        {
                            "id": "task-broad",
                            "description": "Handle the whole request broadly.",
                            "semantic_verb": "read",
                            "object_type": "filesystem",
                            "intent_confidence": 0.7,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "list and summarize files",
                        }
                    ],
                    "global_constraints": {},
                    "unresolved_references": [],
                    "assumptions": ["broad first pass"],
                    "confidence": 0.7,
                },
                {
                    "tasks": [
                        {
                            "id": "task-better",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "filesystem",
                            "intent_confidence": 0.95,
                            "constraints": {"path": "."},
                            "dependencies": [],
                            "raw_evidence": "list files",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
            ],
            "You are critiquing a proposed task decomposition": [
                {
                    "missing_tasks": ["The listing task is too broad and should be narrowed."],
                    "hallucinated_tasks": [],
                    "dependency_issues": [],
                    "overly_broad_tasks": ["task-broad"],
                    "unsafe_tasks": [],
                    "unresolved_references": [],
                    "confidence": 0.9,
                }
            ],
        }
    )
    result = decompose_prompt(
        UserRequest(raw_prompt="list files in this folder"),
        _classification(),
        client,
        available_domains=["filesystem"],
        n_best=1,
    )
    assert result.tasks[0].id == "task-better"
    assert sum("You are decomposing a user prompt" in prompt for prompt in client.prompts) == 2


def test_dag_review_is_advisory_only() -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-1",
                description="read file",
                semantic_verb="read",
                capability_id="filesystem.read_file",
                operation_id="read_file",
                arguments={"path": "README.md"},
                safety_labels=[],
            )
        ]
    )
    original_dump = dag.model_dump(mode="json")
    client = QueueLLMClient(
        {
            "You are reviewing a sanitized action DAG": [
                {
                    "missing_user_intents": [],
                    "suspicious_nodes": ["node-1"],
                    "dependency_warnings": [],
                    "output_expectation_warnings": [],
                    "recommended_repair": {"action": "delete_node", "node_id": "node-1"},
                    "confidence": 0.8,
                }
            ]
        }
    )
    review = review_action_dag(UserRequest(raw_prompt="read README.md"), dag, client)
    assert review.recommended_repair is not None
    assert dag.model_dump(mode="json") == original_dump


def test_failure_repair_can_correct_safe_missing_argument(tmp_path: Path) -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-1",
                description="read a file",
                semantic_verb="read",
                capability_id="filesystem.read_file",
                operation_id="read_file",
                arguments={},
                safety_labels=[],
            )
        ]
    )
    bundle = ResultBundle(
        dag_id=dag.dag_id,
        results=[
            ExecutionResult(
                node_id="node-1",
                status="error",
                error="missing required arguments for filesystem.read_file: path",
                metadata={"capability_id": "filesystem.read_file", "operation_id": "read_file"},
            )
        ],
        status="error",
        safe_summary="failed",
        metadata={},
    )
    client = QueueLLMClient(
        {
            "You are proposing one safe repair": [
                {
                    "strategy": "correct_arguments",
                    "corrected_arguments": {"path": "README.md"},
                    "alternate_capability_id": None,
                    "alternate_operation_id": None,
                    "skip_node": False,
                    "ask_for_clarification": False,
                    "clarification_question": None,
                    "explanation": "Provide the missing file path.",
                    "confidence": 0.93,
                }
            ]
        }
    )
    repaired_dag, metadata = attempt_failure_repair(
        user_request=UserRequest(raw_prompt="read README.md"),
        dag=dag,
        result_bundle=bundle,
        registry=build_default_registry(),
        llm_client=client,
        safety_config=ExecutionEngine(build_default_registry(), {"workspace_root": str(tmp_path), "gateway_url": "http://gateway"}).safety_policy.config,
    )
    assert repaired_dag is not None
    assert repaired_dag.nodes[0].arguments["path"] == "README.md"
    assert metadata["attempted"] is True


def test_failure_repair_cannot_introduce_path_traversal(tmp_path: Path) -> None:
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-1",
                description="read a file",
                semantic_verb="read",
                capability_id="filesystem.read_file",
                operation_id="read_file",
                arguments={},
                safety_labels=[],
            )
        ]
    )
    bundle = ResultBundle(
        dag_id=dag.dag_id,
        results=[ExecutionResult(node_id="node-1", status="error", error="missing path", metadata={})],
        status="error",
        safe_summary="failed",
        metadata={},
    )
    client = QueueLLMClient(
        {
            "You are proposing one safe repair": [
                {
                    "strategy": "correct_arguments",
                    "corrected_arguments": {"path": "../secret.txt"},
                    "alternate_capability_id": None,
                    "alternate_operation_id": None,
                    "skip_node": False,
                    "ask_for_clarification": False,
                    "clarification_question": None,
                    "explanation": "Try a parent path.",
                    "confidence": 0.9,
                }
            ]
        }
    )
    repaired_dag, metadata = attempt_failure_repair(
        user_request=UserRequest(raw_prompt="read ../secret.txt"),
        dag=dag,
        result_bundle=bundle,
        registry=build_default_registry(),
        llm_client=client,
        safety_config=ExecutionEngine(build_default_registry(), {"workspace_root": str(tmp_path), "gateway_url": "http://gateway"}).safety_policy.config,
    )
    assert repaired_dag is None
    assert "blocked_reasons" in metadata or metadata.get("rejected")


def test_failure_repair_cannot_introduce_arbitrary_shell_commands(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(ShellGitStatusCapability())
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-1",
                description="show git status",
                semantic_verb="read",
                capability_id="shell.git_status",
                operation_id="git_status",
                arguments={"path": "."},
                safety_labels=[],
            )
        ]
    )
    bundle = ResultBundle(
        dag_id=dag.dag_id,
        results=[ExecutionResult(node_id="node-1", status="error", error="generic failure", metadata={})],
        status="error",
        safe_summary="failed",
        metadata={},
    )
    client = QueueLLMClient(
        {
            "You are proposing one safe repair": [
                {
                    "strategy": "correct_arguments",
                    "corrected_arguments": {"command": "rm -rf ."},
                    "alternate_capability_id": None,
                    "alternate_operation_id": None,
                    "skip_node": False,
                    "ask_for_clarification": False,
                    "clarification_question": None,
                    "explanation": "Unsafe command injection.",
                    "confidence": 0.9,
                }
            ]
        }
    )
    repaired_dag, metadata = attempt_failure_repair(
        user_request=UserRequest(raw_prompt="show git status"),
        dag=dag,
        result_bundle=bundle,
        registry=registry,
        llm_client=client,
        safety_config=ExecutionEngine(registry, {"workspace_root": str(tmp_path), "gateway_url": "http://gateway", "allow_shell_execution": True}).safety_policy.config,
    )
    assert repaired_dag is None
    assert "arbitrary shell command" in str(metadata.get("rejected"))


def test_execution_refuses_unvalidated_dag() -> None:
    engine = ExecutionEngine(build_default_registry(), {"workspace_root": ".", "gateway_url": "http://gateway"})
    dag = ActionDAG(
        nodes=[
            ActionNode(
                id="node-1",
                task_id="task-1",
                description="read file",
                semantic_verb="read",
                capability_id="filesystem.read_file",
                operation_id="read_file",
                arguments={"path": "README.md"},
                safety_labels=[],
            )
        ]
    )
    with pytest.raises(ValidationError, match="execution_ready"):
        engine.execute(dag, {"confirmation": True})


def test_planning_trace_captures_selected_plan_and_replay_works(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    runtime = _runtime(tmp_path)

    response = runtime.handle_request("list files in this folder", {"workspace_root": str(tmp_path)})

    assert "README.md" in response
    trace = runtime.last_planning_trace
    assert trace is not None
    assert trace.final_dag_hash
    assert trace.safety_decision is not None
    assert trace.validated_dag is not None
    assert any(entry.raw_llm_response is not None for entry in trace.entries)
    assert any(entry.selected_candidate is not None for entry in trace.entries)

    trace_path = save_planning_trace(trace, tmp_path / "planning-trace.json")
    loaded = load_planning_trace(trace_path)
    replay_runtime = AgentRuntime(
        NoLLMClient(),
        build_default_registry(),
        ExecutionEngine(
            build_default_registry(),
            {
                "workspace_root": str(tmp_path),
                "allow_shell_execution": False,
                "allow_network_operations": False,
                "gateway_url": "http://gateway",
            },
            InMemoryResultStore(),
            gateway_client=FakeGatewayClient(tmp_path),
        ),
        OutputPipelineOrchestrator(),
    )
    replayed = replay_runtime.replay_from_trace(loaded, {"confirmation": True})
    assert "README.md" in replayed
