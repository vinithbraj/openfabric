from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import CapabilityRef, TaskFrame, UserRequest
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.input_pipeline.argument_extraction import extract_arguments
from agent_runtime.input_pipeline.capability_fit import assess_capability_fit
from agent_runtime.input_pipeline.dataflow_planning import plan_dataflow
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    hash_action_dag,
    load_planning_trace,
    replay_from_validated_dag,
    save_planning_trace,
)
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from gateway_agent.remote_runner import run_remote_operation


class RoutingLLMClient:
    """Route structured prompts to fixed responses by prompt marker."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = {key: list(value) for key, value in responses.items()}
        self.prompts: list[str] = []
        self.model = "trace-test-model"
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


class NoLLMClient:
    """Fail if replay unexpectedly invokes the LLM."""

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"LLM should not be called during replay: {prompt}")


class FakeGatewayClient:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def invoke(self, *, node, capability, arguments, execution_context):
        result = run_remote_operation(
            capability.manifest.backend_operation or capability.manifest.capability_id,
            arguments,
            workspace_root=self.workspace_root,
        )
        from agent_runtime.core.types import ExecutionResult

        return ExecutionResult(
            node_id=node.id,
            status="success",
            data_preview=result["data_preview"],
            metadata=result["metadata"],
        )


def _selection(task_id: str, capability_id: str, operation_id: str, confidence: float = 0.95) -> CapabilitySelectionResult:
    selected = CapabilityRef(
        capability_id=capability_id,
        operation_id=operation_id,
        confidence=confidence,
        reason="selected for test",
    )
    return CapabilitySelectionResult(
        task_id=task_id,
        candidates=[selected],
        selected=selected,
        unresolved_reason=None,
    )


def _list_runtime(tmp_path: Path) -> AgentRuntime:
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
    llm = RoutingLLMClient(
        {
            "You are classifying a user prompt": [
                {
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
            ],
            "You are decomposing a user prompt": [
                {
                    "tasks": [
                        {
                            "id": "task-list",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "directory",
                            "intent_confidence": 0.97,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "list files in this folder",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.96,
                },
                {
                    "tasks": [
                        {
                            "id": "task-list",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "directory",
                            "intent_confidence": 0.97,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "list files in this folder",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.96,
                },
                {
                    "tasks": [
                        {
                            "id": "task-list",
                            "description": "List files in the current folder.",
                            "semantic_verb": "read",
                            "object_type": "directory",
                            "intent_confidence": 0.97,
                            "constraints": {},
                            "dependencies": [],
                            "raw_evidence": "list files in this folder",
                            "requires_confirmation": False,
                            "risk_level": "low",
                        }
                    ],
                    "global_constraints": {"path": "."},
                    "unresolved_references": [],
                    "assumptions": [],
                    "confidence": 0.96,
                }
            ],
            "You are critiquing a proposed task decomposition": [
                {
                    "missing_tasks": [],
                    "hallucinated_tasks": [],
                    "dependency_issues": [],
                    "overly_broad_tasks": [],
                    "unsafe_tasks": [],
                    "unresolved_references": [],
                    "confidence": 0.0,
                }
            ],
            "You are assigning semantic verbs": [
                {
                    "assignments": [
                        {
                            "task_id": "task-list",
                            "semantic_verb": "read",
                            "object_type": "directory",
                            "intent_confidence": 0.95,
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
                }
            ],
            "You are assessing whether a selected capability truly fits a task": [
                {
                    "task_id": "task-list",
                    "candidate_capability_id": "filesystem.list_directory",
                    "candidate_operation_id": "list_directory",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task is to list files.",
                    "domain_reason": "Filesystem is the correct domain.",
                    "object_type_reason": "The object is a directory.",
                    "argument_reason": "The path can be extracted separately.",
                    "risk_reason": "Low risk and read-only.",
                    "missing_capability_description": None,
                    "suggested_domain": "filesystem",
                    "suggested_object_type": "directory",
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
                    "task_id": "task-list",
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "arguments": {"path": "."},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
                {
                    "task_id": "task-list",
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "arguments": {"path": "."},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
                {
                    "task_id": "task-list",
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "arguments": {"path": "."},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                }
            ],
            "You are reviewing a sanitized action DAG": [
                {
                    "missing_user_intents": [],
                    "suspicious_nodes": [],
                    "dependency_warnings": [],
                    "output_expectation_warnings": [],
                    "recommended_repair": None,
                    "confidence": 0.0,
                }
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


def _read_runtime(tmp_path: Path) -> AgentRuntime:
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
    llm = RoutingLLMClient(
        {
            "You are classifying a user prompt": [
                {
                    "prompt_type": "simple_tool_task",
                    "requires_tools": True,
                    "likely_domains": ["filesystem"],
                    "risk_level": "low",
                    "needs_clarification": False,
                    "clarification_question": None,
                    "reason": "File read request.",
                    "confidence": 0.95,
                    "assumptions": [],
                }
            ],
            "You are decomposing a user prompt": [
                {
                    "tasks": [
                        {
                            "id": "task-read",
                            "description": "Read README.md.",
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
                    "confidence": 0.96,
                },
                {
                    "tasks": [
                        {
                            "id": "task-read",
                            "description": "Read README.md.",
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
                    "confidence": 0.96,
                },
                {
                    "tasks": [
                        {
                            "id": "task-read",
                            "description": "Read README.md.",
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
                    "confidence": 0.96,
                }
            ],
            "You are critiquing a proposed task decomposition": [
                {
                    "missing_tasks": [],
                    "hallucinated_tasks": [],
                    "dependency_issues": [],
                    "overly_broad_tasks": [],
                    "unsafe_tasks": [],
                    "unresolved_references": [],
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
                            "intent_confidence": 0.95,
                            "risk_level": "low",
                            "requires_confirmation": False,
                        }
                    ]
                }
            ],
            "You are selecting capability candidates": [
                {
                    "task_id": "task-read",
                    "candidates": [
                        {
                            "capability_id": "filesystem.read_file",
                            "operation_id": "read_file",
                            "confidence": 0.95,
                            "reason": "File read request.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.read_file",
                        "operation_id": "read_file",
                        "confidence": 0.95,
                        "reason": "File read request.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
                {
                    "task_id": "task-read",
                    "candidates": [
                        {
                            "capability_id": "filesystem.read_file",
                            "operation_id": "read_file",
                            "confidence": 0.95,
                            "reason": "File read request.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.read_file",
                        "operation_id": "read_file",
                        "confidence": 0.95,
                        "reason": "File read request.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                },
                {
                    "task_id": "task-read",
                    "candidates": [
                        {
                            "capability_id": "filesystem.read_file",
                            "operation_id": "read_file",
                            "confidence": 0.95,
                            "reason": "File read request.",
                            "assumptions": [],
                        }
                    ],
                    "selected": {
                        "capability_id": "filesystem.read_file",
                        "operation_id": "read_file",
                        "confidence": 0.95,
                        "reason": "File read request.",
                        "assumptions": [],
                    },
                    "unresolved_reason": None,
                }
            ],
            "You are assessing whether a selected capability truly fits a task": [
                {
                    "task_id": "task-read",
                    "candidate_capability_id": "filesystem.read_file",
                    "candidate_operation_id": "read_file",
                    "proposed_status": "fit",
                    "confidence": 0.96,
                    "semantic_reason": "The task is to read a file.",
                    "domain_reason": "Filesystem is the correct domain.",
                    "object_type_reason": "The object is a file.",
                    "argument_reason": "The path is explicit.",
                    "risk_reason": "Low risk and read-only.",
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
                    "arguments": {"path": "README.md"},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
                {
                    "task_id": "task-read",
                    "capability_id": "filesystem.read_file",
                    "operation_id": "read_file",
                    "arguments": {"path": "README.md"},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                },
                {
                    "task_id": "task-read",
                    "capability_id": "filesystem.read_file",
                    "operation_id": "read_file",
                    "arguments": {"path": "README.md"},
                    "missing_required_arguments": [],
                    "assumptions": [],
                    "confidence": 0.95,
                }
            ],
            "You are reviewing a sanitized action DAG": [
                {
                    "missing_user_intents": [],
                    "suspicious_nodes": [],
                    "dependency_warnings": [],
                    "output_expectation_warnings": [],
                    "recommended_repair": None,
                    "confidence": 0.0,
                }
            ],
            "You are selecting a safe display plan": [
                {
                    "display_type": "code_block",
                    "title": "File Preview",
                    "sections": [
                        {
                            "title": "File Preview",
                            "display_type": "code_block",
                            "source_node_id": "node::task-read",
                            "parameters": {"language": "text"},
                        }
                    ],
                    "constraints": {},
                    "redaction_policy": "standard",
                }
            ],
        }
    )
    return AgentRuntime(llm, registry, engine, OutputPipelineOrchestrator())


def test_trace_captures_capability_candidates_and_fit_decisions(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    runtime = _list_runtime(tmp_path)

    runtime.handle_request("list files in this folder", {"workspace_root": str(tmp_path)})
    trace = runtime.last_planning_trace

    assert trace is not None
    assert "task-list" in trace.capability_candidates_by_task
    assert "task-list" in trace.capability_fit_decisions_by_task
    assert trace.selected_capability_by_task["task-list"]["capability_id"] == "filesystem.list_directory"


def test_trace_captures_dataflow_proposals_and_rejected_refs() -> None:
    registry = build_default_registry()
    task = TaskFrame(
        id="task-list",
        description="List files in this directory.",
        semantic_verb="read",
        object_type="directory",
        intent_confidence=0.95,
        constraints={},
        dependencies=[],
        raw_evidence="list files",
        risk_level="low",
    )
    selection = _selection("task-list", "filesystem.list_directory", "list_directory")
    trace = PlanningTrace(request_id="req-dataflow", raw_prompt="list files and total size")
    llm = RoutingLLMClient(
        {
            "You are proposing typed dataflow": [
                {
                    "refs": [
                        {
                            "consumer_task_id": "task-total",
                            "consumer_argument_name": "input_ref",
                            "producer_task_id": "missing-task",
                            "producer_output_key": None,
                            "expected_data_type": "table",
                            "reason": "Invalid producer for test.",
                            "confidence": 0.95,
                        }
                    ],
                    "derived_tasks": [
                        {
                            "task_id": "task-total",
                            "description": "Compute total file size.",
                            "semantic_verb": "analyze",
                            "object_type": "data.records",
                            "capability_id": "data.aggregate",
                            "operation_id": "aggregate",
                            "arguments": {
                                "operation": "sum",
                                "field": "size",
                                "label": "Total File Size",
                                "unit": "bytes",
                            },
                            "depends_on": ["task-list"],
                            "reason": "Aggregate over the listing.",
                            "confidence": 0.95,
                        }
                    ],
                    "dependency_edges": [("task-list", "task-total")],
                    "assumptions": [],
                    "unresolved_dataflows": [],
                    "confidence": 0.95,
                }
            ]
        }
    )

    validated = plan_dataflow(
        original_prompt="list files and total size",
        tasks=[task],
        capability_selections=[selection],
        registry=registry,
        llm_client=llm,
        trace=trace,
    )

    assert trace.dataflow_plan_raw is not None
    assert trace.dataflow_plan_validated is not None
    assert trace.dataflow_rejected_refs
    assert validated.rejected_refs


def test_trace_captures_skipped_argument_extraction() -> None:
    registry = build_default_registry()
    task = TaskFrame(
        id="task-memory",
        description="how much free memory do i have on this system?",
        semantic_verb="read",
        object_type="system.memory",
        intent_confidence=0.95,
        constraints={},
        dependencies=[],
        raw_evidence="memory",
        risk_level="low",
    )
    selection = _selection("task-memory", "filesystem.list_directory", "list_directory")
    fit_trace = PlanningTrace(request_id="req-fit", raw_prompt=task.description)
    fit_llm = RoutingLLMClient(
        {
            "You are assessing whether a selected capability truly fits a task": [
                {
                    "task_id": "task-memory",
                    "candidate_capability_id": "filesystem.list_directory",
                    "candidate_operation_id": "list_directory",
                    "proposed_status": "domain_mismatch",
                    "confidence": 0.95,
                    "semantic_reason": "Wrong semantic target.",
                    "domain_reason": "Memory is not filesystem listing.",
                    "object_type_reason": "Memory is not a directory.",
                    "argument_reason": "Path would not help.",
                    "risk_reason": "Low risk but wrong.",
                    "missing_capability_description": "A system memory capability is missing.",
                    "suggested_domain": "shell",
                    "suggested_object_type": "system.memory",
                    "requires_clarification": False,
                    "clarification_question": None,
                }
            ]
        }
    )
    fit_decisions, _ = assess_capability_fit(
        [task],
        [selection],
        registry,
        {"original_prompt": task.description, "likely_domains": ["shell"]},
        fit_llm,
        trace=fit_trace,
    )

    extract_arguments(
        [task],
        [selection],
        registry,
        RoutingLLMClient({}),
        trace=fit_trace,
        capability_fit_decisions=fit_decisions,
    )

    assert "task-memory" in fit_trace.skipped_argument_extraction_by_task


def test_final_dag_hash_is_stable() -> None:
    from agent_runtime.core.types import ActionDAG, ActionNode

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

    assert hash_action_dag(dag) == hash_action_dag(dag.model_copy())


def test_replay_does_not_call_llm_and_round_trip_works(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    runtime = _list_runtime(tmp_path)

    runtime.handle_request("list files in this folder", {"workspace_root": str(tmp_path)})
    trace = runtime.last_planning_trace
    assert trace is not None

    path = save_planning_trace(trace, tmp_path / "planning-trace.json")
    loaded = load_planning_trace(path)
    replay_engine = ExecutionEngine(
        build_default_registry(),
        {
            "workspace_root": str(tmp_path),
            "allow_shell_execution": False,
            "allow_network_operations": False,
            "gateway_url": "http://gateway",
        },
        InMemoryResultStore(),
        gateway_client=FakeGatewayClient(tmp_path),
    )

    bundle = replay_from_validated_dag(loaded, replay_engine, {"confirmation": True})

    assert bundle.status == "success"
    assert any(result.node_id == "node::task-list" for result in bundle.results)


def test_replay_fails_if_dag_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    runtime = _list_runtime(tmp_path)

    runtime.handle_request("list files in this folder", {"workspace_root": str(tmp_path)})
    trace = runtime.last_planning_trace
    assert trace is not None
    trace = trace.model_copy(update={"final_dag_hash": "bad-hash"})

    replay_engine = ExecutionEngine(
        build_default_registry(),
        {
            "workspace_root": str(tmp_path),
            "allow_shell_execution": False,
            "allow_network_operations": False,
            "gateway_url": "http://gateway",
        },
        InMemoryResultStore(),
        gateway_client=FakeGatewayClient(tmp_path),
    )

    with pytest.raises(Exception, match="final_dag_hash"):
        replay_from_validated_dag(trace, replay_engine, {"confirmation": True})


def test_trace_does_not_include_raw_tool_output(tmp_path: Path) -> None:
    secret_text = "super secret file contents"
    (tmp_path / "README.md").write_text(secret_text, encoding="utf-8")
    runtime = _read_runtime(tmp_path)

    runtime.handle_request("read README.md", {"workspace_root": str(tmp_path)})
    trace = runtime.last_planning_trace
    assert trace is not None

    serialized = trace.model_dump_json()
    assert secret_text not in serialized
