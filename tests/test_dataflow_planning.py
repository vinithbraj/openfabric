from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.orchestrator import AgentRuntime
from agent_runtime.core.types import CapabilityRef, InputRef, TaskFrame, UserRequest
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.result_store import InMemoryResultStore
from agent_runtime.execution.safety import evaluate_dag_safety
from agent_runtime.input_pipeline.argument_extraction import extract_arguments
from agent_runtime.input_pipeline.dag_builder import build_action_dag
from agent_runtime.input_pipeline.dataflow_planning import (
    DataflowPlanProposal,
    DataflowRefProposal,
    DerivedTaskProposal,
    ValidatedDataflowPlan,
    validate_dataflow_plan,
)
from agent_runtime.input_pipeline.decomposition import DecompositionResult
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator
from gateway_agent.remote_runner import run_remote_operation


class DataflowLLMClient:
    """Prompt-router fake LLM for end-to-end dataflow tests."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.prompts: list[str] = []
        self.model = "fake-model"
        self.temperature = 0.0

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        _ = schema
        if "You are classifying a user prompt" in prompt:
            return {
                "prompt_type": "compound_tool_task",
                "requires_tools": True,
                "likely_domains": ["filesystem", "data"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "Filesystem retrieval followed by deterministic aggregation.",
                "confidence": 0.96,
                "assumptions": [],
            }
        if "You are decomposing a user prompt" in prompt:
            return {
                "tasks": [
                    {
                        "id": "task_list",
                        "description": "List all files in this directory.",
                        "semantic_verb": "read",
                        "object_type": "directory",
                        "intent_confidence": 0.95,
                        "constraints": {},
                        "dependencies": [],
                        "raw_evidence": self.prompt,
                        "requires_confirmation": False,
                        "risk_level": "low",
                    }
                ],
                "global_constraints": {},
                "unresolved_references": [],
                "assumptions": [],
                "confidence": 0.95,
            }
        if "You are critiquing a proposed task decomposition" in prompt:
            return {
                "missing_tasks": [],
                "hallucinated_tasks": [],
                "dependency_issues": [],
                "overly_broad_tasks": [],
                "unsafe_tasks": [],
                "unresolved_references": [],
                "confidence": 0.0,
            }
        if "You are assigning semantic verbs" in prompt:
            return {
                "assignments": [
                    {
                        "task_id": "task_list",
                        "semantic_verb": "read",
                        "object_type": "directory",
                        "intent_confidence": 0.95,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            }
        if "You are selecting capability candidates" in prompt:
            return {
                "task_id": "task_list",
                "candidates": [
                    {
                        "capability_id": "filesystem.list_directory",
                        "operation_id": "list_directory",
                        "confidence": 0.97,
                        "reason": "Listing files requires a directory listing capability.",
                        "assumptions": [],
                    }
                ],
                "selected": {
                    "capability_id": "filesystem.list_directory",
                    "operation_id": "list_directory",
                    "confidence": 0.97,
                    "reason": "Listing files requires a directory listing capability.",
                    "assumptions": [],
                },
                "unresolved_reason": None,
            }
        if "You are assessing whether a selected capability truly fits a task" in prompt:
            return {
                "task_id": "task_list",
                "candidate_capability_id": "filesystem.list_directory",
                "candidate_operation_id": "list_directory",
                "proposed_status": "fit",
                "confidence": 0.97,
                "semantic_reason": "The base task is to list files from a directory.",
                "domain_reason": "Filesystem is the correct domain.",
                "object_type_reason": "The object is a directory listing.",
                "argument_reason": "The path can be extracted separately.",
                "risk_reason": "Read-only and low risk.",
                "missing_capability_description": None,
                "suggested_domain": "filesystem",
                "suggested_object_type": "directory",
                "requires_clarification": False,
                "clarification_question": None,
            }
        if "You are proposing typed dataflow" in prompt:
            if "count files in this directory" in self.prompt.lower():
                return {
                    "refs": [
                        {
                            "consumer_task_id": "task_count",
                            "consumer_argument_name": "input_ref",
                            "producer_task_id": "task_list",
                            "producer_output_key": None,
                            "expected_data_type": "table",
                            "reason": "Count the listed directory entries.",
                            "confidence": 0.95,
                        }
                    ],
                    "derived_tasks": [
                        {
                            "task_id": "task_count",
                            "description": "Count filesystem entries from the directory listing.",
                            "semantic_verb": "analyze",
                            "object_type": "data.records",
                            "capability_id": "data.aggregate",
                            "operation_id": "aggregate",
                            "arguments": {
                                "operation": "count",
                                "filter": {"type": "file"},
                                "label": "File count",
                            },
                            "depends_on": ["task_list"],
                            "reason": "Use a deterministic aggregate to count file rows.",
                            "confidence": 0.96,
                        }
                    ],
                    "dependency_edges": [("task_list", "task_count")],
                    "assumptions": [],
                    "unresolved_dataflows": [],
                    "confidence": 0.95,
                }
            return {
                "refs": [
                    {
                        "consumer_task_id": "task_total_size",
                        "consumer_argument_name": "input_ref",
                        "producer_task_id": "task_list",
                        "producer_output_key": None,
                        "expected_data_type": "table",
                        "reason": "Aggregate over the directory listing output.",
                        "confidence": 0.96,
                    }
                ],
                "derived_tasks": [
                    {
                        "task_id": "task_total_size",
                        "description": "Compute total regular file size from the directory listing.",
                        "semantic_verb": "analyze",
                        "object_type": "data.records",
                        "capability_id": "data.aggregate",
                        "operation_id": "aggregate",
                        "arguments": {
                            "operation": "sum",
                            "field": "size",
                            "filter": {"type": "file"},
                            "label": "Total regular file size",
                            "unit": "bytes",
                        },
                        "depends_on": ["task_list"],
                        "reason": "Use deterministic aggregation over structured directory entries.",
                        "confidence": 0.97,
                    }
                ],
                "dependency_edges": [("task_list", "task_total_size")],
                "assumptions": [],
                "unresolved_dataflows": [],
                "confidence": 0.96,
            }
        if "You are extracting typed arguments" in prompt:
            return {
                "task_id": "task_list",
                "capability_id": "filesystem.list_directory",
                "operation_id": "list_directory",
                "arguments": {"path": "."},
                "missing_required_arguments": [],
                "assumptions": [],
                "confidence": 0.96,
            }
        if "You are reviewing a sanitized action DAG" in prompt:
            return {
                "missing_user_intents": [],
                "suspicious_nodes": [],
                "dependency_warnings": [],
                "output_expectation_warnings": [],
                "recommended_repair": None,
                "confidence": 0.0,
            }
        if "You are selecting a safe display plan" in prompt:
            if "count files in this directory" in self.prompt.lower():
                return {
                    "display_type": "plain_text",
                    "title": "File Count",
                    "sections": [
                        {
                            "title": "File Count",
                            "display_type": "plain_text",
                            "source_node_id": "node::task_count",
                        }
                    ],
                    "constraints": {},
                    "redaction_policy": "standard",
                }
            return {
                "display_type": "plain_text",
                "title": "Directory Size",
                "sections": [
                    {
                        "title": "Directory Size",
                        "display_type": "plain_text",
                        "source_node_id": "node::task_total_size",
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
        from agent_runtime.core.types import ExecutionResult

        return ExecutionResult(
            node_id=node.id,
            status="success",
            data_preview=result["data_preview"],
            metadata=result["metadata"],
        )


def _runtime(tmp_path: Path, prompt: str) -> AgentRuntime:
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
    return AgentRuntime(DataflowLLMClient(prompt), registry, engine, OutputPipelineOrchestrator())


def _base_task() -> TaskFrame:
    return TaskFrame(
        id="task_list",
        description="List all files in this directory.",
        semantic_verb="read",
        object_type="directory",
        intent_confidence=0.95,
        constraints={},
        dependencies=[],
        risk_level="low",
    )


def _base_selection() -> CapabilitySelectionResult:
    selected = CapabilityRef(
        capability_id="filesystem.list_directory",
        operation_id="list_directory",
        confidence=0.97,
        reason="Directory listing request.",
    )
    return CapabilitySelectionResult(
        task_id="task_list",
        candidates=[selected],
        selected=selected,
        unresolved_reason=None,
    )


def _plan_for_sum() -> DataflowPlanProposal:
    return DataflowPlanProposal(
        refs=[
            DataflowRefProposal(
                consumer_task_id="task_total_size",
                consumer_argument_name="input_ref",
                producer_task_id="task_list",
                producer_output_key=None,
                expected_data_type="table",
                reason="Aggregate over directory entries.",
                confidence=0.96,
            )
        ],
        derived_tasks=[
            DerivedTaskProposal(
                task_id="task_total_size",
                description="Compute total regular file size from directory listing.",
                semantic_verb="analyze",
                object_type="data.records",
                capability_id="data.aggregate",
                operation_id="aggregate",
                arguments={
                    "operation": "sum",
                    "field": "size",
                    "filter": {"type": "file"},
                    "label": "Total regular file size",
                    "unit": "bytes",
                },
                depends_on=["task_list"],
                reason="Aggregate file sizes.",
                confidence=0.97,
            )
        ],
        dependency_edges=[("task_list", "task_total_size")],
        assumptions=[],
        unresolved_dataflows=[],
        confidence=0.96,
    )


def test_list_directory_and_total_size_creates_list_plus_aggregate() -> None:
    registry = build_default_registry()
    validated = validate_dataflow_plan(
        proposal=_plan_for_sum(),
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )

    assert len(validated.derived_tasks) == 1
    derived = validated.derived_tasks[0]
    assert derived.capability_id == "data.aggregate"
    assert derived.operation_id == "aggregate"
    assert derived.arguments["operation"] == "sum"
    assert derived.arguments["field"] == "size"
    assert derived.arguments["filter"] == {"type": "file"}
    assert derived.arguments["unit"] == "bytes"


def test_data_aggregate_consumes_input_ref_from_list_directory() -> None:
    registry = build_default_registry()
    validated = validate_dataflow_plan(
        proposal=_plan_for_sum(),
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )

    assert len(validated.refs) == 1
    ref = validated.refs[0]
    assert ref.input_ref.source_node_id == "node::task_list"


def test_final_dag_contains_input_ref() -> None:
    registry = build_default_registry()
    validated = validate_dataflow_plan(
        proposal=_plan_for_sum(),
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )
    extraction_results = extract_arguments(
        [_base_task()],
        [_base_selection()],
        registry,
        DataflowLLMClient("list all files in this directory and compute the total file size"),
        dataflow_plan=validated,
    )
    dag = build_action_dag(
        user_request=UserRequest(
            raw_prompt="list all files in this directory and compute the total file size",
            safety_context={"capability_registry": registry},
        ),
        decomposition_result=DecompositionResult(tasks=[_base_task()]),
        capability_selection_results=[_base_selection()],
        argument_extraction_results=extraction_results,
        dataflow_plan=validated,
    )

    aggregate_node = next(node for node in dag.nodes if node.capability_id == "data.aggregate")
    assert isinstance(aggregate_node.arguments["input_ref"], InputRef)


def test_execution_resolves_input_ref_and_output_contains_numeric_total(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("aa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    registry = build_default_registry()
    llm = DataflowLLMClient("list all files in this directory and compute the total file size")
    validated = validate_dataflow_plan(
        proposal=_plan_for_sum(),
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )
    extraction_results = extract_arguments(
        [_base_task()],
        [_base_selection()],
        registry,
        llm,
        dataflow_plan=validated,
    )
    engine = ExecutionEngine(
        registry,
        {
            "workspace_root": str(tmp_path),
            "allow_shell_execution": False,
            "allow_network_operations": False,
            "gateway_url": "http://gateway",
        },
        InMemoryResultStore(),
        gateway_client=FakeGatewayClient(tmp_path),
    )
    runtime = AgentRuntime(llm, registry, engine, OutputPipelineOrchestrator())
    user_request = runtime._build_user_request(
        "list all files in this directory and compute the total file size",
        {"workspace_root": str(tmp_path)},
    )
    dag = build_action_dag(
        user_request=user_request,
        decomposition_result=DecompositionResult(tasks=[_base_task()]),
        capability_selection_results=[_base_selection()],
        argument_extraction_results=extraction_results,
        dataflow_plan=validated,
    )
    decision = evaluate_dag_safety(dag, registry, engine.safety_policy.config)
    ready_dag = runtime._mark_dag_execution_ready(dag, decision)
    bundle = engine.execute(ready_dag, {"workspace_root": str(tmp_path)})
    aggregate_result = next(result for result in bundle.results if result.node_id == "node::task_total_size")

    assert aggregate_result.data_preview["value"] == 5

    runtime = _runtime(tmp_path, "list all files in this directory and compute the total file size")
    response = runtime.handle_request(
        "list all files in this directory and compute the total file size",
        {"workspace_root": str(tmp_path)},
    )

    assert "5" in response
    assert "a.txt" not in response
    assert "b.txt" not in response


def test_count_files_in_directory_creates_count_aggregate(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("aa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    runtime = _runtime(tmp_path, "count files in this directory")
    response = runtime.handle_request("count files in this directory", {"workspace_root": str(tmp_path)})

    assert "2" in response


def test_invalid_producer_is_rejected() -> None:
    registry = build_default_registry()
    proposal = _plan_for_sum()
    proposal.refs[0].producer_task_id = "missing_task"

    validated = validate_dataflow_plan(
        proposal=proposal,
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )

    assert validated.rejected_refs


def test_invalid_consumer_is_rejected() -> None:
    registry = build_default_registry()
    proposal = _plan_for_sum()
    proposal.refs[0].consumer_task_id = "missing_task"

    validated = validate_dataflow_plan(
        proposal=proposal,
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )

    assert validated.rejected_refs


def test_self_reference_is_rejected() -> None:
    registry = build_default_registry()
    proposal = _plan_for_sum()
    proposal.refs[0].consumer_task_id = "task_list"
    proposal.refs[0].producer_task_id = "task_list"

    validated = validate_dataflow_plan(
        proposal=proposal,
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )

    assert validated.rejected_refs


def test_cycle_is_rejected() -> None:
    registry = build_default_registry()
    proposal = _plan_for_sum()
    proposal.dependency_edges.append(("task_total_size", "task_list"))

    with pytest.raises(Exception):
        validate_dataflow_plan(
            proposal=proposal,
            tasks=[_base_task()],
            capability_selections=[_base_selection()],
            registry=registry,
        )


def test_invalid_consumer_argument_is_rejected() -> None:
    registry = build_default_registry()
    proposal = _plan_for_sum()
    proposal.refs[0].consumer_argument_name = "bogus_argument"

    validated = validate_dataflow_plan(
        proposal=proposal,
        tasks=[_base_task()],
        capability_selections=[_base_selection()],
        registry=registry,
    )

    assert validated.rejected_refs
