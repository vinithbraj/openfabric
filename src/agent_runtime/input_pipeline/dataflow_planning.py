"""LLM-assisted dataflow planning with deterministic validation."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import model_json_schema

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import InputRef, TaskFrame
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)
from agent_runtime.llm.structured_call import structured_call

_REF_PATTERN = re.compile(r"^(?P<source>[A-Za-z0-9_:\-]+)\.(?P<output>[A-Za-z0-9_:\-]+)$")
_ROW_LIKE_KEYS = {"entries", "rows", "matches", "processes", "listeners"}


class DataflowRefProposal(BaseModel):
    """Untrusted proposal for wiring producer output into a consumer argument."""

    model_config = ConfigDict(extra="forbid")

    consumer_task_id: str
    consumer_argument_name: str
    producer_task_id: str
    producer_output_key: str | None = None
    expected_data_type: str | None = None
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class DerivedTaskProposal(BaseModel):
    """Untrusted proposal for a planner-inserted derived task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    description: str
    semantic_verb: str
    object_type: str
    capability_id: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class DataflowPlanProposal(BaseModel):
    """Untrusted LLM proposal describing producer-consumer dataflow."""

    model_config = ConfigDict(extra="forbid")

    refs: list[DataflowRefProposal] = Field(default_factory=list)
    derived_tasks: list[DerivedTaskProposal] = Field(default_factory=list)
    dependency_edges: list[tuple[str, str]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_dataflows: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ValidatedDataflowRef(BaseModel):
    """Trusted validated dataflow reference ready for DAG construction."""

    model_config = ConfigDict(extra="forbid")

    consumer_task_id: str
    consumer_argument_name: str
    producer_task_id: str
    producer_node_id: str
    producer_output_key: str | None = None
    expected_data_type: str | None = None
    input_ref: InputRef
    reasons: list[str] = Field(default_factory=list)


class ValidatedDerivedTask(BaseModel):
    """Trusted validated derived task with safe arguments."""

    model_config = ConfigDict(extra="forbid")

    task: TaskFrame
    capability_id: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ValidatedDataflowPlan(BaseModel):
    """Trusted validated dataflow plan."""

    model_config = ConfigDict(extra="forbid")

    refs: list[ValidatedDataflowRef] = Field(default_factory=list)
    derived_tasks: list[ValidatedDerivedTask] = Field(default_factory=list)
    dependency_edges: list[tuple[str, str]] = Field(default_factory=list)
    rejected_refs: list[dict[str, Any]] = Field(default_factory=list)
    rejected_derived_tasks: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_dataflows: list[str] = Field(default_factory=list)


def _empty_validated_plan() -> ValidatedDataflowPlan:
    """Return an empty validated plan."""

    return ValidatedDataflowPlan()


def _task_output_hints(task_id: str, manifest: CapabilityManifest) -> dict[str, Any]:
    """Build compact output-hint metadata for a selected producer task."""

    return {
        "task_id": task_id,
        "capability_id": manifest.capability_id,
        "operation_id": manifest.operation_id,
        "output_schema": manifest.output_schema,
        "data_hints": sorted(_manifest_data_types(manifest)),
    }


def _consumer_hints(task_id: str, manifest: CapabilityManifest) -> dict[str, Any]:
    """Build compact input-hint metadata for a potential consumer task."""

    return {
        "task_id": task_id,
        "capability_id": manifest.capability_id,
        "operation_id": manifest.operation_id,
        "required_arguments": list(manifest.required_arguments),
        "optional_arguments": list(manifest.optional_arguments),
        "argument_schema": manifest.argument_schema,
    }


def _runtime_data_capability_hints(registry: CapabilityRegistry) -> list[dict[str, Any]]:
    """Return compact hints for internal data capabilities."""

    hints: list[dict[str, Any]] = []
    for capability_id in ("data.aggregate", "data.project", "data.head"):
        try:
            manifest = registry.get(capability_id).manifest
        except Exception:
            continue
        hints.append(
            {
                "capability_id": manifest.capability_id,
                "operation_id": manifest.operation_id,
                "description": manifest.description,
                "required_arguments": list(manifest.required_arguments),
                "optional_arguments": list(manifest.optional_arguments),
                "output_schema": manifest.output_schema,
            }
        )
    return hints


def _build_dataflow_prompt(
    *,
    original_prompt: str,
    tasks: list[TaskFrame],
    capability_selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
) -> str:
    """Build the strict JSON-only prompt for LLM dataflow proposal."""

    selection_map = {result.task_id: result for result in capability_selections}
    producers: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    for task in tasks:
        selection = selection_map.get(task.id)
        if selection is None or selection.selected is None:
            continue
        manifest = registry.get(selection.selected.capability_id).manifest
        producers.append(_task_output_hints(task.id, manifest))
        consumers.append(_consumer_hints(task.id, manifest))

    schema = model_json_schema(DataflowPlanProposal)
    return "\n".join(
        [
            "You are proposing typed dataflow for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "You may propose producer-consumer references, derived data tasks, and dependency edges.",
            "Use only the provided tasks, selected capabilities, compact manifests, producer output schemas, and internal data capabilities.",
            "Do not produce shell commands, Python code, SQL, arbitrary expressions, or executable syntax.",
            "Original prompt:",
            original_prompt,
            "Existing tasks:",
            str([task.model_dump(mode="json") for task in tasks]),
            "Selected producer capability hints:",
            str(producers),
            "Selected consumer capability hints:",
            str(consumers),
            "Available internal data capabilities:",
            str(_runtime_data_capability_hints(registry)),
            "JSON schema:",
            str(schema),
        ]
    )


def propose_dataflow_with_llm(
    *,
    original_prompt: str,
    tasks: list[TaskFrame],
    capability_selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
    llm_client,
) -> DataflowPlanProposal:
    """Ask the LLM for a typed dataflow proposal."""

    prompt = _build_dataflow_prompt(
        original_prompt=original_prompt,
        tasks=tasks,
        capability_selections=capability_selections,
        registry=registry,
    )
    try:
        return structured_call(llm_client, prompt, DataflowPlanProposal)
    except Exception:
        return DataflowPlanProposal(
            refs=[],
            derived_tasks=[],
            dependency_edges=[],
            assumptions=[],
            unresolved_dataflows=[],
            confidence=0.0,
        )


def _manifest_data_types(manifest: CapabilityManifest) -> set[str]:
    """Infer coarse data types exposed by a manifest output schema."""

    keys = set(manifest.output_schema)
    if keys & _ROW_LIKE_KEYS:
        return {"table", "data.records", "list", "structured_output"}
    if "content_preview" in keys or "markdown" in keys or "status_lines" in keys:
        return {"text", "structured_output"}
    if "value" in keys:
        return {"scalar", "summary", "structured_output"}
    if "summary" in keys:
        return {"summary", "structured_output"}
    return {"structured_output"}


def _normalize_input_ref_value(raw_value: Any, known_task_ids: set[str]) -> InputRef | Any:
    """Normalize ref-like string values only after deterministic task validation."""

    if isinstance(raw_value, InputRef):
        return raw_value
    if not isinstance(raw_value, str):
        return raw_value
    match = _REF_PATTERN.match(raw_value.strip())
    if match is None:
        return raw_value
    producer_task_id = match.group("source")
    output = match.group("output")
    if producer_task_id not in known_task_ids:
        return raw_value
    return InputRef(
        source_node_id=f"node::{producer_task_id}",
        output_key=None if output == "output" else output,
    )


def _normalize_argument_refs(arguments: dict[str, Any], known_task_ids: set[str]) -> dict[str, Any]:
    """Normalize explicit ref-shaped argument values in a derived task."""

    normalized: dict[str, Any] = {}
    for key, value in dict(arguments or {}).items():
        if isinstance(value, list):
            normalized[key] = [_normalize_input_ref_value(item, known_task_ids) for item in value]
        elif isinstance(value, dict):
            normalized[key] = {
                item_key: _normalize_input_ref_value(item_value, known_task_ids)
                for item_key, item_value in value.items()
            }
        elif key == "input_ref" or key.endswith("_ref"):
            normalized[key] = _normalize_input_ref_value(value, known_task_ids)
        else:
            normalized[key] = value
    return normalized


def _validate_expected_data_type(expected_data_type: str | None, producer_manifest: CapabilityManifest) -> bool:
    """Return whether an expected data type is compatible with producer output."""

    if not expected_data_type:
        return True
    producer_types = _manifest_data_types(producer_manifest)
    return expected_data_type in producer_types


def _known_output_key(manifest: CapabilityManifest, output_key: str | None) -> bool:
    """Return whether an explicit output key is compatible with the manifest schema."""

    if output_key is None:
        return True
    if not manifest.output_schema:
        return True
    return output_key in manifest.output_schema


def _task_graph_is_acyclic(adjacency: dict[str, set[str]]) -> bool:
    """Return whether a task-level adjacency map is acyclic."""

    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in permanent:
            return True
        if node_id in temporary:
            return False
        temporary.add(node_id)
        for child_id in adjacency.get(node_id, set()):
            if not visit(child_id):
                return False
        temporary.remove(node_id)
        permanent.add(node_id)
        return True

    return all(visit(node_id) for node_id in adjacency)


def _consumer_manifest_for_task(
    task_id: str,
    selection_map: dict[str, CapabilitySelectionResult],
    derived_map: dict[str, ValidatedDerivedTask],
    registry: CapabilityRegistry,
) -> CapabilityManifest | None:
    """Resolve the consumer manifest for an existing or derived task."""

    if task_id in derived_map:
        return registry.get(derived_map[task_id].capability_id).manifest
    selection = selection_map.get(task_id)
    if selection is None or selection.selected is None:
        return None
    return registry.get(selection.selected.capability_id).manifest


def _producer_manifest_for_task(
    task_id: str,
    selection_map: dict[str, CapabilitySelectionResult],
    derived_map: dict[str, ValidatedDerivedTask],
    registry: CapabilityRegistry,
) -> CapabilityManifest | None:
    """Resolve the producer manifest for an existing or derived task."""

    return _consumer_manifest_for_task(task_id, selection_map, derived_map, registry)


def _deterministic_low_confidence_support(
    consumer_manifest: CapabilityManifest,
    consumer_argument_name: str,
    producer_manifest: CapabilityManifest,
) -> bool:
    """Return whether deterministic schema evidence can support a low-confidence dataflow ref."""

    if consumer_argument_name != "input_ref":
        return False
    consumer_is_data_capability = consumer_manifest.domain == "data"
    producer_row_like = bool(_manifest_data_types(producer_manifest) & {"table", "data.records"})
    return consumer_is_data_capability and producer_row_like


def validate_dataflow_plan(
    *,
    proposal: DataflowPlanProposal,
    tasks: list[TaskFrame],
    capability_selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
) -> ValidatedDataflowPlan:
    """Validate one untrusted dataflow plan into trusted runtime structures."""

    base_task_map = {task.id: task for task in tasks}
    selection_map = {selection.task_id: selection for selection in capability_selections}
    known_task_ids = set(base_task_map)
    validated = _empty_validated_plan()
    derived_map: dict[str, ValidatedDerivedTask] = {}
    proposed_ref_arguments: dict[tuple[str, str], Any] = {
        (ref.consumer_task_id, ref.consumer_argument_name): f"{ref.producer_task_id}.output"
        for ref in proposal.refs
    }

    for derived in proposal.derived_tasks:
        rejection_reasons: list[str] = []
        if derived.task_id in known_task_ids or derived.task_id in derived_map:
            rejection_reasons.append(f"derived task id already exists: {derived.task_id}")
        if derived.confidence < 0.60:
            rejection_reasons.append("derived task confidence is below threshold")
        try:
            capability = registry.get(derived.capability_id)
            manifest = capability.manifest
            if manifest.operation_id != derived.operation_id:
                rejection_reasons.append("derived task operation does not match capability manifest")
            if manifest.mutates_state:
                rejection_reasons.append("derived task capability may not mutate state")
            candidate_arguments = dict(derived.arguments)
            for argument_name in set(manifest.required_arguments) | set(manifest.optional_arguments):
                key = (derived.task_id, argument_name)
                if argument_name not in candidate_arguments and key in proposed_ref_arguments:
                    candidate_arguments[argument_name] = proposed_ref_arguments[key]
            normalized_arguments = _normalize_argument_refs(
                candidate_arguments,
                known_task_ids | set(derived_map),
            )
            validated_arguments = capability.validate_arguments(normalized_arguments)
        except Exception as exc:
            rejection_reasons.append(str(exc))
            validated_arguments = {}
            manifest = None  # type: ignore[assignment]

        depends_on = [dependency for dependency in derived.depends_on if dependency]
        if any(dependency == derived.task_id for dependency in depends_on):
            rejection_reasons.append("derived task cannot depend on itself")

        if rejection_reasons:
            validated.rejected_derived_tasks.append(
                {
                    "task_id": derived.task_id,
                    "reasons": rejection_reasons,
                    "proposal": derived.model_dump(mode="json"),
                }
            )
            continue

        task = TaskFrame(
            id=derived.task_id,
            description=derived.description,
            semantic_verb=derived.semantic_verb,  # type: ignore[arg-type]
            object_type=derived.object_type,
            intent_confidence=max(derived.confidence, 0.60),
            constraints={},
            dependencies=list(depends_on),
            raw_evidence=derived.reason,
            requires_confirmation=False,
            risk_level="low",
        )
        validated_derived = ValidatedDerivedTask(
            task=task,
            capability_id=derived.capability_id,
            operation_id=derived.operation_id,
            arguments=validated_arguments,
            depends_on=list(depends_on),
            reasons=[derived.reason],
        )
        validated.derived_tasks.append(validated_derived)
        derived_map[task.id] = validated_derived
        known_task_ids.add(task.id)

    task_graph: dict[str, set[str]] = {task_id: set() for task_id in known_task_ids}
    for task in tasks:
        for dependency in task.dependencies:
            if dependency in task_graph:
                task_graph[dependency].add(task.id)
    invalid_derived_ids: set[str] = set()
    for derived in validated.derived_tasks:
        for dependency in derived.depends_on:
            if dependency not in task_graph:
                validated.rejected_derived_tasks.append(
                    {
                        "task_id": derived.task.id,
                        "reasons": [f"unknown dependency: {dependency}"],
                        "proposal": derived.model_dump(mode="json"),
                    }
                )
                invalid_derived_ids.add(derived.task.id)
            else:
                task_graph[dependency].add(derived.task.id)

    if invalid_derived_ids:
        validated.derived_tasks = [
            derived for derived in validated.derived_tasks if derived.task.id not in invalid_derived_ids
        ]
        for invalid_task_id in invalid_derived_ids:
            derived_map.pop(invalid_task_id, None)
            task_graph.pop(invalid_task_id, None)
            for children in task_graph.values():
                children.discard(invalid_task_id)

    for source, target in proposal.dependency_edges:
        if source not in task_graph or target not in task_graph:
            continue
        task_graph[source].add(target)
        validated.dependency_edges.append((source, target))

    for ref in proposal.refs:
        rejection_reasons: list[str] = []
        if ref.producer_task_id not in known_task_ids:
            rejection_reasons.append(f"unknown producer task: {ref.producer_task_id}")
        if ref.consumer_task_id not in known_task_ids:
            rejection_reasons.append(f"unknown consumer task: {ref.consumer_task_id}")
        if ref.producer_task_id == ref.consumer_task_id:
            rejection_reasons.append("producer and consumer cannot be the same task")

        producer_manifest = (
            _producer_manifest_for_task(ref.producer_task_id, selection_map, derived_map, registry)
            if ref.producer_task_id in known_task_ids
            else None
        )
        consumer_manifest = (
            _consumer_manifest_for_task(ref.consumer_task_id, selection_map, derived_map, registry)
            if ref.consumer_task_id in known_task_ids
            else None
        )
        if producer_manifest is None:
            rejection_reasons.append("producer capability does not exist")
        if consumer_manifest is None:
            rejection_reasons.append("consumer capability does not exist")
        if consumer_manifest is not None:
            allowed_arguments = set(consumer_manifest.required_arguments) | set(consumer_manifest.optional_arguments)
            if ref.consumer_argument_name not in allowed_arguments:
                rejection_reasons.append(
                    f"consumer argument is not declared on manifest: {ref.consumer_argument_name}"
                )
        if producer_manifest is not None and not _known_output_key(producer_manifest, ref.producer_output_key):
            rejection_reasons.append("producer output key is not declared by the manifest schema")
        if (
            producer_manifest is not None
            and ref.expected_data_type
            and not _validate_expected_data_type(ref.expected_data_type, producer_manifest)
        ):
            rejection_reasons.append("expected_data_type is incompatible with producer output")
        if ref.confidence < 0.60:
            if not (
                producer_manifest is not None
                and consumer_manifest is not None
                and _deterministic_low_confidence_support(
                    consumer_manifest,
                    ref.consumer_argument_name,
                    producer_manifest,
                )
            ):
                rejection_reasons.append("dataflow ref confidence is below threshold")

        if rejection_reasons:
            validated.rejected_refs.append(
                {
                    "consumer_task_id": ref.consumer_task_id,
                    "producer_task_id": ref.producer_task_id,
                    "reasons": rejection_reasons,
                    "proposal": ref.model_dump(mode="json"),
                }
            )
            continue

        task_graph.setdefault(ref.producer_task_id, set()).add(ref.consumer_task_id)
        input_ref = InputRef(
            source_node_id=f"node::{ref.producer_task_id}",
            output_key=ref.producer_output_key,
            expected_data_type=ref.expected_data_type,
        )
        validated.refs.append(
            ValidatedDataflowRef(
                consumer_task_id=ref.consumer_task_id,
                consumer_argument_name=ref.consumer_argument_name,
                producer_task_id=ref.producer_task_id,
                producer_node_id=f"node::{ref.producer_task_id}",
                producer_output_key=ref.producer_output_key,
                expected_data_type=ref.expected_data_type,
                input_ref=input_ref,
                reasons=[ref.reason],
            )
        )

    validated_ref_arguments: dict[str, set[str]] = {}
    for ref in validated.refs:
        validated_ref_arguments.setdefault(ref.consumer_task_id, set()).add(ref.consumer_argument_name)

    invalid_missing_ref_ids: set[str] = set()
    for derived in list(validated.derived_tasks):
        manifest = registry.get(derived.capability_id).manifest
        available_arguments = set(derived.arguments) | validated_ref_arguments.get(derived.task.id, set())
        missing_required = [
            argument_name
            for argument_name in manifest.required_arguments
            if argument_name not in available_arguments
        ]
        if missing_required:
            validated.rejected_derived_tasks.append(
                {
                    "task_id": derived.task.id,
                    "reasons": [
                        "derived task is missing required arguments after dataflow validation: "
                        + ", ".join(missing_required)
                    ],
                    "proposal": derived.model_dump(mode="json"),
                }
            )
            invalid_missing_ref_ids.add(derived.task.id)

    if invalid_missing_ref_ids:
        validated.derived_tasks = [
            derived for derived in validated.derived_tasks if derived.task.id not in invalid_missing_ref_ids
        ]
        validated.refs = [
            ref for ref in validated.refs if ref.consumer_task_id not in invalid_missing_ref_ids
        ]
        for invalid_task_id in invalid_missing_ref_ids:
            derived_map.pop(invalid_task_id, None)
            task_graph.pop(invalid_task_id, None)
            for children in task_graph.values():
                children.discard(invalid_task_id)

    if not _task_graph_is_acyclic(task_graph):
        raise ValidationError("Validated dataflow plan would create a cycle.")

    validated.unresolved_dataflows = list(proposal.unresolved_dataflows)
    return validated


def plan_dataflow(
    *,
    original_prompt: str,
    tasks: list[TaskFrame],
    capability_selections: list[CapabilitySelectionResult],
    registry: CapabilityRegistry,
    llm_client,
    trace: PlanningTrace | None = None,
) -> ValidatedDataflowPlan:
    """Propose and validate a dataflow plan, returning an empty plan on failure."""

    proposal = propose_dataflow_with_llm(
        original_prompt=original_prompt,
        tasks=tasks,
        capability_selections=capability_selections,
        registry=registry,
        llm_client=llm_client,
    )
    try:
        validated = validate_dataflow_plan(
            proposal=proposal,
            tasks=tasks,
            capability_selections=capability_selections,
            registry=registry,
        )
        if isinstance(trace, PlanningTrace):
            model_name, temperature = llm_client_metadata(llm_client)
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="dataflow_planning",
                    request_id=str(trace.request_id),
                    model_name=model_name,
                    llm_temperature=temperature,
                    prompt_template_id="dataflow_planning",
                    raw_llm_response=proposal.model_dump(mode="json"),
                    parsed_proposal=proposal.model_dump(mode="json"),
                    selected_candidate=validated.model_dump(mode="json"),
                    rejection_reasons=[
                        str(item)
                        for item in validated.unresolved_dataflows
                    ],
                ),
            )
        return validated
    except Exception:
        validated = _empty_validated_plan()
        if isinstance(trace, PlanningTrace):
            model_name, temperature = llm_client_metadata(llm_client)
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="dataflow_planning",
                    request_id=str(trace.request_id),
                    model_name=model_name,
                    llm_temperature=temperature,
                    prompt_template_id="dataflow_planning",
                    raw_llm_response=proposal.model_dump(mode="json"),
                    parsed_proposal=proposal.model_dump(mode="json"),
                    selected_candidate=validated.model_dump(mode="json"),
                    rejection_reasons=["dataflow validation failed"],
                ),
            )
        return validated


__all__ = [
    "DataflowPlanProposal",
    "DataflowRefProposal",
    "DerivedTaskProposal",
    "ValidatedDataflowPlan",
    "ValidatedDataflowRef",
    "ValidatedDerivedTask",
    "plan_dataflow",
    "propose_dataflow_with_llm",
    "validate_dataflow_plan",
]
