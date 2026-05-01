"""Deterministic DAG construction for selected capability tasks."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from agent_runtime.capabilities.registry import CapabilityRegistry
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ActionDAG, ActionNode, InputRef, TaskFrame, UserRequest
from agent_runtime.input_pipeline.argument_extraction import ArgumentExtractionResult
from agent_runtime.input_pipeline.capability_fit import CapabilityFitDecision
from agent_runtime.input_pipeline.dataflow_planning import ValidatedDataflowPlan
from agent_runtime.input_pipeline.decomposition import DecompositionResult
from agent_runtime.input_pipeline.domain_selection import CapabilitySelectionResult


def _resolve_registry(user_request: UserRequest) -> CapabilityRegistry:
    """Resolve the capability registry from request context."""

    for context in (
        user_request.safety_context,
        user_request.session_context,
        user_request.user_context,
    ):
        registry = context.get("capability_registry")
        if isinstance(registry, CapabilityRegistry):
            return registry
    raise ValidationError(
        "Capability registry is required in user_request.safety_context, "
        "session_context, or user_context."
    )


def _node_id_for_task(task_id: str) -> str:
    """Map one task id to one deterministic action node id."""

    return f"node::{task_id}"


_INPUT_REF_PATTERN = re.compile(r"^(?P<source>[A-Za-z0-9_:\-]+)\.(?P<output>[A-Za-z0-9_:\-]+)$")


def _normalize_string_input_ref(
    raw_value: str,
    known_task_ids: set[str],
    known_node_ids: set[str],
) -> InputRef | str:
    """Convert one deterministic string reference into a typed InputRef when valid."""

    match = _INPUT_REF_PATTERN.match(str(raw_value or "").strip())
    if match is None:
        return raw_value

    source = match.group("source")
    output = match.group("output")
    if source in known_task_ids:
        source_node_id = _node_id_for_task(source)
    elif source in known_node_ids:
        source_node_id = source
    else:
        return raw_value

    return InputRef(
        source_node_id=source_node_id,
        output_key=None if output == "output" else output,
    )


def _normalize_argument_value(
    argument_name: str,
    value: Any,
    known_task_ids: set[str],
    known_node_ids: set[str],
) -> Any:
    """Normalize nested input references only for explicit ref-shaped argument slots."""

    if isinstance(value, InputRef):
        return value
    if isinstance(value, list):
        return [
            _normalize_argument_value(argument_name, item, known_task_ids, known_node_ids)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _normalize_argument_value(key, item, known_task_ids, known_node_ids)
            for key, item in value.items()
        }
    if isinstance(value, str) and (
        argument_name == "input_ref" or argument_name.endswith("_ref")
    ):
        return _normalize_string_input_ref(value, known_task_ids, known_node_ids)
    return value


def _normalize_arguments(
    arguments: dict[str, Any],
    known_task_ids: set[str],
    known_node_ids: set[str],
) -> dict[str, Any]:
    """Normalize argument values that represent typed dataflow references."""

    return {
        key: _normalize_argument_value(key, value, known_task_ids, known_node_ids)
        for key, value in dict(arguments).items()
    }


def _build_safety_labels(task: TaskFrame, manifest: Any) -> list[str]:
    """Build deterministic safety labels from manifest metadata and task risk."""

    labels = [
        f"capability-risk:{manifest.risk_level}",
        f"task-risk:{task.risk_level}",
    ]
    if manifest.read_only:
        labels.append("read-only")
    if manifest.mutates_state:
        labels.append("mutates-state")
    if manifest.requires_confirmation or task.requires_confirmation:
        labels.append("requires-confirmation")
    return labels


def build_action_dag(
    user_request: UserRequest,
    decomposition_result: DecompositionResult,
    capability_selection_results: list[CapabilitySelectionResult],
    argument_extraction_results: list[ArgumentExtractionResult],
    dataflow_plan: ValidatedDataflowPlan | None = None,
    capability_fit_decisions: list[CapabilityFitDecision] | None = None,
) -> ActionDAG:
    """Build a validated action DAG from decomposition, selection, and arguments."""

    registry = _resolve_registry(user_request)
    fit_by_task = {decision.task_id: decision for decision in (capability_fit_decisions or [])}
    base_tasks = [
        task
        for task in decomposition_result.tasks
        if fit_by_task.get(task.id) is None or fit_by_task[task.id].is_fit
    ]
    derived_tasks = [derived.task for derived in dataflow_plan.derived_tasks] if dataflow_plan else []
    tasks = [*base_tasks, *derived_tasks]
    task_by_id = {task.id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValidationError("Decomposition contains duplicate task ids.")

    selections_by_task = {result.task_id: result for result in capability_selection_results}
    if len(selections_by_task) != len(capability_selection_results):
        raise ValidationError("Capability selection contains duplicate task ids.")

    arguments_by_task = {result.task_id: result for result in argument_extraction_results}
    if len(arguments_by_task) != len(argument_extraction_results):
        raise ValidationError("Argument extraction contains duplicate task ids.")

    derived_by_task = {
        derived.task.id: derived for derived in (dataflow_plan.derived_tasks if dataflow_plan else [])
    }

    nodes: list[ActionNode] = []
    edges: list[tuple[str, str]] = []
    dag_requires_confirmation = False
    known_task_ids = set(task_by_id)
    known_node_ids = {_node_id_for_task(task_id) for task_id in known_task_ids}

    for task in tasks:
        derived = derived_by_task.get(task.id)
        if derived is not None:
            capability = registry.get(derived.capability_id)
            manifest = capability.manifest
            if derived.operation_id != manifest.operation_id:
                raise ValidationError(
                    f"Derived task {task.id} operation {derived.operation_id} does not match manifest operation {manifest.operation_id}."
                )
            source_arguments = dict(derived.arguments)
        else:
            selection = selections_by_task.get(task.id)
            if selection is None:
                raise ValidationError(f"Task {task.id} has no capability selection result.")
            if selection.selected is None:
                reason = selection.unresolved_reason or "Capability selection is unresolved."
                raise ValidationError(f"Task {task.id} is unresolved: {reason}")

            capability = registry.get(selection.selected.capability_id)
            manifest = capability.manifest

            if selection.selected.operation_id != manifest.operation_id:
                raise ValidationError(
                    f"Task {task.id} selected operation {selection.selected.operation_id} "
                    f"does not match manifest operation {manifest.operation_id}."
                )

            extraction = arguments_by_task.get(task.id)
            if extraction is None:
                raise ValidationError(f"Task {task.id} has no argument extraction result.")

            if extraction.capability_id != manifest.capability_id:
                raise ValidationError(
                    f"Task {task.id} argument extraction capability {extraction.capability_id} "
                    f"does not match selected capability {manifest.capability_id}."
                )
            if extraction.operation_id != manifest.operation_id:
                raise ValidationError(
                    f"Task {task.id} argument extraction operation {extraction.operation_id} "
                    f"does not match selected operation {manifest.operation_id}."
                )
            if extraction.missing_required_arguments:
                raise ValidationError(
                    f"Task {task.id} is missing required arguments: "
                    f"{', '.join(extraction.missing_required_arguments)}"
                )
            source_arguments = dict(extraction.arguments)

        for ref in dataflow_plan.refs if dataflow_plan else []:
            if ref.consumer_task_id == task.id:
                source_arguments[ref.consumer_argument_name] = ref.input_ref

        normalized_arguments = _normalize_arguments(source_arguments, known_task_ids, known_node_ids)
        validated_arguments = capability.validate_arguments(normalized_arguments)
        node_id = _node_id_for_task(task.id)
        dependency_task_ids = list(task.dependencies)
        if dataflow_plan:
            dependency_task_ids.extend(
                source
                for source, target in dataflow_plan.dependency_edges
                if target == task.id
            )
            dependency_task_ids.extend(
                ref.producer_task_id
                for ref in dataflow_plan.refs
                if ref.consumer_task_id == task.id
            )
        dependency_task_ids = list(dict.fromkeys(dependency_task_ids))
        dependency_node_ids = [_node_id_for_task(dependency) for dependency in dependency_task_ids]
        labels = _build_safety_labels(task, manifest)
        if task.requires_confirmation or manifest.requires_confirmation:
            dag_requires_confirmation = True

        nodes.append(
            ActionNode(
                id=node_id,
                task_id=task.id,
                description=task.description,
                semantic_verb=task.semantic_verb,
                capability_id=manifest.capability_id,
                operation_id=manifest.operation_id,
                arguments=validated_arguments,
                depends_on=dependency_node_ids,
                safety_labels=labels,
                dry_run=bool(task.constraints.get("dry_run", False)),
            )
        )

        for dependency in dependency_node_ids:
            edges.append((dependency, node_id))

    try:
        return ActionDAG(
            nodes=nodes,
            edges=edges,
            global_constraints=dict(decomposition_result.global_constraints),
            requires_confirmation=dag_requires_confirmation,
        )
    except PydanticValidationError as exc:
        raise ValidationError(str(exc)) from exc


class DAGBuilder:
    """Compatibility DAG builder for the lightweight scaffold orchestrator."""

    def build(self, frame: TaskFrame) -> ActionDAG:
        """Build a single-node DAG from one already-populated task frame."""

        capability_id = str(frame.constraints.get("capability_id") or "unknown")
        operation_id = str(frame.constraints.get("operation_id") or frame.semantic_verb)
        labels = ["task-risk:" + frame.risk_level]
        if capability_id == "unknown":
            labels.append("unresolved")
        if frame.requires_confirmation:
            labels.append("requires-confirmation")
        return ActionDAG(
            nodes=[
                ActionNode(
                    id=_node_id_for_task(frame.id),
                    task_id=frame.id,
                    description=frame.description,
                    semantic_verb=frame.semantic_verb,
                    capability_id=capability_id,
                    operation_id=operation_id,
                    arguments=dict(frame.constraints.get("arguments") or {}),
                    depends_on=[_node_id_for_task(dependency) for dependency in frame.dependencies],
                    safety_labels=labels,
                    dry_run=bool(frame.constraints.get("dry_run", False)),
                )
            ],
            edges=[
                (_node_id_for_task(dependency), _node_id_for_task(frame.id))
                for dependency in frame.dependencies
            ],
            global_constraints=dict(frame.constraints),
            requires_confirmation=frame.requires_confirmation,
        )
