from __future__ import annotations

from copy import deepcopy
from typing import Any

from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep


def is_step_reference(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("$ref"), str) and str(value.get("$ref")).strip() != ""


def collect_step_references(value: Any) -> set[str]:
    refs: set[str] = set()
    if is_step_reference(value):
        refs.add(str(value["$ref"]).strip())
        return refs
    if isinstance(value, dict):
        for nested in value.values():
            refs.update(collect_step_references(nested))
    elif isinstance(value, list):
        for nested in value:
            refs.update(collect_step_references(nested))
    return refs


def normalize_execution_plan_dataflow(plan: ExecutionPlan) -> ExecutionPlan:
    output_producers: dict[str, str] = {}
    for step in plan.steps:
        _normalize_python_inputs(step, output_producers)
        referenced_outputs = sorted(collect_step_references(step.args))
        if not referenced_outputs:
            output_name = str(step.output or "").strip()
            if output_name:
                output_producers[output_name] = step.action
            continue
        declared_inputs: list[str] = []
        seen_inputs: set[str] = set()
        for raw_input in step.input:
            input_name = str(raw_input).strip()
            if not input_name or input_name in seen_inputs:
                continue
            declared_inputs.append(input_name)
            seen_inputs.add(input_name)
        for dependency_name in referenced_outputs:
            if dependency_name in seen_inputs:
                continue
            declared_inputs.append(dependency_name)
            seen_inputs.add(dependency_name)
        step.input = declared_inputs
        output_name = str(step.output or "").strip()
        if output_name:
            output_producers[output_name] = step.action
    return plan


def resolve_step_value(value: Any, step_outputs: dict[str, Any]) -> Any:
    if is_step_reference(value):
        alias = str(value["$ref"]).strip()
        if alias not in step_outputs:
            raise ValueError(f"Unknown step output reference: {alias}")
        resolved = deepcopy(step_outputs[alias])
        path = str(value.get("path", "")).strip()
        if path:
            resolved = _resolve_output_path(resolved, path)
        return resolved
    if isinstance(value, dict):
        return {key: resolve_step_value(nested, step_outputs) for key, nested in value.items()}
    if isinstance(value, list):
        return [resolve_step_value(nested, step_outputs) for nested in value]
    return value


def resolve_execution_step(step: ExecutionStep, step_outputs: dict[str, Any]) -> ExecutionStep:
    return ExecutionStep.model_validate(
        {
            "id": step.id,
            "action": step.action,
            "args": resolve_step_value(step.args, step_outputs),
            "input": list(step.input),
            "output": step.output,
        }
    )


def _resolve_output_path(value: Any, path: str) -> Any:
    current = value
    for chunk in path.split("."):
        segment = chunk.strip()
        if not segment:
            raise ValueError(f"Invalid reference path: {path}")
        if isinstance(current, dict):
            if segment not in current and segment == "json" and isinstance(current.get("output"), str):
                return deepcopy(current["output"])
            if segment not in current and "result" in current:
                nested_result = current["result"]
                if isinstance(nested_result, dict) and segment in nested_result:
                    current = nested_result[segment]
                    continue
                if isinstance(nested_result, list):
                    try:
                        index = int(segment)
                    except ValueError:
                        pass
                    else:
                        try:
                            current = nested_result[index]
                            continue
                        except IndexError as exc:
                            raise ValueError(f"Reference path index out of range: {path}") from exc
            if segment not in current:
                raise ValueError(f"Reference path not found: {path}")
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError as exc:
                raise ValueError(f"Reference path segment must be an integer for list access: {segment}") from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise ValueError(f"Reference path index out of range: {path}") from exc
            continue
        raise ValueError(f"Cannot traverse reference path {path!r} through non-container value.")
    return deepcopy(current)


def _normalize_python_inputs(step: ExecutionStep, output_producers: dict[str, str]) -> None:
    if step.action != "python.exec":
        return
    args = dict(step.args)
    inputs_mapping = dict(args.get("inputs") or {})
    mutated = False
    for raw_input in step.input:
        input_alias = str(raw_input).strip()
        if not input_alias or input_alias in inputs_mapping:
            continue
        producer_action = output_producers.get(input_alias, "")
        reference: dict[str, Any] = {"$ref": input_alias}
        path = _default_ref_path_for_action(producer_action)
        if path:
            reference["path"] = path
        inputs_mapping[input_alias] = reference
        mutated = True
    if mutated or ("inputs" in args and args.get("inputs") != inputs_mapping):
        args["inputs"] = inputs_mapping
        step.args = args


def _default_ref_path_for_action(action: str) -> str | None:
    if action == "shell.exec":
        return "stdout"
    if action == "sql.query":
        return "rows"
    if action == "fs.find":
        return "matches"
    if action == "fs.list":
        return "entries"
    if action == "fs.read":
        return "content"
    if action == "fs.size":
        return "size_bytes"
    if action in {"fs.exists", "fs.not_exists"}:
        return "exists"
    return None
