"""OpenFABRIC Runtime Module: aor_runtime.runtime.dataflow

Purpose:
    Resolve references between execution-plan steps.

Responsibilities:
    Canonicalize and dereference $ref values, default paths, formatter inputs, and runtime.return values.

Data flow / Interfaces:
    Consumes action arguments, prior StepLog results, and tool output contracts to produce concrete tool inputs.

Boundaries:
    Invalid references must fail before unsafe execution or user-visible Reference path errors.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep
from aor_runtime.runtime.tool_output_contracts import default_path_for_tool, normalize_tool_ref_path

TEXTUAL_OUTPUT_PATH_ALIASES = {"text", "content", "value", "csv", "csv_string", "json", "json_string", "summary_json"}


def is_step_reference(value: Any) -> bool:
    """Is step reference for the surrounding runtime workflow.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow.is_step_reference.
    """
    return isinstance(value, dict) and isinstance(value.get("$ref"), str) and str(value.get("$ref")).strip() != ""


def collect_step_references(value: Any) -> set[str]:
    """Collect step references for the surrounding runtime workflow.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow.collect_step_references.
    """
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
    """Normalize execution plan dataflow for the surrounding runtime workflow.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow.normalize_execution_plan_dataflow.
    """
    output_producers: dict[str, str] = {}
    for step in plan.steps:
        _normalize_python_inputs(step, output_producers)
        step.args = _canonicalize_reference_paths(step.args, output_producers)
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
    """Resolve step value for the surrounding runtime workflow.

    Inputs:
        Receives value, step_outputs for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow.resolve_step_value.
    """
    if is_step_reference(value):
        alias = str(value["$ref"]).strip()
        if alias not in step_outputs:
            raise ValueError(f"Unknown step output reference: {alias}")
        resolved = deepcopy(step_outputs[alias])
        raw_path = value.get("path", "")
        path = "" if raw_path is None else str(raw_path).strip()
        if path:
            resolved = _resolve_output_path(resolved, path)
        return resolved
    if isinstance(value, dict):
        return {key: resolve_step_value(nested, step_outputs) for key, nested in value.items()}
    if isinstance(value, list):
        return [resolve_step_value(nested, step_outputs) for nested in value]
    return value


def resolve_execution_step(step: ExecutionStep, step_outputs: dict[str, Any]) -> ExecutionStep:
    """Resolve execution step for the surrounding runtime workflow.

    Inputs:
        Receives step, step_outputs for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow.resolve_execution_step.
    """
    resolved_args = resolve_step_value(step.args, step_outputs)
    resolved_args, metadata = _separate_internal_metadata(resolved_args, step.metadata)
    if step.action == "fs.write":
        resolved_args = _normalize_fs_write_args(resolved_args)
    return ExecutionStep.model_validate(
        {
            "id": step.id,
            "action": step.action,
            "args": resolved_args,
            "input": list(step.input),
            "output": step.output,
            "metadata": metadata,
        }
    )


def _separate_internal_metadata(args: dict[str, Any], metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move internal semantic keys out of executable tool arguments.

    Inputs:
        Receives resolved tool arguments and existing execution-step metadata.

    Returns:
        Sanitized tool arguments plus metadata safe for runtime-only use.

    Used by:
        resolve_execution_step before tool invocation.
    """
    if not isinstance(args, dict):
        return args, dict(metadata or {})
    clean_args = dict(args)
    clean_metadata = dict(metadata or {})
    projection = clean_args.pop("__semantic_projection", None)
    for key in list(clean_args):
        if str(key).startswith("__semantic_"):
            clean_args.pop(key, None)
    if projection is not None and "semantic_projection" not in clean_metadata:
        clean_metadata["semantic_projection"] = projection
    return clean_args, clean_metadata


def _resolve_output_path(value: Any, path: str) -> Any:
    """Handle the internal resolve output path helper path for this module.

    Inputs:
        Receives value, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._resolve_output_path.
    """
    current = value
    for chunk in path.split("."):
        segment = chunk.strip()
        if not segment:
            raise ValueError(f"Invalid reference path: {path}")
        if isinstance(current, dict):
            if segment not in current and "result" in current:
                nested_result = current["result"]
                if isinstance(nested_result, dict) and segment in nested_result:
                    current = nested_result[segment]
                    continue
                if isinstance(nested_result, str) and segment in TEXTUAL_OUTPUT_PATH_ALIASES:
                    return deepcopy(nested_result)
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
            if segment not in current and segment in TEXTUAL_OUTPUT_PATH_ALIASES and isinstance(current.get("output"), str):
                return deepcopy(current["output"])
            if segment == "exit_code" and segment not in current and "returncode" in current:
                current = current["returncode"]
                continue
            if segment not in current:
                raise ValueError(f"Reference path not found: {path}. Available fields: {_available_fields(current)}")
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


def _canonicalize_reference_paths(value: Any, output_producers: dict[str, str]) -> Any:
    """Handle the internal canonicalize reference paths helper path for this module.

    Inputs:
        Receives value, output_producers for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._canonicalize_reference_paths.
    """
    if is_step_reference(value):
        normalized = dict(value)
        alias = str(normalized["$ref"]).strip()
        raw_path = normalized.get("path", "")
        path = "" if raw_path is None else str(raw_path).strip()
        if not path:
            default_path = _default_ref_path_for_action(output_producers.get(alias, ""))
            if default_path:
                normalized["path"] = default_path
            else:
                normalized.pop("path", None)
        else:
            normalized["path"] = normalize_tool_ref_path(output_producers.get(alias, ""), path)
        return normalized
    if isinstance(value, dict):
        return {key: _canonicalize_reference_paths(nested, output_producers) for key, nested in value.items()}
    if isinstance(value, list):
        return [_canonicalize_reference_paths(nested, output_producers) for nested in value]
    return value


def _available_fields(value: Any) -> str:
    """Handle the internal available fields helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._available_fields.
    """
    if not isinstance(value, dict) or not value:
        return "<none>"
    fields = sorted(str(key) for key in value.keys())
    preview = ", ".join(fields[:12])
    if len(fields) > 12:
        preview = f"{preview}, ..."
    return preview


def _normalize_python_inputs(step: ExecutionStep, output_producers: dict[str, str]) -> None:
    """Handle the internal normalize python inputs helper path for this module.

    Inputs:
        Receives step, output_producers for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._normalize_python_inputs.
    """
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
    """Handle the internal default ref path for action helper path for this module.

    Inputs:
        Receives action for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._default_ref_path_for_action.
    """
    return default_path_for_tool(action)


def _normalize_fs_write_args(args: dict[str, Any]) -> dict[str, Any]:
    """Handle the internal normalize fs write args helper path for this module.

    Inputs:
        Receives args for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._normalize_fs_write_args.
    """
    content = args.get("content")
    if isinstance(content, str):
        return args
    normalized = dict(args)
    normalized["content"] = _coerce_fs_write_content(content)
    return normalized


def _coerce_fs_write_content(value: Any) -> str:
    """Handle the internal coerce fs write content helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._coerce_fs_write_content.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        if _looks_like_runtime_return_result(value):
            output = value.get("output")
            if isinstance(output, str):
                return output
            return _coerce_fs_write_content(value.get("value"))
        if _looks_like_python_exec_result(value):
            output = value.get("output")
            if isinstance(output, str):
                return output
            return _coerce_fs_write_content(value.get("result"))
        if _looks_like_shell_result(value):
            return str(value.get("stdout") or "")
        return _json_dumps_safe(value)
    if isinstance(value, list):
        return _json_dumps_safe(value)
    return str(value)


def _looks_like_python_exec_result(value: dict[str, Any]) -> bool:
    """Handle the internal looks like python exec result helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._looks_like_python_exec_result.
    """
    return {"success", "result", "error"}.issubset(value.keys()) and "output" in value


def _looks_like_runtime_return_result(value: dict[str, Any]) -> bool:
    """Handle the internal looks like runtime return result helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._looks_like_runtime_return_result.
    """
    return "output" in value and "value" in value and len(value.keys()) == 2


def _looks_like_shell_result(value: dict[str, Any]) -> bool:
    """Handle the internal looks like shell result helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._looks_like_shell_result.
    """
    return {"stdout", "stderr", "returncode"}.issubset(value.keys())


def _json_dumps_safe(value: Any) -> str:
    """Handle the internal json dumps safe helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.dataflow._json_dumps_safe.
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
