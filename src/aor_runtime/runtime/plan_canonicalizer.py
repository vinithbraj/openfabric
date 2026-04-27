from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any

from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep
from aor_runtime.runtime.dataflow import collect_step_references, is_step_reference


TOOL_OUTPUT_MAP: dict[str, tuple[str, ...]] = {
    "sql.query": ("rows",),
    "fs.read": ("content",),
    "fs.find": ("matches",),
    "fs.list": ("entries",),
    "fs.size": ("size_bytes",),
    "fs.aggregate": ("file_count", "total_size_bytes", "matches", "summary_text", "display_size"),
    "fs.exists": ("exists",),
    "fs.not_exists": ("exists",),
    "shell.exec": ("stdout",),
    "python.exec": ("data", "csv", "json", "markdown", "content", "value", "text"),
}
GENERIC_OUTPUT_ALIASES = {
    "data",
    "result",
    "results",
    "output",
    "outputs",
    "rows",
    "csv",
    "json",
    "markdown",
    "content",
    "value",
    "text",
}
TEXT_READBACK_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}
PATH_ARG_KEYS = {"path", "src", "dst"}
DEFAULT_REPAIR_BUDGET = 24
CANONICALIZER_ADDED_ARG = "__canonicalizer_added"
CANONICALIZER_WRITE_PATH_ARG = "__canonicalizer_write_path"
VALID_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RETURN_REQUEST_RE = re.compile(r"\b(return|show|provide|list)\b", re.IGNORECASE)
TEXTUAL_CONTENT_PATHS = {"content", "csv", "json", "markdown", "text", "value"}


@dataclass(slots=True)
class CanonicalizedPlanResult:
    plan: ExecutionPlan
    changed: bool
    repairs: list[str]


@dataclass(slots=True)
class _Producer:
    step_id: int
    action: str
    output: str
    original_output: str


def coerce_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    coerced = deepcopy(payload)
    steps = coerced.get("steps")
    if not isinstance(steps, list):
        return coerced
    for step in steps:
        if not isinstance(step, dict):
            continue
        raw_input = step.get("input")
        if isinstance(raw_input, dict):
            input_aliases = [str(key).strip() for key in raw_input.keys() if str(key).strip()]
            step["input"] = input_aliases
            args = step.setdefault("args", {})
            if isinstance(args, dict):
                inputs_mapping = args.get("inputs")
                if not isinstance(inputs_mapping, dict) or not inputs_mapping:
                    args["inputs"] = deepcopy(raw_input)
    return coerced


def canonicalize_plan(
    plan: ExecutionPlan,
    goal: str,
    allowed_tools: list[str],
    *,
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
) -> CanonicalizedPlanResult:
    working = ExecutionPlan.model_validate(plan.model_dump())
    original_dump = working.model_dump()
    repairs: list[str] = []
    budget_used = 0
    producers: list[_Producer] = []
    alias_rewrites: dict[str, str] = {}
    seen_outputs: set[str] = set()
    alias_counts: dict[str, int] = {}

    for step in working.steps:
        output_name = str(step.output or "").strip()
        if not output_name:
            continue
        canonical_output = _canonical_output_alias(step, output_name, seen_outputs, alias_counts)
        if canonical_output != output_name:
            _consume_budget(repair_budget, budget_used + 1)
            budget_used += 1
            repairs.append(f"output:{output_name}->{canonical_output}")
            alias_rewrites[output_name] = canonical_output
            step.output = canonical_output
        seen_outputs.add(str(step.output))
        producers.append(
            _Producer(
                step_id=step.id,
                action=step.action,
                output=str(step.output),
                original_output=output_name,
            )
        )

    repaired_steps: list[ExecutionStep] = []
    prior_producers: list[_Producer] = []
    prior_paths: list[str] = []
    for step in working.steps:
        rewritten_args, arg_repairs = _rewrite_step_args(step.args, prior_producers, alias_rewrites)
        if arg_repairs:
            _consume_budget(repair_budget, budget_used + len(arg_repairs))
            budget_used += len(arg_repairs)
            repairs.extend(arg_repairs)

        rewritten_args, python_input_repairs = _repair_python_inputs_args(
            step,
            rewritten_args,
            prior_producers,
            alias_rewrites,
        )
        if python_input_repairs:
            _consume_budget(repair_budget, budget_used + len(python_input_repairs))
            budget_used += len(python_input_repairs)
            repairs.extend(python_input_repairs)

        rewritten_inputs, input_repairs = _rewrite_step_inputs(step.input, rewritten_args, prior_producers, alias_rewrites)
        if input_repairs:
            _consume_budget(repair_budget, budget_used + len(input_repairs))
            budget_used += len(input_repairs)
            repairs.extend(input_repairs)

        repaired_write_args, write_repairs = _repair_write_content_from_inputs(
            step,
            rewritten_args,
            rewritten_inputs,
            prior_producers,
        )
        if write_repairs:
            _consume_budget(repair_budget, budget_used + len(write_repairs))
            budget_used += len(write_repairs)
            repairs.extend(write_repairs)
            rewritten_args = repaired_write_args

        repaired_path_args, path_repairs = _repair_step_paths(rewritten_args, prior_paths)
        if path_repairs:
            _consume_budget(repair_budget, budget_used + len(path_repairs))
            budget_used += len(path_repairs)
            repairs.extend(path_repairs)

        repaired_action, repaired_path_args, rewritten_inputs, structured_write_repairs = _rewrite_structured_fs_write_step(
            step,
            repaired_path_args,
            rewritten_inputs,
        )
        if structured_write_repairs:
            _consume_budget(repair_budget, budget_used + len(structured_write_repairs))
            budget_used += len(structured_write_repairs)
            repairs.extend(structured_write_repairs)

        repaired_step = ExecutionStep.model_validate(
            {
                "id": step.id,
                "action": repaired_action,
                "args": repaired_path_args,
                "input": rewritten_inputs,
                "output": step.output,
            }
        )
        repaired_steps.append(repaired_step)
        prior_paths.extend(_literal_paths_for_args(repaired_step.args))
        output_name = str(repaired_step.output or "").strip()
        if output_name:
            prior_producers.append(
                _Producer(
                    step_id=repaired_step.id,
                    action=repaired_step.action,
                    output=output_name,
                    original_output=output_name,
                )
            )

    merged_steps, merge_repairs = _merge_adjacent_python_steps_if_safe(repaired_steps)
    if merge_repairs:
        _consume_budget(repair_budget, budget_used + len(merge_repairs))
        budget_used += len(merge_repairs)
        repairs.extend(merge_repairs)
    repaired_steps = merged_steps

    maybe_with_readback, readback_repairs = _append_final_readback_if_needed(
        repaired_steps,
        goal=goal,
        allowed_tools=allowed_tools,
    )
    if readback_repairs:
        _consume_budget(repair_budget, budget_used + len(readback_repairs))
        budget_used += len(readback_repairs)
        repairs.extend(readback_repairs)

    canonical_plan = ExecutionPlan.model_validate({"steps": [step.model_dump() for step in maybe_with_readback]})
    changed = canonical_plan.model_dump() != original_dump
    if not changed:
        repairs = []
    return CanonicalizedPlanResult(plan=canonical_plan, changed=changed, repairs=repairs)


def _consume_budget(limit: int, proposed_total: int) -> None:
    if proposed_total > limit:
        raise ValueError("Canonicalization repair budget exceeded.")


def _canonical_output_alias(
    step: ExecutionStep,
    output_name: str,
    seen_outputs: set[str],
    alias_counts: dict[str, int],
) -> str:
    normalized = output_name.strip()
    if _is_stable_output_alias(normalized) and normalized not in seen_outputs:
        return normalized
    base_alias = f"step_{step.id}_{_default_output_kind(step.action)}"
    return _dedupe_alias(base_alias, seen_outputs, alias_counts)


def _is_stable_output_alias(alias: str) -> bool:
    if not alias:
        return False
    if alias.lower() in GENERIC_OUTPUT_ALIASES:
        return False
    return bool(VALID_ALIAS_RE.match(alias))


def _dedupe_alias(base_alias: str, seen_outputs: set[str], alias_counts: dict[str, int]) -> str:
    if base_alias not in seen_outputs:
        alias_counts.setdefault(base_alias, 1)
        return base_alias
    counter = alias_counts.get(base_alias, 1)
    while True:
        counter += 1
        candidate = f"{base_alias}_{counter}"
        if candidate not in seen_outputs:
            alias_counts[base_alias] = counter
            return candidate


def _default_output_kind(action: str) -> str:
    return TOOL_OUTPUT_MAP.get(action, ("data",))[0]


def _rewrite_step_args(
    value: Any,
    prior_producers: list[_Producer],
    alias_rewrites: dict[str, str],
) -> tuple[Any, list[str]]:
    repairs: list[str] = []

    def rewrite(current: Any) -> Any:
        if is_step_reference(current):
            ref = deepcopy(current)
            original_alias = str(ref["$ref"]).strip()
            rewritten_alias = alias_rewrites.get(original_alias, original_alias)
            known_outputs = {producer.output for producer in prior_producers}
            if rewritten_alias not in known_outputs:
                repaired_alias = _repair_unknown_ref(rewritten_alias, ref.get("path"), prior_producers)
                if repaired_alias is not None:
                    repairs.append(f"ref:{original_alias}->{repaired_alias}")
                    rewritten_alias = repaired_alias
            if rewritten_alias != original_alias:
                ref["$ref"] = rewritten_alias
            return ref
        if isinstance(current, dict):
            return {key: rewrite(nested) for key, nested in current.items()}
        if isinstance(current, list):
            return [rewrite(nested) for nested in current]
        return current

    return rewrite(value), repairs


def _repair_unknown_ref(alias: str, path: Any, prior_producers: list[_Producer]) -> str | None:
    expected_field = _expected_ref_field(alias, path)
    compatible: list[_Producer] = []
    for producer in prior_producers:
        fields = TOOL_OUTPUT_MAP.get(producer.action, ("data",))
        if expected_field is not None and expected_field not in fields:
            continue
        compatible.append(producer)
    if len(compatible) == 1:
        return compatible[0].output
    return None


def _expected_ref_field(alias: str, path: Any) -> str | None:
    path_text = str(path or "").strip()
    if path_text:
        return path_text.split(".", 1)[0].strip() or None
    alias_text = str(alias or "").strip().lower()
    if alias_text in GENERIC_OUTPUT_ALIASES:
        return alias_text if alias_text != "result" else "data"
    for fields in TOOL_OUTPUT_MAP.values():
        if alias_text in fields:
            return alias_text
    return None


def _rewrite_step_inputs(
    inputs: list[str],
    args: dict[str, Any],
    prior_producers: list[_Producer],
    alias_rewrites: dict[str, str],
) -> tuple[list[str], list[str]]:
    repairs: list[str] = []
    referenced_outputs = list(_ordered_step_references(args))
    known_outputs = {producer.output for producer in prior_producers}
    rewritten_declared: list[str] = []
    for raw_input in inputs:
        input_alias = str(raw_input).strip()
        if not input_alias:
            continue
        rewritten_alias = alias_rewrites.get(input_alias, input_alias)
        if rewritten_alias not in known_outputs:
            repaired_alias = _repair_unknown_ref(rewritten_alias, None, prior_producers)
            if repaired_alias is not None:
                repairs.append(f"input:{input_alias}->{repaired_alias}")
                rewritten_alias = repaired_alias
        rewritten_declared.append(rewritten_alias)

    if referenced_outputs:
        if rewritten_declared != referenced_outputs:
            if rewritten_declared:
                repairs.append("input:normalized_to_refs")
            else:
                repairs.append("input:synthesized_from_refs")
            rewritten_declared = referenced_outputs

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in rewritten_declared:
        if alias in seen:
            continue
        deduped.append(alias)
        seen.add(alias)
    return deduped, repairs


def _repair_python_inputs_args(
    step: ExecutionStep,
    args: dict[str, Any],
    prior_producers: list[_Producer],
    alias_rewrites: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    if step.action != "python.exec":
        return args, []
    inputs_mapping = args.get("inputs")
    if not isinstance(inputs_mapping, dict) or not inputs_mapping:
        return args, []

    known_outputs = {producer.output for producer in prior_producers}
    producer_by_output = {producer.output: producer for producer in prior_producers}
    repaired_args = deepcopy(args)
    repaired_inputs = dict(inputs_mapping)
    repairs: list[str] = []
    declared_inputs = [alias_rewrites.get(str(value).strip(), str(value).strip()) for value in step.input if str(value).strip()]

    for key, value in list(repaired_inputs.items()):
        if is_step_reference(value):
            continue
        candidate_aliases: list[str] = []
        if isinstance(value, str) and value.strip():
            candidate_aliases.append(alias_rewrites.get(value.strip(), value.strip()))
        if isinstance(key, str) and key.strip():
            candidate_aliases.append(alias_rewrites.get(key.strip(), key.strip()))
        candidate_aliases.extend(declared_inputs)

        chosen_alias = ""
        for candidate in candidate_aliases:
            if candidate in known_outputs:
                chosen_alias = candidate
                break
        if not chosen_alias:
            continue
        producer = producer_by_output[chosen_alias]
        repaired_ref: dict[str, Any] = {"$ref": chosen_alias}
        default_field = _default_output_kind(producer.action)
        if default_field and default_field != "data":
            repaired_ref["path"] = default_field
        repaired_inputs[key] = repaired_ref
        repairs.append(f"python_inputs:{key}->{chosen_alias}")

    if repairs:
        repaired_args["inputs"] = repaired_inputs
        return repaired_args, repairs
    return args, []


def _ordered_step_references(value: Any) -> list[str]:
    refs: list[str] = []

    def collect(current: Any) -> None:
        if is_step_reference(current):
            alias = str(current["$ref"]).strip()
            if alias and alias not in refs:
                refs.append(alias)
            return
        if isinstance(current, dict):
            for nested in current.values():
                collect(nested)
            return
        if isinstance(current, list):
            for nested in current:
                collect(nested)

    collect(value)
    return refs


def _repair_step_paths(args: dict[str, Any], prior_paths: list[str]) -> tuple[dict[str, Any], list[str]]:
    repaired = deepcopy(args)
    repairs: list[str] = []
    for key in PATH_ARG_KEYS:
        value = repaired.get(key)
        if not isinstance(value, str):
            continue
        rewritten = _repair_single_path(value, prior_paths)
        if rewritten is None or rewritten == value:
            continue
        repaired[key] = rewritten
        repairs.append(f"path:{value}->{rewritten}")
    return repaired, repairs


def _repair_write_content_from_inputs(
    step: ExecutionStep,
    args: dict[str, Any],
    inputs: list[str],
    prior_producers: list[_Producer],
) -> tuple[dict[str, Any], list[str]]:
    if step.action != "fs.write":
        return args, []
    path_value = args.get("path")
    if not isinstance(path_value, str):
        return args, []

    content_value = args.get("content")
    if is_step_reference(content_value):
        output_alias = str(content_value["$ref"]).strip()
        producer = next((item for item in prior_producers if item.output == output_alias), None)
        if producer is None:
            return args, []
        content_path = _preferred_content_path_for_write(path_value, producer.action)
        if content_path is None:
            return args, []
        current_path = str(content_value.get("path") or "").strip()
        if current_path == content_path:
            return args, []
        repaired = deepcopy(args)
        repaired["content"] = {"$ref": output_alias, "path": content_path}
        label = current_path or "<default>"
        return repaired, [f"write_content_ref:{path_value}:{label}->{content_path}"]

    if collect_step_references(args):
        return args, []
    if len(inputs) != 1:
        return args, []
    if not isinstance(content_value, str):
        return args, []
    output_alias = inputs[0]
    producer = next((item for item in prior_producers if item.output == output_alias), None)
    if producer is None:
        return args, []
    content_path = _preferred_content_path_for_write(path_value, producer.action)
    if content_path is None:
        return args, []
    repaired = deepcopy(args)
    repaired["content"] = {"$ref": output_alias, "path": content_path}
    return repaired, [f"write_content:{path_value}<-{output_alias}.{content_path}"]


def _rewrite_structured_fs_write_step(
    step: ExecutionStep,
    args: dict[str, Any],
    inputs: list[str],
) -> tuple[str, dict[str, Any], list[str], list[str]]:
    if step.action != "fs.write":
        return step.action, args, inputs, []
    path_value = args.get("path")
    content_value = args.get("content")
    if not isinstance(path_value, str):
        return step.action, args, inputs, []
    if not _is_structured_write_content(content_value):
        return step.action, args, inputs, []

    repaired_inputs = list(inputs)
    for alias in _ordered_step_references({"content": content_value}):
        if alias not in repaired_inputs:
            repaired_inputs.append(alias)

    repaired_args = {
        "inputs": {
            "path": path_value,
            "content": deepcopy(content_value),
        },
        "code": "serialized_content = _json_dumps_safe(inputs['content']); result = fs.write(inputs['path'], serialized_content)",
        CANONICALIZER_WRITE_PATH_ARG: path_value,
    }
    return "python.exec", repaired_args, repaired_inputs, [f"write_structured:{path_value}->python.exec"]


def _is_structured_write_content(value: Any) -> bool:
    if isinstance(value, (dict, list)):
        if is_step_reference(value):
            path_text = str(value.get("path") or "").strip().split(".", 1)[0]
            return path_text not in TEXTUAL_CONTENT_PATHS
        return True
    return False


def _repair_single_path(value: str, prior_paths: list[str]) -> str | None:
    path_text = str(value).strip()
    if not path_text or "/" in path_text or path_text.startswith(".") or "*" in path_text:
        return None
    matches = sorted({candidate for candidate in prior_paths if candidate.endswith(f"/{path_text}")})
    if len(matches) == 1:
        return matches[0]
    return None


def _literal_paths_for_args(args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in PATH_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and "/" in value and "*" not in value:
            paths.append(value)
    return paths


def _preferred_content_path_for_write(path_value: str, producer_action: str) -> str | None:
    suffix = PurePosixPath(path_value).suffix.lower()
    if producer_action == "fs.read":
        return "content"
    if producer_action == "shell.exec":
        return "stdout"
    if producer_action == "python.exec":
        if suffix == ".json":
            return "json"
        if suffix == ".csv":
            return "csv"
        if suffix == ".md":
            return "markdown"
        return "value"
    return None


def _append_final_readback_if_needed(
    steps: list[ExecutionStep],
    *,
    goal: str,
    allowed_tools: list[str],
) -> tuple[list[ExecutionStep], list[str]]:
    if "fs.read" not in allowed_tools:
        return steps, []
    if not steps:
        return steps, []
    if not RETURN_REQUEST_RE.search(str(goal or "")):
        return steps, []
    last_step = steps[-1]
    if last_step.action == "fs.read" and bool(last_step.args.get(CANONICALIZER_ADDED_ARG)):
        return steps, []
    path_value: str | None = None
    if last_step.action == "fs.write":
        raw_path = last_step.args.get("path")
        if isinstance(raw_path, str):
            path_value = raw_path
    elif last_step.action == "python.exec":
        raw_path = last_step.args.get(CANONICALIZER_WRITE_PATH_ARG)
        if isinstance(raw_path, str):
            path_value = raw_path
    if path_value is None:
        return steps, []
    suffix = PurePosixPath(path_value).suffix.lower()
    if suffix not in TEXT_READBACK_EXTENSIONS:
        return steps, []
    new_step = ExecutionStep.model_validate(
        {
            "id": max(step.id for step in steps) + 1,
            "action": "fs.read",
            "args": {"path": path_value, CANONICALIZER_ADDED_ARG: True},
        }
    )
    return [*steps, new_step], [f"readback:appended:{path_value}"]


def _merge_adjacent_python_steps_if_safe(steps: list[ExecutionStep]) -> tuple[list[ExecutionStep], list[str]]:
    # Existing step ids and order must remain stable, so aggressive merging is intentionally disabled.
    # We keep the hook in place for future safe transforms, but under the current invariants this is a no-op.
    return steps, []
