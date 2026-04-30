"""OpenFABRIC Runtime Module: aor_runtime.runtime.intent_compiler

Purpose:
    Provide deterministic plan-builder helpers retained for compatibility and domain utilities.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Any
from urllib.parse import unquote, urlparse

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.output_contract import build_output_contract
from aor_runtime.runtime.intents import (
    CompoundIntent,
    CountFilesIntent,
    FileAggregateIntent,
    FetchExtractIntent,
    ListFilesIntent,
    ReadFileLineIntent,
    SearchFileContentsIntent,
    ShellCommandIntent,
    SqlCountIntent,
    SqlSelectIntent,
    TransformChainIntent,
    TransformIntent,
    WriteResultIntent,
    WriteTextIntent,
)
from aor_runtime.tools.sql import resolve_database_selection


INTERNAL_RETURN_ACTION = "runtime.return"


@dataclass
class CompiledFragment:
    """Represent compiled fragment within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CompiledFragment.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intent_compiler.CompiledFragment and related tests.
    """
    steps: list[dict[str, Any]]
    output_alias: str
    output_kind: str
    value_ref: dict[str, Any]
    return_mode: str = "text"
    path_style: str | None = None
    terminal: bool = False


def compile_intent_to_plan(intent: Any, allowed_tools: list[str], settings: Settings) -> ExecutionPlan:
    """Compile intent to plan for the surrounding runtime workflow.

    Inputs:
        Receives intent, allowed_tools, settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler.compile_intent_to_plan.
    """
    if isinstance(intent, CompoundIntent):
        return _compile_compound_intent(intent, allowed_tools, settings)

    if isinstance(intent, ReadFileLineIntent):
        _require_tools(allowed_tools, "fs.read", "python.exec")
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": "fs.read", "args": {"path": intent.path}, "output": "file_text"},
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["file_text"],
                        "args": {
                            "inputs": {"text": {"$ref": "file_text", "path": "content"}},
                            "code": f"lines = inputs['text'].splitlines(); result = lines[{intent.line_number - 1}] if len(lines) >= {intent.line_number} else ''",
                        },
                    },
                ]
            }
        )

    if isinstance(intent, CountFilesIntent):
        action, args, output = _file_discovery_step(intent.path, intent.pattern, recursive=intent.recursive)
        _require_tools(allowed_tools, action)
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": action, "args": args, "output": output},
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": [output],
                        "args": {
                            "value": {"$ref": output, "path": "matches"},
                            "mode": "count",
                            "output_contract": build_output_contract(mode="count"),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, FileAggregateIntent):
        _require_tools(allowed_tools, "fs.aggregate")
        include_matches = intent.output_mode == "json"
        if intent.output_mode == "json":
            value: Any = {
                "file_count": {"$ref": "file_aggregate", "path": "file_count"},
                "matches": {"$ref": "file_aggregate", "path": "matches"},
            }
            if intent.aggregate != "count":
                value["total_size_bytes"] = {"$ref": "file_aggregate", "path": "total_size_bytes"}
        elif intent.output_mode == "count" and intent.aggregate == "count":
            value = {"$ref": "file_aggregate", "path": "file_count"}
        else:
            value = {"$ref": "file_aggregate", "path": "summary_text"}
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "fs.aggregate",
                        "args": {
                            "path": intent.path,
                            "pattern": intent.pattern,
                            "recursive": intent.recursive,
                            "file_only": intent.file_only,
                            "include_matches": include_matches,
                            "path_style": "relative",
                            "size_unit": intent.size_unit,
                            "aggregate": intent.aggregate,
                        },
                        "output": "file_aggregate",
                    },
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": ["file_aggregate"],
                        "args": {
                            "value": value,
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(mode=intent.output_mode),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, ListFilesIntent):
        action, args, output = _file_discovery_step(
            intent.path,
            intent.pattern,
            recursive=intent.recursive,
            file_only=intent.file_only,
            dir_only=intent.dir_only,
            path_style=intent.path_style,
        )
        _require_tools(allowed_tools, action)
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": action, "args": args, "output": output},
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": [output],
                        "args": {
                            "value": {"$ref": output, "path": "matches"},
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(
                                mode=intent.output_mode,
                                path_style=intent.path_style,
                                json_shape="matches" if intent.output_mode == "json" else None,
                            ),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, SearchFileContentsIntent):
        _require_tools(allowed_tools, "fs.search_content")
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "fs.search_content",
                        "args": {
                            "path": intent.path,
                            "needle": intent.needle,
                            "pattern": intent.pattern or "*",
                            "recursive": intent.recursive,
                            "file_only": True,
                            "case_insensitive": False,
                            "path_style": intent.path_style,
                        },
                        "output": "search_results",
                    },
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": ["search_results"],
                        "args": {
                            "value": {"$ref": "search_results", "path": "matches"},
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(
                                mode=intent.output_mode,
                                path_style=intent.path_style,
                                json_shape="matches" if intent.output_mode == "json" else None,
                            ),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, SqlCountIntent):
        _require_tools(allowed_tools, "sql.query")
        database_name, _ = resolve_database_selection(settings, intent.database)
        query = f"SELECT COUNT(*) AS count_value FROM {intent.table}"
        if intent.where:
            query = f"{query} WHERE {intent.where}"
        value: Any = {"$ref": "count_rows", "path": "rows.0.count_value"}
        mode = "count"
        if intent.output_key:
            value = {intent.output_key: {"$ref": "count_rows", "path": "rows.0.count_value"}}
            mode = "json"
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": database_name, "query": query},
                        "output": "count_rows",
                    },
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": ["count_rows"],
                        "args": {
                            "value": value,
                            "mode": mode,
                            "output_contract": build_output_contract(mode=mode, json_shape="count" if mode == "json" and not intent.output_key else None),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, SqlSelectIntent):
        _require_tools(allowed_tools, "sql.query")
        database_name, _ = resolve_database_selection(settings, intent.database)
        query = f"SELECT {', '.join(intent.columns)} FROM {intent.table}"
        if intent.where:
            query = f"{query} WHERE {intent.where}"
        if intent.order_by:
            query = f"{query} ORDER BY {intent.order_by}"
        if intent.limit is not None:
            query = f"{query} LIMIT {intent.limit}"
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {
                        "id": 1,
                        "action": "sql.query",
                        "args": {"database": database_name, "query": query},
                        "output": "select_rows",
                    },
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": ["select_rows"],
                        "args": {
                            "value": {"$ref": "select_rows", "path": "rows"},
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(
                                mode=intent.output_mode,
                                json_shape="rows" if intent.output_mode == "json" else None,
                            ),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, WriteTextIntent):
        _require_tools(allowed_tools, "fs.write")
        steps: list[dict[str, Any]] = [{"id": 1, "action": "fs.write", "args": {"path": intent.path, "content": intent.content}}]
        if intent.return_content:
            _require_tools(allowed_tools, "fs.read")
            steps.append({"id": 2, "action": "fs.read", "args": {"path": intent.path}, "output": "written_content"})
            steps.append(
                {
                    "id": 3,
                    "action": INTERNAL_RETURN_ACTION,
                    "input": ["written_content"],
                    "args": {
                        "value": {"$ref": "written_content", "path": "content"},
                        "mode": "text",
                        "output_contract": build_output_contract(mode="text"),
                    },
                }
            )
        return ExecutionPlan.model_validate({"steps": steps})

    if isinstance(intent, WriteResultIntent):
        _require_tools(allowed_tools, "fs.write")
        steps = [
            {
                "id": 1,
                "action": "fs.write",
                "input": [intent.source_alias],
                "args": {"path": intent.path, "content": {"$ref": intent.source_alias}},
            }
        ]
        if intent.return_content or intent.read_back:
            _require_tools(allowed_tools, "fs.read")
            steps.append({"id": 2, "action": "fs.read", "args": {"path": intent.path}, "output": "written_result_content"})
            steps.append(
                {
                    "id": 3,
                    "action": INTERNAL_RETURN_ACTION,
                    "input": ["written_result_content"],
                    "args": {
                        "value": {"$ref": "written_result_content", "path": "content"},
                        "mode": "text",
                        "output_contract": build_output_contract(mode="text"),
                    },
                }
            )
        return ExecutionPlan.model_validate({"steps": steps})

    if isinstance(intent, ShellCommandIntent):
        _require_tools(allowed_tools, "shell.exec")
        command_args: dict[str, Any] = {"command": intent.command}
        if intent.node:
            command_args["node"] = intent.node
        return ExecutionPlan.model_validate(
            {
                "steps": [
                    {"id": 1, "action": "shell.exec", "args": command_args, "output": "shell_output"},
                    {
                        "id": 2,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": ["shell_output"],
                        "args": {
                            "value": {"$ref": "shell_output", "path": "stdout"},
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(mode=intent.output_mode),
                        },
                    },
                ]
            }
        )

    if isinstance(intent, FetchExtractIntent):
        if intent.extract not in {"title", "head"}:
            raise ValueError("FetchExtractIntent body_text is not supported in the deterministic compiler.")

        parsed_url = urlparse(intent.url)
        if parsed_url.scheme == "file":
            _require_tools(allowed_tools, "fs.read", "python.exec")
            fetch_step = {
                "id": 1,
                "action": "fs.read",
                "args": {"path": unquote(parsed_url.path)},
                "output": "html_content",
            }
            html_ref = {"$ref": "html_content", "path": "content"}
        else:
            _require_tools(allowed_tools, "shell.exec", "python.exec")
            fetch_step = {
                "id": 1,
                "action": "shell.exec",
                "args": {"command": f"curl -sL {shlex.quote(intent.url)}"},
                "output": "html_content",
            }
            html_ref = {"$ref": "html_content", "path": "stdout"}

        return ExecutionPlan.model_validate(
            {
                "steps": [
                    fetch_step,
                    {
                        "id": 2,
                        "action": "python.exec",
                        "input": ["html_content"],
                        "args": {
                            "inputs": {"html": html_ref},
                            "code": _fetch_extract_code(intent.extract),
                        },
                        "output": "fetch_extract",
                    },
                    {
                        "id": 3,
                        "action": INTERNAL_RETURN_ACTION,
                        "input": ["fetch_extract"],
                        "args": {
                            "value": {"$ref": "fetch_extract", "path": "result"},
                            "mode": intent.output_mode,
                            "output_contract": build_output_contract(mode=intent.output_mode),
                        },
                    },
                ]
            }
        )

    raise ValueError(f"Unsupported deterministic intent type: {type(intent).__name__}")


def _compile_compound_intent(intent: CompoundIntent, allowed_tools: list[str], settings: Settings) -> ExecutionPlan:
    """Handle the internal compile compound intent helper path for this module.

    Inputs:
        Receives intent, allowed_tools, settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._compile_compound_intent.
    """
    if not intent.intents:
        raise ValueError("Compound deterministic intent requires at least one nested intent.")
    producer = intent.intents[0]
    fragment = _compile_producer_fragment(producer, allowed_tools, settings)
    saved_path: str | None = None
    saved_mode: str = "text"

    for nested in intent.intents[1:]:
        if isinstance(nested, (TransformIntent, TransformChainIntent)):
            fragment = _compile_transform_fragment(fragment, nested)
            continue
        if isinstance(nested, WriteResultIntent):
            fragment = _ensure_fragment_mode(fragment, nested.output_mode)
            fragment = _append_write_step(fragment, nested.path)
            saved_path = nested.path
            saved_mode = nested.output_mode
            continue
        raise ValueError(f"Unsupported compound nested intent: {type(nested).__name__}")

    steps = list(fragment.steps)
    next_id = len(steps) + 1
    if intent.return_policy == "return_written_file":
        if not saved_path:
            raise ValueError("return_written_file compound intent requires a saved path.")
        _require_tools(allowed_tools, "fs.read")
        steps.append({"id": next_id, "action": "fs.read", "args": {"path": saved_path}, "output": "saved_file_content"})
        next_id += 1
        steps.append(
            {
                "id": next_id,
                "action": INTERNAL_RETURN_ACTION,
                "input": ["saved_file_content"],
                "args": {
                    "value": {"$ref": "saved_file_content", "path": "content"},
                    "mode": "text",
                    "output_contract": build_output_contract(mode="text"),
                },
            }
        )
    elif intent.return_policy in {"return_last", "return_original_result"}:
        steps.append(
            {
                "id": next_id,
                "action": INTERNAL_RETURN_ACTION,
                "input": [fragment.output_alias],
                "args": {
                    "value": fragment.value_ref,
                    "mode": fragment.return_mode if intent.return_policy == "return_last" else saved_mode if saved_path and fragment.return_mode == "text" and saved_mode != "text" else fragment.return_mode,
                    "output_contract": _output_contract_for_fragment(fragment),
                },
            }
        )
    else:
        raise ValueError(f"Unsupported compound return policy: {intent.return_policy}")
    return ExecutionPlan.model_validate({"steps": steps})


def _compile_producer_fragment(intent: Any, allowed_tools: list[str], settings: Settings) -> CompiledFragment:
    """Handle the internal compile producer fragment helper path for this module.

    Inputs:
        Receives intent, allowed_tools, settings for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._compile_producer_fragment.
    """
    if isinstance(intent, CountFilesIntent):
        action, args, output = _file_discovery_step(intent.path, intent.pattern, recursive=intent.recursive)
        _require_tools(allowed_tools, action)
        return CompiledFragment(
            steps=[
                {"id": 1, "action": action, "args": args, "output": "file_matches"},
                {
                    "id": 2,
                    "action": INTERNAL_RETURN_ACTION,
                    "input": ["file_matches"],
                    "output": "count_result",
                    "args": {
                        "value": {"$ref": "file_matches", "path": "matches"},
                        "mode": "count",
                        "output_contract": build_output_contract(mode="count"),
                    },
                },
            ],
            output_alias="count_result",
            output_kind="scalar",
            value_ref={"$ref": "count_result", "path": "value"},
            return_mode="count",
        )

    if isinstance(intent, ListFilesIntent):
        action, args, output = _file_discovery_step(
            intent.path,
            intent.pattern,
            recursive=intent.recursive,
            file_only=intent.file_only,
            dir_only=intent.dir_only,
            path_style=intent.path_style,
        )
        _require_tools(allowed_tools, action)
        steps: list[dict[str, Any]] = [{"id": 1, "action": action, "args": args, "output": "list_result"}]
        fragment = CompiledFragment(
            steps=steps,
            output_alias="list_result",
            output_kind="matches",
            value_ref={"$ref": "list_result", "path": "matches"},
            return_mode="text",
            path_style=intent.path_style,
        )
        if intent.output_mode != "text":
            fragment = _append_shape_formatter(fragment, "transform1_result", intent.output_mode)
        return fragment

    if isinstance(intent, ReadFileLineIntent):
        _require_tools(allowed_tools, "fs.read", "python.exec")
        return CompiledFragment(
            steps=[
                {"id": 1, "action": "fs.read", "args": {"path": intent.path}, "output": "file_text"},
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["file_text"],
                    "output": "read_result",
                    "args": {
                        "inputs": {"text": {"$ref": "file_text", "path": "content"}},
                        "code": f"lines = inputs['text'].splitlines(); result = lines[{intent.line_number - 1}] if len(lines) >= {intent.line_number} else ''",
                    },
                },
            ],
            output_alias="read_result",
            output_kind="text",
            value_ref={"$ref": "read_result", "path": "result"},
            return_mode="text",
        )

    if isinstance(intent, SqlCountIntent):
        _require_tools(allowed_tools, "sql.query")
        database_name, _ = resolve_database_selection(settings, intent.database)
        query = f"SELECT COUNT(*) AS count_value FROM {intent.table}"
        if intent.where:
            query = f"{query} WHERE {intent.where}"
        return CompiledFragment(
            steps=[
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": database_name, "query": query},
                    "output": "count_rows",
                },
                {
                    "id": 2,
                    "action": INTERNAL_RETURN_ACTION,
                    "input": ["count_rows"],
                    "output": "count_result",
                    "args": {"value": {"$ref": "count_rows", "path": "rows.0.count_value"}, "mode": "count"},
                },
            ],
            output_alias="count_result",
            output_kind="scalar",
            value_ref={"$ref": "count_result", "path": "value"},
            return_mode="count",
        )

    if isinstance(intent, SqlSelectIntent):
        _require_tools(allowed_tools, "sql.query")
        database_name, _ = resolve_database_selection(settings, intent.database)
        query = f"SELECT {', '.join(intent.columns)} FROM {intent.table}"
        if intent.where:
            query = f"{query} WHERE {intent.where}"
        if intent.order_by:
            query = f"{query} ORDER BY {intent.order_by}"
        if intent.limit is not None:
            query = f"{query} LIMIT {intent.limit}"
        fragment = CompiledFragment(
            steps=[
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": database_name, "query": query},
                    "output": "sql_result",
                }
            ],
            output_alias="sql_result",
            output_kind="rows",
            value_ref={"$ref": "sql_result", "path": "rows"},
            return_mode="text",
        )
        if intent.output_mode != "text":
            fragment = _append_shape_formatter(fragment, "transform1_result", intent.output_mode)
        return fragment

    if isinstance(intent, ShellCommandIntent):
        _require_tools(allowed_tools, "shell.exec")
        command_args: dict[str, Any] = {"command": intent.command}
        if intent.node:
            command_args["node"] = intent.node
        fragment = CompiledFragment(
            steps=[{"id": 1, "action": "shell.exec", "args": command_args, "output": "shell_result"}],
            output_alias="shell_result",
            output_kind="text",
            value_ref={"$ref": "shell_result", "path": "stdout"},
            return_mode="text",
        )
        if intent.output_mode != "text":
            fragment = _append_shape_formatter(fragment, "transform1_result", intent.output_mode)
        return fragment

    raise ValueError(f"Unsupported compound producer type: {type(intent).__name__}")


def _compile_transform_fragment(fragment: CompiledFragment, intent: TransformIntent | TransformChainIntent) -> CompiledFragment:
    """Handle the internal compile transform fragment helper path for this module.

    Inputs:
        Receives fragment, intent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._compile_transform_fragment.
    """
    operations = [intent.operation] if isinstance(intent, TransformIntent) else list(intent.operations)
    current = fragment
    for index, operation in enumerate(operations):
        alias = intent.output_alias if index == len(operations) - 1 and intent.output_alias else _next_transform_alias(current)
        if operation in {"uppercase", "lowercase", "titlecase"}:
            current = _append_case_transform(current, operation, alias)
            continue
        if operation in {"csv", "newline_text", "json", "count"}:
            mode = "text" if operation == "newline_text" else operation
            current = _append_shape_formatter(current, alias, mode)
            continue
        raise ValueError(f"Unsupported transform operation: {operation}")
    return current


def _append_case_transform(fragment: CompiledFragment, operation: str, output_alias: str) -> CompiledFragment:
    """Handle the internal append case transform helper path for this module.

    Inputs:
        Receives fragment, operation, output_alias for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._append_case_transform.
    """
    code = _case_transform_code(operation, fragment.output_kind)
    steps = list(fragment.steps)
    steps.append(
        {
            "id": len(steps) + 1,
            "action": "python.exec",
            "input": [fragment.output_alias],
            "output": output_alias,
            "args": {
                "inputs": {"value": fragment.value_ref},
                "code": code,
            },
        }
    )
    output_kind = "text"
    if fragment.output_kind in {"matches", "rows", "values"}:
        output_kind = "values"
    return CompiledFragment(
        steps=steps,
        output_alias=output_alias,
        output_kind=output_kind,
        value_ref={"$ref": output_alias, "path": "result"},
        return_mode="text",
        path_style=fragment.path_style,
    )


def _append_shape_formatter(fragment: CompiledFragment, output_alias: str, mode: str) -> CompiledFragment:
    """Handle the internal append shape formatter helper path for this module.

    Inputs:
        Receives fragment, output_alias, mode for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._append_shape_formatter.
    """
    steps = list(fragment.steps)
    steps.append(
        {
            "id": len(steps) + 1,
            "action": INTERNAL_RETURN_ACTION,
            "input": [fragment.output_alias],
            "output": output_alias,
            "args": {
                "value": fragment.value_ref,
                "mode": mode,
                "output_contract": _output_contract_for_fragment(fragment, mode=mode),
            },
        }
    )
    output_kind = "scalar" if mode == "count" else "json" if mode == "json" else "text"
    return CompiledFragment(
        steps=steps,
        output_alias=output_alias,
        output_kind=output_kind,
        value_ref={"$ref": output_alias, "path": "value"},
        return_mode=mode,
        path_style=fragment.path_style,
    )


def _append_write_step(fragment: CompiledFragment, path: str) -> CompiledFragment:
    """Handle the internal append write step helper path for this module.

    Inputs:
        Receives fragment, path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._append_write_step.
    """
    steps = list(fragment.steps)
    steps.append(
        {
            "id": len(steps) + 1,
            "action": "fs.write",
            "input": [fragment.output_alias],
            "args": {"path": path, "content": {"$ref": fragment.output_alias}},
        }
    )
    return CompiledFragment(
        steps=steps,
        output_alias=fragment.output_alias,
        output_kind=fragment.output_kind,
        value_ref=fragment.value_ref,
        return_mode=fragment.return_mode,
        path_style=fragment.path_style,
    )


def _ensure_fragment_mode(fragment: CompiledFragment, mode: str) -> CompiledFragment:
    """Handle the internal ensure fragment mode helper path for this module.

    Inputs:
        Receives fragment, mode for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._ensure_fragment_mode.
    """
    desired_mode = "text" if mode == "newline_text" else mode
    if fragment.output_kind == "text" and desired_mode == "text":
        return fragment
    if fragment.output_kind == "scalar" and desired_mode == "count":
        return fragment
    if fragment.output_kind == "json" and desired_mode == "json":
        return fragment
    if fragment.output_kind == "text" and fragment.return_mode == desired_mode and desired_mode in {"csv", "json", "count"}:
        return fragment
    return _append_shape_formatter(fragment, _next_transform_alias(fragment), desired_mode)


def _case_transform_code(operation: str, input_kind: str) -> str:
    """Handle the internal case transform code helper path for this module.

    Inputs:
        Receives operation, input_kind for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._case_transform_code.
    """
    transform_expression = {
        "uppercase": "str(value).upper()",
        "lowercase": "str(value).lower()",
        "titlecase": "str(value).title()",
    }[operation]
    if input_kind == "rows":
        item_expression = transform_expression.replace("value", "row[key]")
        return (
            "value = inputs['value'];\n"
            "key = list(value[0].keys())[0] if value else '';\n"
            f"result = [{item_expression} for row in value] if value else []\n"
        )
    if input_kind in {"matches", "values"}:
        item_expression = transform_expression.replace("value", "item")
        return (
            "value = inputs['value'];\n"
            f"result = [{item_expression} for item in value]\n"
        )
    return "value = inputs['value']; result = " + transform_expression


def _next_transform_alias(fragment: CompiledFragment) -> str:
    """Handle the internal next transform alias helper path for this module.

    Inputs:
        Receives fragment for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._next_transform_alias.
    """
    existing = [step.get("output") for step in fragment.steps if step.get("output")]
    transform_count = sum(1 for alias in existing if str(alias).startswith("transform"))
    return f"transform{transform_count + 1}_result"


def _fetch_extract_code(extract: str) -> str:
    """Handle the internal fetch extract code helper path for this module.

    Inputs:
        Receives extract for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._fetch_extract_code.
    """
    if extract == "title":
        pattern = r"<title[^>]*>(.*?)</title>"
        group_expression = "match.group(1).strip()"
    elif extract == "head":
        pattern = r"(<head[^>]*>.*?</head>)"
        group_expression = "match.group(1).strip()"
    else:
        raise ValueError(f"Unsupported fetch extraction: {extract}")
    return (
        "import re\n"
        "html = str(inputs.get('html') or '')\n"
        f"match = re.search({pattern!r}, html, re.IGNORECASE | re.DOTALL)\n"
        f"result = {group_expression} if match else ''\n"
    )


def _output_contract_for_fragment(fragment: CompiledFragment, *, mode: str | None = None) -> dict[str, Any]:
    """Handle the internal output contract for fragment helper path for this module.

    Inputs:
        Receives fragment, mode for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._output_contract_for_fragment.
    """
    contract_mode = mode or fragment.return_mode
    json_shape = None
    if contract_mode == "json":
        if fragment.output_kind == "json":
            json_shape = None
        elif fragment.output_kind == "matches":
            json_shape = "matches"
        elif fragment.output_kind == "rows":
            json_shape = "rows"
        elif fragment.output_kind == "scalar":
            json_shape = "count"
        else:
            json_shape = "value"
    return build_output_contract(
        mode=contract_mode,
        path_style=fragment.path_style,
        json_shape=json_shape,
    )


def _file_discovery_step(
    path: str,
    pattern: str | None,
    *,
    recursive: bool,
    file_only: bool = True,
    dir_only: bool = False,
    path_style: str = "relative",
) -> tuple[str, dict[str, Any], str]:
    """Handle the internal file discovery step helper path for this module.

    Inputs:
        Receives path, pattern, recursive, file_only, dir_only, path_style for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._file_discovery_step.
    """
    normalized_pattern = pattern or "*"
    if recursive and not dir_only:
        return "fs.find", {"path": path, "pattern": normalized_pattern}, "file_matches"
    return (
        "fs.glob",
        {
            "path": path,
            "pattern": normalized_pattern,
            "recursive": recursive,
            "file_only": file_only,
            "dir_only": dir_only,
            "path_style": path_style,
        },
        "file_matches",
    )


def _require_tools(allowed_tools: list[str], *required: str) -> None:
    """Handle the internal require tools helper path for this module.

    Inputs:
        Receives allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.intent_compiler._require_tools.
    """
    missing = [name for name in required if name not in allowed_tools]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Deterministic intent requires unavailable tools: {missing_text}")
