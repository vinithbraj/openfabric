"""OpenFABRIC Runtime Module: aor_runtime.core.utils

Purpose:
    Provide small shared serialization and parsing utilities.

Responsibilities:
    Define stable data structures and helpers that keep planning, execution, validation, and persistence aligned.

Data flow / Interfaces:
    Exports dataclasses, models, and utility functions consumed by runtime, tools, API, and tests.

Boundaries:
    Keeps shared primitives dependency-light and free of domain-specific execution policy.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
CODE_KEY_RE = re.compile(r'"code"\s*:\s*"')
VALID_JSON_ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def ensure_jsonable(value: Any) -> Any:
    """Ensure jsonable for the surrounding runtime workflow.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils.ensure_jsonable.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): ensure_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [ensure_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [ensure_jsonable(item) for item in value]
    return value


def dumps_json(value: Any, *, indent: int | None = None) -> str:
    """Dumps json for the surrounding runtime workflow.

    Inputs:
        Receives value, indent for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils.dumps_json.
    """
    return json.dumps(ensure_jsonable(value), ensure_ascii=False, indent=indent, default=str)


def loads_json(text: str) -> Any:
    """Loads json for the surrounding runtime workflow.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils.loads_json.
    """
    return json.loads(text)


def extract_json_object(text: str) -> Any:
    """Extract json object for the surrounding runtime workflow.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils.extract_json_object.
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("Empty LLM response")

    fenced = JSON_BLOCK_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for candidate in _repair_json_candidates(cleaned):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            for candidate in _repair_json_candidates(snippet, include_original=True):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            for candidate in _repair_json_candidates(snippet, include_original=True):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        raise


def _repair_json_candidates(text: str, *, include_original: bool = False) -> list[str]:
    """Handle the internal repair json candidates helper path for this module.

    Inputs:
        Receives text, include_original for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils._repair_json_candidates.
    """
    candidates: list[str] = [text] if include_original else []
    multiline_fixed = _escape_multiline_code_strings(text)
    control_fixed = _escape_control_chars_in_strings(text)
    invalid_escape_fixed = _repair_invalid_json_string_escapes(text)
    combined_fixed = _repair_invalid_json_string_escapes(multiline_fixed)
    combined_control_fixed = _repair_invalid_json_string_escapes(control_fixed)
    for candidate in (multiline_fixed, control_fixed, invalid_escape_fixed, combined_fixed, combined_control_fixed):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _escape_control_chars_in_strings(text: str) -> str:
    """Handle the internal escape control chars in strings helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils._escape_control_chars_in_strings.
    """
    chars: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if not in_string:
            chars.append(ch)
            if ch == '"':
                in_string = True
            continue
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            chars.append(ch)
            in_string = False
            continue
        if ch == "\n":
            chars.append("\\n")
            continue
        if ch == "\r":
            chars.append("\\r")
            continue
        if ch == "\t":
            chars.append("\\t")
            continue
        chars.append(ch)
    return "".join(chars)


def _escape_multiline_code_strings(text: str) -> str:
    """Handle the internal escape multiline code strings helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils._escape_multiline_code_strings.
    """
    if '"code"' not in text:
        return text

    chars: list[str] = []
    index = 0
    while index < len(text):
        match = CODE_KEY_RE.search(text, index)
        if match is None:
            chars.append(text[index:])
            break
        chars.append(text[index : match.end()])
        index = match.end()
        escaped = False
        while index < len(text):
            ch = text[index]
            if ch == "\\" and not escaped:
                chars.append(ch)
                escaped = True
            elif ch == '"' and not escaped:
                chars.append(ch)
                index += 1
                break
            elif ch == "\n":
                chars.append("\\n")
                escaped = False
            elif ch == "\r":
                escaped = False
            else:
                chars.append(ch)
                escaped = False
            index += 1
    return "".join(chars)


def _repair_invalid_json_string_escapes(text: str) -> str:
    """Handle the internal repair invalid json string escapes helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils._repair_invalid_json_string_escapes.
    """
    chars: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        ch = text[index]
        if not in_string:
            chars.append(ch)
            if ch == '"':
                in_string = True
            index += 1
            continue

        if ch == '"':
            chars.append(ch)
            in_string = False
            index += 1
            continue

        if ch == "\\":
            if index + 1 >= len(text):
                chars.append(ch)
                index += 1
                continue
            next_char = text[index + 1]
            if next_char in VALID_JSON_ESCAPE_CHARS:
                chars.append(ch)
                chars.append(next_char)
            else:
                chars.append(next_char)
            index += 2
            continue

        chars.append(ch)
        index += 1

    return "".join(chars)


ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BoolOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.UnaryOp,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Dict,
    ast.List,
    ast.Tuple,
}

SAFE_GLOBALS = {"len": len, "min": min, "max": max, "sum": sum, "any": any, "all": all}


def safe_eval_condition(expression: str, context: dict[str, Any]) -> bool:
    """Safe eval condition for the surrounding runtime workflow.

    Inputs:
        Receives expression, context for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by shared runtime contracts code paths that import or call aor_runtime.core.utils.safe_eval_condition.
    """
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if type(node) not in ALLOWED_AST_NODES:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
    return bool(eval(compile(parsed, "<condition>", "eval"), {"__builtins__": {}}, {**SAFE_GLOBALS, **context}))
