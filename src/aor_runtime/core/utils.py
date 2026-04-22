from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def ensure_jsonable(value: Any) -> Any:
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
    return json.dumps(ensure_jsonable(value), ensure_ascii=False, indent=indent, default=str)


def loads_json(text: str) -> Any:
    return json.loads(text)


def extract_json_object(text: str) -> Any:
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("Empty LLM response")

    fenced = JSON_BLOCK_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


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
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if type(node) not in ALLOWED_AST_NODES:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
    return bool(eval(compile(parsed, "<condition>", "eval"), {"__builtins__": {}}, {**SAFE_GLOBALS, **context}))
