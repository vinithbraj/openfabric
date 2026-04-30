"""OpenFABRIC Runtime Module: aor_runtime.runtime.file_query

Purpose:
    Normalize natural-language file discovery requests into deterministic file-query specifications.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel


NON_RECURSIVE_HINTS = (
    "top-level",
    "direct",
    "directly in",
    "immediate children",
    "immediate-child",
    "non-recursive",
    "not nested",
    "only in this folder",
)
RECURSIVE_HINTS = ("under", "recursively", "anywhere below", "inside this tree")
TRAILING_PUNCTUATION_RE = re.compile(r"[?!,:;]+$")
PATH_CANDIDATE_RE = re.compile(r"(~?/[^,\s]+|\.\.?/[^,\s]+|[A-Za-z]:[\\/][^,\s]+)")


class FileQuerySpec(BaseModel):
    """Represent file query spec within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileQuerySpec.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.file_query.FileQuerySpec and related tests.
    """
    root_path: str
    pattern: str = "*"
    recursive: bool = True
    file_only: bool = True
    dir_only: bool = False
    path_style: Literal["name", "relative", "absolute"] = "relative"


def normalize_file_query(prompt: str, explicit_path: str | None = None) -> FileQuerySpec:
    """Normalize file query for the surrounding runtime workflow.

    Inputs:
        Receives prompt, explicit_path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.file_query.normalize_file_query.
    """
    text = str(prompt or "").strip()
    normalized = text.lower()
    root_path = _clean_path(explicit_path or _extract_path_candidate(text))
    pattern = _normalize_pattern(normalized)
    recursive = _infer_recursive(normalized)
    path_style = _infer_path_style(normalized)
    return FileQuerySpec(root_path=root_path, pattern=pattern, recursive=recursive, path_style=path_style)


def _normalize_pattern(text: str) -> str:
    """Handle the internal normalize pattern helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.file_query._normalize_pattern.
    """
    if re.search(r"\b(?:txt files|\.txt files|txt filenames|\.txt filenames|text files|files ending in txt|files with txt extension)\b", text):
        return "*.txt"
    if "*.txt" in text:
        return "*.txt"
    if re.search(r"\b(?:md files|\.md files|md filenames|\.md filenames|markdown files)\b", text) or "*.md" in text:
        return "*.md"

    match = re.search(r"\*\.(?P<ext>[a-z0-9]+)\b", text)
    if match:
        return f"*.{match.group('ext').lower()}"
    match = re.search(r"\.(?P<ext>[a-z0-9]+)\s+files?\b", text)
    if match:
        return f"*.{match.group('ext').lower()}"
    match = re.search(r"\bfiles?\s+ending\s+in\s+(?P<ext>[a-z0-9]+)\b", text)
    if match:
        return f"*.{match.group('ext').lower()}"
    match = re.search(r"\bfiles?\s+with\s+(?P<ext>[a-z0-9]+)\s+extension\b", text)
    if match:
        return f"*.{match.group('ext').lower()}"
    return "*"


def _infer_recursive(text: str) -> bool:
    """Handle the internal infer recursive helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.file_query._infer_recursive.
    """
    if any(hint in text for hint in NON_RECURSIVE_HINTS):
        return False
    if any(hint in text for hint in RECURSIVE_HINTS):
        return True
    return True


def _infer_path_style(text: str) -> Literal["name", "relative", "absolute"]:
    """Handle the internal infer path style helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.file_query._infer_path_style.
    """
    if any(phrase in text for phrase in ("filenames only", "names only", "matching filenames")):
        return "name"
    if re.search(r"\b(?:txt filenames|md filenames|filenames)\b", text):
        return "name"
    if "absolute paths" in text or "full paths" in text:
        return "absolute"
    if "relative paths" in text:
        return "relative"
    return "relative"


def _extract_path_candidate(text: str) -> str:
    """Handle the internal extract path candidate helper path for this module.

    Inputs:
        Receives text for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.file_query._extract_path_candidate.
    """
    match = PATH_CANDIDATE_RE.search(text)
    if match is None:
        return ""
    return match.group(0)


def _clean_path(value: str | None) -> str:
    """Handle the internal clean path helper path for this module.

    Inputs:
        Receives value for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.file_query._clean_path.
    """
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    candidate = TRAILING_PUNCTUATION_RE.sub("", candidate)
    while candidate.endswith(".") and len(candidate) > 1:
        candidate = candidate[:-1]
    return candidate.strip()
