"""OpenFABRIC Runtime Module: aor_runtime.runtime.text_extract

Purpose:
    Extract text safely from supported inputs for formatting or search flows.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re


QUOTED_CONTENT_PATTERNS = (
    re.compile(r"\bexact\s+text\s+(?P<quote>['\"])(?P<content>.+?)(?P=quote)", re.IGNORECASE),
    re.compile(r"\bexact\s+content\s+(?P<quote>['\"])(?P<content>.+?)(?P=quote)", re.IGNORECASE),
    re.compile(r"\bcontent\s+(?P<quote>['\"])(?P<content>.+?)(?P=quote)", re.IGNORECASE),
    re.compile(r"\bcontaining\s+(?P<quote>['\"])(?P<content>.+?)(?P=quote)", re.IGNORECASE),
    re.compile(r"\bwrite\s+(?:the\s+exact\s+text\s+)?(?P<quote>['\"])(?P<content>.+?)(?P=quote)\s+to\b", re.IGNORECASE),
    re.compile(r"\bsave\s+(?P<quote>['\"])(?P<content>.+?)(?P=quote)\s+to\b", re.IGNORECASE),
)


def extract_quoted_content(prompt: str) -> str | None:
    """Extract quoted content for the surrounding runtime workflow.

    Inputs:
        Receives prompt for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.text_extract.extract_quoted_content.
    """
    text = str(prompt or "").strip()
    if not text:
        return None
    for pattern in QUOTED_CONTENT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return str(match.group("content") or "")
    return None
