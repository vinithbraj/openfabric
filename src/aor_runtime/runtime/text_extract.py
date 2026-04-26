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
    text = str(prompt or "").strip()
    if not text:
        return None
    for pattern in QUOTED_CONTENT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return str(match.group("content") or "")
    return None
