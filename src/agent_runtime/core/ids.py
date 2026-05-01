"""Identifier helpers for runtime objects."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return a compact unique id with a readable prefix."""

    normalized = str(prefix or "id").strip().lower().replace("_", "-")
    return f"{normalized}-{uuid4().hex[:12]}"
