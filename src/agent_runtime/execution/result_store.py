"""Result storage interfaces."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.core.ids import new_id
from agent_runtime.core.types import ExecutionResult


class InMemoryResultStore:
    """Small in-memory result store for tests and placeholders."""

    def __init__(self) -> None:
        self._results: list[ExecutionResult] = []
        self._data: dict[str, Any] = {}

    def add(self, result: ExecutionResult) -> None:
        """Store one result."""

        self._results.append(result)

    def list(self) -> list[ExecutionResult]:
        """Return stored results."""

        return list(self._results)

    def put(self, data: Any) -> str:
        """Store large result data and return an opaque reference."""

        data_ref = new_id("data")
        self._data[data_ref] = data
        return data_ref

    def get(self, data_ref: str) -> Any:
        """Return stored result data by reference."""

        return self._data[data_ref]

    def preview(self, data: Any, max_bytes: int) -> dict[str, Any]:
        """Return a bounded, JSON-safe preview for arbitrary result data."""

        serialized = json.dumps(data, default=str, ensure_ascii=True)
        encoded = serialized.encode("utf-8")
        total_bytes = len(encoded)
        if total_bytes <= max_bytes:
            if isinstance(data, dict):
                return dict(data)
            return {
                "preview_text": serialized,
                "truncated": False,
                "bytes": total_bytes,
            }

        truncated_text = encoded[:max_bytes].decode("utf-8", errors="ignore")
        return {
            "preview_text": truncated_text,
            "truncated": True,
            "bytes": total_bytes,
        }
