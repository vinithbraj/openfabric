"""Result storage interfaces."""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.core.ids import new_id
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import DataRef, ExecutionResult, InputRef


class InMemoryResultStore:
    """Small in-memory result store for tests and placeholders."""

    def __init__(self, default_preview_bytes: int = 4096) -> None:
        self._results: list[ExecutionResult] = []
        self._refs: dict[str, DataRef] = {}
        self._data: dict[str, Any] = {}
        self._node_ref_ids: dict[str, str] = {}
        self._default_preview_bytes = max(1, int(default_preview_bytes))

    def add(self, result: ExecutionResult) -> None:
        """Store one result."""

        self._results.append(result)

    def list(self) -> list[ExecutionResult]:
        """Return stored results."""

        return list(self._results)

    def put(
        self,
        node_id: str,
        data: Any,
        data_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> DataRef:
        """Store one payload and return a typed data reference."""

        ref_id = new_id("data")
        preview = self.preview(data, self._default_preview_bytes)
        data_ref = DataRef(
            ref_id=ref_id,
            producer_node_id=str(node_id),
            data_type=str(data_type or "unknown"),
            preview=preview,
            metadata=dict(metadata or {}),
        )
        self._refs[ref_id] = data_ref
        self._data[ref_id] = data
        self._node_ref_ids[str(node_id)] = ref_id
        return data_ref

    def get(self, data_ref: str) -> Any:
        """Return stored result data by reference."""

        return self._data[data_ref]

    def get_data_ref(self, ref_id: str) -> DataRef:
        """Return typed metadata for one stored reference."""

        return self._refs[ref_id]

    def resolve_input_ref(self, input_ref: InputRef) -> Any:
        """Resolve one typed input reference into stored full data."""

        source_node_id = str(input_ref.source_node_id)
        ref_id = self._node_ref_ids.get(source_node_id)
        if ref_id is None:
            raise ValidationError(f"Producer output is not available for node {source_node_id}.")

        data_ref = self._refs[ref_id]
        if input_ref.expected_data_type and data_ref.data_type != input_ref.expected_data_type:
            raise ValidationError(
                f"InputRef expected data_type {input_ref.expected_data_type!r} "
                f"but producer {source_node_id} has {data_ref.data_type!r}."
            )

        data = self._data[ref_id]
        if input_ref.output_key is None:
            return data
        if not isinstance(data, dict):
            raise ValidationError(
                f"Producer output for node {source_node_id} is not key-addressable."
            )
        if input_ref.output_key not in data:
            raise ValidationError(
                f"Producer output for node {source_node_id} does not contain key {input_ref.output_key!r}."
            )
        return data[input_ref.output_key]

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
