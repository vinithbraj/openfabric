from __future__ import annotations

import inspect
from pathlib import Path

import agent_runtime.capabilities.data as data_module
import pytest

from agent_runtime.capabilities.data import DataAggregateCapability
from agent_runtime.execution.result_store import InMemoryResultStore


def _context(tmp_path: Path, store: InMemoryResultStore) -> dict[str, object]:
    return {
        "node_id": "node-aggregate",
        "result_store": store,
        "execution_context": {"workspace_root": str(tmp_path)},
    }


def test_sum_over_direct_list_dict(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()

    result = capability.execute(
        {
            "input_ref": [{"size": 2}, {"size": 3}],
            "operation": "sum",
            "field": "size",
        },
        _context(tmp_path, store),
    )

    assert result.data_preview["value"] == 5
    assert result.data_preview["used_count"] == 2


def test_sum_over_entries_payload(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()
    data_ref = store.put(
        "node-list",
        {"entries": [{"size": 2, "type": "file"}, {"size": 3, "type": "file"}]},
        "table",
        {},
    )

    result = capability.execute(
        {
            "input_ref": data_ref.ref_id,
            "operation": "sum",
            "field": "size",
        },
        _context(tmp_path, store),
    )

    assert result.data_preview["value"] == 5


def test_count_records(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()

    result = capability.execute(
        {
            "input_ref": [{"path": "a"}, {"path": "b"}, {"path": "c"}],
            "operation": "count",
        },
        _context(tmp_path, store),
    )

    assert result.data_preview["value"] == 3


def test_count_with_filter(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()

    result = capability.execute(
        {
            "input_ref": [
                {"path": "a", "type": "file"},
                {"path": "b", "type": "directory"},
                {"path": "c", "type": "file"},
            ],
            "operation": "count",
            "filter": {"type": "file"},
        },
        _context(tmp_path, store),
    )

    assert result.data_preview["row_count"] == 3
    assert result.data_preview["value"] == 2


def test_skip_missing_non_numeric_values(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()

    result = capability.execute(
        {
            "input_ref": [{"size": 2}, {"size": "x"}, {"name": "missing"}],
            "operation": "sum",
            "field": "size",
        },
        _context(tmp_path, store),
    )

    assert result.data_preview["value"] == 2
    assert result.data_preview["used_count"] == 1
    assert result.data_preview["skipped_count"] == 2


def test_avg_min_max_work(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()
    payload = {"input_ref": [{"size": 2}, {"size": 6}], "field": "size"}

    avg = capability.execute({**payload, "operation": "avg"}, _context(tmp_path, store))
    min_result = capability.execute({**payload, "operation": "min"}, _context(tmp_path, store))
    max_result = capability.execute({**payload, "operation": "max"}, _context(tmp_path, store))

    assert avg.data_preview["value"] == 4
    assert min_result.data_preview["value"] == 2
    assert max_result.data_preview["value"] == 6


def test_arbitrary_expression_is_rejected(tmp_path: Path) -> None:
    capability = DataAggregateCapability()
    store = InMemoryResultStore()

    with pytest.raises(Exception):
        capability.execute(
            {
                "input_ref": [{"size": 2}],
                "operation": "sum",
                "field": "size + 1",
            },
            _context(tmp_path, store),
        )


def test_no_eval_is_used() -> None:
    source = inspect.getsource(data_module)

    assert "eval(" not in source
    assert "exec(" not in source
