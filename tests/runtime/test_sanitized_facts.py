from __future__ import annotations

import json

from aor_runtime.runtime.facts import build_sanitized_facts, validate_facts_for_llm
from aor_runtime.runtime.presentation import PresentationContext


def test_slurm_raw_payload_becomes_compact_facts() -> None:
    result = {
        "results": {
            "cluster_summary": {
                "metric_group": "cluster_summary",
                "payload": {
                    "queue_count": 20,
                    "running_jobs": 2,
                    "pending_jobs": 18,
                    "node_count": 4,
                    "problematic_nodes": 1,
                    "gpu_available": True,
                    "total_gpus": 8,
                },
            }
        },
        "stdout": "raw command output",
        "coverage": {"internal": True},
        "slurm_semantic_frame": {"internal": True},
    }

    facts = build_sanitized_facts(result, [], context=PresentationContext(source_action="slurm.metrics"))
    encoded = json.dumps(facts, sort_keys=True)

    assert facts["domain"] == "slurm"
    assert facts["queue"]["running_jobs"] == 2
    assert facts["queue"]["pending_jobs"] == 18
    assert facts["nodes"]["problematic_nodes"] == 1
    assert facts["gpu"]["available"] is True
    assert "stdout" not in encoded
    assert "coverage" not in encoded
    assert "semantic_frame" not in encoded


def test_sql_rows_are_excluded_by_default() -> None:
    result = {
        "database": "dicom",
        "rows": [{"PatientName": "Alice"}, {"PatientName": "Bob"}],
        "row_count": 2,
        "coverage": {"internal": True},
    }

    facts = build_sanitized_facts(result, [], context=PresentationContext(source_action="sql.query"))
    encoded = json.dumps(facts, sort_keys=True)

    assert facts["domain"] == "sql"
    assert facts["row_count"] == 2
    assert "Alice" not in encoded
    assert "Bob" not in encoded
    assert "PatientName" not in encoded
    assert "coverage" not in encoded


def test_filesystem_paths_are_redacted_unless_allowed() -> None:
    result = {
        "path": "/home/user/private",
        "pattern": "*.mp4",
        "file_count": 3,
        "total_size_bytes": 123,
    }

    redacted = build_sanitized_facts(result, [], context=PresentationContext(source_action="fs.aggregate"))
    included = build_sanitized_facts(result, [], context=PresentationContext(source_action="fs.aggregate", include_paths=True))

    assert redacted["path_scope"] == "requested path"
    assert "/home/user/private" not in json.dumps(redacted)
    assert included["path_scope"] == "/home/user/private"


def test_fact_validation_rejects_unsafe_fact_payloads() -> None:
    assert not validate_facts_for_llm({"stdout": "raw"})
    assert not validate_facts_for_llm({"stderr": "raw"})
    assert not validate_facts_for_llm({"coverage": {"internal": True}})
    assert not validate_facts_for_llm({"semantic_frame": {"internal": True}})
    assert not validate_facts_for_llm({"api_token": "secret"})
    assert not validate_facts_for_llm({"rows": [{"name": "Alice"}]})
    assert not validate_facts_for_llm({"message": "x" * 5000}, max_string_length=100)
