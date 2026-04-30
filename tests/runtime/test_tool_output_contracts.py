from __future__ import annotations

from aor_runtime.runtime.tool_output_contracts import (
    available_paths_for_tool,
    default_path_for_tool,
    formatter_source_path_for_tool,
    normalize_tool_ref_path,
    path_is_declared_for_tool,
    return_value_path_for_tool,
)


def test_core_tool_output_contract_defaults() -> None:
    assert default_path_for_tool("sql.query") == "rows"
    assert default_path_for_tool("sql.schema") == "catalog"
    assert default_path_for_tool("text.format") == "content"
    assert default_path_for_tool("shell.exec") == "stdout"
    assert default_path_for_tool("slurm.metrics") == "payload"
    assert default_path_for_tool("fs.search_content") == "matches"
    assert default_path_for_tool("sql.validate") == "explanation"


def test_formatter_and_return_paths_use_declared_contracts() -> None:
    assert formatter_source_path_for_tool("sql.query") == "rows"
    assert formatter_source_path_for_tool("slurm.queue") == "jobs"
    assert return_value_path_for_tool("text.format") == "content"
    assert return_value_path_for_tool("fs.write") == "path"


def test_path_aliases_are_normalized() -> None:
    assert normalize_tool_ref_path("shell.exec", "exit_code") == "returncode"
    assert normalize_tool_ref_path("slurm.accounting_aggregate", "grouped") == "groups"


def test_path_validation_allows_nested_declared_roots() -> None:
    assert path_is_declared_for_tool("sql.query", "rows")
    assert path_is_declared_for_tool("sql.query", "rows.0.PatientID")
    assert not path_is_declared_for_tool("sql.query", "patient_study_counts")
    assert "rows" in available_paths_for_tool("sql.query")
