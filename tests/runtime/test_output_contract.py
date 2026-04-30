from __future__ import annotations

from aor_runtime.runtime.output_contract import OutputContract, normalize_output, render_output
from aor_runtime.runtime.output_shape import infer_goal_output_contract


def test_list_to_csv_output_contract() -> None:
    contract = OutputContract(mode="csv")
    value = normalize_output(["alpha", "beta"], contract)
    assert value == ["alpha", "beta"]
    assert render_output(value, contract) == "alpha,beta"


def test_list_to_newline_text_output_contract() -> None:
    contract = OutputContract(mode="text")
    value = normalize_output(["alpha", "beta"], contract)
    assert value == ["alpha", "beta"]
    assert render_output(value, contract) == "alpha\nbeta"


def test_list_to_matches_json_output_contract() -> None:
    contract = OutputContract(mode="json", json_shape="matches")
    value = normalize_output(["alpha.txt", "beta.txt"], contract)
    assert value == {"matches": ["alpha.txt", "beta.txt"]}


def test_rows_to_rows_json_output_contract() -> None:
    contract = OutputContract(mode="json", json_shape="rows")
    rows = [{"name": "Alice"}, {"name": "Bob"}]
    value = normalize_output(rows, contract)
    assert value == {"rows": rows}


def test_count_to_scalar_and_count_json_output_contract() -> None:
    scalar_contract = OutputContract(mode="count")
    json_contract = OutputContract(mode="json", json_shape="count")
    assert normalize_output(["a", "b", "c"], scalar_contract) == 3
    assert normalize_output(["a", "b", "c"], json_contract) == {"count": 3}


def test_rows_render_to_csv_when_explicitly_requested() -> None:
    contract = OutputContract(mode="csv", json_shape="rows")
    rows = [{"job_id": "12345", "user": "alice"}, {"job_id": "12346", "user": "bob"}]
    value = normalize_output(rows, contract)
    assert value == rows
    assert render_output(value, contract) == "job_id,user\n12345,alice\n12346,bob"


def test_rows_render_to_text_table_when_explicitly_requested() -> None:
    contract = OutputContract(mode="text", json_shape="rows")
    rows = [{"job_id": "12345", "user": "alice"}, {"job_id": "12346", "user": "bob"}]
    value = normalize_output(rows, contract)
    assert value == rows
    assert "job_id | user " in render_output(value, contract)
    assert "12345  | alice" in render_output(value, contract)


def test_grouped_count_goals_infer_table_contract() -> None:
    assert infer_goal_output_contract("count of jobs in each slurm partition").kind == "table"
    assert infer_goal_output_contract("count jobs by state").kind == "table"
    assert infer_goal_output_contract("count jobs per user").kind == "table"


def test_filtered_count_goal_stays_scalar_contract() -> None:
    assert infer_goal_output_contract("count of jobs in totalseg partition").kind == "scalar"
