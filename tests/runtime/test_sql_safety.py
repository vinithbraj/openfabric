from __future__ import annotations

import pytest

from aor_runtime.runtime.sql_safety import ensure_read_only_sql, validate_read_only_sql


def test_select_allowed() -> None:
    assert validate_read_only_sql("SELECT * FROM patients").valid is True


def test_with_select_allowed() -> None:
    assert validate_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x").valid is True


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO patients VALUES (1)",
        "UPDATE patients SET name = 'x'",
        "DELETE FROM patients",
        "DROP TABLE patients",
        "ALTER TABLE patients ADD COLUMN x int",
        "CALL refresh_patients()",
        "DO $$ BEGIN END $$",
        "COPY patients TO STDOUT",
        "SELECT 1; SELECT 2",
    ],
)
def test_unsafe_sql_rejected(sql: str) -> None:
    result = validate_read_only_sql(sql)
    assert result.valid is False


def test_ensure_read_only_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Multiple SQL statements"):
        ensure_read_only_sql("SELECT * FROM patients; DROP TABLE patients")
