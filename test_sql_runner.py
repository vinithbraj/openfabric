import unittest
import sys
import types
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
sys.modules.setdefault("fastapi", fastapi_stub)

from agent_library.agents.sql_runner import (
    _execute_sql,
    _postgres_safe_sql,
    _repair_sql_with_retries,
    _same_query_specs,
    _sql_repair_max_attempts,
)


POSTGRES_SCHEMA = {
    "dialect": "postgres",
    "tables": [
        {
            "schema": "flathr",
            "name": "Patients",
            "columns": [
                {"name": "PatientId", "type": "uuid", "nullable": False},
                {"name": "LastName", "type": "text", "nullable": True},
            ],
            "foreign_keys": [],
        }
    ],
}


class PostgresSafeSqlTests(unittest.TestCase):
    def test_quotes_mixed_case_schema_table_and_columns(self):
        sql = "select PatientId, LastName from flathr.Patients where PatientId is not null"
        expected = 'select "PatientId", "LastName" from "flathr"."Patients" where "PatientId" is not null'
        self.assertEqual(_postgres_safe_sql(sql, POSTGRES_SCHEMA), expected)

    def test_leaves_strings_and_functions_untouched(self):
        sql = "select count(PatientId) from flathr.Patients where LastName = 'Patients'"
        expected = 'select count("PatientId") from "flathr"."Patients" where "LastName" = \'Patients\''
        self.assertEqual(_postgres_safe_sql(sql, POSTGRES_SCHEMA), expected)

    def test_rolls_back_after_failed_execute(self):
        class FakeCursor:
            def __init__(self):
                self.description = [("PatientId",)]
                self.calls = 0

            def execute(self, _sql):
                self.calls += 1
                raise RuntimeError("UndefinedColumn: bad column")

            def fetchmany(self, _limit):
                return []

            def close(self):
                return None

        class FakeConn:
            def __init__(self):
                self.rollback_calls = 0

            def cursor(self):
                return FakeCursor()

            def rollback(self):
                self.rollback_calls += 1

        conn = FakeConn()
        with self.assertRaisesRegex(RuntimeError, "UndefinedColumn"):
            _execute_sql(conn, "select PatientId from flathr.Patients", 10, POSTGRES_SCHEMA)
        self.assertEqual(conn.rollback_calls, 1)

    def test_detects_noop_repair_sql(self):
        original = [{"label": "Query 1", "sql": "SELECT p.patient_id FROM dicom.patients p"}]
        repaired = [{"label": "fixed", "sql": " select   p.patient_id from dicom.patients p; "}]
        self.assertTrue(_same_query_specs(original, repaired))

    def test_sql_repair_max_attempts_defaults_to_ten(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_sql_repair_max_attempts(), 10)

    def test_sql_repair_retries_up_to_max_attempts(self):
        stats = {}
        attempts = []

        def fake_execute(_conn, query_specs, _limit, _schema):
            attempts.append(query_specs[0]["sql"])
            raise RuntimeError("UndefinedColumn: still broken")

        def fake_repair(_task, _schema, _failing_sql, _error_text, previous_repair_sql="", previous_repair_error=""):
            index = len(attempts)
            return [{"label": f"repair-{index}", "sql": f"SELECT broken_{index}"}]

        with patch("agent_library.agents.sql_runner._execute_sql_queries", side_effect=fake_execute), patch(
            "agent_library.agents.sql_runner._repair_sql_query", side_effect=fake_repair
        ), patch("agent_library.agents.sql_runner._rollback_quietly"), patch.dict(
            "os.environ", {"SQL_AGENT_MAX_REPAIR_ATTEMPTS": "3"}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "UndefinedColumn"):
                _repair_sql_with_retries(
                    conn=object(),
                    schema=POSTGRES_SCHEMA,
                    query_task="count RTPLANS",
                    query_specs=[{"label": "initial", "sql": "SELECT broken_0"}],
                    limit=10,
                    stats=stats,
                )

        self.assertEqual(len(attempts), 4)
        self.assertEqual(stats["sql_repair_attempts"], 3.0)


if __name__ == "__main__":
    unittest.main()
