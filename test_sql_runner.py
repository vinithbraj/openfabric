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

requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.agents.sql_runner import (
    _parse_strict_json_object,
    _execute_sql,
    _postgres_safe_sql,
    _repair_sql_with_retries,
    _schema_tables_result,
    _should_reduce_sql_result,
    _tables_only_schema_request,
    _same_query_specs,
    _single_sql_query_spec_from_object,
    _strict_sql_query_specs_from_object,
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
    def test_tables_only_schema_request_detects_table_listing(self):
        self.assertTrue(_tables_only_schema_request("tables", "give me a list of all the tables in mydb"))
        self.assertFalse(_tables_only_schema_request("tables, columns, and relationships", "show database schema"))

    def test_schema_tables_result_returns_compact_table_rows(self):
        result = _schema_tables_result(
            {
                "dialect": "postgres",
                "tables": [
                    {"schema": "flathr", "name": "Patient", "type": "table"},
                    {"schema": "public", "name": "resources", "type": "table"},
                ],
            }
        )
        self.assertEqual(result["columns"], ["schema", "table", "type"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(
            result["rows"],
            [
                {"schema": "flathr", "table": "Patient", "type": "table"},
                {"schema": "public", "table": "resources", "type": "table"},
            ],
        )

    def test_should_reduce_sql_result_for_full_list_requests(self):
        result = {
            "columns": ["schema", "table", "type"],
            "rows": [{"schema": "flathr", "table": f"Table{i}", "type": "table"} for i in range(10)],
            "row_count": 10,
        }
        self.assertTrue(_should_reduce_sql_result("give me the full list of all tables in mydb", result))
        self.assertFalse(_should_reduce_sql_result("show top 3 tables", {"columns": ["table"], "rows": [{"table": "a"}], "row_count": 1}))

    def test_parse_strict_json_object_rejects_wrapped_sql_markdown(self):
        wrapped = 'Here is the SQL query:\\n```sql\\nSELECT 1\\n```'
        self.assertIsNone(_parse_strict_json_object(wrapped))

    def test_strict_sql_query_specs_require_exact_single_query_fields(self):
        parsed = {"sql": "SELECT 1", "reason": "count rows"}
        self.assertEqual(
            _strict_sql_query_specs_from_object(parsed),
            [{"label": "count rows", "sql": "SELECT 1"}],
        )
        self.assertEqual(_strict_sql_query_specs_from_object({"sql": "SELECT 1", "reason": "ok", "extra": "nope"}), [])

    def test_single_sql_query_spec_requires_only_sql_and_reason(self):
        self.assertEqual(
            _single_sql_query_spec_from_object({"sql": "SELECT 1", "reason": "repair"}),
            [{"label": "repair", "sql": "SELECT 1"}],
        )
        self.assertEqual(_single_sql_query_spec_from_object({"sql": "SELECT 1"}), [])

    def test_strict_sql_query_specs_require_exact_multi_query_fields(self):
        parsed = {
            "queries": [
                {"label": "a", "sql": "SELECT 1", "reason": "first"},
                {"label": "b", "sql": "SELECT 2", "reason": "second"},
            ]
        }
        self.assertEqual(
            _strict_sql_query_specs_from_object(parsed),
            [{"label": "a", "sql": "SELECT 1"}, {"label": "b", "sql": "SELECT 2"}],
        )
        invalid = {"queries": [{"label": "a", "sql": "SELECT 1", "reason": "first", "extra": "nope"}]}
        self.assertEqual(_strict_sql_query_specs_from_object(invalid), [])

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
