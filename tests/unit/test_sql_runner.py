import unittest
import sys
import types
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def get(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.HTTPException = Exception


class _RequestStub:
    pass


fastapi_stub.Request = _RequestStub
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
    handle_event,
    _dsn_health_status,
    _parse_strict_json_object,
    _execute_sql,
    _execute_sql_deterministic_selection,
    _normalize_sql_selection,
    _postgres_safe_sql,
    _repair_sql_with_retries,
    _schema_schemas_result,
    _schema_tables_result,
    _schemas_only_schema_request,
    _should_reduce_sql_result,
    _tables_only_schema_request,
    _same_query_specs,
    _single_sql_query_spec_from_object,
    _strict_sql_query_specs_from_object,
    _sql_repair_max_attempts,
    _sql_llm_transport_settings,
)
from agent_library.common import EventRequest


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


DICOM_COUNT_SCHEMA = {
    "dialect": "postgres",
    "tables": [
        {
            "schema": "dicom",
            "name": "patients",
            "columns": [
                {"name": "patient_id", "type": "text", "nullable": False},
            ],
            "foreign_keys": [],
        },
        {
            "schema": "dicom",
            "name": "studies",
            "columns": [
                {"name": "study_uid", "type": "text", "nullable": False},
                {"name": "patient_id", "type": "text", "nullable": True},
            ],
            "foreign_keys": [],
        },
    ],
}


class PostgresSafeSqlTests(unittest.TestCase):
    def test_dsn_health_status_reports_unconfigured_agent(self):
        with patch.dict("os.environ", {}, clear=True):
            ok, payload = _dsn_health_status()
        self.assertFalse(ok)
        self.assertFalse(payload["configured"])
        self.assertIn("not configured", payload["detail"].lower())

    def test_dsn_health_status_reports_invalid_scheme(self):
        with patch.dict("os.environ", {"SQL_AGENT_DSN": "${SQL_DICOM_MOCK_DATABASE_URL:-postgresql://broken}"}, clear=True):
            ok, payload = _dsn_health_status()
        self.assertFalse(ok)
        self.assertEqual(payload["dsn_scheme"], "unknown")
        self.assertIn("unsupported sql_agent_dsn scheme", payload["detail"].lower())

    def test_sql_llm_transport_uses_dummy_key_for_local_base_url(self):
        with patch.dict(
            "os.environ",
            {"LLM_OPS_BASE_URL": "http://127.0.0.1:8000/v1"},
            clear=True,
        ), patch("agent_library.common._list_openai_compatible_models", return_value=()):
            api_key, base_url, timeout_seconds, model = _sql_llm_transport_settings("gpt-4o-mini")
        self.assertEqual(api_key, "dummy")
        self.assertEqual(base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(timeout_seconds, 300.0)
        self.assertEqual(model, "gpt-4o-mini")

    def test_sql_llm_transport_defaults_to_shared_local_transport_without_key(self):
        with patch.dict("os.environ", {}, clear=True), patch("agent_library.common._list_openai_compatible_models", return_value=()):
            api_key, base_url, timeout_seconds, model = _sql_llm_transport_settings("gpt-4o-mini")
        self.assertEqual(api_key, "dummy")
        self.assertEqual(base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(timeout_seconds, 300.0)
        self.assertEqual(model, "gpt-4o-mini")

    def test_tables_only_schema_request_detects_table_listing(self):
        self.assertTrue(_tables_only_schema_request("tables", "give me a list of all the tables in mydb"))
        self.assertFalse(_tables_only_schema_request("tables, columns, and relationships", "show database schema"))
        self.assertFalse(_tables_only_schema_request(None, "count the tables in the dicom schema of mydb"))

    def test_schemas_only_schema_request_detects_schema_listing(self):
        self.assertTrue(_schemas_only_schema_request("schemas", "list all schemas in dicom_mock"))
        self.assertFalse(_schemas_only_schema_request("schema and tables", "show database schema and tables"))

    def test_schema_schemas_result_returns_unique_schema_rows(self):
        result = _schema_schemas_result(
            {
                "dialect": "postgres",
                "tables": [
                    {"schema": "flathr", "name": "Patient", "type": "table"},
                    {"schema": "flathr", "name": "Study", "type": "table"},
                    {"schema": "public", "name": "resources", "type": "table"},
                ],
            }
        )
        self.assertEqual(result["columns"], ["schema"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["rows"], [{"schema": "flathr"}, {"schema": "public"}])

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

    def test_normalize_sql_selection_rejects_unknown_primitive(self):
        normalized = _normalize_sql_selection(
            {
                "primitive_id": "sql.unknown.primitive",
                "selection_reason": "bad",
                "parameters": {"table_name": "Patients"},
                "fallback_sql": "SELECT 1",
                "fallback_reason": "fallback",
            }
        )
        self.assertEqual(normalized["primitive_id"], "fallback_only")
        self.assertEqual(normalized["fallback_sql"], "SELECT 1")

    def test_execute_sql_deterministic_selection_for_row_count(self):
        executed = {}

        def fake_execute_sql(_conn, sql, limit, _schema):
            executed["sql"] = sql
            executed["limit"] = limit
            return {
                "sql": sql,
                "columns": ["count"],
                "rows": [{"count": 7}],
                "row_count": 1,
                "returned_row_count": 1,
                "total_matching_rows": 1,
                "truncated": False,
                "limit": limit,
            }

        with patch("agent_library.agents.sql_runner._execute_sql", side_effect=fake_execute_sql):
            payload = _execute_sql_deterministic_selection(
                conn=object(),
                dialect="postgres",
                schema=POSTGRES_SCHEMA,
                task="how many rows are in Patients",
                selection={
                    "primitive_id": "sql.table.row_count",
                    "parameters": {"table_name": "Patients"},
                },
                limit=25,
            )

        self.assertIsNotNone(payload)
        self.assertIn('COUNT(*) AS "count"', executed["sql"])
        self.assertEqual(executed["limit"], 1)
        self.assertEqual(payload["result"]["rows"][0]["count"], 7)

    def test_execute_sql_deterministic_selection_for_schema_table_count(self):
        payload = _execute_sql_deterministic_selection(
            conn=object(),
            dialect="postgres",
            schema=DICOM_COUNT_SCHEMA,
            task="count the tables in the dicom schema",
            selection={
                "primitive_id": "sql.schema.count_tables",
                "parameters": {"schema_name": "dicom"},
            },
            limit=25,
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["detail"], "Counted tables for schema dicom.")
        self.assertEqual(payload["result"]["columns"], ["count"])
        self.assertEqual(payload["result"]["rows"], [{"count": 2}])

    def test_handle_event_prefers_deterministic_sql_selection_before_llm_sql(self):
        class _Conn:
            def close(self):
                return None

        req = EventRequest(
            event="task.plan",
            payload={
                "task": "how many rows are in Patients",
                "target_agent": "sql_runner",
                "instruction": {"operation": "query_from_request", "question": "how many rows are in Patients"},
            },
        )

        deterministic_payload = {
            "detail": "SQL deterministic count executed.",
            "sql": 'SELECT COUNT(*) AS "count" FROM "flathr"."Patients"',
            "result": {
                "sql": 'SELECT COUNT(*) AS "count" FROM "flathr"."Patients"',
                "columns": ["count"],
                "rows": [{"count": 7}],
                "row_count": 1,
                "returned_row_count": 1,
                "total_matching_rows": 1,
                "truncated": False,
                "limit": 1,
            },
        }

        with patch("agent_library.agents.sql_runner._dsn", return_value="postgresql://example"), patch(
            "agent_library.agents.sql_runner._connect", return_value=("postgres", _Conn())
        ), patch("agent_library.agents.sql_runner._introspect", return_value=POSTGRES_SCHEMA), patch(
            "agent_library.agents.sql_runner._llm_select_sql_strategy",
            return_value={
                "primitive_id": "sql.table.row_count",
                "selection_reason": "llm selector",
                "parameters": {"table_name": "Patients"},
                "fallback_sql": "",
                "fallback_reason": "",
            },
        ), patch(
            "agent_library.agents.sql_runner._execute_sql_deterministic_selection",
            return_value=deterministic_payload,
        ), patch("agent_library.agents.sql_runner._llm_sql_queries") as llm_sql_queries:
            response = handle_event(req)

        self.assertEqual(response["emits"][0]["event"], "sql.result")
        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["execution_strategy"], "deterministic")
        self.assertEqual(payload["deterministic_primitive"], "sql.table.row_count")
        llm_sql_queries.assert_not_called()

    def test_handle_event_uses_selector_fallback_sql_before_legacy_llm_sql(self):
        class _Conn:
            def close(self):
                return None

        req = EventRequest(
            event="task.plan",
            payload={
                "task": "show top patients by studies",
                "target_agent": "sql_runner",
                "instruction": {"operation": "query_from_request", "question": "show top patients by studies"},
            },
        )

        repaired_result = {
            "sql": 'SELECT "PatientId" FROM "flathr"."Patients" LIMIT 3',
            "columns": ["PatientId"],
            "rows": [{"PatientId": "a"}, {"PatientId": "b"}],
            "row_count": 2,
            "returned_row_count": 2,
            "total_matching_rows": 2,
            "truncated": False,
            "limit": 3,
        }

        with patch("agent_library.agents.sql_runner._dsn", return_value="postgresql://example"), patch(
            "agent_library.agents.sql_runner._connect", return_value=("postgres", _Conn())
        ), patch("agent_library.agents.sql_runner._introspect", return_value=POSTGRES_SCHEMA), patch(
            "agent_library.agents.sql_runner._llm_select_sql_strategy",
            return_value={
                "primitive_id": "fallback_only",
                "selection_reason": "needs join",
                "parameters": {},
                "fallback_sql": 'SELECT "PatientId" FROM "flathr"."Patients" LIMIT 3',
                "fallback_reason": "selector fallback",
            },
        ), patch(
            "agent_library.agents.sql_runner._repair_sql_with_retries",
            return_value=(repaired_result, [{"label": "selector fallback", "sql": 'SELECT "PatientId" FROM "flathr"."Patients" LIMIT 3'}]),
        ), patch("agent_library.agents.sql_runner._llm_sql_queries") as llm_sql_queries:
            response = handle_event(req)

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["execution_strategy"], "selector_fallback_sql")
        self.assertEqual(payload["fallback_sql"], 'SELECT "PatientId" FROM "flathr"."Patients" LIMIT 3')
        llm_sql_queries.assert_not_called()

    def test_handle_event_prefers_selector_fallback_for_average_question(self):
        class _Conn:
            def close(self):
                return None

        req = EventRequest(
            event="task.plan",
            payload={
                "task": "what is the average number of studies per patient",
                "target_agent": "sql_runner",
                "instruction": {"operation": "query_from_request", "question": "what is the average number of studies per patient"},
            },
        )

        repaired_result = {
            "sql": "SELECT AVG(study_count) AS average_studies_per_patient FROM example",
            "columns": ["average_studies_per_patient"],
            "rows": [{"average_studies_per_patient": 2.5}],
            "row_count": 1,
            "returned_row_count": 1,
            "total_matching_rows": 1,
            "truncated": False,
            "limit": 1,
        }

        with patch("agent_library.agents.sql_runner._dsn", return_value="postgresql://example"), patch(
            "agent_library.agents.sql_runner._connect", return_value=("postgres", _Conn())
        ), patch("agent_library.agents.sql_runner._introspect", return_value=POSTGRES_SCHEMA), patch(
            "agent_library.agents.sql_runner._llm_select_sql_strategy",
            return_value={
                "primitive_id": "sql.agg.count",
                "selection_reason": "count then average",
                "parameters": {"table_name": "Patients"},
                "fallback_sql": "SELECT AVG(study_count) AS average_studies_per_patient FROM example",
                "fallback_reason": "average requires fallback",
            },
        ), patch(
            "agent_library.agents.sql_runner._execute_sql_deterministic_selection"
        ) as deterministic_execute, patch(
            "agent_library.agents.sql_runner._repair_sql_with_retries",
            return_value=(repaired_result, [{"label": "average requires fallback", "sql": "SELECT AVG(study_count) AS average_studies_per_patient FROM example"}]),
        ), patch("agent_library.agents.sql_runner._llm_sql_queries") as llm_sql_queries:
            response = handle_event(req)

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["execution_strategy"], "selector_fallback_sql")
        self.assertEqual(payload["sql"], "SELECT AVG(study_count) AS average_studies_per_patient FROM example")
        deterministic_execute.assert_not_called()
        llm_sql_queries.assert_not_called()


if __name__ == "__main__":
    unittest.main()
