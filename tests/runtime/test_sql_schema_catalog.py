from __future__ import annotations

from types import SimpleNamespace

from aor_runtime.runtime.sql_catalog import SqlSchemaCatalog
from aor_runtime.tools import sql as sql_tool


class FakeInspector:
    def get_schema_names(self) -> list[str]:
        return ["public", "flathr", "pg_catalog", "information_schema", "pg_toast"]

    def get_table_names(self, schema: str | None = None) -> list[str]:
        if schema == "flathr":
            return ["Patient"]
        if schema == "public":
            return ["lowercase"]
        return ["system_table"]

    def get_view_names(self, schema: str | None = None) -> list[str]:
        return []

    def get_columns(self, table_name: str, schema: str | None = None) -> list[dict]:
        if schema == "flathr" and table_name == "Patient":
            return [
                {"name": "PatientID", "type": "TEXT", "nullable": False},
                {"name": "PatientBirthDate", "type": "DATE", "nullable": True},
            ]
        return [{"name": "id", "type": "INTEGER", "nullable": False}]

    def get_pk_constraint(self, table_name: str, schema: str | None = None) -> dict:
        if table_name == "Patient":
            return {"constrained_columns": ["PatientID"]}
        return {"constrained_columns": []}

    def get_foreign_keys(self, table_name: str, schema: str | None = None) -> list[dict]:
        return []


def test_catalog_introspection_includes_non_public_and_excludes_system(monkeypatch) -> None:
    monkeypatch.setattr(sql_tool, "inspect", lambda engine: FakeInspector())
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    catalog = sql_tool._inspect_catalog_for_engine(engine, "dicom")

    assert isinstance(catalog, SqlSchemaCatalog)
    assert [table.qualified_name for table in catalog.tables] == ["flathr.Patient", "public.lowercase"]
    patient = catalog.table_by_name("flathr", "Patient")
    assert patient is not None
    assert [column.column_name for column in patient.columns] == ["PatientID", "PatientBirthDate"]
    assert patient.columns[0].primary_key is True


def test_schema_cache_refresh(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    def fake_inspect_catalog(engine, database_name: str) -> SqlSchemaCatalog:
        calls["count"] += 1
        return SqlSchemaCatalog(database=database_name, dialect="postgresql", tables=[])

    monkeypatch.setattr(sql_tool, "_engine_for_url", lambda url: SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    monkeypatch.setattr(sql_tool, "_inspect_catalog_for_engine", fake_inspect_catalog)
    settings = sql_tool.Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://example/db"},
    )

    sql_tool.refresh_schema_cache(settings)
    sql_tool.get_sql_catalog(settings, "dicom")
    sql_tool.get_sql_catalog(settings, "dicom")
    assert calls["count"] == 1

    sql_tool.get_sql_catalog(settings, "dicom", refresh=True)
    assert calls["count"] == 2
