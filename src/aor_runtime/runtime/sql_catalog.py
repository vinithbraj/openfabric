from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SqlColumnRef(BaseModel):
    schema_name: str
    table_name: str
    column_name: str
    data_type: str | None = None
    ordinal_position: int | None = None
    nullable: bool | None = None
    primary_key: bool = False
    foreign_key: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}.{self.column_name}"


class SqlTableRef(BaseModel):
    schema_name: str
    table_name: str
    columns: list[SqlColumnRef] = Field(default_factory=list)
    table_type: str | None = None
    primary_key_columns: list[str] = Field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    def column_by_name(self, name: str) -> SqlColumnRef | None:
        for column in self.columns:
            if column.column_name == name:
                return column
        lowered = name.lower()
        matches = [column for column in self.columns if column.column_name.lower() == lowered]
        return matches[0] if len(matches) == 1 else None


class SqlSchemaCatalog(BaseModel):
    database: str
    dialect: str | None = None
    tables: list[SqlTableRef] = Field(default_factory=list)
    error: str | None = None

    def table_by_name(self, schema_name: str | None, table_name: str) -> SqlTableRef | None:
        if schema_name:
            for table in self.tables:
                if table.schema_name == schema_name and table.table_name == table_name:
                    return table
            lowered_schema = schema_name.lower()
            lowered_table = table_name.lower()
            matches = [
                table
                for table in self.tables
                if table.schema_name.lower() == lowered_schema and table.table_name.lower() == lowered_table
            ]
            return matches[0] if len(matches) == 1 else None

        exact = [table for table in self.tables if table.table_name == table_name]
        if len(exact) == 1:
            return exact[0]
        lowered = table_name.lower()
        matches = [table for table in self.tables if table.table_name.lower() == lowered]
        return matches[0] if len(matches) == 1 else None

    def table_names(self) -> list[str]:
        return sorted(table.qualified_name for table in self.tables)
