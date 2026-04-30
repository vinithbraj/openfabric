"""OpenFABRIC Runtime Module: aor_runtime.runtime.sql_catalog

Purpose:
    Build and cache SQL schema catalogs used by planning and validation.

Responsibilities:
    Introspect schemas, tables, columns, and relationship hints for configured databases.

Data flow / Interfaces:
    Feeds SQL validators, resolvers, prompt context, and schema-answer flows.

Boundaries:
    Catalog data describes schema only and should not include row values or PHI-bearing samples.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SqlColumnRef(BaseModel):
    """Represent sql column ref within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlColumnRef.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_catalog.SqlColumnRef and related tests.
    """
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
        """Qualified name for SqlColumnRef instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through SqlColumnRef.qualified_name calls and related tests.
        """
        return f"{self.schema_name}.{self.table_name}.{self.column_name}"


class SqlTableRef(BaseModel):
    """Represent sql table ref within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlTableRef.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_catalog.SqlTableRef and related tests.
    """
    schema_name: str
    table_name: str
    columns: list[SqlColumnRef] = Field(default_factory=list)
    table_type: str | None = None
    primary_key_columns: list[str] = Field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        """Qualified name for SqlTableRef instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed property value for callers that need this runtime fact.

        Used by:
            Used by planning, execution, validation, and presentation through SqlTableRef.qualified_name calls and related tests.
        """
        return f"{self.schema_name}.{self.table_name}"

    def column_by_name(self, name: str) -> SqlColumnRef | None:
        """Column by name for SqlTableRef instances.

        Inputs:
            Receives name for this SqlTableRef method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlTableRef.column_by_name calls and related tests.
        """
        for column in self.columns:
            if column.column_name == name:
                return column
        lowered = name.lower()
        matches = [column for column in self.columns if column.column_name.lower() == lowered]
        return matches[0] if len(matches) == 1 else None


class SqlSchemaCatalog(BaseModel):
    """Represent sql schema catalog within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlSchemaCatalog.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sql_catalog.SqlSchemaCatalog and related tests.
    """
    database: str
    dialect: str | None = None
    tables: list[SqlTableRef] = Field(default_factory=list)
    error: str | None = None

    def table_by_name(self, schema_name: str | None, table_name: str) -> SqlTableRef | None:
        """Table by name for SqlSchemaCatalog instances.

        Inputs:
            Receives schema_name, table_name for this SqlSchemaCatalog method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlSchemaCatalog.table_by_name calls and related tests.
        """
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
        """Table names for SqlSchemaCatalog instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SqlSchemaCatalog.table_names calls and related tests.
        """
        return sorted(table.qualified_name for table in self.tables)
