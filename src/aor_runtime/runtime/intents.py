"""OpenFABRIC Runtime Module: aor_runtime.runtime.intents

Purpose:
    Define shared typed intent models used by helper and compatibility layers.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReadFileLineIntent(BaseModel):
    """Represent read file line intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ReadFileLineIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.ReadFileLineIntent and related tests.
    """
    path: str
    line_number: int


class CountFilesIntent(BaseModel):
    """Represent count files intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CountFilesIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.CountFilesIntent and related tests.
    """
    path: str
    pattern: str
    recursive: bool = True


class ListFilesIntent(BaseModel):
    """Represent list files intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ListFilesIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.ListFilesIntent and related tests.
    """
    path: str
    pattern: str | None = None
    recursive: bool = False
    file_only: bool = True
    dir_only: bool = False
    path_style: Literal["name", "relative", "absolute"] = "relative"
    output_mode: Literal["text", "csv", "json"] = "text"


class SearchFileContentsIntent(BaseModel):
    """Represent search file contents intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SearchFileContentsIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.SearchFileContentsIntent and related tests.
    """
    path: str
    needle: str
    pattern: str | None = None
    recursive: bool = True
    path_style: Literal["name", "relative", "absolute"] = "relative"
    output_mode: Literal["text", "csv", "json"] = "text"


class FileAggregateIntent(BaseModel):
    """Represent file aggregate intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FileAggregateIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.FileAggregateIntent and related tests.
    """
    path: str
    pattern: str = "*"
    recursive: bool = True
    file_only: bool = True
    aggregate: Literal["total_size", "count", "count_and_total_size"] = "total_size"
    output_mode: Literal["text", "json", "count"] = "text"
    size_unit: Literal["bytes", "kb", "mb", "gb", "auto"] = "auto"


class SqlCountIntent(BaseModel):
    """Represent sql count intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlCountIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.SqlCountIntent and related tests.
    """
    database: str | None = None
    table: str
    where: str | None = None
    output_key: str | None = None


class SqlSelectIntent(BaseModel):
    """Represent sql select intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlSelectIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.SqlSelectIntent and related tests.
    """
    database: str | None = None
    table: str
    columns: list[str]
    where: str | None = None
    order_by: str | None = None
    limit: int | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class SqlGeneratedQueryIntent(BaseModel):
    """Represent sql generated query intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlGeneratedQueryIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.SqlGeneratedQueryIntent and related tests.
    """
    database: str
    query: str
    output_mode: Literal["text", "csv", "json", "count"] = "json"
    scalar_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SqlCatalogReturnIntent(BaseModel):
    """Represent sql catalog return intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlCatalogReturnIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.SqlCatalogReturnIntent and related tests.
    """
    database: str
    value: Any
    output_mode: Literal["text", "csv", "json", "count"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SqlFailureIntent(BaseModel):
    """Represent sql failure intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SqlFailureIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.SqlFailureIntent and related tests.
    """
    message: str
    suggestions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WriteTextIntent(BaseModel):
    """Represent write text intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by WriteTextIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.WriteTextIntent and related tests.
    """
    path: str
    content: str
    return_content: bool = False


class WriteResultIntent(BaseModel):
    """Represent write result intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by WriteResultIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.WriteResultIntent and related tests.
    """
    path: str
    source_alias: str
    output_mode: Literal["text", "csv", "json", "count"] = "text"
    return_content: bool = True
    read_back: bool = False


class ShellCommandIntent(BaseModel):
    """Represent shell command intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by ShellCommandIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.ShellCommandIntent and related tests.
    """
    command: str
    node: str | None = None
    output_mode: Literal["text", "csv", "count"] = "text"


class FetchExtractIntent(BaseModel):
    """Represent fetch extract intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by FetchExtractIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.FetchExtractIntent and related tests.
    """
    url: str
    extract: Literal["title", "head", "body_text"]
    output_mode: Literal["text"] = "text"


class TransformIntent(BaseModel):
    """Represent transform intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TransformIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.TransformIntent and related tests.
    """
    source_alias: str | None = None
    operation: Literal["uppercase", "lowercase", "titlecase", "csv", "newline_text", "json", "count"]
    output_alias: str | None = None


class TransformChainIntent(BaseModel):
    """Represent transform chain intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TransformChainIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.TransformChainIntent and related tests.
    """
    source_alias: str | None = None
    operations: list[Literal["uppercase", "lowercase", "titlecase", "csv", "newline_text", "json", "count"]] = Field(
        default_factory=list
    )
    output_alias: str | None = None


class CompoundIntent(BaseModel):
    """Represent compound intent within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by CompoundIntent.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.CompoundIntent and related tests.
    """
    intents: list[Any] = Field(default_factory=list)
    return_policy: Literal["return_last", "return_written_file", "return_original_result", "return_all"] = "return_last"


class IntentResult(BaseModel):
    """Represent intent result within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by IntentResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.intents.IntentResult and related tests.
    """
    matched: bool
    intent: Any | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
