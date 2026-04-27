from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReadFileLineIntent(BaseModel):
    path: str
    line_number: int


class CountFilesIntent(BaseModel):
    path: str
    pattern: str
    recursive: bool = True


class ListFilesIntent(BaseModel):
    path: str
    pattern: str | None = None
    recursive: bool = False
    file_only: bool = True
    path_style: Literal["name", "relative", "absolute"] = "relative"
    output_mode: Literal["text", "csv", "json"] = "text"


class SearchFileContentsIntent(BaseModel):
    path: str
    needle: str
    pattern: str | None = None
    recursive: bool = True
    path_style: Literal["name", "relative", "absolute"] = "relative"
    output_mode: Literal["text", "csv", "json"] = "text"


class SqlCountIntent(BaseModel):
    database: str | None = None
    table: str
    where: str | None = None
    output_key: str | None = None


class SqlSelectIntent(BaseModel):
    database: str | None = None
    table: str
    columns: list[str]
    where: str | None = None
    order_by: str | None = None
    limit: int | None = None
    output_mode: Literal["text", "csv", "json"] = "text"


class WriteTextIntent(BaseModel):
    path: str
    content: str
    return_content: bool = False


class WriteResultIntent(BaseModel):
    path: str
    source_alias: str
    output_mode: Literal["text", "csv", "json", "count"] = "text"
    return_content: bool = True
    read_back: bool = False


class ShellCommandIntent(BaseModel):
    command: str
    node: str | None = None
    output_mode: Literal["text", "csv", "count"] = "text"


class FetchExtractIntent(BaseModel):
    url: str
    extract: Literal["title", "head", "body_text"]
    output_mode: Literal["text"] = "text"


class TransformIntent(BaseModel):
    source_alias: str | None = None
    operation: Literal["uppercase", "lowercase", "titlecase", "csv", "newline_text", "json", "count"]
    output_alias: str | None = None


class TransformChainIntent(BaseModel):
    source_alias: str | None = None
    operations: list[Literal["uppercase", "lowercase", "titlecase", "csv", "newline_text", "json", "count"]] = Field(
        default_factory=list
    )
    output_alias: str | None = None


class CompoundIntent(BaseModel):
    intents: list[Any] = Field(default_factory=list)
    return_policy: Literal["return_last", "return_written_file", "return_original_result", "return_all"] = "return_last"


class IntentResult(BaseModel):
    matched: bool
    intent: Any | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
