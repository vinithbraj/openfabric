from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from aor_runtime.runtime.sql_catalog import SqlSchemaCatalog, SqlTableRef


RESERVED_WORDS = {
    "all",
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "desc",
    "distinct",
    "else",
    "end",
    "false",
    "from",
    "full",
    "group",
    "having",
    "in",
    "inner",
    "interval",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "right",
    "select",
    "then",
    "true",
    "union",
    "when",
    "where",
    "with",
}
UNSAFE_SQL_KEYWORDS = {
    "alter",
    "analyze",
    "attach",
    "call",
    "copy",
    "create",
    "delete",
    "detach",
    "do",
    "drop",
    "execute",
    "grant",
    "insert",
    "lock",
    "merge",
    "reindex",
    "replace",
    "revoke",
    "set",
    "truncate",
    "update",
    "vacuum",
}
RELATION_START_KEYWORDS = {"from", "join"}
CLAUSE_END_KEYWORDS = {
    "where",
    "on",
    "group",
    "order",
    "having",
    "limit",
    "offset",
    "union",
    "except",
    "intersect",
    "returning",
}


@dataclass(frozen=True)
class SqlValidationResult:
    valid: bool
    reason: str | None = None
    normalized_sql: str | None = None


@dataclass
class _Token:
    kind: str
    text: str
    value: str | None = None


def quote_pg_identifier(name: str) -> str:
    text = str(name or "")
    if re.fullmatch(r"[a-z_][a-z0-9_]*", text) and text not in RESERVED_WORDS:
        return text
    return '"' + text.replace('"', '""') + '"'


def quote_pg_relation(schema: str | None, table: str) -> str:
    if schema:
        return f"{quote_pg_identifier(schema)}.{quote_pg_identifier(table)}"
    return quote_pg_identifier(table)


def validate_read_only_sql(sql: str) -> SqlValidationResult:
    normalized = str(sql or "").strip()
    if not normalized:
        return SqlValidationResult(False, "Empty SQL query.")

    tokens = _tokenize_sql(normalized)
    if any(token.kind == "comment" for token in tokens):
        return SqlValidationResult(False, "SQL comments are not allowed in generated queries.")

    statements = _split_statements(tokens)
    if not statements:
        return SqlValidationResult(False, "Empty SQL query.")
    if len(statements) > 1:
        return SqlValidationResult(False, "Multiple SQL statements are not allowed.")

    statement_tokens = [token for token in statements[0] if token.kind not in {"ws", "comment"}]
    if not statement_tokens:
        return SqlValidationResult(False, "Empty SQL query.")
    first_word = _word_value(statement_tokens[0])
    if first_word not in {"select", "with"}:
        return SqlValidationResult(False, "Only SELECT and WITH queries are allowed.")

    words = [_word_value(token) for token in statement_tokens]
    unsafe = [word for word in words if word in UNSAFE_SQL_KEYWORDS]
    if unsafe:
        return SqlValidationResult(False, f"Unsafe SQL keyword is not allowed: {unsafe[0].upper()}.")
    if first_word == "with" and "select" not in words:
        return SqlValidationResult(False, "WITH queries must contain a SELECT statement.")

    return SqlValidationResult(True, normalized.rstrip(";"), normalized.rstrip(";"))


def ensure_read_only_sql(sql: str) -> str:
    result = validate_read_only_sql(sql)
    if not result.valid:
        raise ValueError(result.reason or "SQL query failed read-only validation.")
    return str(result.normalized_sql or sql).strip()


def normalize_pg_relation_quoting(sql: str, catalog: SqlSchemaCatalog | None = None) -> str:
    tokens = _tokenize_sql(str(sql or ""))
    normalized_tokens, alias_columns = _normalize_relation_tokens(tokens, catalog)
    if catalog is not None:
        normalized_tokens = _normalize_column_reference_tokens(normalized_tokens, catalog, alias_columns)
    return "".join(token.text for token in normalized_tokens).strip()


def _normalize_relation_tokens(
    tokens: list[_Token],
    catalog: SqlSchemaCatalog | None,
) -> tuple[list[_Token], dict[str, set[str]]]:
    output: list[_Token] = []
    alias_columns: dict[str, set[str]] = {}
    index = 0
    expect_relation = False
    while index < len(tokens):
        token = tokens[index]
        word = _word_value(token)
        if word in RELATION_START_KEYWORDS or (expect_relation and token.text == ","):
            output.append(token)
            expect_relation = True
            index += 1
            continue
        if expect_relation and token.kind == "ws":
            output.append(token)
            index += 1
            continue
        if expect_relation and word in CLAUSE_END_KEYWORDS:
            expect_relation = False
            output.append(token)
            index += 1
            continue
        if expect_relation:
            parsed = _parse_relation_at(tokens, index, catalog)
            if parsed is not None:
                replacement, consumed, table_ref = parsed
                output.append(_Token("raw", replacement))
                alias, alias_consumed = _parse_alias_after_relation(tokens, index + consumed)
                if table_ref is not None:
                    _register_alias_columns(alias_columns, table_ref.table_name, table_ref)
                    if alias:
                        _register_alias_columns(alias_columns, alias, table_ref)
                for offset in range(alias_consumed):
                    output.append(tokens[index + consumed + offset])
                index += consumed + alias_consumed
                expect_relation = True
                continue
            expect_relation = False
        output.append(token)
        index += 1
    return output, alias_columns


def _normalize_column_reference_tokens(
    tokens: list[_Token],
    catalog: SqlSchemaCatalog,
    alias_columns: dict[str, set[str]],
) -> list[_Token]:
    output: list[_Token] = []
    all_columns = _catalog_column_names(catalog)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.kind == "word"
            and index + 2 < len(tokens)
            and tokens[index + 1].text == "."
            and tokens[index + 2].kind == "word"
        ):
            left = str(token.value or token.text)
            right = str(tokens[index + 2].value or tokens[index + 2].text)
            matching_columns = alias_columns.get(left.lower(), set()) or all_columns
            replacement = _canonical_case_match(right, matching_columns)
            quoted_replacement = quote_pg_identifier(replacement) if replacement is not None else None
            if replacement is not None and quoted_replacement != right:
                output.extend([token, tokens[index + 1], _Token("raw", str(quoted_replacement))])
                index += 3
                continue
        if token.kind == "word" and _word_value(token) not in RESERVED_WORDS:
            previous_word = _previous_word_value(tokens, index)
            next_non_ws = _next_non_ws_token(tokens, index)
            if previous_word != "as" and (next_non_ws is None or next_non_ws.text != "("):
                value = str(token.value or token.text)
                replacement = _unique_canonical_case_match(value, all_columns)
                quoted_replacement = quote_pg_identifier(replacement) if replacement is not None else None
                if replacement is not None and quoted_replacement != value:
                    output.append(_Token("raw", str(quoted_replacement)))
                    index += 1
                    continue
        output.append(token)
        index += 1
    return output


def _parse_relation_at(
    tokens: list[_Token],
    index: int,
    catalog: SqlSchemaCatalog | None,
) -> tuple[str, int, SqlTableRef | None] | None:
    token = tokens[index]
    if token.text == "(":
        return None
    if token.kind == "quoted_identifier" and token.value and "." in token.value:
        schema, table = token.value.split(".", 1)
        table_ref = catalog.table_by_name(schema, table) if catalog is not None else None
        if catalog is None or table_ref is not None:
            return quote_pg_relation(schema, table_ref.table_name if table_ref else table), 1, table_ref
        return None
    if token.kind not in {"word", "quoted_identifier"}:
        return None

    first = str(token.value or token.text)
    if index + 2 < len(tokens) and tokens[index + 1].text == "." and tokens[index + 2].kind in {"word", "quoted_identifier"}:
        schema = first
        table = str(tokens[index + 2].value or tokens[index + 2].text)
        table_ref = catalog.table_by_name(schema, table) if catalog is not None else None
        if catalog is not None and table_ref is None:
            return None
        return quote_pg_relation(schema, table_ref.table_name if table_ref else table), 3, table_ref

    table_ref = catalog.table_by_name(None, first) if catalog is not None else None
    if catalog is not None and table_ref is None:
        return None
    if table_ref is not None:
        return quote_pg_relation(table_ref.schema_name if table_ref.schema_name != "public" else table_ref.schema_name, table_ref.table_name), 1, table_ref
    return quote_pg_identifier(first), 1, None


def _parse_alias_after_relation(tokens: list[_Token], index: int) -> tuple[str | None, int]:
    start = index
    while index < len(tokens) and tokens[index].kind == "ws":
        index += 1
    if index < len(tokens) and _word_value(tokens[index]) == "as":
        index += 1
        while index < len(tokens) and tokens[index].kind == "ws":
            index += 1
    if index >= len(tokens):
        return None, 0
    token = tokens[index]
    word = _word_value(token)
    if token.kind not in {"word", "quoted_identifier"} or word in CLAUSE_END_KEYWORDS or word in RELATION_START_KEYWORDS:
        return None, 0
    return str(token.value or token.text), index + 1 - start


def _register_alias_columns(alias_columns: dict[str, set[str]], alias: str, table_ref: SqlTableRef) -> None:
    alias_columns[alias.lower()] = {column.column_name for column in table_ref.columns}


def _catalog_column_names(catalog: SqlSchemaCatalog) -> set[str]:
    return {column.column_name for table in catalog.tables for column in table.columns}


def _canonical_case_match(value: str, candidates: Iterable[str]) -> str | None:
    if value in candidates:
        return value
    lowered = value.lower()
    matches = [candidate for candidate in candidates if candidate.lower() == lowered]
    return matches[0] if len(matches) == 1 else None


def _unique_canonical_case_match(value: str, candidates: Iterable[str]) -> str | None:
    lowered = value.lower()
    matches = sorted({candidate for candidate in candidates if candidate.lower() == lowered})
    return matches[0] if len(matches) == 1 else None


def _previous_word_value(tokens: list[_Token], index: int) -> str:
    cursor = index - 1
    while cursor >= 0:
        token = tokens[cursor]
        if token.kind == "ws":
            cursor -= 1
            continue
        return _word_value(token)
    return ""


def _next_non_ws_token(tokens: list[_Token], index: int) -> _Token | None:
    cursor = index + 1
    while cursor < len(tokens):
        token = tokens[cursor]
        if token.kind == "ws":
            cursor += 1
            continue
        return token
    return None


def _word_value(token: _Token) -> str:
    if token.kind not in {"word", "quoted_identifier"}:
        return ""
    return str(token.value or token.text).lower()


def _split_statements(tokens: list[_Token]) -> list[list[_Token]]:
    statements: list[list[_Token]] = []
    current: list[_Token] = []
    for token in tokens:
        if token.kind == "symbol" and token.text == ";":
            if any(item.kind != "ws" for item in current):
                statements.append(current)
            current = []
            continue
        current.append(token)
    if any(item.kind != "ws" for item in current):
        statements.append(current)
    return statements


def _tokenize_sql(sql: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    length = len(sql)
    while index < length:
        char = sql[index]
        if char.isspace():
            start = index
            while index < length and sql[index].isspace():
                index += 1
            tokens.append(_Token("ws", sql[start:index]))
            continue
        if char == "'" or (char in {"E", "e"} and index + 1 < length and sql[index + 1] == "'"):
            start = index
            if char in {"E", "e"}:
                index += 1
            index += 1
            while index < length:
                if sql[index] == "'" and index + 1 < length and sql[index + 1] == "'":
                    index += 2
                    continue
                if sql[index] == "'":
                    index += 1
                    break
                index += 1
            tokens.append(_Token("string", sql[start:index]))
            continue
        if char == '"':
            start = index
            index += 1
            value_parts: list[str] = []
            while index < length:
                if sql[index] == '"' and index + 1 < length and sql[index + 1] == '"':
                    value_parts.append('"')
                    index += 2
                    continue
                if sql[index] == '"':
                    index += 1
                    break
                value_parts.append(sql[index])
                index += 1
            tokens.append(_Token("quoted_identifier", sql[start:index], "".join(value_parts)))
            continue
        if char == "-" and index + 1 < length and sql[index + 1] == "-":
            start = index
            index += 2
            while index < length and sql[index] not in "\r\n":
                index += 1
            tokens.append(_Token("comment", sql[start:index]))
            continue
        if char == "/" and index + 1 < length and sql[index + 1] == "*":
            start = index
            index += 2
            while index + 1 < length and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index = min(length, index + 2)
            tokens.append(_Token("comment", sql[start:index]))
            continue
        if char == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[index:])
            if match:
                tag = match.group(0)
                start = index
                index += len(tag)
                end = sql.find(tag, index)
                index = length if end == -1 else end + len(tag)
                tokens.append(_Token("string", sql[start:index]))
                continue
        if re.match(r"[A-Za-z_]", char):
            start = index
            index += 1
            while index < length and re.match(r"[A-Za-z0-9_$]", sql[index]):
                index += 1
            tokens.append(_Token("word", sql[start:index], sql[start:index]))
            continue
        if char.isdigit():
            start = index
            index += 1
            while index < length and re.match(r"[A-Za-z0-9_.]", sql[index]):
                index += 1
            tokens.append(_Token("number", sql[start:index]))
            continue
        if index + 1 < length and sql[index : index + 2] in {"::", "->", "->>", "<=", ">=", "<>", "!=", "||"}:
            tokens.append(_Token("symbol", sql[index : index + 2]))
            index += 2
            continue
        tokens.append(_Token("symbol", char))
        index += 1
    return tokens
