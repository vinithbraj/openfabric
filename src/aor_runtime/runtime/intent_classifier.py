from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from aor_runtime.runtime.file_query import normalize_file_query
from aor_runtime.runtime.intents import (
    CompoundIntent,
    CountFilesIntent,
    FileAggregateIntent,
    FetchExtractIntent,
    IntentResult,
    ListFilesIntent,
    ReadFileLineIntent,
    SearchFileContentsIntent,
    ShellCommandIntent,
    SqlCountIntent,
    SqlSelectIntent,
    TransformChainIntent,
    TransformIntent,
    WriteResultIntent,
    WriteTextIntent,
)
from aor_runtime.runtime.text_extract import extract_quoted_content


ORDINAL_TO_NUMBER = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
CARDINAL_TO_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
NON_RECURSIVE_HINTS = (
    "top-level",
    "directly in",
    "immediate children",
    "immediate-child",
    "not nested",
    "non-recursive",
    "only in this folder",
)
RECURSIVE_HINTS = ("under", "recursively", "anywhere below", "inside this tree")
PATH_STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "meeting",
    "notes",
    "result",
    "content",
    "file",
    "folder",
    "directory",
    "saved",
}
WRITE_CONTENT_STOPWORDS = {
    "it",
    "them",
    "the result",
    "result",
    "the csv",
    "csv",
    "the content",
    "content",
}
SUPPORTED_COMPOUND_PRODUCERS = (
    CountFilesIntent,
    ListFilesIntent,
    ReadFileLineIntent,
    SqlCountIntent,
    SqlSelectIntent,
    ShellCommandIntent,
)
CASE_TRANSFORM_OPERATIONS = {"uppercase", "lowercase", "titlecase"}
SHAPE_TRANSFORM_OPERATIONS = {"csv", "newline_text", "json", "count"}
RETURN_FILE_PHRASES = ("return the file contents", "return the saved file", "return the content")
RETURN_RESULT_PHRASES = ("return the result only", "no extra text")
RELATION_PATTERN = (
    r"inside\s+this\s+tree|anywhere\s+below|directly\s+in|only\s+in\s+this\s+folder|under|in"
)
TRAILING_PUNCTUATION_RE = re.compile(r"[?!,:;]+$")
PATH_CANDIDATE_RE = re.compile(r"(~?/[^,\s]+|\.\.?/[^,\s]+|[A-Za-z]:[\\/][^,\s]+)")
READ_LINE_PATTERNS = (
    re.compile(
        r"\bread\s+(?:the\s+)?(?:(?:line|line\s+numbered)\s+(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)|(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+line)\s+from\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:from|take|open)\s+(?P<path>\S+),\s*read\s+(?:the\s+)?(?:(?:line|line\s+numbered)\s+(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)|(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+line)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:give\s+me|return(?:\s+only)?)\s+line\s+(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+from\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btake\s+line\s+(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+from\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom\s+(?P<path>\S+),\s*return\s+line\s+(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+is\s+on\s+the\s+(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+line\s+of\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+line\s+of\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breturn\s+(?:only\s+)?the\s+(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+line\s+(?:in|from)\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
)
COUNT_FILES_RE = re.compile(
    rf"\b(?:how\s+many|count|tell\s+me\s+the\s+number\s+of)\s+(?P<body>.+?)\s+(?:are\s+)?(?P<relation>{RELATION_PATTERN})\s+(?P<path>\S+)",
    re.IGNORECASE,
)
LIST_FILES_RE = re.compile(
    rf"\b(?:list|show|return|give\s+me)\s+(?P<body>.+?)\s+(?P<relation>{RELATION_PATTERN})\s+(?P<path>\S+)",
    re.IGNORECASE,
)
SEARCH_FILES_PATTERNS = (
    re.compile(
        rf"\b(?:find|search)\s+(?P<body>.+?)\s+containing\s+(?P<needle>.+?)\s+(?P<relation>{RELATION_PATTERN})\s+(?P<path>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:please\s+search|search|look\s+through)\s+(?P<path>\S+)\s+for\s+(?P<body>.+?)\s+containing\s+(?P<needle>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfind\s+(?P<body>.+?)\s+in\s+(?P<path>\S+)\s+whose\s+contents\s+(?:include|contain)\s+(?P<needle>.+)",
        re.IGNORECASE,
    ),
)
WRITE_TO_PATH_RE = re.compile(
    r"\b(?:write|save)\s+(?P<content>.+?)\s+to\s+(?P<path>\S+)(?P<tail>\s+(?:and(?:\s+then)?\s+return(?:\s+.+)?|return(?:\s+.+)?|and\s+no\s+extra\s+text)?)?\s*$",
    re.IGNORECASE,
)
CREATE_WITH_CONTENT_RE = re.compile(
    r"\bcreate\s+(?P<path>\S+)\s+(?:with\s+(?:exact\s+)?content|containing)\s+(?P<content>.+?)(?P<tail>\s+(?:and(?:\s+then)?\s+return(?:\s+.+)?|return(?:\s+.+)?|and\s+no\s+extra\s+text)?)?\s*$",
    re.IGNORECASE,
)
FETCH_EXTRACT_RE = re.compile(
    r"\b(?:curl|fetch)\s+(?P<url>\S+)\s+and\s+extract\s+(?:the\s+)?(?P<extract><title>|title|<head>|head|body[\s_]?text)\b",
    re.IGNORECASE,
)
USING_SHELL_RE = re.compile(r"^\s*using\s+shell,\s*(?P<body>.+?)\s*$", re.IGNORECASE)
WITH_SHELL_RE = re.compile(r"^\s*with\s+shell[:,]?\s*(?P<body>.+?)\s*$", re.IGNORECASE)
SHELL_PREFIX_RE = re.compile(r"^\s*shell:\s*(?P<body>.+?)\s*$", re.IGNORECASE)
RUN_SHELL_RE = re.compile(r"^\s*run\s+a\s+shell\s+command\s+that\s+(?P<body>.+?)\s*$", re.IGNORECASE)
PRINT_LINES_RE = re.compile(
    r"\bprints?\s+(?P<first>.+?)\s+then\s+(?P<second>.+?)\s+on\s+separate\s+lines(?:\s+and\s+return(?:\s+them)?(?:\s+as)?\s+(?P<mode>csv|count|text))?\s*$",
    re.IGNORECASE,
)
PRINT_AND_LINES_RE = re.compile(
    r"\bprints?\s+(?P<first>.+?)\s+and\s+(?P<second>.+?)\s+on\s+separate\s+lines(?:\s+and\s+return(?:\s+them)?(?:\s+as)?\s+(?P<mode>csv|count|text))?\s*$",
    re.IGNORECASE,
)
OUTPUT_LINES_RE = re.compile(
    r"\boutput\s+(?P<first>.+?)\s+then\s+(?P<second>.+?),\s*one\s+per\s+line(?:,\s*and\s+give\s+(?P<mode>csv|count|text))?\s*$",
    re.IGNORECASE,
)
PRINT_NEWLINE_RE = re.compile(
    r"\bprints?\s+(?P<first>.+?)\s+newline\s+(?P<second>[^;]+?)(?:\s*;\s*return\s+as\s+(?P<mode>csv|count|text))?\s*$",
    re.IGNORECASE,
)
PRINT_RETURN_RE = re.compile(
    r"\bprints?\s+(?P<first>.+?)\s+then\s+(?P<second>.+?)\s+and\s+return\s+(?P<mode>comma\s+separated|csv|count|text)\s*$",
    re.IGNORECASE,
)
SQL_COUNT_RE = re.compile(
    r"\bcount\s+(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)\s+(?:in|from)\s+(?P<database>[a-zA-Z_][a-zA-Z0-9_]*)\b",
    re.IGNORECASE,
)
SQL_SELECT_RE = re.compile(
    r"\b(?:list|show|return|query|get|select)\s+(?:top\s+(?P<top>\d+)\s+)?(?P<columns>[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s+from\s+(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)(?:\s+in\s+(?P<database>[a-zA-Z_][a-zA-Z0-9_]*))?(?:\s+order\s+by\s+(?P<order_by>[a-zA-Z_][a-zA-Z0-9_]*))?(?:\s+limit\s+(?P<limit>\d+))?",
    re.IGNORECASE,
)
AGGREGATE_TRIGGER_RE = re.compile(
    r"\b(?:total\s+file\s+size|total\s+size|sum(?:\s+the)?\s+size|how\s+much\s+space|disk\s+space\s+used\s+by|size\s+of\s+all|total\s+bytes|count\s+and\s+total\s+size)\b",
    re.IGNORECASE,
)
AGGREGATE_COUNT_AND_SIZE_RE = re.compile(r"\bcount\s+and\s+total\s+size\b", re.IGNORECASE)
AGGREGATE_COUNT_ONLY_RE = re.compile(r"\b(?:count\s+only|number\s+only)\b", re.IGNORECASE)
SIZE_UNIT_RE = re.compile(r"\b(bytes|kb|mb|gb)\b", re.IGNORECASE)
OUTPUT_JSON_RE = re.compile(r"\bjson\b", re.IGNORECASE)
OUTPUT_CSV_RE = re.compile(r"\bcsv\b|comma\s+separated", re.IGNORECASE)
OUTPUT_COUNT_RE = re.compile(r"\bcount\s+only\b|\bnumber\s+only\b", re.IGNORECASE)
OUTPUT_NEWLINE_RE = re.compile(r"\b(?:as\s+plain\s+text|newline\s+text|one\s+per\s+line)\b", re.IGNORECASE)
RETURN_HINT_RE = re.compile(r"\breturn\b|\bno extra text\b", re.IGNORECASE)
SAVE_RESULT_RE = re.compile(
    r"\b(?:save|write)\s+(?:the\s+result|it|them)\s+(?:to|as)\s+(?P<path>\S+)\s*$",
    re.IGNORECASE,
)
CASE_TRANSFORM_PATTERNS = (
    (re.compile(r"\b(?:uppercase|upper\s+case|capital\s+letters)\b", re.IGNORECASE), "uppercase"),
    (re.compile(r"\b(?:lowercase|lower\s+case)\b", re.IGNORECASE), "lowercase"),
    (re.compile(r"\btitle\s+case\b", re.IGNORECASE), "titlecase"),
)
SHAPE_TRANSFORM_PATTERNS = (
    (re.compile(r"\b(?:as\s+uppercase\s+csv|uppercase\s+csv)\b", re.IGNORECASE), ("uppercase", "csv")),
    (re.compile(r"\b(?:as\s+csv|csv(?:\s+string)?|comma\s+separated)\b", re.IGNORECASE), "csv"),
    (re.compile(r"\b(?:as\s+plain\s+text|newline\s+text|one\s+per\s+line)\b", re.IGNORECASE), "newline_text"),
    (re.compile(r"\b(?:as\s+json|json\s+only)\b", re.IGNORECASE), "json"),
    (re.compile(r"\b(?:count\s+only|number\s+only)\b", re.IGNORECASE), "count"),
)


def classify_intent(goal: str, *, schema_payload: dict[str, Any] | None = None) -> IntentResult:
    text = _normalize_goal_text(goal)
    if not text:
        return IntentResult(matched=False, reason="empty_goal")
    compound = classify_compound_intent(text, schema_payload=schema_payload)
    if compound.matched:
        return compound
    single = classify_single_intent(text, schema_payload=schema_payload)
    if single.matched:
        return single
    return IntentResult(matched=False, reason="no_deterministic_intent")


def classify_single_intent(goal: str, *, schema_payload: dict[str, Any] | None = None) -> IntentResult:
    text = _normalize_goal_text(goal)
    if not text:
        return IntentResult(matched=False, reason="empty_goal")
    for classifier in (
        _classify_read_file_line,
        _classify_file_aggregate,
        _classify_count_files,
        _classify_list_files,
        _classify_search_file_contents,
        _classify_write_text,
        _classify_fetch_extract,
        _classify_shell_command,
        lambda value: _classify_sql_count(value, schema_payload=schema_payload),
        lambda value: _classify_sql_select(value, schema_payload=schema_payload),
    ):
        result = classifier(text)
        if result.matched:
            return result
    return IntentResult(matched=False, reason="no_single_intent")


def classify_compound_intent(goal: str, *, schema_payload: dict[str, Any] | None = None) -> IntentResult:
    text = _normalize_goal_text(goal)
    clauses = _split_compound_clauses(text)
    if len(clauses) < 2:
        inline_compound = _classify_inline_transform_compound(text, schema_payload=schema_payload)
        if inline_compound.matched:
            return inline_compound
        return IntentResult(matched=False, reason="compound_not_enough_clauses")

    inline_operations, producer_clause = _extract_inline_transform_operations(clauses[0])
    producer_result = classify_single_intent(producer_clause, schema_payload=schema_payload)
    clause_start_index = 1
    if not producer_result.matched:
        if len(clauses) >= 2:
            combined_clause = f"{clauses[0]}, {clauses[1]}"
            inline_operations, producer_clause = _extract_inline_transform_operations(combined_clause)
            producer_result = classify_single_intent(producer_clause, schema_payload=schema_payload)
            clause_start_index = 2
        if not producer_result.matched:
            return IntentResult(matched=False, reason="compound_producer_not_supported")
    producer_intent = producer_result.intent
    if not isinstance(producer_intent, SUPPORTED_COMPOUND_PRODUCERS):
        return IntentResult(matched=False, reason="compound_producer_type_not_supported")

    intents: list[Any] = [producer_intent]
    return_policy = "return_last"
    transform_operations: list[str] = list(inline_operations)
    saw_case_transform = any(operation in CASE_TRANSFORM_OPERATIONS for operation in transform_operations)
    saw_shape_transform = any(operation in SHAPE_TRANSFORM_OPERATIONS for operation in transform_operations)
    save_index: int | None = None

    for position, clause in enumerate(clauses[clause_start_index:], start=clause_start_index):
        save_intent = _classify_save_clause(clause, intents)
        if save_intent is not None:
            if save_index is not None:
                return IntentResult(matched=False, reason="compound_multiple_save_clauses")
            intents.append(save_intent)
            save_index = len(intents) - 1
            continue

        clause_policy = _classify_return_clause(clause, has_save=save_index is not None)
        if clause_policy is not None:
            if position != len(clauses) - 1:
                return IntentResult(matched=False, reason="compound_return_clause_must_be_last")
            return_policy = clause_policy
            continue

        operations = _classify_transform_clause(clause)
        if operations:
            if save_index is not None:
                return IntentResult(matched=False, reason="compound_transform_after_save_not_supported")
            for operation in operations:
                if operation in CASE_TRANSFORM_OPERATIONS:
                    if saw_case_transform or saw_shape_transform:
                        return IntentResult(matched=False, reason="compound_case_transform_order_invalid")
                    saw_case_transform = True
                elif operation in SHAPE_TRANSFORM_OPERATIONS:
                    if saw_shape_transform:
                        return IntentResult(matched=False, reason="compound_duplicate_shape_transform")
                    saw_shape_transform = True
                transform_operations.append(operation)
            continue

        return IntentResult(matched=False, reason="compound_clause_not_supported")

    if transform_operations:
        transform_intent = TransformChainIntent(operations=transform_operations)
        if save_index is not None:
            intents.insert(save_index, transform_intent)
            save_index += 1
        else:
            intents.append(transform_intent)

    if len(intents) == 1:
        return IntentResult(matched=False, reason="compound_no_supported_tail")

    if save_index is not None:
        save_intent = intents[save_index]
        if isinstance(save_intent, WriteResultIntent):
            save_intent.output_mode = _compound_output_mode(intents)
            save_intent.read_back = return_policy == "return_written_file"

    return IntentResult(matched=True, intent=CompoundIntent(intents=intents, return_policy=return_policy))


def _classify_read_file_line(goal: str) -> IntentResult:
    for pattern in READ_LINE_PATTERNS:
        match = pattern.search(goal)
        if match is None:
            continue
        line_number = _line_number_from_match(match)
        path = _clean_path(match.group("path"))
        if line_number <= 0 or not _looks_like_path_candidate(path):
            return IntentResult(matched=False, reason="read_line_incomplete")
        return IntentResult(matched=True, intent=ReadFileLineIntent(path=path, line_number=line_number))
    return IntentResult(matched=False, reason="read_line_no_match")


def _classify_count_files(goal: str) -> IntentResult:
    match = COUNT_FILES_RE.search(goal)
    if match is not None:
        path = _clean_path(match.group("path"))
        body = str(match.group("body") or "")
        if _looks_like_path_candidate(path) and "file" in body.lower():
            query = normalize_file_query(goal, explicit_path=path)
            return IntentResult(matched=True, intent=CountFilesIntent(path=path, pattern=query.pattern, recursive=query.recursive))
    return _classify_count_files_by_path_hint(goal)


def _classify_file_aggregate(goal: str) -> IntentResult:
    if AGGREGATE_TRIGGER_RE.search(goal) is None:
        return IntentResult(matched=False, reason="file_aggregate_no_match")
    path = _extract_first_path_candidate(goal)
    if not _looks_like_path_candidate(path):
        return IntentResult(matched=False, reason="file_aggregate_missing_path")
    if not _contains_file_phrase(goal) and _extract_glob_pattern(goal) is None:
        return IntentResult(matched=False, reason="file_aggregate_missing_file_phrase")

    query = normalize_file_query(goal, explicit_path=path)
    pattern = _extract_glob_pattern(goal) or query.pattern
    aggregate = _infer_file_aggregate_mode(goal)
    return IntentResult(
        matched=True,
        intent=FileAggregateIntent(
            path=path,
            pattern=pattern,
            recursive=query.recursive,
            file_only=True,
            aggregate=aggregate,
            output_mode=_infer_file_aggregate_output_mode(goal, aggregate),
            size_unit=_infer_file_aggregate_size_unit(goal),
        ),
    )


def _classify_list_files(goal: str) -> IntentResult:
    match = LIST_FILES_RE.search(goal)
    if match is not None:
        path = _clean_path(match.group("path"))
        body = str(match.group("body") or "")
        if _looks_like_path_candidate(path) and "file" in body.lower() and not _looks_like_content_diagnostic_body(body):
            query = normalize_file_query(goal, explicit_path=path)
            output_mode = _infer_output_mode(goal)
            return IntentResult(
                matched=True,
                intent=ListFilesIntent(
                    path=path,
                    pattern=query.pattern if query.pattern != "*" else None,
                    recursive=query.recursive,
                    file_only=True,
                    path_style=query.path_style,
                    output_mode=output_mode,
                ),
            )
    return _classify_list_files_by_path_hint(goal)


def _classify_search_file_contents(goal: str) -> IntentResult:
    for pattern in SEARCH_FILES_PATTERNS:
        match = pattern.search(goal)
        if match is None:
            continue
        path = _clean_path(match.group("path"))
        needle = _normalize_search_needle(match.group("needle"))
        if not _looks_like_path_candidate(path) or not needle:
            return IntentResult(matched=False, reason="search_files_incomplete")
        body = str(match.group("body") or "")
        if "file" not in body.lower():
            return IntentResult(matched=False, reason="search_files_missing_file_phrase")
        query = normalize_file_query(goal, explicit_path=path)
        return IntentResult(
            matched=True,
            intent=SearchFileContentsIntent(
                path=path,
                needle=needle,
                pattern=query.pattern if query.pattern != "*" else None,
                recursive=query.recursive,
                path_style=query.path_style,
                output_mode=_infer_output_mode(goal),
            ),
        )
    return IntentResult(matched=False, reason="search_files_no_match")


def _classify_write_text(goal: str) -> IntentResult:
    match = WRITE_TO_PATH_RE.search(goal)
    if match is not None:
        path = _clean_path(match.group("path"))
        content = extract_quoted_content(goal) or _normalize_phrase(match.group("content"))
        if _looks_like_path_candidate(path) and _looks_like_literal_write_content(content):
            return IntentResult(
                matched=True,
                intent=WriteTextIntent(path=path, content=content, return_content=_has_return_clause(match.group("tail"))),
            )

    match = CREATE_WITH_CONTENT_RE.search(goal)
    if match is None:
        return IntentResult(matched=False, reason="write_text_no_match")
    path = _clean_path(match.group("path"))
    content = extract_quoted_content(goal) or _normalize_phrase(match.group("content"))
    if not _looks_like_path_candidate(path) or not _looks_like_literal_write_content(content):
        return IntentResult(matched=False, reason="write_text_incomplete")
    return IntentResult(
        matched=True,
        intent=WriteTextIntent(path=path, content=content, return_content=_has_return_clause(match.group("tail"))),
    )


def _classify_fetch_extract(goal: str) -> IntentResult:
    match = FETCH_EXTRACT_RE.search(goal)
    if match is None:
        return IntentResult(matched=False, reason="fetch_extract_no_match")
    extract = str(match.group("extract") or "").lower().strip("<>").replace(" ", "_")
    if extract == "body_text":
        return IntentResult(matched=False, reason="fetch_extract_body_text_deferred")
    if extract not in {"title", "head"}:
        return IntentResult(matched=False, reason="fetch_extract_unsupported")
    url = _normalize_url(match.group("url"))
    if not url:
        return IntentResult(matched=False, reason="fetch_extract_missing_url")
    return IntentResult(matched=True, intent=FetchExtractIntent(url=url, extract=extract))


def _classify_shell_command(goal: str) -> IntentResult:
    body = _extract_shell_body(goal)
    if not body:
        return IntentResult(matched=False, reason="shell_no_match")
    for pattern in (PRINT_LINES_RE, PRINT_AND_LINES_RE, OUTPUT_LINES_RE, PRINT_NEWLINE_RE, PRINT_RETURN_RE):
        print_match = pattern.search(body)
        if print_match is None:
            continue
        first = _normalize_phrase(print_match.group("first"))
        second = _normalize_phrase(print_match.group("second"))
        if not first or not second:
            return IntentResult(matched=False, reason="shell_print_incomplete")
        mode = str(print_match.groupdict().get("mode") or "").lower() or _infer_shell_output_mode(goal)
        if mode == "comma separated":
            mode = "csv"
        if mode not in {"text", "csv", "count"}:
            mode = "text"
        command = "printf '%s\\n%s\\n' {} {}".format(_shell_quote(first), _shell_quote(second))
        return IntentResult(matched=True, intent=ShellCommandIntent(command=command, output_mode=mode))
    return IntentResult(matched=False, reason="shell_body_not_supported")


def _classify_sql_count(goal: str, *, schema_payload: dict[str, Any] | None = None) -> IntentResult:
    match = SQL_COUNT_RE.search(goal)
    if match is None:
        return IntentResult(matched=False, reason="sql_count_no_match")
    table = str(match.group("table") or "").strip()
    database = _normalize_identifier(match.group("database"))
    if not table or not database:
        return IntentResult(matched=False, reason="sql_count_incomplete")
    if not _schema_supports_table(schema_payload, database=database, table=table):
        return IntentResult(matched=False, reason="sql_count_schema_unconfirmed")
    output_key = _extract_single_json_key(goal)
    return IntentResult(matched=True, intent=SqlCountIntent(database=database, table=table, output_key=output_key))


def _classify_sql_select(goal: str, *, schema_payload: dict[str, Any] | None = None) -> IntentResult:
    match = SQL_SELECT_RE.search(goal)
    if match is None:
        return IntentResult(matched=False, reason="sql_select_no_match")
    raw_columns = str(match.group("columns") or "").strip()
    columns = [part.strip() for part in raw_columns.split(",") if part.strip()]
    if not columns:
        return IntentResult(matched=False, reason="sql_select_missing_columns")
    database = _normalize_identifier(match.group("database"))
    table = str(match.group("table") or "").strip()
    if not table:
        return IntentResult(matched=False, reason="sql_select_missing_table")
    if OUTPUT_JSON_RE.search(goal):
        output_mode = "json"
    elif OUTPUT_CSV_RE.search(goal):
        output_mode = "csv"
    else:
        output_mode = "text"
    if output_mode != "json" and len(columns) != 1:
        return IntentResult(matched=False, reason="sql_select_multicolumn_not_supported")
    if not _schema_supports_columns(schema_payload, database=database, table=table, columns=columns):
        return IntentResult(matched=False, reason="sql_select_schema_unconfirmed")
    limit = match.group("limit") or match.group("top")
    return IntentResult(
        matched=True,
        intent=SqlSelectIntent(
            database=database,
            table=table,
            columns=columns,
            order_by=str(match.group("order_by") or "").strip() or None,
            limit=int(limit) if limit else None,
            output_mode=output_mode,
        ),
    )


def _classify_transform_clause(clause: str) -> list[str]:
    operations: list[str] = []
    normalized = clause.strip()
    for pattern, operation in CASE_TRANSFORM_PATTERNS:
        if pattern.search(normalized):
            operations.append(operation)
            break
    for pattern, operation in SHAPE_TRANSFORM_PATTERNS:
        if pattern.search(normalized):
            if isinstance(operation, tuple):
                operations.extend(list(operation))
            else:
                operations.append(operation)
            break
    deduped: list[str] = []
    for operation in operations:
        if operation not in deduped:
            deduped.append(operation)
    return deduped


def _classify_save_clause(clause: str, intents: list[Any]) -> WriteResultIntent | None:
    match = SAVE_RESULT_RE.search(clause)
    if match is None:
        return None
    path = _clean_path(match.group("path"))
    if not _looks_like_path_candidate(path):
        return None
    return WriteResultIntent(path=path, source_alias="__previous__", output_mode=_compound_output_mode(intents), return_content=False)


def _classify_return_clause(clause: str, *, has_save: bool) -> str | None:
    normalized = str(clause or "").strip().lower()
    if not normalized:
        return None
    if any(phrase in normalized for phrase in RETURN_FILE_PHRASES):
        return "return_written_file" if has_save else "return_last"
    if any(phrase in normalized for phrase in RETURN_RESULT_PHRASES):
        return "return_original_result" if has_save else "return_last"
    if normalized.startswith("reply with only") or normalized.startswith("send back just"):
        return "return_last"
    if normalized.startswith("return only "):
        return "return_last"
    if normalized.startswith("return "):
        return "return_written_file" if has_save else "return_last"
    return None


def _classify_inline_transform_compound(goal: str, *, schema_payload: dict[str, Any] | None = None) -> IntentResult:
    operations, stripped_goal = _extract_inline_transform_operations(goal)
    if not operations:
        return IntentResult(matched=False, reason="inline_transform_not_found")
    if not any(operation in CASE_TRANSFORM_OPERATIONS for operation in operations):
        return IntentResult(matched=False, reason="inline_transform_no_case_operation")
    producer_result = classify_single_intent(stripped_goal, schema_payload=schema_payload)
    if not producer_result.matched or not isinstance(producer_result.intent, SUPPORTED_COMPOUND_PRODUCERS):
        return IntentResult(matched=False, reason="inline_transform_producer_not_supported")
    return IntentResult(
        matched=True,
        intent=CompoundIntent(
            intents=[producer_result.intent, TransformChainIntent(operations=operations)],
            return_policy="return_last",
        ),
    )


def _extract_inline_transform_operations(clause: str) -> tuple[list[str], str]:
    operations = _classify_transform_clause(clause)
    stripped = clause
    if operations:
        stripped = re.sub(r"\bas\s+uppercase\s+csv\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\buppercase\s+csv\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:uppercase|upper\s+case|capital\s+letters)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:lowercase|lower\s+case)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\btitle\s+case\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:as\s+csv|csv(?:\s+string)?|comma\s+separated)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:as\s+plain\s+text|newline\s+text|one\s+per\s+line)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:as\s+json|json\s+only)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\b(?:count\s+only|number\s+only)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\bconvert\s+(.+?)\s+to\b", r"\1", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\bchange\s+(.+?)\s+to\b", r"\1", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip(" ,")
    return operations, stripped


def _compound_output_mode(intents: list[Any]) -> str:
    shape_operations: list[str] = []
    for intent in intents:
        if isinstance(intent, TransformIntent) and intent.operation in SHAPE_TRANSFORM_OPERATIONS:
            shape_operations.append(intent.operation)
        if isinstance(intent, TransformChainIntent):
            shape_operations.extend([operation for operation in intent.operations if operation in SHAPE_TRANSFORM_OPERATIONS])
    if shape_operations:
        operation = shape_operations[-1]
        if operation == "newline_text":
            return "text"
        return operation
    producer = intents[0] if intents else None
    if isinstance(producer, (CountFilesIntent, SqlCountIntent)):
        return "count"
    if isinstance(producer, ListFilesIntent):
        return producer.output_mode
    if isinstance(producer, SqlSelectIntent):
        return producer.output_mode
    if isinstance(producer, ShellCommandIntent):
        return producer.output_mode
    return "text"


def _split_compound_clauses(goal: str) -> list[str]:
    clauses: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(goal):
        character = goal[index]
        if quote:
            current.append(character)
            if character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
            index += 1
            continue
        lower_tail = goal[index:].lower()
        if character == ",":
            _push_clause(clauses, current)
            index += 1
            continue
        if lower_tail.startswith(" and "):
            _push_clause(clauses, current)
            index += 5
            continue
        if lower_tail.startswith(" then "):
            _push_clause(clauses, current)
            index += 6
            continue
        current.append(character)
        index += 1
    _push_clause(clauses, current)
    return [clause for clause in clauses if clause]


def _push_clause(clauses: list[str], current: list[str]) -> None:
    text = "".join(current).strip(" ,")
    current.clear()
    if not text:
        return
    clauses.append(text.strip())


def _clean_path(raw_path: str | None) -> str:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    candidate = TRAILING_PUNCTUATION_RE.sub("", candidate)
    while candidate.endswith(".") and len(candidate) > 1:
        candidate = candidate[:-1]
    return candidate.strip()


def _normalize_phrase(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = TRAILING_PUNCTUATION_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _normalize_search_needle(value: str | None) -> str:
    text = _normalize_phrase(value)
    text = re.sub(r"\s+(?:and\s+)?(?:return|give)\b.*$", "", text, flags=re.IGNORECASE).strip()
    return text


def _classify_count_files_by_path_hint(goal: str) -> IntentResult:
    normalized = goal.lower().strip()
    if not re.match(r"^(?:how\s+many|count|tell\s+me\s+the\s+number\s+of)\b", normalized):
        return IntentResult(matched=False, reason="count_files_no_match")
    path = _extract_first_path_candidate(goal)
    if not _looks_like_path_candidate(path):
        return IntentResult(matched=False, reason="count_files_missing_path")
    if not _contains_file_phrase(goal):
        return IntentResult(matched=False, reason="count_files_missing_file_phrase")
    query = normalize_file_query(goal, explicit_path=path)
    return IntentResult(matched=True, intent=CountFilesIntent(path=path, pattern=query.pattern, recursive=query.recursive))


def _classify_list_files_by_path_hint(goal: str) -> IntentResult:
    normalized = goal.lower().strip()
    if not re.match(r"^(?:list|show|return|give\s+me)\b", normalized):
        return IntentResult(matched=False, reason="list_files_no_match")
    path = _extract_first_path_candidate(goal)
    if not _looks_like_path_candidate(path):
        return IntentResult(matched=False, reason="list_files_missing_path")
    if not _contains_file_phrase(goal):
        return IntentResult(matched=False, reason="list_files_missing_file_phrase")
    query = normalize_file_query(goal, explicit_path=path)
    return IntentResult(
        matched=True,
        intent=ListFilesIntent(
            path=path,
            pattern=query.pattern if query.pattern != "*" else None,
            recursive=query.recursive,
            file_only=True,
            path_style=query.path_style,
            output_mode=_infer_output_mode(goal),
        ),
    )


def _extract_glob_pattern(body: str | None) -> str | None:
    text = str(body or "").strip().lower()
    if not text:
        return None
    if re.search(r"\btext\s+files?\b", text):
        return "*.txt"
    stopwords = {
        "all",
        "direct",
        "directly",
        "file",
        "files",
        "folder",
        "folders",
        "immediate",
        "matching",
        "nested",
        "non",
        "not",
        "only",
        "recursive",
        "recursively",
        "size",
        "text",
        "the",
        "top",
        "total",
    }
    for pattern in (
        re.compile(r"\*(\.[a-z0-9]+)\b", re.IGNORECASE),
        re.compile(r"\.(?P<ext>[a-z0-9]+)\s+files?\b", re.IGNORECASE),
        re.compile(r"\bfiles?\s+ending\s+in\s+(?P<ext>[a-z0-9]+)\b", re.IGNORECASE),
        re.compile(r"\bfiles?\s+with\s+(?P<ext>[a-z0-9]+)\s+extension\b", re.IGNORECASE),
        re.compile(r"\b(?P<ext>[a-z0-9]{2,8})\s+files?\b", re.IGNORECASE),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        extension = match.groupdict().get("ext")
        if extension:
            extension = extension.lower()
            if extension in stopwords:
                continue
            return f"*.{extension.lower()}"
        if match.group(0).startswith("*."):
            return match.group(0).lower()
    return None


def _contains_file_phrase(text: str) -> bool:
    normalized = str(text or "").lower()
    scrubbed = PATH_CANDIDATE_RE.sub(" ", normalized)
    return re.search(r"\b(?:file|files|filename|filenames)\b", scrubbed) is not None


def _looks_like_content_diagnostic_body(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(
        token in normalized
        for token in (
            "containing",
            "contents include",
            "contents contain",
            "matching lines",
            "line containing",
            "lines containing",
            "mentioning",
            "mentions",
        )
    )


def _infer_output_mode(goal: str) -> str:
    if OUTPUT_JSON_RE.search(goal):
        return "json"
    if OUTPUT_CSV_RE.search(goal):
        return "csv"
    return "text"


def _infer_file_aggregate_output_mode(goal: str, aggregate: str) -> str:
    if OUTPUT_JSON_RE.search(goal):
        return "json"
    if aggregate == "count" and AGGREGATE_COUNT_ONLY_RE.search(goal):
        return "count"
    return "text"


def _infer_file_aggregate_mode(goal: str) -> str:
    if AGGREGATE_COUNT_AND_SIZE_RE.search(goal):
        return "count_and_total_size"
    return "total_size"


def _infer_file_aggregate_size_unit(goal: str) -> str:
    match = SIZE_UNIT_RE.search(goal)
    if match is None:
        return "auto"
    return match.group(1).lower()


def _infer_shell_output_mode(goal: str) -> str:
    if OUTPUT_CSV_RE.search(goal):
        return "csv"
    if OUTPUT_COUNT_RE.search(goal):
        return "count"
    return "text"


def _infer_recursive(body: str, relation: str, *, default: bool) -> bool:
    combined = f"{body} {relation}".lower()
    if any(hint in combined for hint in NON_RECURSIVE_HINTS):
        return False
    if any(hint in combined for hint in RECURSIVE_HINTS):
        return True
    return default


def _has_return_clause(text: str | None) -> bool:
    return bool(text and RETURN_HINT_RE.search(text))


def _looks_like_path_candidate(candidate: str) -> bool:
    value = str(candidate or "").strip()
    if not value:
        return False
    normalized = value.lower()
    if normalized in PATH_STOPWORDS:
        return False
    if " " in value:
        return False
    if value.startswith(("~", "./", "../", "/")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return True
    if "/" in value or "\\" in value or "." in value:
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", value) and normalized not in PATH_STOPWORDS)


def _extract_first_path_candidate(text: str) -> str:
    for match in PATH_CANDIDATE_RE.findall(str(text or "")):
        candidate = _clean_path(match)
        if _looks_like_path_candidate(candidate):
            return candidate
    return ""


def _looks_like_literal_write_content(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    return text.lower() not in WRITE_CONTENT_STOPWORDS


def _normalize_url(raw_url: str | None) -> str:
    value = _normalize_phrase(raw_url)
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme:
        return value
    return f"https://{value}"


def _normalize_goal_text(value: str) -> str:
    text = str(value or "").strip()
    while text.endswith((".", "!", "?")):
        text = text[:-1].rstrip()
    return text


def _normalize_identifier(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _schema_supports_table(schema_payload: dict[str, Any] | None, *, database: str | None, table: str) -> bool:
    table_map = _schema_table_map(schema_payload)
    if not table_map:
        return False
    normalized_table = table.lower()
    if database:
        tables = table_map.get(database.lower())
        if tables is None:
            return False
        return normalized_table in tables
    matched_databases = [tables for tables in table_map.values() if normalized_table in tables]
    return len(matched_databases) == 1


def _schema_supports_columns(
    schema_payload: dict[str, Any] | None,
    *,
    database: str | None,
    table: str,
    columns: list[str],
) -> bool:
    table_map = _schema_table_map(schema_payload)
    normalized_table = table.lower()
    normalized_columns = {column.lower() for column in columns}
    if not table_map:
        return False
    candidate_tables: list[dict[str, set[str]]] = []
    if database:
        tables = table_map.get(database.lower())
        if tables is None:
            return False
        candidate_tables.append(tables)
    else:
        candidate_tables = [tables for tables in table_map.values() if normalized_table in tables]
        if len(candidate_tables) != 1:
            return False
    for tables in candidate_tables:
        available_columns = tables.get(normalized_table)
        if available_columns is None:
            return False
        if not normalized_columns.issubset(available_columns):
            return False
    return True


def _schema_table_map(schema_payload: dict[str, Any] | None) -> dict[str, dict[str, set[str]]]:
    if not isinstance(schema_payload, dict):
        return {}
    databases: dict[str, dict[str, set[str]]] = {}
    for database in schema_payload.get("databases", []):
        if not isinstance(database, dict):
            continue
        database_name = str(database.get("name") or "").strip().lower()
        if not database_name:
            continue
        tables: dict[str, set[str]] = {}
        for table in database.get("tables", []):
            if not isinstance(table, dict):
                continue
            table_name = str(table.get("name") or "").strip().lower()
            if not table_name:
                continue
            tables[table_name] = {
                str(column.get("name") or "").strip().lower()
                for column in table.get("columns", [])
                if isinstance(column, dict) and str(column.get("name") or "").strip()
            }
        databases[database_name] = tables
    return databases


def _extract_single_json_key(goal: str) -> str | None:
    match = re.search(r"\bjson\s+with\s+key\s+([A-Za-z_][A-Za-z0-9_]*)\b", goal, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _line_number_from_match(match: re.Match[str]) -> int:
    raw_number = str(match.groupdict().get("number") or "").lower()
    if raw_number.isdigit():
        return int(raw_number)
    if raw_number:
        return CARDINAL_TO_NUMBER.get(raw_number, 0)
    ordinal = str(match.groupdict().get("ordinal") or "").lower()
    return ORDINAL_TO_NUMBER.get(ordinal, 0)


def _extract_shell_body(goal: str) -> str:
    for pattern in (USING_SHELL_RE, WITH_SHELL_RE, SHELL_PREFIX_RE, RUN_SHELL_RE):
        match = pattern.match(goal)
        if match is not None:
            return str(match.group("body") or "").strip()
    return ""
