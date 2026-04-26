from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolExecutionError, ToolResultModel
from aor_runtime.tools.filesystem import resolve_path


MAX_MATCHED_LINES_PER_FILE = 20


def fs_search_content(
    settings: Settings,
    path: str,
    needle: str,
    pattern: str = "*",
    recursive: bool = True,
    file_only: bool = True,
    case_insensitive: bool = False,
    path_style: Literal["name", "relative", "absolute"] = "relative",
    max_matches: int | None = None,
) -> dict[str, Any]:
    root = resolve_path(settings, path)
    if not root.exists():
        raise ToolExecutionError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {root}")
    normalized_needle = str(needle or "")
    if not normalized_needle.strip():
        raise ToolExecutionError("Needle must be non-empty.")
    normalized_pattern = str(pattern or "*").strip() or "*"
    if path_style not in {"name", "relative", "absolute"}:
        raise ToolExecutionError("fs.search_content path_style must be one of: name, relative, absolute.")
    if max_matches is not None and max_matches < 0:
        raise ToolExecutionError("fs.search_content max_matches must be non-negative when provided.")

    candidate_entries = _candidate_entries(root=root, pattern=normalized_pattern, recursive=bool(recursive), file_only=bool(file_only))
    matches: list[str] = []
    entries: list[dict[str, Any]] = []

    for entry in candidate_entries:
        if max_matches is not None and len(matches) >= max_matches:
            break
        matched_lines = _matched_lines_for_file(
            Path(str(entry["path"])),
            needle=normalized_needle,
            case_insensitive=bool(case_insensitive),
        )
        if not matched_lines:
            continue
        matches.append(_format_match(entry, path_style))
        entries.append(
            {
                "name": entry["name"],
                "path": entry["path"],
                "relative_path": entry["relative_path"],
                "matched_lines": matched_lines,
            }
        )

    return {
        "path": str(root),
        "needle": normalized_needle,
        "pattern": normalized_pattern,
        "recursive": bool(recursive),
        "matches": matches,
        "entries": entries,
    }


class MatchedLineModel(BaseModel):
    line_number: int
    text: str


class SearchEntryModel(BaseModel):
    name: str
    path: str
    relative_path: str
    matched_lines: list[MatchedLineModel]


class SearchContentTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        path: str
        needle: str
        pattern: str = "*"
        recursive: bool = True
        file_only: bool = True
        case_insensitive: bool = False
        path_style: Literal["name", "relative", "absolute"] = "relative"
        max_matches: int | None = None

    class ToolResult(ToolResultModel):
        path: str
        needle: str
        pattern: str
        recursive: bool
        matches: list[str]
        entries: list[SearchEntryModel]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fs.search_content",
            description="Search text file contents under a directory with deterministic filtering and path shaping.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "needle": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "file_only": {"type": "boolean"},
                    "case_insensitive": {"type": "boolean"},
                    "path_style": {"type": "string", "enum": ["name", "relative", "absolute"]},
                    "max_matches": {"type": ["integer", "null"], "minimum": 0},
                },
                "required": ["path", "needle"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(
            fs_search_content(
                self.settings,
                arguments.path,
                arguments.needle,
                pattern=arguments.pattern,
                recursive=arguments.recursive,
                file_only=arguments.file_only,
                case_insensitive=arguments.case_insensitive,
                path_style=arguments.path_style,
                max_matches=arguments.max_matches,
            )
        )


def _candidate_entries(*, root: Path, pattern: str, recursive: bool, file_only: bool) -> list[dict[str, Any]]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    entries: list[dict[str, Any]] = []
    for item in iterator:
        try:
            resolved_item = item.resolve()
            resolved_item.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if not item.is_file():
            continue
        if file_only and not item.is_file():
            continue
        if not _matches_pattern(item=item, root=root, pattern=pattern):
            continue
        relative_path = str(item.relative_to(root))
        entries.append(
            {
                "name": item.name,
                "path": str(item),
                "relative_path": relative_path,
            }
        )
    entries.sort(key=lambda value: str(value["relative_path"]))
    return entries


def _matches_pattern(*, item: Path, root: Path, pattern: str) -> bool:
    relative_path = str(item.relative_to(root))
    if "/" in pattern or "\\" in pattern:
        normalized_relative = relative_path.replace("\\", "/")
        normalized_pattern = pattern.replace("\\", "/")
        return fnmatch(normalized_relative, normalized_pattern)
    return fnmatch(item.name, pattern)


def _format_match(entry: dict[str, Any], path_style: str) -> str:
    if path_style == "name":
        return str(entry["name"])
    if path_style == "absolute":
        return str(entry["path"])
    return str(entry["relative_path"])


def _matched_lines_for_file(path: Path, *, needle: str, case_insensitive: bool) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return []
    if b"\x00" in sample:
        return []

    lines: list[dict[str, Any]] = []
    haystack_needle = needle.casefold() if case_insensitive else needle
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                haystack = line.casefold() if case_insensitive else line
                if haystack_needle not in haystack:
                    continue
                lines.append({"line_number": line_number, "text": line.rstrip("\r\n")})
                if len(lines) >= MAX_MATCHED_LINES_PER_FILE:
                    break
    except (OSError, UnicodeDecodeError):
        return []
    return lines
