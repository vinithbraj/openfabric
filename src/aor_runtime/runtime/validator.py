"""OpenFABRIC Runtime Module: aor_runtime.runtime.validator

Purpose:
    Re-validate executed tool outputs against deterministic expectations and fixtures.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re
from pathlib import Path

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import StepLog, ValidationResult
from aor_runtime.core.utils import extract_json_object
from aor_runtime.tools.filesystem import fs_aggregate, fs_exists, fs_find, fs_glob, fs_list, fs_read, fs_size, resolve_path
from aor_runtime.tools.runtime_return import runtime_return
from aor_runtime.tools.search_content import fs_search_content
from aor_runtime.tools.slurm import (
    _canonical_job_state,
    _canonical_node_state,
    _is_fixture_mode,
    slurm_accounting_aggregate,
    slurm_accounting,
    slurm_job_detail,
    slurm_metrics,
    slurm_node_detail,
    slurm_nodes,
    slurm_partitions,
    slurm_queue,
    slurm_slurmdbd_health,
    is_problematic_node_state,
)
from aor_runtime.tools.sql import resolve_sql_databases
from aor_runtime.runtime.tool_surfaces import registered_tool_result_valid


ALIAS_RE = re.compile(r'\bas\s+("?)([a-zA-Z_][a-zA-Z0-9_]*)\1', re.IGNORECASE)
SQL_TYPE_ALIAS_FALSE_POSITIVES = {
    "bigint",
    "boolean",
    "date",
    "decimal",
    "double",
    "float",
    "integer",
    "int",
    "numeric",
    "real",
    "text",
    "timestamp",
    "timestamptz",
    "uuid",
    "varchar",
}
STORAGE_TOKEN_RE = re.compile(r"[a-z0-9_]+")
DU_OUTPUT_RE = re.compile(r"^\s*[0-9.]+[A-Za-z]+\s+\S+")
SQL_IDENTIFIER_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _top_level_select_list(query: str) -> str:
    """Handle the internal top level select list helper path for this module.

    Inputs:
        Receives query for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.validator._top_level_select_list.
    """
    text = str(query or "")
    select_start = _find_top_level_keyword(text, "select")
    if select_start < 0:
        return text
    list_start = select_start + len("select")
    from_start = _find_top_level_keyword(text, "from", start=list_start)
    if from_start < 0:
        return text[list_start:]
    return text[list_start:from_start]


def _find_top_level_keyword(text: str, keyword: str, *, start: int = 0) -> int:
    """Handle the internal find top level keyword helper path for this module.

    Inputs:
        Receives text, keyword, start for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.validator._find_top_level_keyword.
    """
    keyword_lower = keyword.lower()
    depth = 0
    quote: str | None = None
    index = max(0, start)
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                if quote == "'" and index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0 and text[index : index + len(keyword)].lower() == keyword_lower:
            before = text[index - 1] if index > 0 else ""
            after_index = index + len(keyword)
            after = text[after_index] if after_index < len(text) else ""
            if before not in SQL_IDENTIFIER_CHARS and after not in SQL_IDENTIFIER_CHARS:
                return index
        index += 1
    return -1


class RuntimeValidator:
    """Represent runtime validator within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeValidator.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.validator.RuntimeValidator and related tests.
    """
    def __init__(self, settings: Settings | None = None, tools: object | None = None) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings, tools for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator.__init__ calls and related tests.
        """
        self.settings = settings or get_settings()
        if tools is None:
            from aor_runtime.tools.factory import build_tool_registry

            tools = build_tool_registry(self.settings)
        self.tools = tools

    def validate(self, history: list[StepLog], goal: str | None = None) -> tuple[ValidationResult, list[dict[str, str | bool]]]:
        """Validate for RuntimeValidator instances.

        Inputs:
            Receives history, goal for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator.validate calls and related tests.
        """
        checks: list[dict[str, str | bool]] = []
        for item in history:
            checks.append(self._validate_step(item, goal=goal))
        failed = [check for check in checks if not bool(check["success"])]
        if failed:
            first = failed[0]
            return ValidationResult(success=False, reason=str(first["detail"])), checks
        return ValidationResult(success=True, reason=None), checks

    def _validate_step(self, item: StepLog, *, goal: str | None = None) -> dict[str, str | bool]:
        """Handle the internal validate step helper path for this module.

        Inputs:
            Receives item, goal for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._validate_step calls and related tests.
        """
        step = item.step
        if not item.success:
            return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": item.error or "step failed"}

        try:
            if step.action == "fs.exists":
                observed_exists = bool(item.result.get("exists"))
                observed_path = str(item.result.get("path", step.args["path"]))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": observed_exists,
                    "detail": f"exists={observed_exists} path={observed_path}",
                }

            if step.action == "fs.not_exists":
                observed_exists = bool(item.result.get("exists"))
                observed_path = str(item.result.get("path", step.args["path"]))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": not observed_exists,
                    "detail": f"exists={observed_exists} path={observed_path}",
                }

            if step.action == "fs.copy":
                src = str(step.args["src"])
                dst = str(step.args["dst"])
                src_check = fs_exists(self.settings, src)
                dst_check = fs_exists(self.settings, dst)
                if not src_check["exists"] or not dst_check["exists"]:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "source or destination missing after copy"}
                src_content = str(fs_read(self.settings, src)["content"])
                dst_content = str(fs_read(self.settings, dst)["content"])
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": src_content == dst_content,
                    "detail": "copied file matches source exactly" if src_content == dst_content else "copied file content mismatch",
                }

            if step.action == "fs.read":
                actual = str(fs_read(self.settings, str(step.args["path"]))["content"])
                observed = str(item.result.get("content", ""))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == observed,
                    "detail": "read content matches filesystem" if actual == observed else "read content mismatch",
                }

            if step.action == "fs.write":
                actual = str(fs_read(self.settings, str(step.args["path"]))["content"])
                expected = str(step.args["content"])
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == expected,
                    "detail": "write content exact match" if actual == expected else "write content mismatch",
                }

            if step.action == "fs.mkdir":
                resolved = resolve_path(self.settings, str(step.args["path"]))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": resolved.exists() and resolved.is_dir(),
                    "detail": f"directory exists at {resolved}",
                }

            if step.action == "fs.list":
                actual = list(fs_list(self.settings, str(step.args["path"]))["entries"])
                observed = [str(entry) for entry in item.result.get("entries", [])]
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == observed,
                    "detail": "directory listing matches filesystem" if actual == observed else "directory listing mismatch",
                }

            if step.action == "fs.find":
                actual = list(fs_find(self.settings, str(step.args["path"]), str(step.args["pattern"]))["matches"])
                observed = [str(entry) for entry in item.result.get("matches", [])]
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == observed,
                    "detail": "file search matches filesystem" if actual == observed else "file search mismatch",
                }

            if step.action == "fs.glob":
                actual_result = fs_glob(
                    self.settings,
                    str(step.args["path"]),
                    pattern=str(step.args.get("pattern", "*")),
                    recursive=bool(step.args.get("recursive", False)),
                    file_only=bool(step.args.get("file_only", True)),
                    dir_only=bool(step.args.get("dir_only", False)),
                    path_style=str(step.args.get("path_style", "relative")),
                )
                actual_matches = list(actual_result["matches"])
                observed_matches = [str(entry) for entry in item.result.get("matches", [])]
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual_matches == observed_matches,
                    "detail": "glob matches filesystem" if actual_matches == observed_matches else "glob mismatch",
                }

            if step.action == "fs.search_content":
                actual_result = fs_search_content(
                    self.settings,
                    str(step.args["path"]),
                    str(step.args["needle"]),
                    pattern=str(step.args.get("pattern", "*")),
                    recursive=bool(step.args.get("recursive", True)),
                    file_only=bool(step.args.get("file_only", True)),
                    case_insensitive=bool(step.args.get("case_insensitive", False)),
                    path_style=str(step.args.get("path_style", "relative")),
                    max_matches=step.args.get("max_matches"),
                )
                actual_matches = list(actual_result["matches"])
                observed_matches = [str(entry) for entry in item.result.get("matches", [])]
                actual_entries = list(actual_result["entries"])
                observed_entries = list(item.result.get("entries", []))
                success = actual_matches == observed_matches and actual_entries == observed_entries
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "search content matches filesystem" if success else "search content mismatch",
                }

            if step.action == "fs.size":
                actual = int(fs_size(self.settings, str(step.args["path"]))["size_bytes"])
                observed = int(item.result.get("size_bytes", -1))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == observed,
                    "detail": "file size matches filesystem" if actual == observed else "file size mismatch",
                }

            if step.action == "fs.aggregate":
                actual_result = fs_aggregate(
                    self.settings,
                    str(step.args["path"]),
                    pattern=str(step.args.get("pattern", "*")),
                    recursive=bool(step.args.get("recursive", True)),
                    file_only=bool(step.args.get("file_only", True)),
                    include_matches=bool(step.args.get("include_matches", True)),
                    path_style=str(step.args.get("path_style", "relative")),
                    size_unit=str(step.args.get("size_unit", "auto")),
                    aggregate=str(step.args.get("aggregate", "total_size")),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "file aggregate matches filesystem" if success else "file aggregate mismatch",
                }

            if step.action == "slurm.queue":
                if not _is_fixture_mode():
                    success, detail = self._validate_live_slurm_queue(step.args, item.result)
                    return {"name": f"step_{step.id}_{step.action}", "success": success, "detail": detail}
                actual_result = slurm_queue(
                    self.settings,
                    user=step.args.get("user"),
                    state=step.args.get("state"),
                    partition=step.args.get("partition"),
                    group_by=step.args.get("group_by"),
                    limit=step.args.get("limit"),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm queue matches expected fixture output" if success else "slurm queue mismatch",
                }

            if step.action == "slurm.job_detail":
                actual_result = slurm_job_detail(self.settings, job_id=str(step.args["job_id"]))
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm job detail matches expected fixture output" if success else "slurm job detail mismatch",
                }

            if step.action == "slurm.nodes":
                if not _is_fixture_mode():
                    success, detail = self._validate_live_slurm_nodes(step.args, item.result)
                    return {"name": f"step_{step.id}_{step.action}", "success": success, "detail": detail}
                actual_result = slurm_nodes(
                    self.settings,
                    node=step.args.get("node"),
                    partition=step.args.get("partition"),
                    state=step.args.get("state"),
                    state_group=step.args.get("state_group"),
                    gpu_only=bool(step.args.get("gpu_only") or False),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm nodes match expected fixture output" if success else "slurm nodes mismatch",
                }

            if step.action == "slurm.node_detail":
                actual_result = slurm_node_detail(self.settings, node=str(step.args["node"]))
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm node detail matches expected fixture output" if success else "slurm node detail mismatch",
                }

            if step.action == "slurm.partitions":
                actual_result = slurm_partitions(self.settings, partition=step.args.get("partition"))
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm partitions match expected fixture output" if success else "slurm partitions mismatch",
                }

            if step.action == "slurm.accounting":
                actual_result = slurm_accounting(
                    self.settings,
                    user=step.args.get("user"),
                    state=step.args.get("state"),
                    partition=step.args.get("partition"),
                    start=step.args.get("start"),
                    end=step.args.get("end"),
                    min_elapsed_seconds=step.args.get("min_elapsed_seconds"),
                    max_elapsed_seconds=step.args.get("max_elapsed_seconds"),
                    limit=step.args.get("limit"),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm accounting matches expected fixture output" if success else "slurm accounting mismatch",
                }

            if step.action == "slurm.accounting_aggregate":
                if not _is_fixture_mode():
                    success, detail = self._validate_live_slurm_accounting_aggregate(step.args, item.result)
                    return {"name": f"step_{step.id}_{step.action}", "success": success, "detail": detail}
                actual_result = slurm_accounting_aggregate(
                    self.settings,
                    user=step.args.get("user"),
                    state=step.args.get("state"),
                    include_all_states=bool(step.args.get("include_all_states") or False),
                    excluded_states=list(step.args.get("excluded_states") or []),
                    default_state_applied=bool(step.args.get("default_state_applied") or False),
                    partition=step.args.get("partition"),
                    start=step.args.get("start"),
                    end=step.args.get("end"),
                    metric=str(step.args.get("metric", "average_elapsed")),
                    group_by=step.args.get("group_by"),
                    threshold_seconds=step.args.get("threshold_seconds"),
                    limit=step.args.get("limit"),
                    time_window_label=step.args.get("time_window_label"),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm accounting aggregate matches expected fixture output"
                    if success
                    else "slurm accounting aggregate mismatch",
                }

            if step.action == "slurm.metrics":
                actual_result = slurm_metrics(
                    self.settings,
                    metric_group=str(step.args.get("metric_group", "cluster_summary")),
                    start=step.args.get("start"),
                    end=step.args.get("end"),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm metrics match expected fixture output" if success else "slurm metrics mismatch",
                }

            if step.action == "slurm.slurmdbd_health":
                actual_result = slurm_slurmdbd_health(self.settings)
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm accounting health matches expected fixture output" if success else "slurm accounting health mismatch",
                }

            if step.action == "shell.exec":
                returncode = int(item.result.get("returncode", 0))
                if returncode != 0:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": f"returncode={returncode}"}

                shell_semantics = self._validate_shell_semantics(step=step, stdout=str(item.result.get("stdout", "")), goal=goal or "")
                if shell_semantics is not None:
                    return {"name": f"step_{step.id}_{step.action}", "success": shell_semantics[0], "detail": shell_semantics[1]}
                return {"name": f"step_{step.id}_{step.action}", "success": True, "detail": f"returncode={returncode}"}

            if step.action == "sql.schema":
                catalog = item.result.get("catalog")
                configured_databases = resolve_sql_databases(self.settings)
                if not isinstance(catalog, dict):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "sql.schema result missing catalog"}
                database_name = str(catalog.get("database") or "").strip()
                if not database_name:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "sql.schema catalog missing database"}
                if database_name not in configured_databases:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": f"unknown database {database_name!r}"}
                if catalog.get("error"):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": f"sql.schema catalog error: {catalog.get('error')}"}
                tables = catalog.get("tables")
                if not isinstance(tables, list):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "sql.schema catalog missing tables"}
                return {"name": f"step_{step.id}_{step.action}", "success": True, "detail": f"database={database_name} tables={len(tables)}"}

            if step.action == "sql.query":
                database_name = item.result.get("database")
                rows = item.result.get("rows")
                row_count = item.result.get("row_count")
                configured_databases = resolve_sql_databases(self.settings)
                if not isinstance(database_name, str) or not database_name:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "invalid database"}
                if database_name not in configured_databases:
                    return {
                        "name": f"step_{step.id}_{step.action}",
                        "success": False,
                        "detail": f"unknown database {database_name!r}",
                    }
                if not isinstance(rows, list):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "invalid rows"}
                if not isinstance(row_count, int):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "invalid row count"}
                if row_count < 0:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "invalid row count"}
                if row_count != len(rows):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "row_count does not match rows"}
                if rows:
                    aliases = self._extract_sql_aliases(str(step.args.get("query", "")))
                    if aliases:
                        first_row = rows[0]
                        if not isinstance(first_row, dict):
                            return {
                                "name": f"step_{step.id}_{step.action}",
                                "success": False,
                                "detail": "sql rows must contain objects when aliases are requested",
                            }
                        missing = [alias for alias in aliases if alias not in first_row]
                        if missing:
                            return {
                                "name": f"step_{step.id}_{step.action}",
                                "success": False,
                                "detail": f"missing aliased columns: {', '.join(missing)}",
                            }
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": True,
                    "detail": f"database={database_name} row_count={row_count}",
                }

            if step.action == "sql.validate":
                database_name = item.result.get("database")
                query = item.result.get("query")
                valid = item.result.get("valid")
                explanation = item.result.get("explanation")
                reason = item.result.get("reason")
                configured_databases = resolve_sql_databases(self.settings)
                if not isinstance(database_name, str) or not database_name:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "invalid database"}
                if database_name not in configured_databases:
                    return {
                        "name": f"step_{step.id}_{step.action}",
                        "success": False,
                        "detail": f"unknown database {database_name!r}",
                    }
                if not isinstance(query, str) or not query.strip():
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "sql.validate result missing query"}
                if not isinstance(valid, bool):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "sql.validate result missing valid flag"}
                if reason is not None and not isinstance(reason, str):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "sql.validate result has invalid reason"}
                if not isinstance(explanation, str) or not explanation.strip():
                    return {
                        "name": f"step_{step.id}_{step.action}",
                        "success": False,
                        "detail": "sql.validate result missing explanation",
                    }
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": True,
                    "detail": f"database={database_name} valid={str(valid).lower()}",
                }

            if step.action == "python.exec":
                success = bool(item.result.get("success", False))
                detail = str(item.result.get("error") or "python.exec returned structured result")
                if not success:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": detail}
                output = item.result.get("output")
                if isinstance(output, str):
                    manifest_check = self._validate_python_manifest(output)
                    if manifest_check is not None:
                        return {"name": f"step_{step.id}_{step.action}", "success": manifest_check[0], "detail": manifest_check[1]}
                return {"name": f"step_{step.id}_{step.action}", "success": True, "detail": "python.exec returned structured result"}

            if step.action == "text.format":
                content = item.result.get("content")
                row_count = item.result.get("row_count")
                output_format = str(item.result.get("format") or "")
                if not isinstance(content, str):
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "text.format result missing content"}
                if not isinstance(row_count, int) or row_count < 0:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "text.format result has invalid row_count"}
                if output_format not in {"txt", "csv", "json", "markdown"}:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "text.format result has invalid format"}
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": True,
                    "detail": f"text.format produced {output_format} rows={row_count}",
                }

            if step.action == "runtime.return":
                expected = runtime_return(
                    step.args.get("value"),
                    str(step.args.get("mode", "text")),
                    step.args.get("output_contract"),
                )
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": expected["output"] == item.result.get("output"),
                    "detail": "runtime.return output matches mode shaping"
                    if expected["output"] == item.result.get("output")
                    else "runtime.return output mismatch",
                }

            if getattr(self.tools, "contains", lambda _name: False)(step.action):
                success, detail = registered_tool_result_valid(step.action, item.result, self.tools)
                return {"name": f"step_{step.id}_{step.action}", "success": success, "detail": detail}
        except Exception as exc:  # noqa: BLE001
            return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": str(exc)}

        return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "unknown action"}

    def _validate_live_slurm_queue(self, args: dict, result: dict) -> tuple[bool, str]:
        """Handle the internal validate live slurm queue helper path for this module.

        Inputs:
            Receives args, result for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._validate_live_slurm_queue calls and related tests.
        """
        jobs = result.get("jobs")
        if not isinstance(jobs, list):
            return False, "slurm queue result missing jobs"
        expected_state = str(args.get("state") or "").strip()
        expected_user = str(args.get("user") or "").strip()
        expected_partition = str(args.get("partition") or "").strip()
        for job in jobs:
            if not isinstance(job, dict):
                return False, "slurm queue job row is not an object"
            if expected_state and _canonical_job_state(str(job.get("state", ""))) != _canonical_job_state(expected_state):
                return False, f"slurm queue row does not match state={expected_state}"
            if expected_user and str(job.get("user", "")) != expected_user:
                return False, f"slurm queue row does not match user={expected_user}"
            if expected_partition and str(job.get("partition", "")) != expected_partition:
                return False, f"slurm queue row does not match partition={expected_partition}"
        returned_count = int(result.get("returned_count", len(jobs)) or 0)
        total_count = int(result.get("total_count", result.get("count", returned_count)) or 0)
        if returned_count != len(jobs):
            return False, "slurm queue returned_count does not match jobs"
        if total_count < returned_count:
            return False, "slurm queue total_count is less than returned_count"
        if bool(result.get("truncated")) != (returned_count < total_count):
            return False, "slurm queue truncated flag is inconsistent"
        return True, f"slurm queue semantic validation passed rows={returned_count} total={total_count}"

    def _validate_live_slurm_accounting_aggregate(self, args: dict, result: dict) -> tuple[bool, str]:
        """Handle the internal validate live slurm accounting aggregate helper path for this module.

        Inputs:
            Receives args, result for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._validate_live_slurm_accounting_aggregate calls and related tests.
        """
        if not isinstance(result, dict):
            return False, "slurm accounting aggregate result is not an object"
        if str(result.get("result_kind") or "") != "accounting_aggregate":
            return False, "slurm accounting aggregate result_kind mismatch"
        if str(result.get("source") or "sacct") != "sacct":
            return False, "slurm accounting aggregate source must be sacct"
        expected_metric = str(args.get("metric") or "average_elapsed")
        if str(result.get("metric") or "") != expected_metric:
            return False, f"slurm accounting aggregate metric mismatch expected={expected_metric}"
        include_all_states = bool(args.get("include_all_states") or False)
        if bool(result.get("include_all_states") or False) != include_all_states:
            return False, "slurm accounting aggregate include_all_states mismatch"
        expected_state = str(args.get("state") or "").strip()
        observed_state = str(result.get("state") or "").strip()
        if include_all_states:
            if expected_state or observed_state:
                return False, "slurm accounting aggregate all-states request must not apply a state filter"
        elif expected_state and _canonical_job_state(observed_state) != _canonical_job_state(expected_state):
            return False, f"slurm accounting aggregate state mismatch expected={expected_state}"
        for field in ("user", "partition", "start", "end", "group_by", "threshold_seconds"):
            expected = args.get(field)
            if expected not in (None, "") and result.get(field) != expected:
                return False, f"slurm accounting aggregate {field} mismatch"
        job_count = int(result.get("job_count") or 0)
        total_count = int(result.get("total_count", job_count) or 0)
        returned_count = int(result.get("returned_count", job_count) or 0)
        if job_count < 0 or total_count < 0 or returned_count < 0:
            return False, "slurm accounting aggregate counts must be non-negative"
        if total_count < returned_count:
            return False, "slurm accounting aggregate total_count is less than returned_count"
        if bool(result.get("truncated")) and total_count <= returned_count:
            return False, "slurm accounting aggregate truncated flag is inconsistent"
        for field in ("average_elapsed_seconds", "min_elapsed_seconds", "max_elapsed_seconds", "sum_elapsed_seconds"):
            value = result.get(field)
            if value is not None and float(value) < 0:
                return False, f"slurm accounting aggregate {field} must be non-negative"
        return True, f"slurm accounting aggregate semantic validation passed jobs={job_count} total={total_count}"

    def _validate_live_slurm_nodes(self, args: dict, result: dict) -> tuple[bool, str]:
        """Handle the internal validate live slurm nodes helper path for this module.

        Inputs:
            Receives args, result for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._validate_live_slurm_nodes calls and related tests.
        """
        nodes = result.get("nodes")
        if not isinstance(nodes, list):
            return False, "slurm nodes result missing nodes"
        expected_state = str(args.get("state") or "").strip()
        expected_state_group = str(args.get("state_group") or "").strip().lower()
        expected_partition = str(args.get("partition") or "").strip()
        expected_node = str(args.get("node") or "").strip()
        for node in nodes:
            if not isinstance(node, dict):
                return False, "slurm node row is not an object"
            state = str(node.get("state", ""))
            if expected_node and str(node.get("name", "")) != expected_node:
                return False, f"slurm node row does not match node={expected_node}"
            if expected_partition and str(node.get("partition", "")) != expected_partition:
                return False, f"slurm node row does not match partition={expected_partition}"
            if expected_state and _canonical_node_state(state) != _canonical_node_state(expected_state):
                return False, f"slurm node row does not match state={expected_state}"
            if expected_state_group:
                if expected_state_group == "problematic":
                    if not is_problematic_node_state(state):
                        return False, "slurm node row is not problematic"
                elif expected_state_group != "all" and _canonical_node_state(state) != expected_state_group:
                    return False, f"slurm node row does not match state_group={expected_state_group}"
        partition_rows = int(result.get("partition_row_count", result.get("count", len(nodes))) or 0)
        if partition_rows != len(nodes):
            return False, "slurm nodes partition_row_count does not match nodes"
        return True, f"slurm nodes semantic validation passed partition_rows={partition_rows}"

    def _validate_python_manifest(self, output: str) -> tuple[bool, str] | None:
        """Handle the internal validate python manifest helper path for this module.

        Inputs:
            Receives output for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._validate_python_manifest calls and related tests.
        """
        try:
            payload = extract_json_object(output)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None

        operation = payload.get("operation")
        if operation != "bulk_copy":
            return None

        src_dir = payload.get("src_dir")
        dst_dir = payload.get("dst_dir")
        copied_files = payload.get("copied_files")
        if not isinstance(src_dir, str) or not isinstance(dst_dir, str) or not isinstance(copied_files, list):
            return False, "python.exec bulk_copy manifest is missing required fields"

        src_entries = [name for name in fs_list(self.settings, src_dir)["entries"] if str(name).endswith(".txt")]
        dst_entries = fs_list(self.settings, dst_dir)["entries"]
        normalized_copied = [str(name) for name in copied_files]
        if sorted(src_entries) != sorted(normalized_copied):
            return False, "python.exec bulk_copy manifest does not match source txt files"
        for name in src_entries:
            if name not in dst_entries:
                return False, f"bulk copy missing destination file {name}"
            src_content = fs_read(self.settings, f"{src_dir}/{name}")["content"]
            dst_content = fs_read(self.settings, f"{dst_dir}/{name}")["content"]
            if src_content != dst_content:
                return False, f"bulk copy content mismatch for {name}"
        return True, "python.exec bulk_copy manifest verified"

    def _extract_sql_aliases(self, query: str) -> list[str]:
        """Handle the internal extract sql aliases helper path for this module.

        Inputs:
            Receives query for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._extract_sql_aliases calls and related tests.
        """
        aliases: list[str] = []
        select_list = _top_level_select_list(str(query or ""))
        for match in ALIAS_RE.finditer(select_list):
            alias = match.group(2)
            if alias.lower() in SQL_TYPE_ALIAS_FALSE_POSITIVES:
                continue
            if alias not in aliases:
                aliases.append(alias)
        return aliases

    def _validate_shell_semantics(self, *, step, stdout: str, goal: str) -> tuple[bool, str] | None:
        """Handle the internal validate shell semantics helper path for this module.

        Inputs:
            Receives step, stdout, goal for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._validate_shell_semantics calls and related tests.
        """
        intent = self._classify_storage_intent(goal)
        if intent is None:
            return None

        command = str(step.args.get("command", ""))
        lines = [line for line in stdout.splitlines() if line.strip()]
        if intent.startswith("folder_usage"):
            if re.search(r"\bdf\b", command) or (lines and lines[0].startswith("Filesystem")):
                return False, "filesystem capacity output does not answer a folder-usage question"
            if not lines or not DU_OUTPUT_RE.match(lines[0]):
                return False, "shell output does not look like folder disk-usage output"
            return True, "folder usage output matched expected disk-usage shape"

        if intent == "filesystem_capacity":
            if re.search(r"\bdu\b", command):
                return False, "folder disk-usage output does not answer a filesystem-capacity question"
            if not lines or not lines[0].startswith("Filesystem"):
                return False, "shell output does not look like filesystem capacity output"
            return True, "filesystem capacity output matched expected shape"

        return None

    def _classify_storage_intent(self, goal: str) -> str | None:
        """Handle the internal classify storage intent helper path for this module.

        Inputs:
            Receives goal for this RuntimeValidator method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RuntimeValidator._classify_storage_intent calls and related tests.
        """
        tokens = set(STORAGE_TOKEN_RE.findall(str(goal or "").lower()))
        if not tokens:
            return None

        folder_terms = {"folder", "folders", "directory", "directories"}
        filesystem_terms = {"disk", "disks", "filesystem", "filesystems", "partition", "partitions", "mount", "mounted", "drive", "drives"}
        usage_terms = {"space", "size", "usage", "used", "consuming", "largest", "biggest", "heaviest", "most"}
        system_scope_terms = {"computer", "system", "root", "whole", "entire"}

        mentions_usage = bool(tokens & usage_terms)
        mentions_folder = bool(tokens & folder_terms)
        mentions_filesystem = bool(tokens & filesystem_terms)
        mentions_system_scope = bool(tokens & system_scope_terms)

        if mentions_folder and mentions_usage:
            return "folder_usage_system" if mentions_system_scope else "folder_usage_workspace"
        if mentions_filesystem and mentions_usage:
            return "filesystem_capacity"
        return None
