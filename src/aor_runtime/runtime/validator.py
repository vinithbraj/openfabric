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
    slurm_accounting,
    slurm_job_detail,
    slurm_metrics,
    slurm_node_detail,
    slurm_nodes,
    slurm_partitions,
    slurm_queue,
    slurm_slurmdbd_health,
)
from aor_runtime.tools.sql import resolve_sql_databases


ALIAS_RE = re.compile(r'\bas\s+("?)([a-zA-Z_][a-zA-Z0-9_]*)\1', re.IGNORECASE)
STORAGE_TOKEN_RE = re.compile(r"[a-z0-9_]+")
DU_OUTPUT_RE = re.compile(r"^\s*[0-9.]+[A-Za-z]+\s+\S+")


class RuntimeValidator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, history: list[StepLog], goal: str | None = None) -> tuple[ValidationResult, list[dict[str, str | bool]]]:
        checks: list[dict[str, str | bool]] = []
        for item in history:
            checks.append(self._validate_step(item, goal=goal))
        failed = [check for check in checks if not bool(check["success"])]
        if failed:
            first = failed[0]
            return ValidationResult(success=False, reason=str(first["detail"])), checks
        return ValidationResult(success=True, reason=None), checks

    def _validate_step(self, item: StepLog, *, goal: str | None = None) -> dict[str, str | bool]:
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
                actual_result = slurm_queue(
                    self.settings,
                    user=step.args.get("user"),
                    state=step.args.get("state"),
                    partition=step.args.get("partition"),
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
                actual_result = slurm_nodes(
                    self.settings,
                    node=step.args.get("node"),
                    partition=step.args.get("partition"),
                    state=step.args.get("state"),
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
                    limit=step.args.get("limit"),
                )
                success = actual_result == item.result
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": success,
                    "detail": "slurm accounting matches expected fixture output" if success else "slurm accounting mismatch",
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
        except Exception as exc:  # noqa: BLE001
            return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": str(exc)}

        return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "unknown action"}

    def _validate_python_manifest(self, output: str) -> tuple[bool, str] | None:
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
        aliases: list[str] = []
        for match in ALIAS_RE.finditer(query):
            alias = match.group(2)
            if alias not in aliases:
                aliases.append(alias)
        return aliases

    def _validate_shell_semantics(self, *, step, stdout: str, goal: str) -> tuple[bool, str] | None:
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
