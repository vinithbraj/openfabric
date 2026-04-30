from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolOutputContract:
    default_path: str | None = None
    collection_paths: tuple[str, ...] = ()
    scalar_paths: tuple[str, ...] = ()
    text_paths: tuple[str, ...] = ()
    file_paths: tuple[str, ...] = ()
    formatter_source_path: str | None = None
    return_value_path: str | None = None
    path_aliases: dict[str, str] | None = None

    @property
    def declared_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for path in (
            self.default_path,
            self.formatter_source_path,
            self.return_value_path,
            *self.collection_paths,
            *self.scalar_paths,
            *self.text_paths,
            *self.file_paths,
        ):
            if path and path not in paths:
                paths.append(path)
        aliases = self.path_aliases or {}
        for alias in aliases:
            if alias and alias not in paths:
                paths.append(alias)
        return tuple(paths)


RefUse = Literal["default", "formatter", "return", "scalar"]


TOOL_OUTPUT_CONTRACTS: dict[str, ToolOutputContract] = {
    "fs.aggregate": ToolOutputContract(
        default_path="summary_text",
        scalar_paths=("file_count", "total_size_bytes"),
        text_paths=("summary_text",),
    ),
    "fs.exists": ToolOutputContract(default_path="exists", scalar_paths=("exists",)),
    "fs.find": ToolOutputContract(default_path="matches", collection_paths=("matches",)),
    "fs.glob": ToolOutputContract(default_path="matches", collection_paths=("matches",)),
    "fs.list": ToolOutputContract(default_path="entries", collection_paths=("entries",)),
    "fs.not_exists": ToolOutputContract(default_path="exists", scalar_paths=("exists",)),
    "fs.read": ToolOutputContract(default_path="content", text_paths=("content",)),
    "fs.search_content": ToolOutputContract(default_path="matches", collection_paths=("matches", "entries")),
    "fs.size": ToolOutputContract(default_path="size_bytes", scalar_paths=("size_bytes",)),
    "fs.write": ToolOutputContract(
        default_path="path",
        scalar_paths=("bytes_written",),
        file_paths=("path",),
        return_value_path="path",
    ),
    "python.exec": ToolOutputContract(
        default_path="result",
        collection_paths=("result",),
        scalar_paths=("result",),
        text_paths=("output",),
        return_value_path="result",
    ),
    "runtime.return": ToolOutputContract(
        default_path="value",
        scalar_paths=("value",),
        text_paths=("output", "value"),
        return_value_path="value",
    ),
    "shell.exec": ToolOutputContract(
        default_path="stdout",
        scalar_paths=("stdout", "returncode"),
        text_paths=("stdout", "stderr"),
        path_aliases={"exit_code": "returncode"},
    ),
    "slurm.accounting": ToolOutputContract(
        default_path="jobs",
        collection_paths=("jobs",),
        scalar_paths=("count", "total_count", "returned_count"),
    ),
    "slurm.accounting_aggregate": ToolOutputContract(
        default_path="count",
        collection_paths=("groups", "grouped"),
        scalar_paths=("count", "job_count", "count_longer_than", "total_count", "returned_count"),
        path_aliases={"grouped": "groups"},
    ),
    "slurm.job_detail": ToolOutputContract(default_path="fields", collection_paths=("fields",)),
    "slurm.metrics": ToolOutputContract(
        default_path="payload",
        collection_paths=("jobs", "nodes", "partitions"),
        scalar_paths=("queue_count", "job_count", "pending_jobs", "running_jobs", "node_count", "problematic_nodes"),
    ),
    "slurm.node_detail": ToolOutputContract(default_path="fields", collection_paths=("fields",)),
    "slurm.nodes": ToolOutputContract(
        default_path="nodes",
        collection_paths=("nodes",),
        scalar_paths=("unique_count", "partition_row_count", "count"),
    ),
    "slurm.partitions": ToolOutputContract(
        default_path="partitions",
        collection_paths=("partitions",),
        scalar_paths=("count",),
    ),
    "slurm.queue": ToolOutputContract(
        default_path="jobs",
        collection_paths=("jobs",),
        scalar_paths=("count", "total_count", "returned_count"),
    ),
    "sql.query": ToolOutputContract(default_path="rows", collection_paths=("rows",)),
    "sql.schema": ToolOutputContract(default_path="catalog", collection_paths=("catalog", "tables")),
    "text.format": ToolOutputContract(default_path="content", text_paths=("content",), return_value_path="content"),
}


def contract_for_tool(tool: str) -> ToolOutputContract | None:
    return TOOL_OUTPUT_CONTRACTS.get(str(tool or ""))


def default_path_for_tool(tool: str) -> str | None:
    contract = contract_for_tool(tool)
    return contract.default_path if contract else None


def formatter_source_path_for_tool(tool: str) -> str | None:
    contract = contract_for_tool(tool)
    if contract is None:
        return None
    return contract.formatter_source_path or contract.default_path


def return_value_path_for_tool(tool: str) -> str | None:
    contract = contract_for_tool(tool)
    if contract is None:
        return None
    return contract.return_value_path or contract.default_path


def available_paths_for_tool(tool: str) -> tuple[str, ...]:
    contract = contract_for_tool(tool)
    return contract.declared_paths if contract else ()


def normalize_tool_ref_path(tool: str, path: str | None, *, use: RefUse = "default") -> str | None:
    contract = contract_for_tool(tool)
    if contract is None:
        return None if path is None else str(path).strip() or None
    normalized = None if path is None else str(path).strip()
    if normalized:
        root, suffix = _split_path(normalized)
        canonical_root = (contract.path_aliases or {}).get(root, root)
        if canonical_root in contract.declared_paths:
            return f"{canonical_root}.{suffix}" if suffix else canonical_root
        return normalized
    if use == "formatter":
        return contract.formatter_source_path or contract.default_path
    if use == "return":
        return contract.return_value_path or contract.default_path
    return contract.default_path


def path_is_declared_for_tool(tool: str, path: str | None) -> bool:
    contract = contract_for_tool(tool)
    if contract is None:
        return True
    normalized = None if path is None else str(path).strip()
    if not normalized:
        return True
    root, _suffix = _split_path(normalized)
    root = (contract.path_aliases or {}).get(root, root)
    return root in contract.declared_paths


def root_path(path: str | None) -> str | None:
    normalized = None if path is None else str(path).strip()
    if not normalized:
        return None
    return normalized.split(".", 1)[0]


def _split_path(path: str) -> tuple[str, str]:
    root, dot, suffix = str(path or "").strip().partition(".")
    return root, suffix if dot else ""
