from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.runtime.tool_output_contracts import ToolOutputContract, contract_for_tool
from aor_runtime.tools.base import BaseTool, ToolRegistry


ToolSurfaceCategory = Literal["sql", "filesystem", "shell", "slurm", "text", "runtime", "python", "generic"]


@dataclass(frozen=True)
class ToolSurfaceContract:
    name: str
    category: ToolSurfaceCategory
    friendly_label: str
    argument_schema: dict[str, Any] = field(default_factory=dict)
    result_model: type[Any] | None = None
    output_contract: ToolOutputContract | None = None
    default_display_path: str | None = None
    formatter_source_path: str | None = None
    result_kind: str = "generic"
    presentation_category: ToolSurfaceCategory = "generic"


def build_tool_surface(tool: BaseTool) -> ToolSurfaceContract:
    name = str(tool.spec.name)
    output_contract = contract_for_tool(name)
    category = category_for_tool(name)
    return ToolSurfaceContract(
        name=name,
        category=category,
        friendly_label=friendly_label_for_tool(name),
        argument_schema=dict(tool.spec.arguments_schema or {}),
        result_model=tool.result_model,
        output_contract=output_contract,
        default_display_path=output_contract.default_path if output_contract else None,
        formatter_source_path=(output_contract.formatter_source_path or output_contract.default_path) if output_contract else None,
        result_kind=_result_kind_for_tool(name, output_contract),
        presentation_category=category,
    )


def build_tool_surfaces(registry: ToolRegistry) -> dict[str, ToolSurfaceContract]:
    return {name: build_tool_surface(registry.get(name)) for name in registry.names()}


def category_for_tool(name: str) -> ToolSurfaceCategory:
    tool = str(name or "")
    if tool.startswith("sql."):
        return "sql"
    if tool.startswith("fs."):
        return "filesystem"
    if tool.startswith("shell."):
        return "shell"
    if tool.startswith("slurm."):
        return "slurm"
    if tool.startswith("text."):
        return "text"
    if tool.startswith("runtime."):
        return "runtime"
    if tool.startswith("python."):
        return "python"
    return "generic"


def friendly_label_for_tool(name: str, args: dict[str, Any] | None = None) -> str:
    tool = str(name or "")
    values = dict(args or {})
    if tool == "sql.query":
        database = _clean(values.get("database"))
        return f"Query {database}" if database else "Query database"
    if tool == "sql.schema":
        database = _clean(values.get("database") or values.get("domain"))
        return f"Inspect {database} schema" if database else "Inspect database schema"
    if tool == "sql.validate":
        return "Validate SQL"
    if tool == "text.format":
        output_format = _clean(values.get("format")) or "output"
        return f"Format results as {output_format}"
    if tool == "fs.write":
        path = _clean(values.get("path")) or "file"
        return f"Write {path}"
    if tool == "fs.read":
        path = _clean(values.get("path")) or "file"
        return f"Read {path}"
    if tool.startswith("fs."):
        return f"Use {tool}"
    if tool == "shell.exec":
        return "Run safe shell inspection"
    if tool.startswith("slurm."):
        return f"Inspect SLURM with {tool}"
    if tool == "runtime.return":
        return "Return answer"
    return f"Run {tool}"


def registered_tool_result_valid(tool_name: str, result: Any, registry: ToolRegistry) -> tuple[bool, str]:
    try:
        tool = registry.get(tool_name)
    except KeyError:
        return False, "unknown action"
    try:
        tool.result_model.model_validate(result)
    except Exception as exc:  # noqa: BLE001
        return False, f"{tool_name} result schema mismatch: {exc}"
    return True, f"{tool_name} result matches registered schema"


def _result_kind_for_tool(name: str, contract: ToolOutputContract | None) -> str:
    if contract is None:
        return category_for_tool(name)
    if contract.collection_paths:
        return "table"
    if contract.file_paths:
        return "file"
    if contract.text_paths:
        return "text"
    if contract.scalar_paths:
        return "scalar"
    return category_for_tool(name)


def _clean(value: Any) -> str:
    return str(value or "").strip()
