from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from aor_runtime.core.utils import extract_json_object
from aor_runtime.runtime.markdown import cell as md_cell
from aor_runtime.runtime.markdown import section as md_section
from aor_runtime.runtime.markdown import table as md_table
from aor_runtime.runtime.presentation import PresentationContext, strip_internal_telemetry
from aor_runtime.runtime.slurm_result_normalizer import SlurmResultKind, normalize_slurm_result


IntelligentOutputMode = Literal["off", "compare", "replace"]
RenderStyle = Literal["table", "key_value", "bullets"]


@dataclass(frozen=True)
class DisplayField:
    id: str
    label: str
    type: str = "text"
    unit: str | None = None
    description: str = ""
    domain: str = "generic"

    def prompt_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "description": self.description,
            "domain": self.domain,
        }
        if self.unit:
            data["unit"] = self.unit
        return data


@dataclass
class DisplayFieldCatalog:
    domain: str
    title: str
    fields: list[DisplayField]
    records: list[dict[str, Any]] = field(default_factory=list)
    default_fields: list[str] = field(default_factory=list)
    render_style: RenderStyle = "table"
    source_tools: list[str] = field(default_factory=list)

    def field_map(self) -> dict[str, DisplayField]:
        return {field.id: field for field in self.fields}

    def prompt_payload(self, *, goal: str, max_fields: int) -> dict[str, Any]:
        return {
            "user_goal": str(goal or ""),
            "domain": self.domain,
            "title": self.title,
            "source_tools": list(self.source_tools),
            "record_count": len(self.records),
            "max_fields": max_fields,
            "available_fields": [field.prompt_dict() for field in self.fields],
            "default_fields": list(self.default_fields),
            "allowed_render_styles": ["table", "key_value", "bullets"],
            "data_visibility_rule": "Only field metadata is provided. Actual values are local-only and unavailable to the LLM.",
        }


@dataclass
class IntelligentOutputSelection:
    selected_fields: list[str]
    title: str = "Intelligent Output"
    render_style: RenderStyle = "table"
    rationale: str = ""
    confidence: float | None = None


@dataclass
class IntelligentOutputResult:
    markdown: str
    selected_fields: list[str]
    prompt_payload: dict[str, Any]
    llm_used: bool = True


def render_intelligent_output(
    result: Any,
    actions: list[Any],
    context: PresentationContext,
    settings: Any,
    *,
    mode: str = "off",
    max_fields: int = 8,
) -> IntelligentOutputResult | None:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode not in {"compare", "replace"} or settings is None:
        return None
    catalog = build_display_field_catalog(result, actions, context)
    if catalog is None or not catalog.fields or not catalog.records:
        return None
    max_field_count = max(1, int(max_fields or 8))
    prompt_payload = catalog.prompt_payload(goal=context.goal, max_fields=max_field_count)
    selection = _select_fields_with_llm(prompt_payload, catalog, settings, max_fields=max_field_count)
    if selection is None:
        return None
    markdown = _render_selection(catalog, selection, max_rows=context.max_rows)
    if not markdown:
        return None
    return IntelligentOutputResult(markdown=markdown, selected_fields=selection.selected_fields, prompt_payload=prompt_payload)


def build_display_field_catalog(
    result: Any,
    actions: list[Any],
    context: PresentationContext,
) -> DisplayFieldCatalog | None:
    clean = strip_internal_telemetry(result)
    source_tools = _action_tools(actions)
    source_action = str(context.source_action or (source_tools[-1] if source_tools else "") or "")

    slurm_catalog = _slurm_catalog(clean, source_action, context, source_tools)
    if slurm_catalog is not None:
        return slurm_catalog
    sql_catalog = _sql_catalog(clean, source_tools)
    if sql_catalog is not None:
        return sql_catalog
    fs_catalog = _filesystem_catalog(clean, source_tools)
    if fs_catalog is not None:
        return fs_catalog
    return _generic_catalog(clean, source_tools)


def _select_fields_with_llm(
    prompt_payload: dict[str, Any],
    catalog: DisplayFieldCatalog,
    settings: Any,
    *,
    max_fields: int,
) -> IntelligentOutputSelection | None:
    try:
        from aor_runtime.llm.client import LLMClient

        client = LLMClient(settings)
        system_prompt = (
            "You select which display fields best answer the user's request. "
            "You receive field metadata only, never data values. "
            "Return one JSON object with keys: title, render_style, selected_fields, rationale, confidence. "
            "selected_fields must contain only ids from available_fields and should be concise. "
            "Do not invent fields. Do not request raw data."
        )
        if hasattr(client, "complete_json"):
            raw = client.complete_json(system_prompt=system_prompt, user_prompt=json.dumps(prompt_payload, sort_keys=True), temperature=0.0)
        else:
            text = client.complete(system_prompt=system_prompt, user_prompt=json.dumps(prompt_payload, sort_keys=True), temperature=0.0)
            raw = extract_json_object(text)
        if not isinstance(raw, dict):
            return None
        return _coerce_selection(raw, catalog, max_fields=max_fields)
    except Exception:
        return None


def _coerce_selection(raw: dict[str, Any], catalog: DisplayFieldCatalog, *, max_fields: int) -> IntelligentOutputSelection | None:
    allowed = catalog.field_map()
    selected: list[str] = []
    for item in raw.get("selected_fields") or []:
        field_id = str(item or "").strip()
        if field_id in allowed and field_id not in selected:
            selected.append(field_id)
        if len(selected) >= max_fields:
            break
    if not selected:
        return None
    render_style = str(raw.get("render_style") or catalog.render_style or "table").strip().lower()
    if render_style not in {"table", "key_value", "bullets"}:
        render_style = catalog.render_style
    confidence = raw.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except Exception:
        confidence_value = None
    return IntelligentOutputSelection(
        selected_fields=selected,
        title=str(raw.get("title") or catalog.title or "Intelligent Output").strip() or "Intelligent Output",
        render_style=render_style,  # type: ignore[arg-type]
        rationale=str(raw.get("rationale") or "").strip(),
        confidence=confidence_value,
    )


def _render_selection(catalog: DisplayFieldCatalog, selection: IntelligentOutputSelection, *, max_rows: int) -> str:
    fields = [catalog.field_map()[field_id] for field_id in selection.selected_fields if field_id in catalog.field_map()]
    if not fields:
        return ""
    rows = list(catalog.records or [])
    limit = max(1, int(max_rows or 20))
    lines = md_section("Intelligent Output")
    lines.append("")
    if selection.title and selection.title != "Intelligent Output":
        lines.append(f"**{md_cell(selection.title)}**")
        lines.append("")
    if selection.render_style == "key_value" and len(rows) == 1:
        lines.extend(md_table(["Field", "Value"], [[_code_cell(field.label), _code_cell(rows[0].get(field.id))] for field in fields]))
    elif selection.render_style == "bullets" and len(rows) == 1:
        for field in fields:
            lines.append(f"- **{md_cell(field.label)}:** `{_inline_code(rows[0].get(field.id))}`")
    else:
        rendered_rows = rows[:limit]
        lines.extend(md_table([field.label for field in fields], [[_code_cell(row.get(field.id)) for field in fields] for row in rendered_rows]))
        if len(rows) > limit:
            lines.append("")
            lines.append(f"Showing first {limit} of {len(rows)} rows.")
    if selection.rationale:
        lines.append("")
        lines.append(f"Selection: `{_inline_code(selection.rationale)}`")
    return "\n".join(lines).strip()


def _slurm_catalog(result: Any, source_action: str, context: PresentationContext, source_tools: list[str]) -> DisplayFieldCatalog | None:
    if not (source_action.startswith("slurm.") or _looks_like_slurm_payload(result)):
        return None
    normalized = normalize_slurm_result(result, context)
    if normalized.kind == SlurmResultKind.ACCOUNTING_AGGREGATE:
        metric_rows = [dict(row) for row in list(normalized.grouped.get("metrics") or []) if isinstance(row, dict)]
        if metric_rows:
            summary = dict(normalized.summary or {})
            records = [
                {
                    **row,
                    "partition": row.get("partition") or summary.get("partition"),
                    "state": row.get("state") or summary.get("state"),
                    "time_window_label": summary.get("time_window_label") or _time_window(summary),
                    "source": summary.get("source"),
                }
                for row in metric_rows
            ]
            return DisplayFieldCatalog(
                domain="slurm",
                title=normalized.title,
                fields=[
                    DisplayField("metric", "Metric", description="Runtime metric name", domain="slurm"),
                    DisplayField("value", "Value", description="Requested metric value", domain="slurm"),
                    DisplayField("jobs", "Jobs", type="number", description="Number of jobs included", domain="slurm"),
                    DisplayField("average", "Average", unit="duration", description="Average elapsed runtime", domain="slurm"),
                    DisplayField("minimum", "Min", unit="duration", description="Minimum elapsed runtime", domain="slurm"),
                    DisplayField("maximum", "Max", unit="duration", description="Maximum elapsed runtime", domain="slurm"),
                    DisplayField("total", "Total", unit="duration", description="Total elapsed runtime", domain="slurm"),
                    DisplayField("partition", "Partition", description="SLURM partition filter", domain="slurm"),
                    DisplayField("state", "State", description="SLURM state filter", domain="slurm"),
                    DisplayField("time_window_label", "Time Window", description="Resolved time window", domain="slurm"),
                    DisplayField("source", "Source", description="SLURM source command family", domain="slurm"),
                ],
                records=records,
                default_fields=["metric", "value", "jobs", "average", "minimum", "maximum", "total", "time_window_label"],
                source_tools=source_tools,
            )
        summary = dict(normalized.summary or {})
        record = {
            "metric": summary.get("metric"),
            "value_human": summary.get("value_human") or summary.get("average_elapsed_human"),
            "job_count": summary.get("job_count"),
            "average_elapsed_human": summary.get("average_elapsed_human"),
            "min_elapsed_human": summary.get("min_elapsed_human"),
            "max_elapsed_human": summary.get("max_elapsed_human"),
            "sum_elapsed_human": summary.get("sum_elapsed_human"),
            "partition": summary.get("partition"),
            "state": summary.get("state"),
            "time_window_label": summary.get("time_window_label") or _time_window(summary),
            "source": summary.get("source"),
        }
        return DisplayFieldCatalog(
            domain="slurm",
            title=normalized.title,
            fields=[
                DisplayField("metric", "Metric", description="Runtime metric name", domain="slurm"),
                DisplayField("value_human", "Value", unit="duration", description="Requested metric value", domain="slurm"),
                DisplayField("job_count", "Jobs", type="number", description="Number of jobs included", domain="slurm"),
                DisplayField("average_elapsed_human", "Average", unit="duration", description="Average elapsed runtime", domain="slurm"),
                DisplayField("min_elapsed_human", "Min", unit="duration", description="Minimum elapsed runtime", domain="slurm"),
                DisplayField("max_elapsed_human", "Max", unit="duration", description="Maximum elapsed runtime", domain="slurm"),
                DisplayField("sum_elapsed_human", "Total", unit="duration", description="Total elapsed runtime", domain="slurm"),
                DisplayField("partition", "Partition", description="SLURM partition filter", domain="slurm"),
                DisplayField("state", "State", description="SLURM state filter", domain="slurm"),
                DisplayField("time_window_label", "Time Window", description="Resolved time window", domain="slurm"),
                DisplayField("source", "Source", description="SLURM source command family", domain="slurm"),
            ],
            records=[_drop_empty(record)],
            default_fields=["metric", "value_human", "job_count", "partition", "time_window_label"],
            render_style="key_value",
            source_tools=source_tools,
        )
    rows = list(normalized.rows or [])
    if rows:
        return _catalog_from_records("slurm", normalized.title, rows, source_tools=source_tools)
    if normalized.summary:
        return _catalog_from_records("slurm", normalized.title, [dict(normalized.summary)], source_tools=source_tools, render_style="key_value")
    return None


def _sql_catalog(result: Any, source_tools: list[str]) -> DisplayFieldCatalog | None:
    if not isinstance(result, dict) or "rows" not in result:
        return None
    rows = [dict(row) for row in list(result.get("rows") or []) if isinstance(row, dict)]
    if not rows:
        row_count = result.get("row_count")
        rows = [_drop_empty({"row_count": row_count, "database": result.get("database")})]
    return _catalog_from_records("sql", "SQL Results", rows, source_tools=source_tools)


def _filesystem_catalog(result: Any, source_tools: list[str]) -> DisplayFieldCatalog | None:
    if not isinstance(result, dict):
        return None
    collection = result.get("matches") or result.get("entries")
    if isinstance(collection, list) and collection:
        rows = [dict(item) if isinstance(item, dict) else {"value": item} for item in collection]
        return _catalog_from_records("filesystem", "Filesystem Results", rows, source_tools=source_tools)
    keys = {"file_count", "total_size_bytes", "display_size", "path", "pattern"}
    if keys & set(result):
        return _catalog_from_records("filesystem", "Filesystem Results", [_drop_empty({key: result.get(key) for key in keys})], source_tools=source_tools)
    return None


def _generic_catalog(result: Any, source_tools: list[str]) -> DisplayFieldCatalog | None:
    if isinstance(result, list) and result and all(isinstance(item, dict) for item in result):
        return _catalog_from_records("generic", "Result", [dict(item) for item in result], source_tools=source_tools)
    if isinstance(result, dict):
        scalar_items = {
            str(key): value
            for key, value in result.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        if scalar_items:
            return _catalog_from_records("generic", "Result", [scalar_items], source_tools=source_tools, render_style="key_value")
    return None


def _catalog_from_records(
    domain: str,
    title: str,
    records: list[dict[str, Any]],
    *,
    source_tools: list[str],
    render_style: RenderStyle = "table",
) -> DisplayFieldCatalog | None:
    if not records:
        return None
    field_ids: list[str] = []
    for record in records:
        for key in record:
            field_id = str(key)
            if field_id not in field_ids and _is_safe_field_id(field_id):
                field_ids.append(field_id)
    if not field_ids:
        return None
    fields = [
        DisplayField(field_id, field_id.replace("_", " ").title(), _field_type(records, field_id), description=f"{field_id} from {domain} result", domain=domain)
        for field_id in field_ids
    ]
    return DisplayFieldCatalog(
        domain=domain,
        title=title,
        fields=fields,
        records=[_drop_empty({field_id: record.get(field_id) for field_id in field_ids}) for record in records],
        default_fields=field_ids[:8],
        render_style=render_style,
        source_tools=source_tools,
    )


def _looks_like_slurm_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "metric_group" in value or "results" in value:
        return True
    metric_children = [
        item
        for item in value.values()
        if isinstance(item, dict)
        and (item.get("result_kind") == "accounting_aggregate" or {"average_elapsed_seconds", "min_elapsed_seconds", "max_elapsed_seconds"} & set(item))
    ]
    if metric_children and len(metric_children) == len(value):
        return True
    return any(key in value for key in ("jobs", "nodes", "partitions"))


def _action_tools(actions: list[Any]) -> list[str]:
    tools: list[str] = []
    for action in actions:
        tool = str(getattr(action, "tool", "") or (action.get("tool") if isinstance(action, dict) else "") or "")
        if tool and tool != "runtime.return" and tool not in tools:
            tools.append(tool)
    return tools


def _field_type(records: list[dict[str, Any]], field_id: str) -> str:
    for record in records:
        value = record.get(field_id)
        if value is None:
            continue
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        return "text"
    return "text"


def _is_safe_field_id(field_id: str) -> bool:
    lowered = field_id.lower()
    forbidden = ("password", "token", "secret", "credential", "raw", "payload", "stdout", "stderr")
    return bool(field_id.strip()) and not any(part in lowered for part in forbidden)


def _time_window(summary: dict[str, Any]) -> str:
    start = summary.get("start")
    end = summary.get("end")
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"Since {start}"
    if end:
        return f"Until {end}"
    return ""


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _inline_code(value: Any) -> str:
    return md_cell(value).replace("`", "'")


def _code_cell(value: Any) -> str:
    return f"`{_inline_code(value)}`"
