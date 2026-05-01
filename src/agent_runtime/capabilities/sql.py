"""Structured placeholder SQL capability."""

from __future__ import annotations

from typing import Any

from agent_runtime.capabilities.base import BaseCapability
from agent_runtime.capabilities.schemas import CapabilityManifest
from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ExecutionResult

READ_ONLY_TEMPLATES = {"select_rows", "count_rows", "group_count"}


class ReadQueryCapability(BaseCapability):
    """Validate a structured read-only query intent without executing raw SQL."""

    manifest = CapabilityManifest(
        capability_id="sql.read_query",
        domain="sql",
        operation_id="read_query",
        name="Read Query Intent",
        description="Validate and plan a structured read-only query intent.",
        semantic_verbs=["read", "analyze"],
        object_types=["table", "database", "query"],
        argument_schema={"query_intent": {"type": "object"}},
        required_arguments=["query_intent"],
        optional_arguments=["limit"],
        output_schema={"query_plan": {"type": "object"}, "rows": {"type": "array"}},
        risk_level="medium",
        read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        examples=[
            {
                "arguments": {
                    "query_intent": {
                        "template": "count_rows",
                        "table": "patients",
                    },
                    "limit": 10,
                }
            }
        ],
        safety_notes=["Only structured read-only query intents are accepted."],
    )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ExecutionResult:
        """Accept only simple read-only query-intent templates."""

        payload = {"limit": 100}
        payload.update(dict(arguments or {}))
        validated = self.validate_arguments(payload)
        query_intent = validated["query_intent"]
        if not isinstance(query_intent, dict):
            raise ValidationError("query_intent must be an object.")

        if any(key in query_intent for key in {"sql", "query", "command"}):
            raise ValidationError("arbitrary SQL text is not allowed in query_intent.")

        template = str(query_intent.get("template") or "").strip()
        table = str(query_intent.get("table") or "").strip()
        if template not in READ_ONLY_TEMPLATES:
            raise ValidationError(f"unsupported query template: {template or 'missing'}")
        if not table:
            raise ValidationError("query_intent.table is required.")

        columns = query_intent.get("columns", [])
        if columns and not isinstance(columns, list):
            raise ValidationError("query_intent.columns must be a list when provided.")
        filters = query_intent.get("filters", {})
        if filters and not isinstance(filters, dict):
            raise ValidationError("query_intent.filters must be an object when provided.")
        group_by = query_intent.get("group_by")
        if group_by is not None and not isinstance(group_by, list):
            raise ValidationError("query_intent.group_by must be a list when provided.")

        limit = max(1, int(validated.get("limit", 100)))
        query_plan = {
            "template": template,
            "table": table,
            "columns": list(columns or []),
            "filters": dict(filters or {}),
            "group_by": list(group_by or []),
            "limit": limit,
        }

        return ExecutionResult(
            node_id=str(context.get("node_id") or ""),
            status="success",
            data_preview={"query_plan": query_plan, "rows": []},
            metadata={"planned_only": True},
        )
