"""Open WebUI-friendly event formatting."""

from __future__ import annotations

from typing import Any

from agent_runtime.observability.events import PipelineEvent


def _pretty_stage(stage: str) -> str:
    return str(stage or "").replace("_", " ").strip().title() or "Pipeline"


def _render_detail_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        preview = ", ".join(str(item) for item in value[:5])
        if len(value) > 5:
            preview += f", ... ({len(value)} items)"
        return preview
    if isinstance(value, dict):
        if not value:
            return None
        pairs = []
        for index, (key, item) in enumerate(value.items()):
            if index >= 4:
                pairs.append("...")
                break
            pairs.append(f"{key}={item}")
        return ", ".join(pairs)
    return str(value)


def format_event_for_openwebui(event: PipelineEvent) -> str:
    """Render one pipeline event as concise markdown."""

    lines = [f"[{_pretty_stage(event.stage)}] {event.title}", "", event.summary]
    for key, value in event.details.items():
        rendered = _render_detail_value(value)
        if rendered:
            lines.append(f"- {str(key).replace('_', ' ').title()}: {rendered}")
    return "\n".join(lines).strip() + "\n\n"

