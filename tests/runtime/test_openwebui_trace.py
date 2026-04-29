from __future__ import annotations

from types import SimpleNamespace

from aor_runtime.runtime.openwebui_trace import OpenWebUITraceRenderer, resolve_openwebui_trace_mode, sanitize_detail


def test_openwebui_trace_mode_uses_explicit_mode() -> None:
    assert resolve_openwebui_trace_mode(SimpleNamespace(openwebui_trace_mode="summary")) == "summary"
    assert resolve_openwebui_trace_mode(SimpleNamespace(openwebui_trace_mode="diagnostic")) == "diagnostic"
    assert resolve_openwebui_trace_mode(SimpleNamespace(openwebui_trace_mode="off", show_tool_events=True)) == "off"


def test_openwebui_trace_mode_maps_legacy_event_flags_to_summary() -> None:
    settings = SimpleNamespace(openwebui_trace_mode="", show_planner_events=False, show_tool_events=True, show_validation_events=False)
    assert resolve_openwebui_trace_mode(settings) == "summary"


def test_sanitize_detail_redacts_common_secret_shapes() -> None:
    text = sanitize_detail("curl -H 'Authorization: Bearer abcdef123456' https://x.test?api_key=secret")
    assert "abcdef123456" not in text
    assert "secret" not in text
    assert "<redacted>" in text


def test_summary_renderer_does_not_stream_tool_output_payloads() -> None:
    renderer = OpenWebUITraceRenderer(mode="summary")
    event = {
        "event_type": "executor.step.output",
        "payload": {"channel": "stdout", "text": "raw row payload"},
    }
    assert renderer.render(event) is None
