import json
import os
import sys


RESET = "\033[0m"
BOLD = "\033[1m"

_NAMED_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "bright_cyan": "\033[96m",
}

_AGENT_COLORS = [
    "\033[33m",
    "\033[32m",
    "\033[36m",
    "\033[35m",
    "\033[34m",
    "\033[92m",
    "\033[93m",
    "\033[94m",
]


def _colors_enabled() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    force = os.getenv("FORCE_COLOR", "").lower()
    if force in {"1", "true", "yes", "on"}:
        return True
    return sys.stdout.isatty()


def _paint(text: str, color: str, *, bold: bool = False) -> str:
    if not _colors_enabled():
        return text
    prefix = _NAMED_COLORS.get(color, "")
    if bold:
        prefix = BOLD + prefix
    return f"{prefix}{text}{RESET}"


def _agent_color(name: str) -> str:
    if not name:
        return "yellow"
    idx = sum(ord(ch) for ch in name) % len(_AGENT_COLORS)
    color_code = _AGENT_COLORS[idx]
    if not _colors_enabled():
        return ""
    return color_code


def _format_payload(payload) -> str:
    try:
        return json.dumps(_compact_payload(payload), ensure_ascii=True)
    except TypeError:
        return str(payload)


def _full_event_logs_enabled() -> bool:
    return os.getenv("OPENFABRIC_FULL_EVENT_LOGS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _truncate_text(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _compact_payload(value, *, _seen=None, _depth: int = 0):
    if _full_event_logs_enabled():
        return value
    if _depth > 12:
        return "<max-depth>"
    if _seen is None:
        _seen = set()
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return "<cycle>"
        nested_seen = set(_seen)
        nested_seen.add(value_id)
        return [_compact_payload(item, _seen=nested_seen, _depth=_depth + 1) for item in value]
    if not isinstance(value, dict):
        if isinstance(value, str):
            return _truncate_text(value)
        return value

    value_id = id(value)
    if value_id in _seen:
        return "<cycle>"
    nested_seen = set(_seen)
    nested_seen.add(value_id)
    compact = {}
    for key, item in value.items():
        if key in {"stdout", "stderr"} and isinstance(item, str):
            compact[key] = {
                "chars": len(item),
                "excerpt": _truncate_text(item.replace("\b", ""), 180) if item else "",
            }
            continue
        if key == "emitted" and isinstance(item, list):
            compact[key] = f"<{len(item)} emitted event(s)>"
            continue
        if key == "result" and isinstance(item, str):
            compact[key] = {
                "chars": len(item),
                "excerpt": _truncate_text(item.replace("\b", ""), 180) if item else "",
            }
            continue
        compact[key] = _compact_payload(item, _seen=nested_seen, _depth=_depth + 1)
    return compact


def _tag(label: str, color: str) -> str:
    return _paint(f"[{label}]", color, bold=True)


def style_agent_name(agent_name: str) -> str:
    if not _colors_enabled():
        return agent_name
    color_code = _agent_color(agent_name)
    return f"{BOLD}{color_code}{agent_name}{RESET}"


def log_boot(message: str):
    print(f"{_tag('BOOT', 'cyan')} {message}")


def log_event(event_name: str, payload: dict, depth: int = 0):
    indent = "  " * depth
    event_text = _paint(event_name, "blue", bold=True)
    if event_name == "answer.final" and isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        print(f"{indent}{_tag('EVENT', 'blue')} {event_text} ->")
        extra_payload = {key: value for key, value in payload.items() if key != "answer"}
        if extra_payload:
            extra_text = _paint(_format_payload(extra_payload), "gray")
            print(f"{indent}  {_paint('meta:', 'gray')} {extra_text}")
        print(f"{indent}  {_paint('answer:', 'gray')}")
        for line in payload["answer"].splitlines():
            print(f"{indent}    {_paint(line, 'gray')}")
        return

    payload_text = _paint(_format_payload(payload), "gray")
    print(f"{indent}{_tag('EVENT', 'blue')} {event_text} -> {payload_text}")


def log_event_handler(agent_name: str, depth: int = 0):
    indent = "  " * depth
    print(f"{indent}  {_paint('↳ handled by:', 'gray')} {style_agent_name(agent_name)}")


def log_debug(channel: str, message: str):
    print(f"{_tag(channel, 'yellow')} {_paint(message, 'gray')}")


def log_raw(channel: str, message: str):
    print(f"{_tag(channel, 'magenta')} {_paint(message, 'gray')}")


def log_error(message: str):
    print(f"{_tag('ERROR', 'red')} {_paint(message, 'red')}")
