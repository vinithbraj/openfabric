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
        return json.dumps(payload, ensure_ascii=True)
    except TypeError:
        return str(payload)


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
