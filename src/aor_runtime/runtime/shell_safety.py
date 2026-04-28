from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Literal


ShellRiskLevel = Literal["safe_read_only", "low_risk", "medium_risk", "high_risk", "destructive", "forbidden"]


@dataclass(frozen=True)
class ShellCommandPolicy:
    risk: ShellRiskLevel
    allowed: bool
    requires_approval: bool
    reason: str
    detected_operations: list[str] = field(default_factory=list)
    redacted_command: str | None = None


SAFE_READ_ONLY_COMMANDS = {
    "awk",
    "cat",
    "cut",
    "curl",
    "df",
    "du",
    "env",
    "file",
    "find",
    "findmnt",
    "free",
    "grep",
    "head",
    "hostname",
    "id",
    "ip",
    "journalctl",
    "lscpu",
    "ls",
    "lsof",
    "netstat",
    "ps",
    "pwd",
    "printf",
    "sed",
    "sort",
    "ss",
    "stat",
    "systemctl",
    "tail",
    "top",
    "tr",
    "uname",
    "uniq",
    "uptime",
    "wc",
    "wget",
    "whoami",
    "xargs",
}

MUTATING_COMMANDS = {
    "apt",
    "apt-get",
    "chgrp",
    "chmod",
    "chown",
    "cp",
    "docker",
    "git",
    "kill",
    "kubectl",
    "mount",
    "mv",
    "pkill",
    "pip",
    "rm",
    "rmdir",
    "service",
    "systemctl",
    "tee",
    "truncate",
    "umount",
    "yum",
}

FORBIDDEN_COMMANDS = {
    "dd",
    "iptables",
    "mkfs",
    "nft",
    "passwd",
    "poweroff",
    "reboot",
    "shutdown",
    "su",
    "sudo",
    "useradd",
    "userdel",
    "usermod",
}

SYSTEMCTL_READONLY = {"status", "show", "is-active", "is-enabled", "list-units", "list-unit-files"}
SYSTEMCTL_MUTATING = {"restart", "start", "stop", "reload", "enable", "disable", "mask", "unmask"}
SERVICE_MUTATING = {"restart", "start", "stop", "reload"}
GIT_MUTATING = {"clean", "reset", "checkout", "switch", "restore", "commit", "push", "pull", "merge", "rebase"}
DOCKER_MUTATING = {"rm", "rmi", "stop", "restart", "kill", "exec", "run"}
KUBECTL_MUTATING = {"apply", "delete", "patch", "replace", "scale", "cordon", "drain"}

CHAIN_RE = re.compile(r"(?:;|&&|\|\||\n)")
SUBSTITUTION_RE = re.compile(r"(?:`|\$\(|<\(|>\()")
REDIRECT_RE = re.compile(r"(?:^|[^0-9])(?:>>?|<)")
FORK_BOMB_RE = re.compile(r":\(\)\s*\{")


def classify_shell_command(command: str, *, mode: str = "read_only", allow_mutation_with_approval: bool = False) -> ShellCommandPolicy:
    text = str(command or "").strip()
    normalized_mode = str(mode or "read_only").strip().lower()
    if normalized_mode == "disabled":
        return ShellCommandPolicy("forbidden", False, False, "Shell execution is disabled.", [], _redact_command(text))
    if not text:
        return ShellCommandPolicy("forbidden", False, False, "Empty shell command.", [], "")

    forbidden_reason = _forbidden_text_reason(text)
    if forbidden_reason:
        return ShellCommandPolicy("forbidden", False, False, forbidden_reason, ["forbidden"], _redact_command(text))

    if SUBSTITUTION_RE.search(text):
        return ShellCommandPolicy("high_risk", False, True, "Command substitution is not allowed in read-only shell mode.", ["substitution"], _redact_command(text))
    if _has_unquoted_redirect(text):
        return _policy_for_risk("medium_risk", normalized_mode, allow_mutation_with_approval, "Shell redirection can write or read unintended files.", ["redirection"], text)
    if CHAIN_RE.search(text):
        return _policy_for_risk("medium_risk", normalized_mode, allow_mutation_with_approval, "Command chaining is not allowed for automatic shell execution.", ["chaining"], text)

    segments = _split_pipeline(text)
    if not segments:
        return ShellCommandPolicy("forbidden", False, False, "Could not parse shell command.", ["parse_error"], _redact_command(text))

    detected: list[str] = []
    saw_pipe = len(segments) > 1
    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return ShellCommandPolicy("high_risk", False, True, "Could not safely parse shell command.", ["parse_error"], _redact_command(text))
        if not tokens:
            return ShellCommandPolicy("forbidden", False, False, "Empty shell pipeline segment.", ["parse_error"], _redact_command(text))
        command_name = _base_command(tokens[0])
        detected.append(command_name)
        command_risk, reason = _segment_risk(command_name, tokens)
        if command_risk == "forbidden":
            return ShellCommandPolicy("forbidden", False, False, reason, detected, _redact_command(text))
        if command_risk in {"destructive", "high_risk", "medium_risk"}:
            return _policy_for_risk(command_risk, normalized_mode, allow_mutation_with_approval, reason, detected, text)

    risk: ShellRiskLevel = "low_risk" if saw_pipe else "safe_read_only"
    return _policy_for_risk(risk, normalized_mode, allow_mutation_with_approval, "Read-only shell inspection command.", detected, text)


def _segment_risk(command_name: str, tokens: list[str]) -> tuple[ShellRiskLevel, str]:
    if command_name in FORBIDDEN_COMMANDS:
        return "forbidden", f"`{command_name}` is forbidden by shell policy."
    if command_name == "find" and "-delete" in tokens:
        return "destructive", "find -delete is destructive."
    if command_name == "find" and any(token in {"-exec", "-execdir"} for token in tokens):
        return "medium_risk", "find -exec is not allowed for automatic read-only execution."
    if command_name == "rm":
        if any(token.startswith("-") and "r" in token and "f" in token for token in tokens[1:]):
            return "destructive", "Recursive forced delete is destructive."
        return "high_risk", "File deletion requires approval."
    if command_name in {"kill", "pkill"}:
        return "high_risk", "Killing processes requires approval."
    if command_name in {"chmod", "chown", "chgrp", "truncate", "tee", "mv", "cp", "rmdir", "mount", "umount"}:
        return "medium_risk", f"`{command_name}` can modify files or system state."
    if command_name == "systemctl":
        action = _first_non_option(tokens[1:])
        if action in SYSTEMCTL_READONLY:
            return "safe_read_only", "Read-only systemctl inspection."
        if action in SYSTEMCTL_MUTATING:
            return "high_risk", f"systemctl {action} modifies service state."
        return "medium_risk", "systemctl action is not recognized as read-only."
    if command_name == "service":
        if any(token in SERVICE_MUTATING for token in tokens[1:]):
            return "high_risk", "service start/stop/restart modifies service state."
        return "medium_risk", "service command is not recognized as read-only."
    if command_name == "git" and any(token in GIT_MUTATING for token in tokens[1:]):
        return "high_risk", "git mutation commands require approval."
    if command_name == "docker" and any(token in DOCKER_MUTATING for token in tokens[1:]):
        return "high_risk", "docker mutation commands require approval."
    if command_name == "kubectl" and any(token in KUBECTL_MUTATING for token in tokens[1:]):
        return "high_risk", "kubectl mutation commands require approval."
    if command_name == "xargs" and any(_base_command(token) in MUTATING_COMMANDS | FORBIDDEN_COMMANDS for token in tokens[1:]):
        return "high_risk", "xargs invoking mutating commands requires approval."
    if command_name == "top" and not ("-b" in tokens and "-n" in tokens):
        return "medium_risk", "top must be run in non-interactive batch mode."
    if command_name in MUTATING_COMMANDS and command_name not in SAFE_READ_ONLY_COMMANDS:
        return "medium_risk", f"`{command_name}` is not known to be read-only."
    if command_name not in SAFE_READ_ONLY_COMMANDS:
        return "medium_risk", f"`{command_name}` is not in the read-only allowlist."
    if command_name in {"curl", "wget"} and _pipes_to_shell(tokens):
        return "forbidden", "Piping downloaded content to a shell is forbidden."
    return "safe_read_only", "Read-only command."


def _policy_for_risk(
    risk: ShellRiskLevel,
    mode: str,
    allow_mutation_with_approval: bool,
    reason: str,
    detected: list[str],
    command: str,
) -> ShellCommandPolicy:
    if risk in {"safe_read_only", "low_risk"}:
        return ShellCommandPolicy(risk, mode != "disabled", False, reason, detected, _redact_command(command))
    if risk == "forbidden":
        return ShellCommandPolicy(risk, False, False, reason, detected, _redact_command(command))
    if mode == "permissive" and risk not in {"destructive", "forbidden"}:
        return ShellCommandPolicy(risk, True, False, reason, detected, _redact_command(command))
    if mode == "approval_required" and allow_mutation_with_approval:
        return ShellCommandPolicy(risk, False, True, reason, detected, _redact_command(command))
    return ShellCommandPolicy(risk, False, mode == "approval_required", reason, detected, _redact_command(command))


def _forbidden_text_reason(text: str) -> str | None:
    if FORK_BOMB_RE.search(text):
        return "Fork-bomb pattern is forbidden."
    if re.search(r"\brm\s+-[A-Za-z]*r[A-Za-z]*f[A-Za-z]*\s+/(?:\s|$)", text):
        return "Recursive deletion of root is forbidden."
    if re.search(r"\bcurl\b.+\|\s*(?:sh|bash|zsh)\b", text) or re.search(r"\bwget\b.+\|\s*(?:sh|bash|zsh)\b", text):
        return "Piping downloaded content to a shell is forbidden."
    return None


def _split_pipeline(text: str) -> list[str]:
    if "|" not in text:
        return [text]
    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


def _has_unquoted_redirect(text: str) -> bool:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(str(text or "")):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote:
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            continue
        if character == ">":
            return True
        if character == "<":
            # Process substitution is handled separately. A bare unquoted
            # less-than is input redirection.
            if index + 1 < len(text) and text[index + 1] == "(":
                continue
            return True
    return False


def _base_command(command: str) -> str:
    return command.rsplit("/", 1)[-1].strip().lower()


def _first_non_option(tokens: list[str]) -> str:
    for token in tokens:
        if token.startswith("-"):
            continue
        return token.lower()
    return ""


def _pipes_to_shell(tokens: list[str]) -> bool:
    return any(_base_command(token) in {"sh", "bash", "zsh", "fish"} for token in tokens[1:])


def _redact_command(command: str) -> str:
    text = str(command or "")
    text = re.sub(r"(?i)(password|token|secret|credential|api[_-]?key)=\\S+", r"\1=<redacted>", text)
    return text
