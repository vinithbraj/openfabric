import re
import json
import os
import shlex
import subprocess
from typing import Any

from fastapi import FastAPI
import requests

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Executes safe bash commands from explicit shell input or LLM-derived task plans.",
    "capability_domains": ["shell", "operations", "workspace_inspection", "service_control"],
    "action_verbs": ["run", "execute", "list", "find", "search", "inspect", "start", "stop", "restart"],
    "side_effect_policy": "allow_mutating_commands_with_safety_checks",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Use for command-line operations such as search, list, grep, and process inspection in the workspace.",
        "Can derive shell commands from natural-language task plans using an LLM preprocessing step.",
        "If a task does not map to shell capabilities, emit nothing for task.plan.",
    ],
    "methods": [
        {
            "name": "execute_explicit_command",
            "event": "shell.exec",
            "when": "Runs explicitly provided shell command strings.",
            "intent_tags": ["cli_exec"],
            "risk_level": "medium",
            "examples": [
                "find . -iname \"*vinith*\"",
                "ls -la agent_library/agents",
            ],
            "anti_patterns": ["read a specific file's contents"],
        },
        {
            "name": "execute_llm_derived_command",
            "event": "task.plan",
            "when": "Derives a shell command from natural language and executes it when safely processable.",
            "intent_tags": ["cli_exec", "file_search", "workspace_inspection"],
            "risk_level": "medium",
            "examples": ["list files under agent_library", "find python files containing FastAPI"],
            "anti_patterns": ["delete system files", "install packages globally"],
        }
    ],
}

SHELL_CAPABILITIES = {
    "allowed_operations": [
        "list files and directories in current workspace",
        "search file names and file contents (find, rg, ls, cat, head, tail, wc, sort)",
        "inspect git/workspace state (git status, git branch, git log --oneline)",
        "show process/network basics (ps, pgrep, netstat/ss) when read-only",
        "manage local developer services and containers when explicitly requested (e.g., docker ps/stop/start/restart)",
    ],
    "disallowed_operations": [
        "destructive commands (rm, mkfs, dd, reboot/shutdown)",
        "privileged execution (sudo)",
        "commands that modify system configuration outside workspace",
        "fork bombs or process-kill-all patterns",
    ],
}

BLOCKED_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\brm\s+-fr\b",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bdd\s+if=",
    r"\bsudo\b",
    r":\(\)\s*{\s*:\|:&\s*};:",
    r">\s*/dev/sd[a-z]",
]


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        print(f"[SHELL_DEBUG] {message}")


def _is_blocked(command: str) -> bool:
    command_lc = command.lower()
    return any(re.search(pattern, command_lc) for pattern in BLOCKED_PATTERNS)


def _extract_json_object(text: str):
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end < start:
        return text[start:]
    return text[start : end + 1]


def _repair_json_blob(blob: str):
    repaired = blob.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    normalized_chars = []
    stack = []
    in_string = False
    escaped = False
    for ch in repaired:
        if escaped:
            normalized_chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            normalized_chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            normalized_chars.append(ch)
            in_string = not in_string
            continue
        if in_string:
            normalized_chars.append(ch)
            continue
        if ch in "{[":
            normalized_chars.append(ch)
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                normalized_chars.append(ch)
                stack.pop()
            else:
                continue
        elif ch == "]":
            if stack and stack[-1] == "[":
                normalized_chars.append(ch)
                stack.pop()
            else:
                continue
        else:
            normalized_chars.append(ch)

    repaired = "".join(normalized_chars)
    closing = {"{": "}", "[": "]"}
    while stack:
        repaired += closing[stack.pop()]
    return repaired


def _parse_json(content: str):
    json_blob = _extract_json_object(content)
    if not json_blob:
        return None
    for candidate in (json_blob, _repair_json_blob(json_blob)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _build_preprocess_prompt(task: str):
    return (
        "You are a strict shell-command preprocessor.\n"
        "Do NOT execute commands. Do NOT explain. Only decide if a safe shell command can be formed.\n"
        "Map the request to exactly one shell command when possible.\n"
        "Capabilities and limits:\n"
        f"{json.dumps(SHELL_CAPABILITIES, indent=2)}\n"
        "Return ONLY JSON with this exact shape:\n"
        '{"processable":true|false,"command":"shell command or null","reason":"short reason"}\n'
        "Rules:\n"
        "- If unprocessable, set processable=false and command=null.\n"
        "- If processable, set processable=true with a single non-empty shell command string.\n"
        "- Prefer read-only, workspace-scoped commands.\n"
        "- Never output dangerous commands (rm -rf, mkfs, dd, reboot, shutdown, sudo).\n"
        "- Never include markdown or extra keys.\n"
        "Valid examples:\n"
        '{"processable":true,"command":"find . -type f","reason":"list files request"}\n'
        '{"processable":true,"command":"rg -n \\"FastAPI\\" agent_library","reason":"content search request"}\n'
        '{"processable":true,"command":"docker stop $(docker ps -q)","reason":"explicit request to stop running containers"}\n'
        '{"processable":false,"command":null,"reason":"requires destructive or privileged action"}\n'
        f'User request: "{task}"'
    )


def _llm_preprocess(task: str):
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "10"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

    if api_key.startswith("sk-") and api_key.lower() != "dummy":
        base_url = "https://api.openai.com/v1"
        if not model:
            model = "gpt-4.1"
    elif not base_url:
        base_url = "https://api.openai.com/v1"
    if not model:
        model = "gpt-4o-mini"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_prompt = _build_preprocess_prompt(task)
    messages = [
        {"role": "system", "content": "You produce strict JSON only."},
        {"role": "user", "content": user_prompt},
    ]
    payload = {"model": model, "messages": messages, "temperature": 0}

    _debug_log("Constructed shell preprocessing prompt:")
    _debug_log(user_prompt)

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(f"[SHELL_LLM_RAW] {content}")
    _debug_log("Raw preprocessing response:")
    _debug_log(content)
    return _parse_json(content)


def _parse_decision(raw: Any):
    if not isinstance(raw, dict):
        return None

    processable = raw.get("processable")
    command = raw.get("command")
    reason = raw.get("reason", "")

    if not isinstance(processable, bool):
        return None
    if command is not None and not isinstance(command, str):
        return None
    if not isinstance(reason, str):
        reason = ""

    if not processable:
        return {"processable": False, "command": None, "reason": reason.strip()}

    if not command or not command.strip():
        return None
    return {"processable": True, "command": command.strip(), "reason": reason.strip()}


def _looks_like_shell_command(text: str) -> bool:
    if not text or not text.strip():
        return False
    token = text.strip().split()[0].lower()
    common_commands = {
        "ls",
        "find",
        "rg",
        "grep",
        "cat",
        "head",
        "tail",
        "pwd",
        "echo",
        "wc",
        "sort",
        "uniq",
        "git",
        "ps",
        "pgrep",
        "ss",
        "netstat",
    }
    return token in common_commands or text.strip().startswith("./")


def _derive_command_from_task(task: str):
    decision_raw = _llm_preprocess(task)
    decision = _parse_decision(decision_raw)
    if decision is None or not decision["processable"]:
        return None
    return decision["command"]


def _execute_command(command: str):
    if _is_blocked(command):
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": "Rejected potentially destructive command"},
                }
            ]
        }

    try:
        shlex.split(command)
    except ValueError as exc:
        return {
            "emits": [
                {"event": "task.result", "payload": {"detail": f"Invalid command: {exc}"}}
            ]
        }

    if not command.strip():
        return {
            "emits": [
                {"event": "task.result", "payload": {"detail": "Empty command rejected"}}
            ]
        }

    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": "Command timed out after 10 seconds"},
                }
            ]
        }

    return {
        "emits": [
            {
                "event": "shell.result",
                "payload": {
                    "command": command,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "returncode": completed.returncode,
                },
            }
        ]
    }


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    command = None
    if req.event == "shell.exec":
        command_raw = req.payload.get("command")
        if not isinstance(command_raw, str):
            return {"emits": []}
        if _looks_like_shell_command(command_raw):
            command = command_raw.strip()
        else:
            try:
                command = _derive_command_from_task(command_raw)
            except Exception as exc:
                _debug_log(f"Shell preprocessing failed for shell.exec: {type(exc).__name__}: {exc}")
                return {"emits": []}
            if not command:
                return {"emits": []}
    elif req.event == "task.plan":
        task = req.payload.get("task")
        if not isinstance(task, str):
            return {"emits": []}
        try:
            command = _derive_command_from_task(task)
        except Exception as exc:
            _debug_log(f"Shell preprocessing failed for task.plan: {type(exc).__name__}: {exc}")
            return {"emits": []}
        if not command:
            return {"emits": []}
    else:
        return {"emits": []}
    return _execute_command(command)
