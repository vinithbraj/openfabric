import re
import json
import os
import shlex
import subprocess
from typing import Any

from fastapi import FastAPI
import requests

from agent_library.common import EventRequest, EventResponse, task_plan_context
from runtime.console import log_debug, log_raw

app = FastAPI()

AGENT_METADATA = {
    "description": (
        "General-purpose local shell executor for safe machine, workspace, filesystem, "
        "repository, process, service, container, network, build, test, arithmetic, "
        "and text/data transformation operations."
    ),
    "capability_domains": [
        "general_shell",
        "machine_operations",
        "workspace_operations",
        "filesystem_operations",
        "process_management",
        "service_control",
        "network_inspection",
        "container_operations",
        "repository_operations",
        "build_and_test",
        "package_management",
        "text_processing",
        "data_transformation",
        "arithmetic",
    ],
    "action_verbs": [
        "run",
        "execute",
        "inspect",
        "list",
        "find",
        "search",
        "read",
        "write",
        "create",
        "append",
        "copy",
        "move",
        "rename",
        "delete",
        "grep",
        "count",
        "filter",
        "sort",
        "transform",
        "calculate",
        "compute",
        "show",
        "start",
        "stop",
        "restart",
        "test",
        "build",
        "install",
        "check",
        "diagnose",
        "monitor",
        "stage",
        "commit",
        "push",
    ],
    "side_effect_policy": "general_machine_operations_with_safety_checks",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Use for any request that can be expressed as safe local shell commands.",
        "Covers workspace and filesystem operations, process and network inspection, services, containers, repositories, builds, tests, package commands, arithmetic, and text/data transformations.",
        "Prefer workspace-scoped commands and explicit user intent for mutating operations.",
        "Reject destructive, privileged, ambiguous, or unsafe commands according to runtime safety checks.",
        "Can derive shell commands from natural-language task plans using an LLM preprocessing step.",
        "If a task cannot be mapped to a safe local shell command, emit nothing for task.plan.",
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
            "intent_tags": ["cli_exec", "machine_operations", "workspace_operations", "data_transformation"],
            "risk_level": "medium",
            "examples": [
                "list files under agent_library",
                "find python files containing FastAPI",
                "find all files with extension sh in current directory",
                "show listening ports",
                "open Readme.md",
                "calculate 2+3*5",
                "create file vinith.txt with Hello world",
                "show running docker containers",
                "restart container named web",
                "commit all git changes with message 'update shell prompt'",
            ],
            "anti_patterns": ["delete system files", "install packages globally"],
        }
    ],
}

SHELL_CAPABILITIES = {
    "allowed_operations": [
        "execute safe local shell commands that satisfy explicit user intent",
        "inspect, read, create, overwrite, append, copy, move, and rename workspace files when explicitly requested",
        "search, filter, sort, count, aggregate, and transform text/data with standard shell tools",
        "perform arithmetic and lightweight data calculations with shell tools such as awk, bc, python -c, printf, and expr",
        "inspect and manage local processes, ports, services, and containers when explicitly requested",
        "inspect and operate on the current repository, including git status, branch, log, stage, commit, and push",
        "run project-local build, test, lint, package, and diagnostic commands",
        "inspect system state with read-only commands such as ps, pgrep, ss, netstat, df, du, env, uname, and date",
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

METADATA_COMMAND_PATTERNS = [
    r"^[a-zA-Z_][\w.-]*\s*->\s*[a-zA-Z_][\w.-]*$",
    r"^[a-zA-Z_][\w.-]*\s+emits\s+event\s+[a-zA-Z_][\w.-]*$",
]


def _needs_decomposition(detail: str):
    return {
        "emits": [
            {
                "event": "task.result",
                "payload": {
                    "detail": detail,
                    "status": "needs_decomposition",
                    "error": detail,
                    "replan_hint": {
                        "reason": detail,
                        "failure_class": "needs_decomposition",
                        "suggested_capabilities": ["shell_runner"],
                    },
                },
            }
        ]
    }


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("SHELL_DEBUG", message)


def _llm_api_settings():
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL") or "https://api.openai.com/v1"
    model = os.getenv("LLM_OPS_MODEL") or "gpt-4o-mini"
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))
    return api_key, base_url.rstrip("/"), timeout_seconds, model


def _llm_generate_reduction_command(task: str, original_cmd: str, sample_stdout: str) -> str:
    api_key, base_url, timeout, model = _llm_api_settings()
    if not api_key:
        return ""

    prompt = (
        "You are an expert shell data analyst. I have a large output from a shell command but I only want to "
        "send the relevant data back for analysis. Your job is to provide exactly ONE shell command "
        "(using awk, grep, tail, head, or jq) that will extract or calculate the necessary information "
        "from the FULL output locally.\n\n"
        f"User Intent: {task}\n"
        f"Original Command: {original_cmd}\n"
        "Sample Data (first few lines):\n"
        "```\n"
        f"{sample_stdout}\n"
        "```\n"
        "Instructions:\n"
        "- Return ONLY the shell command string, no markdown, no explanations.\n"
        "- The command will receive the full output via STDIN.\n"
        "- If the user wants a sum of a column, use awk.\n"
        "- If you cannot generate a reliable processing command, return 'NONE'.\n"
        "Shell Command:"
    )

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a shell command generator. Return only the command."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        cmd = response.json()["choices"][0]["message"]["content"].strip().strip("`").strip()
        return cmd if cmd and cmd.upper() != "NONE" else ""
    except Exception:
        return ""


def _llm_summarize_locally(task: str, original_cmd: str, stdout: str) -> str:
    # Phase 9: Local Data Processing Loop
    if len(stdout) < 5000:
        return ""

    sample = stdout[:5000]
    reduction_cmd = _llm_generate_reduction_command(task, original_cmd, sample)
    if not reduction_cmd:
        return ""

    try:
        proc_result = subprocess.run(
            reduction_cmd,
            input=stdout,
            capture_output=True,
            text=True,
            shell=True,
            timeout=30
        )
        if proc_result.returncode == 0:
            return proc_result.stdout.strip()
    except Exception:
        pass
    return ""


def _is_blocked(command: str) -> bool:
    command_lc = command.lower()
    return any(re.search(pattern, command_lc) for pattern in BLOCKED_PATTERNS)


def _looks_like_metadata_not_command(command: str) -> bool:
    command = command.strip()
    return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in METADATA_COMMAND_PATTERNS)


def _is_introspection_request(text: str) -> bool:
    text_lc = text.lower()
    if not re.search(r"\b(you|your|system|agent|agents|tool|tools|capabilit\w*|supported|available|introspect|introspection)\b", text_lc):
        return False
    return any(
        phrase in text_lc
        for phrase in (
            "capabilities",
            "capability",
            "introspect",
            "introspection",
            "what can you do",
            "available tools",
            "available agents",
            "list your tools",
            "list tools",
            "list agents",
            "supported operations",
            "available operations",
            "what tools",
            "what agents",
            "show tools",
            "show agents",
            "show capabilities",
        )
    )


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


def _build_preprocess_prompt(task: str, structured_input: Any = None):
    prompt = (
        "You are a general-purpose local shell-command planner.\n"
        "Do NOT execute commands. Convert the user request into one safe local shell command when possible.\n"
        "Return JSON only.\n"
        "Capabilities and limits:\n"
        f"{json.dumps(SHELL_CAPABILITIES, indent=2)}\n"
        "Output shape (exact keys):\n"
        '{"processable":true|false,"command":"shell command or null","reason":"short reason"}\n'
        "Rules:\n"
        "- Prefer processable=true whenever the request can be satisfied by a safe local shell command.\n"
        "- Shell-suitable requests include OS, workspace, filesystem, process, network, service, container, repository, build/test, package, arithmetic, and text/data transformation tasks.\n"
        "- If processable=true, command must be one non-empty bash command string.\n"
        "- If processable=false, set command=null.\n"
        "- If structured workflow input JSON is provided below, the derived command will receive that exact JSON on stdin at execution time.\n"
        "- When the task asks to inspect, filter, count, search, compare, summarize, or validate prior results, prefer operating on the provided stdin JSON instead of fetching data again.\n"
        "- Treat references such as 'this list', 'these rows', 'that output', or 'the previous result' as referring to the provided workflow input JSON.\n"
        "- Use workspace-scoped commands by default.\n"
        "- Use standard local tools directly when they fit; examples include sh/bash, printf, awk, sed, grep, rg, find, sort, uniq, wc, cat, head, tail, xargs, python -c, git, docker, ps, ss, date, du, df, make, npm, pytest, and project-local scripts.\n"
        "- Interpret 'current directory' as '.'.\n"
        "- Interpret 'files with extension sh' as name pattern '*.sh'.\n"
        "- 'open/read/show <file>' should usually map to cat/head/tail/sed if path appears valid.\n"
        "- Explicit create/write/save file requests in the workspace may use printf/tee/redirection with a relative path.\n"
        "- If the user asks to create a file with content and return its path, write the file and print its absolute path.\n"
        "- 'where is X implemented' should usually map to rg/find over repository paths.\n"
        "- For Docker natural-language requests, map directly to docker CLI equivalents.\n"
        "- Interpret 'running docker containers' or 'running dockers' as docker ps.\n"
        "- Interpret 'all docker containers' as docker ps -a.\n"
        "- For git commit requests, map to explicit git commands in current repo.\n"
        "- For arithmetic or formula requests, produce a shell command that prints only the answer.\n"
        "- If user asks 'commit all changes' and no message is provided, use a neutral message like "
        "'update files'.\n"
        "- Example mappings: 'running containers' -> docker ps, 'all containers' -> docker ps -a,\n"
        "  'images' -> docker images, 'logs for <name>' -> docker logs --tail 200 <name>,\n"
        "  'restart <name>' -> docker restart <name>, 'stop all running containers' -> docker stop $(docker ps -q).\n"
        "- Example git mappings: 'commit all git changes' -> git add -A && git commit -m \"update files\".\n"
        "  'commit only staged changes with message fix parser' -> git commit -m \"fix parser\".\n"
        "- Tolerate minor typos and shorthand (e.g., 'directoryt', 'pls', 'u').\n"
        "- Never output dangerous commands: rm -rf, mkfs, dd if=, shutdown, reboot, sudo.\n"
        "- No markdown. No extra keys.\n"
        "Examples:\n"
        '{"processable":true,"command":"find . -type f -name \\"*.sh\\"","reason":"file extension search"}\n'
        '{"processable":true,"command":"rg -n \\"FastAPI\\" agent_library","reason":"content search"}\n'
        '{"processable":true,"command":"cat Readme.md","reason":"read file request"}\n'
        '{"processable":true,"command":"printf \\"%s\\\\n\\" \\"Hello world\\" > vinith.txt && realpath vinith.txt","reason":"create file and return path"}\n'
        '{"processable":true,"command":"printf \\"%s\\\\n\\" \\"$((2 + 3 * 5))\\"","reason":"arithmetic"}\n'
        '{"processable":true,"command":"docker ps","reason":"running containers"}\n'
        '{"processable":true,"command":"docker logs --tail 200 vllm","reason":"container logs request"}\n'
        '{"processable":true,"command":"git status","reason":"git inspection"}\n'
        '{"processable":true,"command":"git add -A && git commit -m \\"update files\\"","reason":"commit all changes"}\n'
        '{"processable":true,"command":"ss -ltnp","reason":"list listening ports"}\n'
        '{"processable":true,"command":"python3 -c \\"import json,sys; data=json.load(sys.stdin); rows=data.get(\\\"dependency_results\\\", [{}])[-1].get(\\\"rows\\\", []); print(any(str(row.get(\\\"PatientName\\\", \\\"\\\")).lower() == \\\"test\\\" for row in rows))\\"","reason":"analyze prior JSON rows from stdin"}\n'
        '{"processable":false,"command":null,"reason":"not a shell-operational task"}\n'
        f'User request: "{task}"'
    )
    if structured_input not in (None, "", [], {}):
        prompt += "\nStructured workflow input JSON (available on stdin to the command):\n" + json.dumps(
            structured_input, ensure_ascii=True, indent=2, default=str
        )
    return prompt


def _llm_preprocess(task: str, structured_input: Any = None):
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))

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
    user_prompt = _build_preprocess_prompt(task, structured_input)
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
    log_raw("SHELL_LLM_RAW", content)
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
        "docker",
        "ps",
        "pgrep",
        "ss",
        "netstat",
    }
    return token in common_commands or text.strip().startswith("./")


def _derive_command_from_task(task: str, structured_input: Any = None):
    if _is_introspection_request(task):
        return None
    decision_raw = _llm_preprocess(task, structured_input)
    decision = _parse_decision(decision_raw)
    if decision is None or not decision["processable"]:
        return None
    return decision["command"]


def _serialize_stdin_data(stdin_data: Any) -> str | None:
    if stdin_data is None:
        return None
    if isinstance(stdin_data, str):
        return stdin_data
    if isinstance(stdin_data, (dict, list, tuple, int, float, bool)):
        return json.dumps(stdin_data, ensure_ascii=True, default=str)
    return str(stdin_data)


def _execute_command(command: str, stdin_data: Any = None, task: str = None):
    if _looks_like_metadata_not_command(command):
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {"detail": "Rejected descriptive capability metadata because it is not an executable shell command."},
                }
            ]
        }

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
        stdin_text = _serialize_stdin_data(stdin_data)
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=30,
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

    # Attempt local reduction for large outputs (Phase 9)
    refined_answer = ""
    if task and completed.returncode == 0 and len(completed.stdout) > 5000:
        refined_answer = _llm_summarize_locally(task, command, completed.stdout)
        if refined_answer:
            _debug_log(f"Local reduction successful. Summary: {refined_answer[:100]}...")
    
    return {
        "emits": [
            {
                "event": "shell.result",
                "payload": {
                    "command": command,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "returncode": completed.returncode,
                    "refined_answer": refined_answer or None,
                },
            }
        ]
    }


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    command = None
    stdin_data = None
    if req.event == "shell.exec":
        command_raw = req.payload.get("command")
        if not isinstance(command_raw, str):
            return {"emits": []}
        if _is_introspection_request(command_raw):
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
        plan_context = task_plan_context(req.payload)
        instruction = req.payload.get("instruction")
        structured_input = None
        if isinstance(instruction, dict) and "input" in instruction:
            structured_input = instruction.get("input")
        else:
            structured_input = plan_context.structured_context
        stdin_data = structured_input

        task = plan_context.classification_task
        execution_task = plan_context.execution_task

        if isinstance(instruction, dict) and instruction.get("operation") == "run_command":
            command_raw = instruction.get("command")
            if isinstance(command_raw, str) and command_raw.strip():
                return _execute_command(command_raw.strip(), structured_input, task=task)
        
        if not task:
            return {"emits": []}
        if _is_introspection_request(task):
            return {"emits": []}
        try:
            command = _derive_command_from_task(execution_task, structured_input)
        except Exception as exc:
            _debug_log(f"Shell preprocessing failed for task.plan: {type(exc).__name__}: {exc}")
            return _needs_decomposition(f"Shell agent could not derive a safe command: {type(exc).__name__}: {exc}")
        if not command:
            return _needs_decomposition("Shell agent needs the task broken into smaller executable operations.")
    else:
        return {"emits": []}
    return _execute_command(command, stdin_data, task=task)
