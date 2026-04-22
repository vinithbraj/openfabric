import re
import json
import os
import shlex
import subprocess
import sys
from typing import Any

from fastapi import FastAPI
import requests

from agent_library.common import EventRequest, EventResponse, serialize_for_stdin, shared_llm_api_settings, task_plan_context, with_node_envelope
from agent_library.reduction import build_shell_reduction_request, execute_reduction_request, generate_shell_reduction_command
from agent_library.template import (
    agent_api,
    agent_descriptor,
    emit,
    needs_decomposition as shared_needs_decomposition,
    noop,
    task_result,
)
from agent_library.agents.llm_operations_planner import _derive_shell_command as _planner_derive_shell_command
from runtime.console import log_debug, log_raw

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="shell_runner",
    role="executor",
    description=(
        "General-purpose local shell executor for safe machine, workspace, filesystem, "
        "repository, process, service, container, network, build, test, arithmetic, "
        "and text/data transformation operations."
    ),
    capability_domains=[
        "general_shell",
        "machine_operations",
        "workspace_operations",
        "filesystem_operations",
        "process_management",
        "service_control",
        "network_inspection",
        "hardware_inspection",
        "container_operations",
        "repository_operations",
        "build_and_test",
        "package_management",
        "text_processing",
        "data_transformation",
        "arithmetic",
    ],
    action_verbs=[
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
    side_effect_policy="general_machine_operations_with_safety_checks",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Use for any request that can be expressed as safe local shell commands.",
        "Covers workspace and filesystem operations, process and network inspection, services, containers, repositories, builds, tests, package commands, arithmetic, and text/data transformations.",
        "Use for local machine hardware inspection such as checking whether a CLI is installed, inspecting GPUs with nvidia-smi, and reading local driver or CUDA details.",
        "Prefer workspace-scoped commands and explicit user intent for mutating operations.",
        "Reject destructive, privileged, ambiguous, or unsafe commands according to runtime safety checks.",
        "If runtime policy allows it, this agent may install missing Python packages into the active Python environment when that is necessary to complete a user task.",
        "Can derive shell commands from natural-language task plans using an LLM preprocessing step.",
        "If a task cannot be mapped to a safe local shell command, emit nothing for task.plan.",
    ],
    apis=[
        agent_api(
            name="execute_explicit_command",
            event="shell.exec",
            summary="Runs explicitly provided shell command strings.",
            when="Runs explicitly provided shell command strings.",
            intent_tags=["cli_exec"],
            risk_level="medium",
            examples=[
                "find . -iname \"*vinith*\"",
                "ls -la agent_library/agents",
            ],
            anti_patterns=["read a specific file's contents"],
            deterministic=True,
            side_effect_level="variable",
        ),
        agent_api(
            name="execute_llm_derived_command",
            event="task.plan",
            summary="Derives a shell command from natural language and executes it when safely processable.",
            when="Derives a shell command from natural language and executes it when safely processable.",
            intent_tags=["cli_exec", "machine_operations", "workspace_operations", "data_transformation"],
            risk_level="medium",
            examples=[
                "list files under agent_library",
                "find python files containing FastAPI",
                "find all files with extension sh in current directory",
                "show listening ports",
                "open Readme.md",
                "calculate 2+3*5",
                "create file vinith.txt with Hello world",
                "show running docker containers",
                "check if nvidia-smi is installed",
                "show GPU specs on this machine",
                "restart container named web",
                "commit all git changes with message 'update shell prompt'",
            ],
            anti_patterns=["delete system files", "modify system package managers without explicit need"],
            deterministic=False,
            side_effect_level="variable",
        ),
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR

SHELL_CAPABILITIES = {
    "allowed_operations": [
        "execute safe local shell commands that satisfy explicit user intent",
        "inspect, read, create, overwrite, append, copy, move, and rename workspace files when explicitly requested",
        "search, filter, sort, count, aggregate, and transform text/data with standard shell tools",
        "perform arithmetic and lightweight data calculations with shell tools such as awk, bc, python -c, printf, and expr",
        "inspect and manage local processes, ports, services, and containers when explicitly requested",
        "inspect and operate on the current repository, including git status, branch, log, stage, commit, and push",
        "run project-local build, test, lint, package, and diagnostic commands",
        "install Python packages into the current runtime when needed to satisfy a task",
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

ENVIRONMENT_MUTATION_PATTERNS = [
    r"\bdocker\s+(?:rm|rmi|kill|stop|system\s+prune|container\s+prune|image\s+prune|volume\s+prune|network\s+prune)\b",
    r"\bdocker\s+(?:container|image|volume|network)\s+(?:rm|prune)\b",
    r"\bdocker\s+compose\s+(?:down|rm|stop)\b",
    r"\b(?:killall|pkill)\b",
    r"\bsystemctl\s+(?:stop|restart|reload|disable)\b",
    r"\bservice\s+\S+\s+(?:stop|restart|reload)\b",
    r"\bscancel\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[^\n]*f",
    r"\bgit\s+push\b[^\n]*--force(?:-with-lease)?\b",
]

METADATA_COMMAND_PATTERNS = [
    r"^[a-zA-Z_][\w.-]*\s*->\s*[a-zA-Z_][\w.-]*$",
    r"^[a-zA-Z_][\w.-]*\s+emits\s+event\s+[a-zA-Z_][\w.-]*$",
]


def _needs_decomposition(detail: str):
    return shared_needs_decomposition(detail, suggested_capabilities=["shell_runner"])


NONFATAL_SCAN_ERROR_PATTERNS = (
    "permission denied",
    "operation not permitted",
)


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("SHELL_DEBUG", message)


def _python_package_installs_allowed() -> bool:
    return os.getenv("OPENFABRIC_ALLOW_PYTHON_PACKAGE_INSTALLS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _is_python_command(command: str) -> bool:
    if not isinstance(command, str):
        return False
    command_lc = command.lower()
    return bool(re.search(r"(?:^|[;&(]\s*)(python(?:3(?:\.\d+)?)?)\b", command_lc)) or "py -" in command_lc


def _extract_missing_python_module(stdout: str, stderr: str) -> str | None:
    combined = "\n".join(part for part in (stdout, stderr) if isinstance(part, str) and part.strip())
    patterns = [
        r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
        r"ImportError:\s+No module named ['\"]?([A-Za-z0-9_.-]+)['\"]?",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            module = match.group(1).strip()
            if module:
                return module
    return None


def _module_package_candidates(module_name: str) -> list[str]:
    module = str(module_name or "").strip()
    if not module:
        return []
    base = module.split(".", 1)[0]
    mapped = {
        "yaml": "PyYAML",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "bs4": "beautifulsoup4",
        "sklearn": "scikit-learn",
        "Crypto": "pycryptodome",
        "dotenv": "python-dotenv",
        "dateutil": "python-dateutil",
    }.get(base)
    candidates = [mapped] if mapped else []
    candidates.extend([base, base.replace("_", "-")])
    seen = set()
    unique = []
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _python_install_command(command: str, package_name: str) -> str:
    interpreter = "python3"
    if isinstance(command, str):
        match = re.search(r"(python(?:3(?:\.\d+)?)?)\b", command)
        if match:
            interpreter = match.group(1)
    configured_tool = os.getenv("OPENFABRIC_PYTHON_PACKAGE_INSTALL_TOOL", "").strip()
    if configured_tool:
        if "{interpreter}" in configured_tool:
            tool = configured_tool.format(interpreter=interpreter)
        else:
            tool = configured_tool
    else:
        tool = f"{interpreter} -m pip"
    return f"{tool} install {shlex.quote(package_name)}"


def _run_shell_subprocess(command: str, stdin_text: str = "", timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-lc", command],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _attempt_python_dependency_repair(command: str, stdin_text: str, completed: subprocess.CompletedProcess[str]) -> tuple[subprocess.CompletedProcess[str], str | None, str | None]:
    if not _python_package_installs_allowed() or not _is_python_command(command):
        return completed, None, None
    missing_module = _extract_missing_python_module(completed.stdout or "", completed.stderr or "")
    if not missing_module:
        return completed, None, None

    for package_name in _module_package_candidates(missing_module):
        install_command = _python_install_command(command, package_name)
        try:
            install_result = _run_shell_subprocess(install_command, timeout=180)
        except subprocess.TimeoutExpired:
            continue
        if install_result.returncode != 0:
            continue
        try:
            repaired = _run_shell_subprocess(command, stdin_text=stdin_text, timeout=30)
        except subprocess.TimeoutExpired:
            return completed, install_command, f"Installed Python package `{package_name}` but rerun timed out."
        detail = f"Automatically installed Python package `{package_name}` to satisfy missing module `{missing_module}`."
        return repaired, install_command, detail

    return completed, None, None


def _normalize_partial_success(command: str, completed: subprocess.CompletedProcess[str]) -> tuple[int, str | None]:
    if completed.returncode in (0, None):
        return int(completed.returncode or 0), None
    if not isinstance(command, str) or not command.strip():
        return completed.returncode, None
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if not stdout.strip() or not stderr.strip():
        return completed.returncode, None

    command_lc = command.strip().lower()
    if not command_lc.startswith("find "):
        return completed.returncode, None

    stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not stderr_lines:
        return completed.returncode, None
    if all(any(pattern in line.lower() for pattern in NONFATAL_SCAN_ERROR_PATTERNS) for line in stderr_lines):
        return 0, "find returned partial results and only reported non-fatal permission warnings."
    return completed.returncode, None


def _llm_api_settings():
    return shared_llm_api_settings("gpt-4o-mini")


def _llm_generate_reduction_command(
    task: str,
    original_cmd: str,
    sample_stdout: str,
    previous_command: str = "",
    previous_error: str = "",
) -> str:
    return generate_shell_reduction_command(
        task,
        original_cmd,
        sample_stdout,
        previous_command,
        previous_error,
    )


def _llm_summarize_locally(task: str, original_cmd: str, stdout: str) -> tuple[str, str]:
    reduction_request = build_shell_reduction_request(task, original_cmd, stdout)
    if not isinstance(reduction_request, dict):
        return "", ""
    reduction = execute_reduction_request(reduction_request, stdout)
    reduced_result = reduction.reduced_result if isinstance(reduction.reduced_result, str) else ""
    return reduced_result, reduction.local_reduction_command


def _is_blocked(command: str) -> bool:
    command_lc = command.lower()
    return any(re.search(pattern, command_lc) for pattern in BLOCKED_PATTERNS)


def _is_environment_mutation(command: str) -> bool:
    command_lc = command.lower()
    return any(re.search(pattern, command_lc) for pattern in ENVIRONMENT_MUTATION_PATTERNS)


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
        "- When the task is generation-heavy or needs stable parsing, aggregation, formatting, or file creation, prefer a single python3 heredoc snippet over brittle shell pipelines.\n"
        "- Python snippets should print the final user-visible answer or the absolute saved file path explicitly.\n"
        "- If a Python command needs a missing third-party package, you may use python -m pip install to add it to the current Python environment when necessary.\n"
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
        "{\"processable\":true,\"command\":\"python3 - <<'PY'\\nimport json,sys\\nrows=json.load(sys.stdin)\\nprint(sum(1 for row in rows if row.get(\\\"status\\\") == \\\"active\\\"))\\nPY\",\"reason\":\"structured aggregation from stdin\"}\n"
        '{"processable":true,"command":"python3 -m pip install pyyaml && python3 - <<\\"PY\\"\\nimport yaml\\nprint(yaml.safe_load(\\"a: 1\\")[\\"a\\"])\\nPY","reason":"install missing python dependency when required"}\n'
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
    api_key, base_url, timeout_seconds, model = _llm_api_settings()

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
    if structured_input in (None, "", [], {}):
        deterministic = _planner_derive_shell_command(task, 1)
        if isinstance(deterministic, str) and deterministic.strip():
            return deterministic.strip()
    decision_raw = _llm_preprocess(task, structured_input)
    decision = _parse_decision(decision_raw)
    if decision is None or not decision["processable"]:
        return None
    return decision["command"]


def _execute_command(command: str, stdin_data: Any = None, task: str = None):
    if _looks_like_metadata_not_command(command):
        return task_result("Rejected descriptive capability metadata because it is not an executable shell command.")

    if _is_blocked(command):
        return task_result("Rejected potentially destructive command")

    if _is_environment_mutation(command):
        return task_result(
            (
                "Rejected environment-altering command. "
                "This shell agent currently allows inspection plus workspace-local file operations, "
                "but blocks destructive or service/container control commands."
            )
        )

    try:
        shlex.split(command)
    except ValueError as exc:
        return task_result(f"Invalid command: {exc}")

    if not command.strip():
        return task_result("Empty command rejected")

    try:
        stdin_text = serialize_for_stdin(stdin_data)
        completed = _run_shell_subprocess(command, stdin_text=stdin_text, timeout=30)
    except subprocess.TimeoutExpired:
        return task_result("Command timed out after 30 seconds")

    completed, install_command, install_detail = _attempt_python_dependency_repair(command, stdin_text, completed)

    effective_returncode, partial_success_reason = _normalize_partial_success(command, completed)
    reduction_request = None
    if task and effective_returncode == 0:
        reduction_request = build_shell_reduction_request(task, command, completed.stdout)
    
    payload = {
        "command": command,
        "install_command": install_command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": effective_returncode,
    }
    if isinstance(reduction_request, dict):
        payload["reduction_request"] = reduction_request
    if effective_returncode != completed.returncode:
        payload["raw_returncode"] = completed.returncode
    details = [item for item in (install_detail, partial_success_reason) if isinstance(item, str) and item.strip()]
    if details:
        payload["detail"] = " ".join(details)

    return emit("shell.result", payload)


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("shell_runner", "executor")
def handle_event(req: EventRequest):
    command = None
    stdin_data = None
    if req.event == "shell.exec":
        command_raw = req.payload.get("command")
        if not isinstance(command_raw, str):
            return noop()
        if _is_introspection_request(command_raw):
            return noop()
        if _looks_like_shell_command(command_raw):
            command = command_raw.strip()
        else:
            try:
                command = _derive_command_from_task(command_raw)
            except Exception as exc:
                _debug_log(f"Shell preprocessing failed for shell.exec: {type(exc).__name__}: {exc}")
                return noop()
            if not command:
                return noop()
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

        command_raw = req.payload.get("command")
        if isinstance(command_raw, str) and command_raw.strip():
            return _execute_command(command_raw.strip(), structured_input, task=task)
        
        if not task:
            return noop()
        if _is_introspection_request(task):
            return noop()
        try:
            command = _derive_command_from_task(execution_task, structured_input)
        except Exception as exc:
            _debug_log(f"Shell preprocessing failed for task.plan: {type(exc).__name__}: {exc}")
            return _needs_decomposition(f"Shell agent could not derive a safe command: {type(exc).__name__}: {exc}")
        if not command:
            return _needs_decomposition("Shell agent needs the task broken into smaller executable operations.")
    else:
        return noop()
    return _execute_command(command, stdin_data, task=task)
