import json
import os
import re
from typing import Any, Dict, List

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse
from runtime.console import log_debug

app = FastAPI()

AGENT_METADATA = {
    "description": "LLM planner that decides whether a request is processable by discovered system capabilities.",
    "capability_domains": ["planning", "routing", "operations"],
    "action_verbs": ["plan", "route", "assess"],
    "side_effect_policy": "read_only",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Only decide if request is processable by discovered capabilities.",
        "If processable, emit task.plan with original task and the best target agent for focused execution.",
        "If not processable, emit task.result with reason.",
    ],
    "methods": [
        {
            "name": "assess_processable_request",
            "event": "task.plan",
            "when": "When request can be handled by at least one discovered agent capability, including multi-step chains.",
            "intent_tags": ["processable", "capability_match"],
        },
        {
            "name": "reject_unprocessable_request",
            "event": "task.result",
            "when": "When request cannot be handled by discovered capabilities.",
            "intent_tags": ["unprocessable"],
        },
    ],
}

SUPPORTED_EVENT_SCHEMAS = {
    "task.plan": '{"task":"...","steps":[{"id":"step1","target_agent":"shell_runner","task":"...","command":"docker ps","result_mode":"stdout_first_line"}]}',
    "task.result": '{"detail":"..."}',
}
DEFAULT_ALLOWED_EVENTS = set(SUPPORTED_EVENT_SCHEMAS.keys())
CAPABILITIES = {"agents": [], "available_events": sorted(DEFAULT_ALLOWED_EVENTS)}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("LLM_OPS_DEBUG", message)


def _format_discovered_agents(capabilities: dict) -> str:
    lines = []
    for item in capabilities.get("agents", []):
        name = item.get("name")
        if not name:
            continue
        description = item.get("description", "").strip() or "No description provided."
        domains = item.get("capability_domains", [])
        verbs = item.get("action_verbs", [])
        domain_text = ", ".join(entry for entry in domains if isinstance(entry, str)) if isinstance(domains, list) else ""
        verb_text = ", ".join(entry for entry in verbs if isinstance(entry, str)) if isinstance(verbs, list) else ""
        methods = []
        method_list = item.get("methods", [])
        if isinstance(method_list, list):
            for method in method_list:
                if not isinstance(method, dict):
                    continue
                method_name = method.get("name")
                method_event = method.get("event")
                if isinstance(method_name, str) and isinstance(method_event, str):
                    methods.append(f"{method_name} emits event {method_event}")
        method_text = ", ".join(methods) if methods else "none"
        lines.append(
            f"- {name}: {description} Domains[{domain_text or 'none'}] "
            f"Verbs[{verb_text or 'none'}] Methods[{method_text}]"
        )
    return "\n".join(lines) if lines else "- unknown: No discovered agents"


def _build_prompt(question: str, capabilities: dict) -> str:
    return (
        "You are the routing planner for an operations assistant.\n"
        "Task: decide whether the request is processable by at least one discovered agent capability and, when processable, "
        "produce the best execution plan and presentation intent.\n"
        "Output JSON only with exact keys: "
        '{"processable":true|false,"reason":"short reason","steps":[{"id":"step1","target_agent":"agent_name","task":"step task","command":"optional shell command","result_mode":"optional capture mode","steps":[{"id":"step1_1","task":"optional nested substep"}]}],"presentation":{"task":"how to present the result","format":"markdown|markdown_table|json|bullets|plain","audience":"openwebui","include_context":true|false,"include_internal_steps":true|false}}\n'
        "No markdown, no extra keys, no prose outside JSON.\n"
        "Decision policy:\n"
        "1) processable=true if any discovered agent can attempt the request.\n"
        "2) For processable=true, steps MUST be a non-empty ordered list.\n"
        "3) Each step MUST have id and task. Leaf steps MUST have target_agent. Group steps may omit target_agent when they contain nested steps.\n"
        "4) command is optional and should be included for shell_runner when you can specify one safe atomic bash command exactly.\n"
        "5) result_mode is optional. Use stdout_first_line for shell steps that should pass only the first non-empty line to later steps.\n"
        "   You may also use stdout_last_line or json_field:<key> when that is a better fit.\n"
        "6) target_agent MUST be one discovered runtime agent name that can handle task.plan; never use ops_planner or synthesizer.\n"
        "7) Break chained requests into multiple atomic steps. Do not combine unrelated actions into one step. Use nested steps for grouped subtasks.\n"
        "8) If a later step depends on an earlier step result, reference it with {{prev}} for the immediately previous result "
        "or {{step_id}} for a specific earlier step.\n"
        "9) For arithmetic chains with explicit numeric operands, use shell_runner with safe arithmetic commands. Do not invent extra arithmetic operations.\n"
        "10) For any safe local machine, workspace, repository, process, service, container, network, build/test, package, arithmetic, text, or data operation that can be expressed as shell commands, prefer shell_runner.\n"
        "11) For SQL/database/schema/table/column/relationship/data questions against a configured database, prefer sql_runner.\n"
        "12) When using shell_runner, prefer explicit command for deterministic operations like docker ps, docker logs, rg, find, ls, git status.\n"
        "13) When using sql_runner, do not invent SQL unless it is clearly requested or needed; sql_runner can introspect schema and generate safe read-only SQL itself.\n"
        "14) Do not invent nonexistent specialized agents; use discovered agents only.\n"
        "15) Extract presentation intent separately from execution steps. Examples: table, JSON, bullets, concise, detailed, include commands, hide internals.\n"
        "16) For Open WebUI, prefer format=markdown_table when the user asks for a table/list/comparison/status report with rows and columns.\n"
        "17) Set include_internal_steps=false unless the user explicitly asks for workflow details, commands, debug output, SQL query text, or how it was done.\n"
        "18) If the user asks about available capabilities, tools, agents, supported operations, or what this system can do, answer from discovered capabilities with task.result; do not call shell_runner.\n"
        "19) Capability metadata is descriptive, not executable. Never put method names, event names, arrows, or capability labels in shell commands.\n"
        "20) Tolerate minor typos and informal phrasing.\n"
        "21) Do NOT mark false only because the request might fail at execution time.\n"
        "22) processable=false only when no discovered capability can attempt it; in that case steps=[].\n"
        "Calibration examples:\n"
        '- "find all files with extension sh in the current directory" -> {"processable":true,"reason":"shell file search","steps":[{"id":"step1","target_agent":"shell_runner","task":"find all files with extension sh in the current directory","command":"find . -type f -name \\"*.sh\\""}],"presentation":{"task":"List matching files clearly.","format":"bullets","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "list all running docker containers" -> {"processable":true,"reason":"docker container listing","steps":[{"id":"step1","target_agent":"shell_runner","task":"list all running docker containers","command":"docker ps"}],"presentation":{"task":"Show running containers in a concise Markdown table.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "list running containers and tell me how long they have been running" -> {"processable":true,"reason":"docker uptime listing","steps":[{"id":"step1","target_agent":"shell_runner","task":"list running containers with uptime status","command":"docker ps --format \\"table {{.Names}}\\\\t{{.Status}}\\\\t{{.Image}}\\""}],"presentation":{"task":"Render container names, images, and runtime status as a Markdown table.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "show the last 50 log lines for vllm" -> {"processable":true,"reason":"docker logs request","steps":[{"id":"step1","target_agent":"shell_runner","task":"show the last 50 log lines for vllm","command":"docker logs --tail 50 vllm"}]}\n'
        '- "find the vllm container id and then show the last 50 log lines" -> {"processable":true,"reason":"two-step shell workflow","steps":[{"id":"step1","target_agent":"shell_runner","task":"find the vllm container id","command":"docker ps --filter name=vllm --format \\"{{.ID}}\\"","result_mode":"stdout_first_line"},{"id":"step2","target_agent":"shell_runner","task":"show the last 50 log lines for container {{prev}}","command":"docker logs --tail 50 {{prev}}"}]}\n'
        '- "show listening ports and then grep for 8000" -> {"processable":true,"reason":"ports inspection workflow","steps":[{"id":"step1","target_agent":"shell_runner","task":"show listening ports","command":"ss -ltnp"},{"id":"step2","target_agent":"shell_runner","task":"filter step1 for 8000","command":"printf \\"%s\\\\n\\" \\"{{step1}}\\" | grep 8000"}]}\n'
        '- "find the newest log file and then show the last 20 lines" -> {"processable":true,"reason":"file discovery workflow","steps":[{"id":"step1","target_agent":"shell_runner","task":"find the newest log file","command":"find . -type f -name \\"*.log\\" -printf \\"%T@ %p\\\\n\\" | sort -nr | head -n 1 | cut -d\\" \\" -f2-","result_mode":"stdout_first_line"},{"id":"step2","target_agent":"shell_runner","task":"show the last 20 lines of {{prev}}","command":"tail -n 20 {{prev}}"}]}\n'
        '- "show git status and current branch" -> {"processable":true,"reason":"multi-step git inspection","steps":[{"id":"step1","target_agent":"shell_runner","task":"show git status","command":"git status --short"},{"id":"step2","target_agent":"shell_runner","task":"show current branch","command":"git branch --show-current","result_mode":"stdout_first_line"}]}\n'
        '- "find python files mentioning FastAPI" -> {"processable":true,"reason":"code search","steps":[{"id":"step1","target_agent":"shell_runner","task":"find python files mentioning FastAPI","command":"rg -n \\"FastAPI\\" --glob \\"*.py\\" ."}]}\n'
        '- "show database schema and relationships" -> {"processable":true,"reason":"SQL schema introspection","steps":[{"id":"step1","target_agent":"sql_runner","task":"show database schema and relationships"}],"presentation":{"task":"Summarize schemas, tables, columns, and relationships clearly.","format":"markdown","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "which customers spent the most money last month" -> {"processable":true,"reason":"SQL database question","steps":[{"id":"step1","target_agent":"sql_runner","task":"which customers spent the most money last month"}],"presentation":{"task":"Show the query results as a readable Markdown table.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "add 12 and 30" -> {"processable":true,"reason":"shell arithmetic","steps":[{"id":"step1","target_agent":"shell_runner","task":"add 12 and 30","command":"printf \\"%s\\\\n\\" \\"$((12 + 30))\\""}],"presentation":{"task":"Give the numeric result directly.","format":"plain","audience":"openwebui","include_context":false,"include_internal_steps":false}}\n'
        '- "open Readme.md" -> {"processable":true,"reason":"shell file read","steps":[{"id":"step1","target_agent":"shell_runner","task":"open Readme.md","command":"cat Readme.md"}]}\n'
        '- "create a file named vinith.txt with Hello world and return the path" -> {"processable":true,"reason":"shell file write","steps":[{"id":"step1","target_agent":"shell_runner","task":"create vinith.txt with Hello world and return the path","command":"printf \\"%s\\\\n\\" \\"Hello world\\" > vinith.txt && realpath vinith.txt","result_mode":"stdout_last_line"}],"presentation":{"task":"Return the created file path directly.","format":"plain","audience":"openwebui","include_context":false,"include_internal_steps":false}}\n'
        '- "add 1 and 2 and then multiply by 230 and then divide by 2" -> {"processable":true,"reason":"multi-step shell arithmetic","steps":[{"id":"step1","target_agent":"shell_runner","task":"add 1 and 2","command":"printf \\"%s\\\\n\\" \\"$((1 + 2))\\"","result_mode":"stdout_stripped"},{"id":"step2","target_agent":"shell_runner","task":"multiply {{prev}} by 230","command":"printf \\"%s\\\\n\\" \\"$(({{prev}} * 230))\\"","result_mode":"stdout_stripped"},{"id":"step3","target_agent":"shell_runner","task":"divide {{prev}} by 2","command":"printf \\"%s\\\\n\\" \\"$(({{prev}} / 2))\\"","result_mode":"stdout_stripped"}]}\n'
        '- "book a flight to NYC" -> {"processable":false,"reason":"no travel booking capability","steps":[]}\n'
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        f'User request: "{question}"'
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
    # Apply conservative repairs for common LLM formatting mistakes.
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


def _parse_planner_json(content: str):
    json_blob = _extract_json_object(content)
    if not json_blob:
        return None, None
    for candidate in (json_blob, _repair_json_blob(json_blob)):
        try:
            return json_blob, json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json_blob, None


def _parse_decision(raw: Any):
    if not isinstance(raw, dict):
        return None
    processable = raw.get("processable")
    reason = raw.get("reason", "")
    steps = raw.get("steps", [])
    presentation = raw.get("presentation")
    if not isinstance(processable, bool):
        return None
    if not isinstance(reason, str):
        reason = ""
    if not isinstance(steps, list):
        return None

    normalized_steps = _parse_steps(steps)
    if normalized_steps is None:
        return None
    return {
        "processable": processable,
        "reason": reason.strip(),
        "steps": normalized_steps,
        "presentation": _parse_presentation(presentation),
    }


def _parse_presentation(raw: Any):
    if not isinstance(raw, dict):
        return {}

    parsed = {}
    for key in ("task", "format", "audience"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            parsed[key] = value.strip()
    for key in ("include_context", "include_internal_steps"):
        value = raw.get(key)
        if isinstance(value, bool):
            parsed[key] = value
    return parsed


def _parse_steps(steps: list):
    normalized_steps = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return None
        step_id = step.get("id") or f"step{index}"
        target_agent = step.get("target_agent")
        task = step.get("task")
        command = step.get("command")
        result_mode = step.get("result_mode")
        nested_steps = step.get("steps")
        depends_on = step.get("depends_on")
        if not isinstance(step_id, str) or not step_id.strip():
            return None
        if not isinstance(task, str) or not task.strip():
            return None
        normalized_step = {
            "id": step_id.strip(),
            "task": task.strip(),
        }
        if isinstance(target_agent, str) and target_agent.strip():
            normalized_step["target_agent"] = target_agent.strip()
        if command is not None:
            if not isinstance(command, str) or not command.strip():
                return None
            normalized_step["command"] = command.strip()
        if result_mode is not None:
            if not isinstance(result_mode, str) or not result_mode.strip():
                return None
            normalized_step["result_mode"] = result_mode.strip()
        if depends_on is not None:
            if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
                return None
            normalized_step["depends_on"] = [item for item in depends_on if item.strip()]
        if nested_steps is not None:
            if not isinstance(nested_steps, list):
                return None
            normalized_nested = _parse_steps(nested_steps)
            if normalized_nested is None:
                return None
            normalized_step["steps"] = normalized_nested
        if "target_agent" not in normalized_step and not normalized_step.get("steps"):
            return None
        normalized_steps.append(normalized_step)
    return normalized_steps


def _llm_decide(question: str, capabilities):
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = float(os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300"))

    if not api_key:
        raise RuntimeError("LLM_OPS_API_KEY is not set")

    # If a real OpenAI key is present, prefer OpenAI endpoint over local defaults.
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
    user_prompt = _build_prompt(question, capabilities)
    messages = [
        {"role": "system", "content": "You produce strict JSON only."},
        {"role": "user", "content": user_prompt},
    ]

    _debug_log("Constructed planner prompt:")
    _debug_log(user_prompt)
    _debug_log("Messages sent to LLM:")
    _debug_log(json.dumps(messages, indent=2))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if not response.ok:
        _debug_log(f"Planner LLM HTTP error status: {response.status_code}")
        _debug_log("Planner LLM HTTP error body:")
        _debug_log(response.text[:4000])
        response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    _debug_log("Raw LLM response content:")
    _debug_log(content)
    json_blob, parsed = _parse_planner_json(content)
    if not json_blob:
        _debug_log("No JSON object found in LLM response content.")
        return None
    _debug_log("Extracted JSON object from LLM response:")
    _debug_log(json_blob)
    if parsed is None:
        _debug_log("Could not parse planner JSON after repair attempts.")
        return None
    decision = _parse_decision(parsed)
    if decision is None:
        _debug_log("Parsed planner JSON missing valid processable/reason fields.")
        return None
    _debug_log("Planner decision:")
    _debug_log(json.dumps(decision, indent=2))
    return decision


def _derive_available_events(agents: List[Dict[str, Any]]) -> List[str]:
    available = set()
    for agent in agents:
        for event in agent.get("subscribes_to", []):
            if event in SUPPORTED_EVENT_SCHEMAS and event != "user.ask":
                available.add(event)
    return sorted(available or DEFAULT_ALLOWED_EVENTS)


def _agent_names_with_task_plan(capabilities: dict) -> set[str]:
    names = set()
    for agent in capabilities.get("agents", []):
        name = agent.get("name")
        subscribes_to = agent.get("subscribes_to", [])
        if isinstance(name, str) and isinstance(subscribes_to, list) and "task.plan" in subscribes_to:
            names.add(name)
    return names


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.-]+", text.lower()))


def _agent_search_blob(agent: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(agent.get("name", "")),
        str(agent.get("description", "")),
        " ".join(entry for entry in agent.get("capability_domains", []) if isinstance(entry, str)),
        " ".join(entry for entry in agent.get("action_verbs", []) if isinstance(entry, str)),
        " ".join(entry for entry in agent.get("routing_notes", []) if isinstance(entry, str)),
    ]
    methods = agent.get("methods", [])
    if isinstance(methods, list):
        for method in methods:
            if not isinstance(method, dict):
                continue
            parts.extend(
                str(method.get(key, ""))
                for key in ("name", "event", "when")
            )
            for list_key in ("intent_tags", "examples", "anti_patterns"):
                values = method.get(list_key, [])
                if isinstance(values, list):
                    parts.append(" ".join(entry for entry in values if isinstance(entry, str)))
    return " ".join(part for part in parts if part)


def _select_target_agent(question: str, capabilities: dict) -> str | None:
    question_lc = question.lower()
    question_tokens = _tokenize(question)
    candidates = []

    for agent in capabilities.get("agents", []):
        name = agent.get("name")
        if not isinstance(name, str):
            continue
        if name in {"ops_planner", "synthesizer"}:
            continue
        subscribes_to = agent.get("subscribes_to", [])
        if not isinstance(subscribes_to, list) or "task.plan" not in subscribes_to:
            continue
        candidates.append(agent)

    if not candidates:
        return None

    priority_boosts = {
        "shell_runner": 0,
        "sql_runner": 0,
        "notifier": 0,
    }

    if any(token in question_tokens for token in {"docker", "container", "containers", "git", "grep", "rg", "find", "list", "logs", "restart", "stop", "start", "ps", "ports"}):
        priority_boosts["shell_runner"] += 10
    if any(token in question_tokens for token in {"sql", "database", "db", "schema", "schemas", "table", "tables", "column", "columns", "query", "queries", "join", "relationships"}):
        priority_boosts["sql_runner"] += 12
    if re.search(r"\b(add|subtract|multiply|divide|calculate|compute|sum)\b", question_lc) or len(re.findall(r"\b\d+(?:\.\d+)?\b", question_lc)) >= 2:
        priority_boosts["shell_runner"] += 10
    if any(token in question_tokens for token in {"notify", "notification", "alert", "remind", "message"}):
        priority_boosts["notifier"] += 10

    best_name = None
    best_score = float("-inf")
    for agent in candidates:
        name = agent["name"]
        blob_tokens = _tokenize(_agent_search_blob(agent))
        score = len(question_tokens & blob_tokens)
        score += priority_boosts.get(name, 0)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def _derive_shell_command(task: str, step_index: int = 1) -> str | None:
    task_lc = task.lower().strip()
    if "{{prev}}" in task_lc:
        if "log" in task_lc:
            tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
            tail = tail_match.group(1) if tail_match else "50"
            return f"docker logs --tail {tail} {{{{prev}}}}"
        if "tail" in task_lc:
            tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
            tail = tail_match.group(1) if tail_match else "20"
            return f"tail -n {tail} {{{{prev}}}}"
        if "grep" in task_lc or "filter" in task_lc:
            grep_match = re.search(r"(?:grep|filter)(?:\s+for)?\s+(.+)$", task, flags=re.IGNORECASE)
            if grep_match:
                term = grep_match.group(1).strip().strip("\"'")
                if term:
                    safe_term = term.replace('"', '\\"')
                    return f'printf "%s\\n" "{{{{prev}}}}" | grep "{safe_term}"'
        if "inspect" in task_lc and "container" in task_lc:
            return "docker inspect {{prev}}"
        if "count" in task_lc and ("error" in task_lc or "errors" in task_lc):
            return 'printf "%s\\n" "{{prev}}" | grep -ic "error"'

    if "docker" in task_lc or "container" in task_lc or "containers" in task_lc:
        if "running" in task_lc and "all" not in task_lc:
            return "docker ps"
        if "all" in task_lc and ("container" in task_lc or "containers" in task_lc):
            return "docker ps -a"
        log_match = re.search(r"logs?(?:\s+for|\s+of)?\s+([a-zA-Z0-9_.-]+)", task, flags=re.IGNORECASE)
        tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
        if "log" in task_lc and log_match:
            tail = tail_match.group(1) if tail_match else "200"
            return f"docker logs --tail {tail} {log_match.group(1)}"
        if "container id" in task_lc and "vllm" in task_lc:
            return 'docker ps --filter name=vllm --format "{{.ID}}"'

    if "newest log file" in task_lc or ("latest log file" in task_lc):
        return 'find . -type f -name "*.log" -printf "%T@ %p\\n" | sort -nr | head -n 1 | cut -d" " -f2-'

    if "show listening ports" in task_lc or "listening ports" in task_lc:
        return "ss -ltnp"

    if "current branch" in task_lc:
        return "git branch --show-current"

    if "git status" in task_lc or (task_lc == "status" and step_index == 1):
        return "git status --short"

    if "git log" in task_lc or "recent commits" in task_lc:
        return "git log --oneline -n 10"

    if "docker images" in task_lc or (task_lc == "images"):
        return "docker images"

    if "docker compose" in task_lc and "ps" in task_lc:
        return "docker compose ps"

    ext_match = re.search(r"files?\s+with\s+extension\s+([a-z0-9]+)\b", task_lc)
    if ext_match:
        return f'find . -type f -name "*.{ext_match.group(1)}"'

    rg_match = re.search(r"(?:find|search).*(?:mentioning|containing|with)\s+([a-zA-Z0-9_.-]+)", task, flags=re.IGNORECASE)
    if rg_match and any(token in task_lc for token in {"python", ".py", "py files"}):
        term = rg_match.group(1).strip().strip("\"'")
        if term:
            safe_term = term.replace('"', '\\"')
            return f'rg -n "{safe_term}" --glob "*.py" .'
    if rg_match:
        term = rg_match.group(1).strip().strip("\"'")
        if term:
            safe_term = term.replace('"', '\\"')
            return f'rg -n "{safe_term}" .'

    if re.search(r"\bopen\s+([./a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9]+)?)", task):
        match = re.search(r"\bopen\s+([./a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9]+)?)", task)
        if match:
            return f'cat {match.group(1)}'

    if task_lc.startswith("list ") and "files" in task_lc:
        return "find . -maxdepth 2 -type f | sort"

    return None


def _derive_shell_result_mode(task: str, command: str | None) -> str | None:
    task_lc = task.lower()
    command_lc = (command or "").lower()
    if "container id" in task_lc or "--format" in command_lc:
        return "stdout_first_line"
    if "current branch" in task_lc or "newest log file" in task_lc or "latest log file" in task_lc:
        return "stdout_first_line"
    return None


def _derive_presentation(question: str) -> dict:
    question_lc = question.lower()
    presentation = {
        "task": "Answer the user request directly using clean Markdown.",
        "format": "markdown",
        "audience": "openwebui",
        "include_context": True,
        "include_internal_steps": False,
    }
    if any(token in question_lc for token in ("table", "tabulate", "columns", "rows", "compare", "comparison", "status", "list")):
        presentation["format"] = "markdown_table"
        presentation["task"] = "Render the result as a clean Markdown table with helpful context."
    elif any(token in question_lc for token in ("json", "raw json")):
        presentation["format"] = "json"
        presentation["task"] = "Return valid JSON matching the user's requested shape."
        presentation["include_context"] = False
    elif any(token in question_lc for token in ("bullet", "bullets")):
        presentation["format"] = "bullets"
        presentation["task"] = "Render the result as concise bullets."
    elif any(token in question_lc for token in ("brief", "concise", "short")):
        presentation["include_context"] = False

    if any(token in question_lc for token in ("show command", "commands used", "how did", "debug", "workflow", "steps")):
        presentation["include_internal_steps"] = True
    return presentation


def _looks_like_metadata_command(command: str) -> bool:
    command = command.strip()
    return any(
        re.search(pattern, command, flags=re.IGNORECASE)
        for pattern in (
            r"^[a-zA-Z_][\w.-]*\s*->\s*[a-zA-Z_][\w.-]*$",
            r"^[a-zA-Z_][\w.-]*\s+emits\s+event\s+[a-zA-Z_][\w.-]*$",
        )
    )


def _is_capability_question(question: str) -> bool:
    question_lc = question.lower()
    if not re.search(r"\b(you|your|system|agent|agents|tool|tools|capabilit|can you|can this|supported|available|introspect|introspection)\b", question_lc):
        return False
    return any(
        phrase in question_lc
        for phrase in (
            "capabilities",
            "capability",
            "introspect",
            "introspection",
            "what can you do",
            "what are you able",
            "available tools",
            "available agents",
            "list your tools",
            "list tools",
            "list agents",
            "list available",
            "supported operations",
            "available operations",
            "what tools",
            "what agents",
            "what operations",
            "show tools",
            "show agents",
            "show capabilities",
        )
    )


def _capability_summary(capabilities: dict) -> dict:
    agents = []
    for agent in capabilities.get("agents", []):
        if not isinstance(agent, dict):
            continue
        name = agent.get("name")
        if not isinstance(name, str) or name in {"ops_planner", "synthesizer"}:
            continue
        entry = {
            "name": name,
            "description": agent.get("description", ""),
            "domains": agent.get("capability_domains", []),
            "actions": agent.get("action_verbs", []),
            "subscribes_to": agent.get("subscribes_to", []),
            "emits": agent.get("emits", []),
        }
        agents.append(entry)
    return {"agents": agents}


def _format_capability_answer(capabilities: dict) -> str:
    summary = _capability_summary(capabilities)
    agents = summary.get("agents", [])
    if not agents:
        return "No runtime capabilities have been discovered yet."

    lines = ["Available runtime capabilities:"]
    for agent in agents:
        name = agent.get("name", "unknown")
        description = agent.get("description") or "No description provided."
        domains = agent.get("domains", [])
        actions = agent.get("actions", [])
        domain_text = ", ".join(item for item in domains if isinstance(item, str)) if isinstance(domains, list) else ""
        action_text = ", ".join(item for item in actions if isinstance(item, str)) if isinstance(actions, list) else ""
        detail = description
        if domain_text:
            detail += f" Domains: {domain_text}."
        if action_text:
            detail += f" Actions: {action_text}."
        lines.append(f"- {name}: {detail}")
    return "\n".join(lines)


def _fallback_steps(question: str, capabilities: dict):
    parts = [part.strip(" ,") for part in re.split(r"\s+(?:and then|then|after that|next)\s+", question, flags=re.IGNORECASE) if part.strip(" ,")]
    if not parts:
        parts = [question.strip()]
    steps = []
    for index, part in enumerate(parts, start=1):
        target_agent = _select_target_agent(part, capabilities)
        task = part
        if target_agent:
            step = {"id": f"step{index}", "target_agent": target_agent, "task": task}
            if target_agent == "shell_runner":
                command = _derive_shell_command(task, index)
                if command:
                    step["command"] = command
                    result_mode = _derive_shell_result_mode(task, command)
                    if result_mode:
                        step["result_mode"] = result_mode
            steps.append(step)
    return steps


def _normalize_steps(question: str, steps: List[Dict[str, str]], capabilities: dict):
    valid_names = _agent_names_with_task_plan(capabilities)
    normalized = []
    available_step_ids = set()
    for index, step in enumerate(steps, start=1):
        nested_steps = step.get("steps")
        if isinstance(nested_steps, list) and nested_steps:
            normalized_nested = _normalize_steps(question, nested_steps, capabilities)
            if normalized_nested:
                group_id = step.get("id") or f"step{index}"
                normalized.append(
                    {
                        "id": group_id,
                        "task": step.get("task", "").strip() or question,
                        "steps": normalized_nested,
                    }
                )
                available_step_ids.add(group_id)
            continue

        target_agent = step.get("target_agent")
        task = step.get("task", "").strip()
        command = step.get("command")
        referenced_keys = set(re.findall(r"\{\{([a-zA-Z0-9_.-]+)\}\}", json.dumps(step)))
        missing_refs = {
            key
            for key in referenced_keys
            if key != "prev" and key not in available_step_ids
        }
        if missing_refs or ("prev" in referenced_keys and not available_step_ids):
            continue
        if target_agent not in valid_names:
            target_agent = _select_target_agent(task or question, capabilities)
        if not target_agent or target_agent in {"ops_planner", "synthesizer"}:
            continue
        normalized_step = {
            "id": step.get("id") or f"step{index}",
            "target_agent": target_agent,
            "task": task or question,
        }
        if target_agent == "shell_runner":
            if isinstance(command, str) and command.strip():
                command = command.strip()
                if not _looks_like_metadata_command(command):
                    normalized_step["command"] = command
            else:
                derived = _derive_shell_command(normalized_step["task"], index)
                if derived:
                    normalized_step["command"] = derived
            result_mode = step.get("result_mode")
            if isinstance(result_mode, str) and result_mode.strip():
                normalized_step["result_mode"] = result_mode.strip()
            else:
                derived_mode = _derive_shell_result_mode(normalized_step["task"], normalized_step.get("command"))
                if derived_mode:
                    normalized_step["result_mode"] = derived_mode
        depends_on = step.get("depends_on")
        if isinstance(depends_on, list):
            clean_depends_on = [item for item in depends_on if isinstance(item, str) and item.strip()]
            if clean_depends_on:
                normalized_step["depends_on"] = clean_depends_on
        normalized.append(normalized_step)
        available_step_ids.add(normalized_step["id"])
    return normalized


def _normalize_presentation(question: str, presentation: dict | None):
    normalized = _derive_presentation(question)
    if not isinstance(presentation, dict):
        return normalized

    allowed_formats = {"markdown", "markdown_table", "json", "bullets", "plain"}
    task = presentation.get("task")
    if isinstance(task, str) and task.strip():
        normalized["task"] = task.strip()
    fmt = presentation.get("format")
    if isinstance(fmt, str) and fmt.strip():
        fmt = fmt.strip().lower()
        normalized["format"] = fmt if fmt in allowed_formats else "markdown"
    audience = presentation.get("audience")
    if isinstance(audience, str) and audience.strip():
        normalized["audience"] = audience.strip()
    for key in ("include_context", "include_internal_steps"):
        value = presentation.get(key)
        if isinstance(value, bool):
            normalized[key] = value
    return normalized


def _flatten_plan_steps(steps: list, prefix: str = "") -> list[dict]:
    flattened = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or f"step{index}")
        display_id = f"{prefix}.{step_id}" if prefix else step_id
        nested = step.get("steps")
        item = {
            "id": display_id,
            "task": step.get("task", ""),
        }
        target_agent = step.get("target_agent")
        if isinstance(target_agent, str) and target_agent:
            item["target_agent"] = target_agent
        command = step.get("command")
        if isinstance(command, str) and command:
            item["command"] = command
        flattened.append(item)
        if isinstance(nested, list) and nested:
            flattened.extend(_flatten_plan_steps(nested, display_id))
    return flattened


def _plan_progress_payload(question: str, steps: list, presentation: dict):
    flat_steps = _flatten_plan_steps(steps)
    message = "I will run this as a workflow."
    if flat_steps:
        step_count = len([step for step in flat_steps if step.get("target_agent")])
        if step_count == 1:
            message = "I found 1 action to run."
        elif step_count > 1:
            message = f"I found {step_count} actions to run in order."
    return {
        "stage": "planned",
        "message": message,
        "steps": flat_steps,
        "presentation": presentation,
    }


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "system.capabilities":
        agents = req.payload.get("agents", [])
        if isinstance(agents, list):
            CAPABILITIES["agents"] = agents
            CAPABILITIES["available_events"] = _derive_available_events(agents)
        return {"emits": []}

    if req.event != "user.ask":
        return {"emits": []}

    question = req.payload["question"]
    allowed_events = set(CAPABILITIES.get("available_events", DEFAULT_ALLOWED_EVENTS))
    if _is_capability_question(question) and "task.result" in allowed_events:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": _format_capability_answer(CAPABILITIES),
                        "result": _capability_summary(CAPABILITIES),
                    },
                }
            ]
        }
    try:
        decision = _llm_decide(question, CAPABILITIES)
        if decision is not None:
            if decision["processable"] and "task.plan" in allowed_events:
                steps = _normalize_steps(question, decision.get("steps", []), CAPABILITIES)
                if not steps:
                    steps = _fallback_steps(question, CAPABILITIES)
                payload = {"task": question}
                if steps:
                    payload["steps"] = steps
                    presentation = _normalize_presentation(question, decision.get("presentation"))
                    payload["presentation"] = presentation
                    if len(steps) == 1:
                        payload["target_agent"] = steps[0]["target_agent"]
                    _debug_log("Planner steps:")
                    _debug_log(json.dumps(steps, indent=2))
                else:
                    _debug_log("Planner found no valid target steps after normalization.")
                emits = []
                if steps and "plan.progress" in allowed_events:
                    emits.append(
                        {
                            "event": "plan.progress",
                            "payload": _plan_progress_payload(question, steps, payload.get("presentation", {})),
                        }
                    )
                emits.append({"event": "task.plan", "payload": payload})
                return {"emits": emits}
            if "task.result" in allowed_events:
                reason = decision["reason"] or "No matching capability found."
                return {"emits": [{"event": "task.result", "payload": {"detail": reason}}]}
            return {"emits": []}
        _debug_log("LLM planner decision was invalid.")
    except Exception as exc:
        _debug_log(f"LLM planning failed. Error: {type(exc).__name__}: {exc}")

    if "task.result" in allowed_events:
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": "Planner could not determine if the request is processable. "
                        "Check LLM connectivity/response format and retry."
                    },
                }
            ]
        }
    return {"emits": []}
