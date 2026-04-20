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
        "When an executing step fails or requests decomposition, emit planner.replan.result with a smaller replacement subplan.",
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
        {
            "name": "decompose_failed_step",
            "event": "planner.replan.result",
            "when": "When a running step requests further decomposition or clarification.",
            "intent_tags": ["replan", "decomposition"],
        },
    ],
}

SUPPORTED_EVENT_SCHEMAS = {
    "task.plan": '{"task":"...","steps":[{"id":"step1","target_agent":"shell_runner","task":"...","instruction":{"operation":"run_command","command":"docker ps","capture":{"mode":"stdout_first_line"}}}]}',
    "task.result": '{"detail":"..."}',
    "plan.progress": '{"stage":"planned","message":"...","steps":[...],"presentation":{...}}',
    "planner.replan.result": '{"replace_step_id":"step1","reason":"...","steps":[{"id":"step1_1","target_agent":"shell_runner","task":"..."}]}',
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
        metadata = []
        database_name = item.get("database_name")
        if isinstance(database_name, str) and database_name.strip():
            metadata.append(f"database_name={database_name.strip()}")
        database_aliases = item.get("database_aliases")
        if isinstance(database_aliases, list):
            aliases = ", ".join(alias for alias in database_aliases if isinstance(alias, str) and alias.strip())
            if aliases:
                metadata.append(f"database_aliases=[{aliases}]")
        template_agent = item.get("template_agent")
        if isinstance(template_agent, str) and template_agent.strip():
            metadata.append(f"template_agent={template_agent.strip()}")
        metadata_text = f" Metadata[{'; '.join(metadata)}]" if metadata else ""
        lines.append(
            f"- {name}: {description} Domains[{domain_text or 'none'}] "
            f"Verbs[{verb_text or 'none'}] Methods[{method_text}]{metadata_text}"
        )
    return "\n".join(lines) if lines else "- unknown: No discovered agents"


def _build_prompt(question: str, capabilities: dict) -> str:
    return (
        "You are the routing planner for an operations assistant.\n"
        "Task: decide whether the request is processable by at least one discovered agent capability and, when processable, "
        "produce the best execution plan and presentation intent.\n"
        "Output JSON only with exact keys: "
        '{"processable":true|false,"reason":"short reason","steps":[{"id":"step1","target_agent":"agent_name","task":"step task","instruction":{"operation":"agent_specific_operation"},"depends_on":["optional_step_id"],"when":{"$from":"optional.path","equals":"optional value"},"steps":[{"id":"step1_1","task":"optional nested substep"}]}],"presentation":{"task":"how to present the result","format":"markdown|markdown_table|json|bullets|plain","audience":"openwebui","include_context":true|false,"include_internal_steps":true|false}}\n'
        "No markdown, no extra keys, no prose outside JSON.\n"
        "Decision policy:\n"
        "1) processable=true if any discovered agent can attempt the request.\n"
        "2) For processable=true, steps MUST be a non-empty ordered list.\n"
        "3) Each step MUST have id and task. Leaf steps MUST have target_agent. Group steps may omit target_agent when they contain nested steps.\n"
        "4) Every leaf step MUST include an instruction object with an agent-native operation and inputs.\n"
        "5) For shell_runner use instruction.operation=run_command with fields such as command, input, and capture.\n"
        "   Example capture objects: {\"mode\":\"stdout_first_line\"}, {\"mode\":\"stdout_stripped\"}, {\"mode\":\"json\"}, {\"mode\":\"json_field\",\"field\":\"envs\"}.\n"
        "   For conditional checks, prefer machine-readable JSON output and set allow_returncodes when a nonzero exit code is expected and should not fail the workflow.\n"
        "6) For sql_runner use instruction.operation values like inspect_schema, query_from_request, execute_sql, or sample_rows.\n"
        "7) For slurm_runner use instruction.operation values like query_from_request, execute_command, cluster_status, list_jobs, or job_details.\n"
        "8) For filesystem use instruction.operation=read_file with a path.\n"
        "9) For notifier use instruction.operation=send_notification with channel/message.\n"
        "10) target_agent MUST be one discovered runtime agent name that can handle task.plan; never use ops_planner or synthesizer.\n"
        "11) Break chained requests into multiple atomic steps. Do not combine unrelated actions into one step. Use nested steps for grouped subtasks.\n"
        "12) Rewrite each step as a direct, concrete instruction for the target agent. Do not simply copy long stretches of the user's wording.\n"
        "13) Each step should be self-contained, operational, and phrased so another LLM can execute it without guessing.\n"
        "14) If a later step depends on an earlier step result, set depends_on and use a when condition or instruction inputs that reference prior results.\n"
        "15) References to previous step data must use objects of the form {\"$from\":\"step_id.field.path\"}. Do not splice raw prose into commands.\n"
        "15a) When a later step needs to inspect or transform rows returned by an earlier step, prefer shell_runner with instruction.input referencing prior rows or the full prior step result.\n"
        "15b) For tasks like 'in this list', 'from those rows', 'check the previous result', or 'search the output', do not issue a fresh SQL query unless the user explicitly asks to query again.\n"
        "16) when is optional. Use it for conditional execution such as {\"$from\":\"step1.returncode\",\"not_equals\":0}.\n"
        "17) For arithmetic chains with explicit numeric operands, use shell_runner with safe arithmetic commands. Do not invent extra arithmetic operations.\n"
        "18) For any safe local machine, workspace, repository, process, service, container, network, build/test, package, arithmetic, text, or data operation that can be expressed as shell commands, prefer shell_runner.\n"
        "19) For SQL/database/schema/table/column/relationship/data questions against a configured database, prefer sql_runner.\n"
        "20) For Slurm, HPC cluster, scheduler, queue, node, partition, reservation, fairshare, or job-control questions, prefer slurm_runner.\n"
        "20a) For Slurm and SQL requests involving counts, averages, stats, or filtering, prefer a SINGLE step using the native agent (slurm_runner or sql_runner) with operation=query_from_request. Avoid decomposing these into shell_runner pipes (grep, wc, etc.) as native agents handle large datasets more reliably.\n"
        "21) For simple SQL/database questions, one sql_runner step is fine. For complex SQL tasks, decompose into concrete sql_runner steps such as inspect schema, sample values, determine join path, and run the final query.\n"
        "22) When you emit multiple SQL steps, each later SQL step should clearly state what it is trying to learn or produce, and it must declare depends_on when it uses earlier findings.\n"
        "23) When using shell_runner, prefer explicit machine-readable commands for deterministic checks.\n"
        "24) For existence checks, avoid substring grep against broad human-readable output. Use exact or JSON-based checks.\n"
        "25) When using sql_runner or slurm_runner, do not invent shell commands or fake CLI syntax for other agents.\n"
        "26) Do not invent nonexistent specialized agents; use discovered agents only.\n"
        "27) Extract presentation intent separately from execution steps. Examples: table, JSON, bullets, concise, detailed, include commands, hide internals.\n"
        "28) NEVER create execution steps for presentation, summarization, formatting, synthesis, or making output readable. That happens after execution in the synthesizer layer and must not appear in steps.\n"
        "29) For Open WebUI, prefer format=markdown_table when the user asks for a table/list/comparison/status report with rows and columns.\n"
        "30) Set include_internal_steps=false unless the user explicitly asks for workflow details, commands, debug output, SQL query text, or how it was done.\n"
        "31) If the user asks about available capabilities, tools, agents, supported operations, or what this system can do, answer from discovered capabilities with task.result; do not call shell_runner.\n"
        "32) Capability metadata is descriptive, not executable. Never turn method names, event names, or capability labels into shell commands.\n"
        "33) Tolerate minor typos and informal phrasing.\n"
        "34) Do NOT mark false only because the request might fail at execution time.\n"
        "35) processable=false only when no discovered capability can attempt it; in that case steps=[].\n"
        "Calibration examples:\n"
        '- "find all files with extension sh in the current directory" -> {"processable":true,"reason":"shell file search","steps":[{"id":"step1","target_agent":"shell_runner","task":"find all files with extension sh in the current directory","instruction":{"operation":"run_command","command":"find . -type f -name \\"*.sh\\"","capture":{"mode":"stdout_stripped"}}}],"presentation":{"task":"List matching files clearly.","format":"bullets","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "list all running docker containers" -> {"processable":true,"reason":"docker container listing","steps":[{"id":"step1","target_agent":"shell_runner","task":"list all running docker containers","instruction":{"operation":"run_command","command":"docker ps","capture":{"mode":"stdout_stripped"}}}],"presentation":{"task":"Show running containers in a concise Markdown table.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "find the newest log file and then show the last 20 lines" -> {"processable":true,"reason":"file discovery workflow","steps":[{"id":"step1","target_agent":"shell_runner","task":"find the newest log file","instruction":{"operation":"run_command","command":"find . -type f -name \\"*.log\\" -printf \\"%T@ %p\\\\n\\" | sort -nr | head -n 1 | cut -d\\" \\" -f2-","capture":{"mode":"stdout_first_line"}}},{"id":"step2","target_agent":"shell_runner","task":"show the last 20 lines of the newest log file","instruction":{"operation":"run_command","command":{"$from":"step1.value"}},"depends_on":["step1"]}]}\n'
        '- "list patients with more than 20 studies and then check whether any listed patient is named Test case-insensitively" -> {"processable":true,"reason":"query then inspect prior rows","steps":[{"id":"step1","target_agent":"sql_runner","task":"list patients with more than 20 studies","instruction":{"operation":"query_from_request","question":"list patients with more than 20 studies and include patient names in a neat table"}},{"id":"step2","target_agent":"shell_runner","task":"check whether any patient in the previous SQL result is named Test using a case-insensitive exact match","instruction":{"operation":"run_command","command":"python3 -c \\"import json,sys; rows=json.load(sys.stdin); print(any(str(row.get(\\\\\\\"PatientName\\\\\\\", \\\\\\\"\\\\\\\")).strip().lower() == \\\\\\\"test\\\\\\\" for row in rows))\\"","input":{"$from":"step1.rows"},"capture":{"mode":"stdout_stripped"}},"depends_on":["step1"]}],"presentation":{"task":"Show the patient table and clearly report whether any listed patient name matches Test case-insensitively.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "show database schema and relationships" -> {"processable":true,"reason":"SQL schema introspection","steps":[{"id":"step1","target_agent":"sql_runner","task":"inspect the database schema and relationships","instruction":{"operation":"inspect_schema","focus":"tables, columns, and relationships"}}],"presentation":{"task":"Summarize schemas, tables, columns, and relationships clearly.","format":"markdown","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "which customers spent the most money last month" -> {"processable":true,"reason":"SQL database question","steps":[{"id":"step1","target_agent":"sql_runner","task":"query which customers spent the most money last month","instruction":{"operation":"query_from_request","question":"which customers spent the most money last month"}}],"presentation":{"task":"Show the query results as a readable Markdown table.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "show queued slurm jobs for user vinith" -> {"processable":true,"reason":"Slurm queue query","steps":[{"id":"step1","target_agent":"slurm_runner","task":"show queued Slurm jobs for user vinith","instruction":{"operation":"query_from_request","question":"show queued Slurm jobs for user vinith"}}],"presentation":{"task":"Show the Slurm results clearly in Markdown.","format":"markdown","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "how many jobs are in a non running state" -> {"processable":true,"reason":"Slurm state count","steps":[{"id":"step1","target_agent":"slurm_runner","task":"how many jobs are in a non running state","instruction":{"operation":"query_from_request","question":"how many jobs are in a non running state"}}],"presentation":{"task":"Report the count of non-running jobs clearly.","format":"plain","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "open Readme.md" -> {"processable":true,"reason":"file read","steps":[{"id":"step1","target_agent":"filesystem","task":"read Readme.md","instruction":{"operation":"read_file","path":"Readme.md"}}]}\n'
        '- "book a flight to NYC" -> {"processable":false,"reason":"no travel booking capability","steps":[]}\n'
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        f'User request: "{question}"'
    )


def _build_replan_prompt(payload: dict, capabilities: dict) -> str:
    return (
        "You are the routing planner for an operations assistant.\n"
        "Task: replace one failed or blocked execution step with a smaller dependency-aware subplan.\n"
        "Output JSON only with exact keys: "
        '{"replace_step_id":"step id","reason":"short reason","steps":[{"id":"step1","target_agent":"agent_name","task":"step task","command":"optional shell command","result_mode":"optional capture mode","depends_on":["optional_step_id"],"steps":[{"id":"step1_1","task":"optional nested substep"}]}]}\n'
        "Rules:\n"
        "1) Return a non-empty replacement steps array.\n"
        "2) Break the failed task into smaller atomic steps that can be attempted by discovered agents.\n"
        "3) Use depends_on when a step requires outputs from earlier steps.\n"
        "3a) Steps without depends_on may run in parallel, so mark sequential steps explicitly.\n"
        "3b) When a replacement step needs to inspect earlier results, prefer shell_runner with instruction.input referencing prior rows or result objects instead of repeating the original query.\n"
        "4) Use only discovered runtime agent names that can handle task.plan; never use ops_planner or synthesizer.\n"
        "5) Replace only the failed step, not the full workflow.\n"
        "6) Prefer diagnostic or introspection subtasks before repeating the same failed action.\n"
        "7) For SQL/database/schema/table/column/relationship/data questions, prefer sql_runner.\n"
        "8) For shell/machine/workspace/container/process/repo operations, prefer shell_runner and explicit commands when possible.\n"
        "9) Tolerate minor typos and informal phrasing.\n"
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        "Replan request JSON:\n"
        f"{json.dumps(payload, indent=2)}"
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


def _parse_replan_decision(raw: Any):
    if not isinstance(raw, dict):
        return None
    replace_step_id = raw.get("replace_step_id")
    reason = raw.get("reason", "")
    steps = raw.get("steps", [])
    if not isinstance(replace_step_id, str) or not replace_step_id.strip():
        return None
    if not isinstance(reason, str):
        reason = ""
    if not isinstance(steps, list):
        return None
    normalized_steps = _parse_steps(steps)
    if normalized_steps is None or not normalized_steps:
        return None
    return {
        "replace_step_id": replace_step_id.strip(),
        "reason": reason.strip(),
        "steps": normalized_steps,
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
        instruction = step.get("instruction")
        nested_steps = step.get("steps")
        depends_on = step.get("depends_on")
        when = step.get("when")
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
        if instruction is not None:
            if not isinstance(instruction, dict) or not instruction:
                return None
            normalized_step["instruction"] = instruction
        elif command is not None:
            if not isinstance(command, str) or not command.strip():
                return None
            shell_instruction = {"operation": "run_command", "command": command.strip()}
            if result_mode is not None:
                if not isinstance(result_mode, str) or not result_mode.strip():
                    return None
                if result_mode.startswith("json_field:"):
                    shell_instruction["capture"] = {
                        "mode": "json_field",
                        "field": result_mode.split(":", 1)[1].strip(),
                    }
                else:
                    shell_instruction["capture"] = {"mode": result_mode.strip()}
            normalized_step["instruction"] = shell_instruction
        if depends_on is not None:
            if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
                return None
            normalized_step["depends_on"] = [item for item in depends_on if item.strip()]
        if when is not None:
            if not isinstance(when, dict) or not when:
                return None
            normalized_step["when"] = when
        if nested_steps is not None:
            if not isinstance(nested_steps, list):
                return None
            normalized_nested = _parse_steps(nested_steps)
            if normalized_nested is None:
                return None
            normalized_step["steps"] = normalized_nested
        if "target_agent" not in normalized_step and not normalized_step.get("steps"):
            return None
        if "target_agent" in normalized_step and "instruction" not in normalized_step and not normalized_step.get("steps"):
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


def _llm_replan(payload: dict, capabilities: dict):
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

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": _build_replan_prompt(payload, capabilities)},
            ],
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    _debug_log("Raw replan LLM response content:")
    _debug_log(content)
    json_blob, parsed = _parse_planner_json(content)
    if not json_blob or parsed is None:
        return None
    decision = _parse_replan_decision(parsed)
    if decision is None:
        return None
    _debug_log("Planner replan decision:")
    _debug_log(json.dumps(decision, indent=2))
    return decision


def _derive_available_events(agents: List[Dict[str, Any]]) -> List[str]:
    available = set()
    for agent in agents:
        for event in agent.get("subscribes_to", []):
            if event in SUPPORTED_EVENT_SCHEMAS and event != "user.ask":
                available.add(event)
        for event in agent.get("emits", []):
            if event in SUPPORTED_EVENT_SCHEMAS:
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


def _extract_agent_invocation(command: str, valid_names: set[str]):
    if not isinstance(command, str):
        return None, None
    stripped = command.strip()
    if not stripped:
        return None, None
    parts = stripped.split(None, 1)
    head = parts[0]
    tail = parts[1].strip() if len(parts) > 1 else ""
    if head in valid_names:
        return head, tail
    return None, None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.-]+", text.lower()))


SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "get",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "list",
    "me",
    "my",
    "name",
    "names",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "what",
    "which",
    "with",
}

FAMILY_COMMON_TOKENS = {
    "slurm_runner": {
        "slurm",
        "cluster",
        "clusters",
        "command",
        "commands",
        "query",
        "queries",
    },
    "sql_runner": {
        "sql",
        "database",
        "databases",
        "db",
        "query",
        "queries",
    },
}

SLURM_ANCHOR_GROUPS = {
    "nodes": {"node", "nodes", "partition", "partitions", "sinfo"},
    "jobs": {"job", "jobs", "queue", "queued", "pending", "running", "squeue", "sacct"},
    "users": {"user", "users"},
    "reservations": {"reservation", "reservations"},
    "fairshare": {"fairshare", "qos", "sshare", "sprio"},
}

SQL_ANCHOR_GROUPS = {
    "schema": {"schema", "schemas", "table", "tables", "column", "columns", "relationship", "relationships"},
    "rows": {"row", "rows", "count", "counts", "list", "show"},
}


def _filtered_semantic_tokens(text: str, family: str) -> set[str]:
    tokens = _tokenize(text)
    family_common = FAMILY_COMMON_TOKENS.get(family, set())
    normalized = set()
    for token in tokens:
        normalized.add(token)
        if len(token) > 3 and token.endswith("s"):
            normalized.add(token[:-1])
    return {
        token
        for token in normalized
        if token not in SEMANTIC_STOPWORDS and token not in family_common and len(token) > 2
    }


def _present_anchor_groups(tokens: set[str], groups: dict[str, set[str]]) -> set[str]:
    present = set()
    for name, values in groups.items():
        if tokens & values:
            present.add(name)
    return present


def _semantic_anchor_drift(original_tokens: set[str], candidate_tokens: set[str], family: str) -> bool:
    if family == "slurm_runner":
        original_groups = _present_anchor_groups(original_tokens, SLURM_ANCHOR_GROUPS)
        candidate_groups = _present_anchor_groups(candidate_tokens, SLURM_ANCHOR_GROUPS)
        if "nodes" in original_groups and "nodes" not in candidate_groups:
            return True
        if "jobs" in original_groups and "jobs" not in candidate_groups:
            return True
        if "users" not in original_groups and "users" in candidate_groups and "nodes" in original_groups:
            return True
        if "nodes" in original_groups and "jobs" in candidate_groups and "jobs" not in original_groups:
            return True
    if family == "sql_runner":
        original_groups = _present_anchor_groups(original_tokens, SQL_ANCHOR_GROUPS)
        candidate_groups = _present_anchor_groups(candidate_tokens, SQL_ANCHOR_GROUPS)
        if "schema" in original_groups and "schema" not in candidate_groups:
            return True
    return False


def _step_semantic_drift(question: str, target_agent: str, task: str, instruction: dict | None) -> bool:
    family = _agent_family(target_agent)
    if family not in {"slurm_runner", "sql_runner"}:
        return False
    candidate_text = task
    if isinstance(instruction, dict):
        candidate_text += "\n" + json.dumps(instruction, ensure_ascii=True, sort_keys=True)
    original_tokens = _tokenize(question)
    candidate_tokens = _tokenize(candidate_text)
    if _semantic_anchor_drift(original_tokens, candidate_tokens, family):
        return True
    original_filtered = _filtered_semantic_tokens(question, family)
    candidate_filtered = _filtered_semantic_tokens(candidate_text, family)
    if original_filtered and candidate_filtered and not (original_filtered & candidate_filtered):
        return True
    return False


def _agent_search_blob(agent: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(agent.get("name", "")),
        str(agent.get("description", "")),
        " ".join(entry for entry in agent.get("capability_domains", []) if isinstance(entry, str)),
        " ".join(entry for entry in agent.get("action_verbs", []) if isinstance(entry, str)),
        " ".join(entry for entry in agent.get("routing_notes", []) if isinstance(entry, str)),
    ]
    for key in ("database_name", "database_aliases", "argument_name", "template_agent"):
        value = agent.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.append(" ".join(entry for entry in value if isinstance(entry, str)))
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


def _agent_family(name: str) -> str:
    if name.startswith("sql_runner"):
        return "sql_runner"
    if name.startswith("slurm_runner"):
        return "slurm_runner"
    return name


def _has_agent_family(capabilities: dict, family: str) -> bool:
    return any(
        isinstance(agent, dict)
        and isinstance(agent.get("name"), str)
        and _agent_family(agent["name"]) == family
        and "task.plan" in agent.get("subscribes_to", [])
        for agent in capabilities.get("agents", [])
    )


def _configured_database_tokens(capabilities: dict) -> set[str]:
    tokens = set()
    for agent in capabilities.get("agents", []):
        if not isinstance(agent, dict):
            continue
        for key in ("database_name", "database_aliases", "argument_name"):
            value = agent.get(key)
            if isinstance(value, str):
                tokens.update(_tokenize(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        tokens.update(_tokenize(item))
    return tokens


def _looks_like_sql_question(question: str, capabilities: dict) -> bool:
    tokens = _tokenize(question)
    sql_tokens = {
        "sql",
        "database",
        "db",
        "schema",
        "schemas",
        "table",
        "tables",
        "column",
        "columns",
        "query",
        "queries",
        "join",
        "relationships",
    }
    if tokens & sql_tokens:
        return True
    if tokens & _configured_database_tokens(capabilities):
        return True
    return False


def _looks_like_slurm_question(question: str) -> bool:
    tokens = _tokenize(question)
    return bool(
        tokens
        & {
            "slurm",
            "sinfo",
            "squeue",
            "sacct",
            "scontrol",
            "slurmdbd",
            "partition",
            "partitions",
            "node",
            "nodes",
            "job",
            "jobs",
            "reservation",
            "reservations",
            "cluster",
            "qos",
            "fairshare",
            "scancel",
            "hpc",
        }
    )


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
        "slurm_runner": 0,
        "notifier": 0,
    }

    if any(token in question_tokens for token in {"docker", "container", "containers", "git", "grep", "rg", "find", "list", "logs", "restart", "stop", "start", "ps", "ports"}):
        priority_boosts["shell_runner"] += 10
    if _looks_like_sql_question(question, capabilities):
        priority_boosts["sql_runner"] += 12
    if _looks_like_slurm_question(question):
        priority_boosts["slurm_runner"] += 12
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
        score += priority_boosts.get(_agent_family(name), 0)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def _derive_shell_command(task: str, step_index: int = 1) -> str | None:
    task_lc = task.lower().strip()
    save_path = _extract_save_output_path(task)

    if save_path and any(token in task_lc for token in {"list", "save", "write", "create"}):
        return _build_save_rows_command(save_path)

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


def _extract_save_output_path(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    match = re.search(
        r"\b(?:save|write)(?:\s+it|\s+them|\s+these(?:\s+\w+)*)?\s+(?:in|to)\s+([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"\bfile\s+named\s+([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Za-z0-9_./-]+\.txt)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _build_save_rows_command(path: str) -> str:
    safe_path = json.dumps(path)
    return (
        "python3 -c 'import json,sys,pathlib; "
        "rows=json.load(sys.stdin); "
        "lines=[(f\"{row.get(\\\"PatientID\\\", \\\"\\\")}: {row.get(\\\"PatientName\\\", \\\"\\\")}\".strip(\": \") "
        "if isinstance(row, dict) and (row.get(\"PatientID\") or row.get(\"PatientName\")) "
        "else (json.dumps(row, ensure_ascii=True) if isinstance(row, dict) else str(row))) for row in rows]; "
        f"path=pathlib.Path({safe_path}); "
        "path.write_text(\"\\n\".join(lines)); "
        "print(path.resolve())'"
    )


def _build_save_rows_step(source_step_id: str, path: str, step_index: int = 2) -> dict:
    return {
        "id": f"step{step_index}",
        "target_agent": "shell_runner",
        "task": f"save the rows from {source_step_id} into {path} and print the final absolute path",
        "instruction": {
            "operation": "run_command",
            "command": _build_save_rows_command(path),
            "input": {"$from": f"{source_step_id}.rows"},
            "capture": {"mode": "stdout_stripped"},
        },
        "depends_on": [source_step_id],
    }


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


def _planner_is_schema_request(task: str) -> bool:
    task_lc = task.lower()
    return any(
        token in task_lc
        for token in (
            "schema",
            "schemas",
            "table",
            "tables",
            "column",
            "columns",
            "relationship",
            "relationships",
            "relation",
            "relations",
            "foreign key",
            "foreign keys",
            "describe database",
            "inspect the database schema",
        )
    )


PRESENTATION_ONLY_PATTERNS = [
    r"\bsynthesi[sz]e\b",
    r"\bsummariz(?:e|ing|ation)\b",
    r"\bformat\b",
    r"\brender\b",
    r"\breadable format\b",
    r"\bmake (?:the )?result readable\b",
    r"\bpresent the result\b",
    r"\bconvert .* to markdown\b",
]


def _is_presentation_only_task(task: str) -> bool:
    task_lc = task.lower().strip()
    if not task_lc:
        return False
    return any(re.search(pattern, task_lc) for pattern in PRESENTATION_ONLY_PATTERNS)


def _normalize_agent_instruction(target_agent: str, task: str, instruction: dict | None, command: str | None, index: int):
    family = _agent_family(target_agent)
    if family == "sql_runner":
        if _is_presentation_only_task(task):
            return None
        if isinstance(instruction, dict) and instruction:
            op = instruction.get("operation")
            if op in {"inspect_schema", "query_from_request", "execute_sql", "sample_rows"}:
                return instruction
        if isinstance(command, str) and command.strip():
            stripped = command.strip()
            if stripped.lower().startswith(("select ", "with ", "show ", "describe ", "explain ")):
                return {"operation": "execute_sql", "sql": stripped}
        if _planner_is_schema_request(task):
            return {"operation": "inspect_schema", "focus": task}
        return {"operation": "query_from_request", "question": task}

    if family == "slurm_runner":
        if _is_presentation_only_task(task):
            return None
        if isinstance(instruction, dict) and instruction:
            op = instruction.get("operation")
            if op in {"query_from_request", "execute_command", "cluster_status", "list_jobs", "job_details"}:
                return instruction
        if isinstance(command, str) and command.strip():
            return {"operation": "execute_command", "command": command.strip()}
        return {"operation": "query_from_request", "question": task}

    if target_agent == "shell_runner":
        shell_instruction = instruction if isinstance(instruction, dict) and instruction.get("operation") == "run_command" else None
        if shell_instruction is None:
            if isinstance(command, str) and command.strip():
                stripped = command.strip()
                if _looks_like_metadata_command(stripped):
                    return None
                shell_instruction = {"operation": "run_command", "command": stripped}
            else:
                derived = _derive_shell_command(task, index)
                if derived:
                    shell_instruction = {"operation": "run_command", "command": derived}
        if not isinstance(shell_instruction, dict):
            return None
        capture = shell_instruction.get("capture")
        if not isinstance(capture, dict):
            derived_mode = _derive_shell_result_mode(task, shell_instruction.get("command"))
            if derived_mode:
                shell_instruction = {
                    **shell_instruction,
                    "capture": {"mode": derived_mode},
                }
        env_name = _extract_conda_env_name(task, shell_instruction.get("command"))
        if env_name and _is_conda_env_existence_task(task, shell_instruction.get("command")):
            shell_instruction = {
                **shell_instruction,
                "command": _build_conda_env_exists_command(env_name),
                "capture": {"mode": "json"},
                "allow_returncodes": [0, 1],
            }
        return shell_instruction

    if target_agent == "filesystem":
        if _is_presentation_only_task(task):
            return None
        if isinstance(instruction, dict) and instruction.get("operation") == "read_file":
            return instruction
        return {"operation": "read_file"}

    if target_agent == "notifier":
        if isinstance(instruction, dict) and instruction.get("operation") == "send_notification":
            return instruction
        return {"operation": "send_notification"}

    return instruction if isinstance(instruction, dict) and instruction else None


def _looks_like_metadata_command(command: str) -> bool:
    command = command.strip()
    return any(
        re.search(pattern, command, flags=re.IGNORECASE)
        for pattern in (
            r"^[a-zA-Z_][\w.-]*\s*->\s*[a-zA-Z_][\w.-]*$",
            r"^[a-zA-Z_][\w.-]*\s+emits\s+event\s+[a-zA-Z_][\w.-]*$",
        )
    )


def _extract_conda_env_name(task: str, command: str | None = None) -> str | None:
    candidates = [task]
    if isinstance(command, str) and command.strip():
        candidates.append(command)
    patterns = (
        r"\bconda environment named\s+([A-Za-z0-9._-]+)\b",
        r"\benv(?:ironment)? named\s+([A-Za-z0-9._-]+)\b",
        r"\b-n\s+([A-Za-z0-9._-]+)\b",
    )
    for text in candidates:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
    return None


def _is_conda_env_existence_task(task: str, command: str | None = None) -> bool:
    haystacks = [task.lower()]
    if isinstance(command, str):
        haystacks.append(command.lower())
    return any("conda" in text and "env" in text and "exist" in text for text in haystacks)


def _build_conda_env_exists_command(env_name: str) -> str:
    return (
        "conda env list --json | "
        "python3 -c "
        f"'import json,sys,os; name={json.dumps(env_name)}; "
        "envs=json.load(sys.stdin).get(\"envs\", []); "
        "exists=any(os.path.basename(path.rstrip(\"/\")) == name for path in envs); "
        "print(json.dumps({\"exists\": exists, \"name\": name})); "
        "raise SystemExit(0 if exists else 1)'"
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
    compound_steps = _compound_fallback_steps(question, capabilities)
    if compound_steps:
        return compound_steps
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


def _sql_fallback_steps(question: str, capabilities: dict):
    return _sql_fallback_steps_for_task(question, question, capabilities)


def _sql_fallback_steps_for_task(planning_question: str, task_question: str, capabilities: dict):
    if not _looks_like_sql_question(planning_question, capabilities):
        return []
    target_agent = _select_target_agent(planning_question, capabilities)
    if not target_agent or _agent_family(target_agent) != "sql_runner":
        return []
    return [
        {
            "id": "step1",
            "target_agent": target_agent,
            "task": task_question.strip(),
            "instruction": {
                "operation": "query_from_request",
                "question": task_question.strip(),
            },
        }
    ]


def _collapse_sql_steps(question: str, steps: list[dict], capabilities: dict):
    return steps


def _current_request_from_context(question: str) -> str:
    match = re.search(r"(?:^|\n)Current user request:\s*(.+)\s*$", question, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return question
    current = match.group(1).strip()
    return current or question


COMPOUND_REQUEST_MARKER_RE = re.compile(
    r"\b(?:how many|count|list|show|find|which|what(?:\s+is|\s+are)?)\b",
    flags=re.IGNORECASE,
)


def _split_compound_request(question: str) -> list[str]:
    text = re.sub(r"\s+", " ", question).strip()
    if not text:
        return []

    matches = list(COMPOUND_REQUEST_MARKER_RE.finditer(text))
    if len(matches) < 2:
        return [text]

    split_points = [0]
    lower_text = text.lower()
    for match in matches[1:]:
        prefix = lower_text[split_points[-1] : match.start()]
        if any(separator in prefix for separator in (" and ", ", and ", ";", ",")):
            split_points.append(match.start())

    if len(split_points) < 2:
        return [text]

    parts = []
    split_points.append(len(text))
    for start, end in zip(split_points, split_points[1:]):
        part = text[start:end].strip(" ,;")
        part = re.sub(r"\s+(?:,?\s*and|;)\s*$", "", part, flags=re.IGNORECASE)
        part = re.sub(r"^(?:and|then|also)\s+", "", part, flags=re.IGNORECASE)
        if part:
            parts.append(part)
    return parts or [text]


def _compound_fallback_steps(question: str, capabilities: dict):
    parts = _split_compound_request(question)
    if len(parts) < 2:
        return []

    steps = []
    for index, part in enumerate(parts, start=1):
        target_agent = _select_target_agent(part, capabilities)
        if not target_agent or target_agent in {"ops_planner", "synthesizer"}:
            return []
        instruction = _normalize_agent_instruction(target_agent, part, None, None, index)
        if not isinstance(instruction, dict) or not instruction:
            return []
        step = {
            "id": f"step{index}",
            "target_agent": target_agent,
            "task": part,
            "instruction": instruction,
        }
        steps.append(step)
    return steps


def _recover_compound_steps(question: str, steps: list[dict], capabilities: dict):
    if len(steps) != 1:
        return steps
    compound_steps = _compound_fallback_steps(question, capabilities)
    if len(compound_steps) < 2:
        return steps

    original_task = str(steps[0].get("task") or "").strip().lower()
    split_tasks = {str(step.get("task") or "").strip().lower() for step in compound_steps}
    if original_task and original_task in split_tasks:
        return compound_steps
    if original_task != question.strip().lower():
        return compound_steps
    return steps


def _looks_like_save_list_sql_request(question: str, capabilities: dict) -> bool:
    question_lc = question.lower()
    return (
        _looks_like_sql_question(question, capabilities)
        and bool(re.search(r"\b(?:save|write)\b", question_lc))
        and bool(re.search(r"\b(?:list|llist|users|patients|rows|results|tables|schema)\b", question_lc))
    )


def _list_query_from_request(question: str) -> str:
    compact = re.sub(r"\s+", " ", question).strip()
    compact = re.split(r"\s*,\s*(?:create|save|write)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    compact = re.split(r"\b(?:and then|then)\b\s+(?:create|save|write)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return (
        f"{compact}. Return one row per matching patient and include PatientID and PatientName. "
        "Do not return only an aggregate count."
    )


def _align_sql_steps_with_downstream_needs(question: str, steps: list[dict], capabilities: dict) -> list[dict]:
    if not steps or not _looks_like_save_list_sql_request(question, capabilities):
        return steps

    save_path = _extract_save_output_path(question)
    if not save_path:
        return steps

    aligned = []
    saw_sql_list_rewrite = False
    saw_save_step = False
    first_sql_step_id = None
    for index, step in enumerate(steps):
        updated = dict(step)
        target_agent = updated.get("target_agent")
        instruction = updated.get("instruction")
        if first_sql_step_id is None and _agent_family(str(target_agent)) == "sql_runner":
            first_sql_step_id = str(updated.get("id") or f"step{index+1}")
        if (
            not saw_sql_list_rewrite
            and _agent_family(str(target_agent)) == "sql_runner"
            and isinstance(instruction, dict)
            and instruction.get("operation") == "query_from_request"
        ):
            downstream = steps[index + 1] if index + 1 < len(steps) else None
            if isinstance(downstream, dict) and str(downstream.get("target_agent")) == "shell_runner":
                downstream_task = str(downstream.get("task") or "")
                if _extract_save_output_path(downstream_task):
                    updated_instruction = dict(instruction)
                    updated_instruction["question"] = _list_query_from_request(question)
                    updated["instruction"] = updated_instruction
                    updated["task"] = "List the qualifying patients from mydb with identifying columns for export"
                    saw_sql_list_rewrite = True

        if (
            str(updated.get("target_agent")) == "shell_runner"
            and _extract_save_output_path(str(updated.get("task") or ""))
        ):
            saw_save_step = True
            command = _derive_shell_command(str(updated.get("task") or ""), index + 1)
            if command:
                input_ref = None
                if isinstance(updated.get("instruction"), dict):
                    input_ref = updated["instruction"].get("input")
                updated["instruction"] = {
                    "operation": "run_command",
                    "command": command,
                    "capture": {"mode": "stdout_stripped"},
                }
                if input_ref is not None:
                    updated["instruction"]["input"] = input_ref
        aligned.append(updated)
    if first_sql_step_id and not saw_save_step:
        aligned.append(_build_save_rows_step(first_sql_step_id, save_path, len(aligned) + 1))
    return aligned


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
        instruction = step.get("instruction")
        invoked_agent = None
        invoked_tail = None
        if isinstance(command, str):
            invoked_agent, invoked_tail = _extract_agent_invocation(command, valid_names)
            if invoked_agent:
                target_agent = invoked_agent
                command = invoked_tail
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
        if _is_presentation_only_task(task or question):
            continue
        normalized_step = {
            "id": step.get("id") or f"step{index}",
            "target_agent": target_agent,
            "task": task or question,
        }
        result_mode = step.get("result_mode")
        if target_agent == "shell_runner" and isinstance(instruction, dict) and isinstance(result_mode, str) and result_mode.strip():
            normalized_instruction = dict(instruction)
            if normalized_instruction.get("operation") == "run_command" and not isinstance(normalized_instruction.get("capture"), dict):
                mode = result_mode.strip()
                normalized_instruction["capture"] = (
                    {"mode": "json_field", "field": mode.split(":", 1)[1].strip()}
                    if mode.startswith("json_field:")
                    else {"mode": mode}
                )
            instruction = normalized_instruction
        normalized_instruction = _normalize_agent_instruction(
            target_agent,
            normalized_step["task"],
            instruction if isinstance(instruction, dict) else None,
            command if isinstance(command, str) else None,
            index,
        )
        if not isinstance(normalized_instruction, dict) or not normalized_instruction:
            continue
        if _step_semantic_drift(question, target_agent, normalized_step["task"], normalized_instruction):
            _debug_log(
                "Rejecting semantically drifted step and recovering to original request: "
                + json.dumps(
                    {
                        "question": question,
                        "target_agent": target_agent,
                        "task": normalized_step["task"],
                        "instruction": normalized_instruction,
                    },
                    ensure_ascii=True,
                )
            )
            recovered_instruction = _normalize_agent_instruction(target_agent, question, None, None, index)
            if not isinstance(recovered_instruction, dict) or not recovered_instruction:
                continue
            normalized_step["task"] = question
            normalized_instruction = recovered_instruction
        normalized_step["instruction"] = normalized_instruction
        depends_on = step.get("depends_on")
        if isinstance(depends_on, list):
            clean_depends_on = [item for item in depends_on if isinstance(item, str) and item.strip()]
            if clean_depends_on:
                normalized_step["depends_on"] = clean_depends_on
        when = step.get("when")
        if isinstance(when, dict) and when:
            normalized_step["when"] = when
        normalized.append(normalized_step)
        available_step_ids.add(normalized_step["id"])
    normalized = _recover_compound_steps(question, normalized, capabilities)
    return _align_sql_steps_with_downstream_needs(question, normalized, capabilities)


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
        instruction = step.get("instruction")
        if not isinstance(command, str) and isinstance(instruction, dict):
            maybe_command = instruction.get("command")
            if isinstance(maybe_command, str):
                command = maybe_command
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


def _sql_plan_emits(question: str, allowed_events: set[str], task_question: str | None = None):
    task_question = task_question or _current_request_from_context(question)
    sql_steps = _sql_fallback_steps_for_task(question, task_question, CAPABILITIES)
    if not sql_steps or "task.plan" not in allowed_events:
        return None
    presentation = _normalize_presentation(
        task_question,
        {
            "task": "Show the query result as a readable Markdown table and include the SQL used.",
            "format": "markdown_table",
            "audience": "openwebui",
            "include_context": True,
            "include_internal_steps": False,
        },
    )
    payload = {
        "task": task_question,
        "steps": sql_steps,
        "presentation": presentation,
        "target_agent": sql_steps[0]["target_agent"],
    }
    emits = []
    if "plan.progress" in allowed_events:
        emits.append(
            {
                "event": "plan.progress",
                "payload": _plan_progress_payload(task_question, sql_steps, presentation),
            }
        )
    emits.append({"event": "task.plan", "payload": payload})
    return emits


def _fallback_replan_steps(payload: dict, capabilities: dict):
    task = str(payload.get("task") or "").strip()
    if not task:
        return []
    steps = _fallback_steps(task, capabilities)
    if not steps:
        return []
    failed_task = str(payload.get("task") or "").strip().lower()
    if len(steps) == 1:
        only = steps[0]
        if (
            str(only.get("task") or "").strip().lower() == failed_task
            and not only.get("steps")
            and not only.get("depends_on")
            and not only.get("command")
        ):
            return []
    return steps


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "system.capabilities":
        agents = req.payload.get("agents", [])
        if isinstance(agents, list):
            CAPABILITIES["agents"] = agents
            CAPABILITIES["available_events"] = _derive_available_events(agents)
        return {"emits": []}

    if req.event == "planner.replan.request":
        allowed_events = set(CAPABILITIES.get("available_events", DEFAULT_ALLOWED_EVENTS))
        if "planner.replan.result" not in allowed_events:
            return {"emits": []}
        replace_step_id = str(req.payload.get("step_id") or "").strip()
        if not replace_step_id:
            return {"emits": []}
        steps = []
        try:
            decision = _llm_replan(req.payload, CAPABILITIES)
            if decision is not None and decision["replace_step_id"] == replace_step_id:
                steps = _normalize_steps(str(req.payload.get("task") or ""), decision.get("steps", []), CAPABILITIES)
        except Exception as exc:
            _debug_log(f"Planner replan failed. Error: {type(exc).__name__}: {exc}")
        if not steps:
            steps = _fallback_replan_steps(req.payload, CAPABILITIES)
        if not steps:
            return {
                "emits": [
                    {
                        "event": "task.result",
                        "payload": {
                            "detail": req.payload.get("reason") or "Planner could not further decompose the failed step.",
                            "status": "failed",
                            "error": req.payload.get("error") or req.payload.get("reason"),
                        },
                    }
                ]
            }
        emits = []
        if "plan.progress" in allowed_events:
            emits.append(
                {
                    "event": "plan.progress",
                    "payload": _plan_progress_payload(
                        str(req.payload.get("task") or ""),
                        steps,
                        req.payload.get("presentation", {}),
                    ),
                }
            )
        emits.append(
            {
                "event": "planner.replan.result",
                "payload": {
                    "replace_step_id": replace_step_id,
                    "reason": str(req.payload.get("reason") or "Replanned failed step."),
                    "steps": steps,
                },
            }
        )
        return {"emits": emits}

    if req.event != "user.ask":
        return {"emits": []}

    question = req.payload["question"]
    current_question = _current_request_from_context(question)
    allowed_events = set(CAPABILITIES.get("available_events", DEFAULT_ALLOWED_EVENTS))
    if _is_capability_question(current_question) and "task.result" in allowed_events:
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
                steps = _normalize_steps(current_question, decision.get("steps", []), CAPABILITIES)
                if not steps:
                    steps = _fallback_steps(current_question, CAPABILITIES)
                if not steps and _looks_like_sql_question(question, CAPABILITIES):
                    steps = _sql_fallback_steps_for_task(question, current_question, CAPABILITIES) or steps
                payload = {"task": current_question}
                if steps:
                    payload["steps"] = steps
                    presentation = _normalize_presentation(current_question, decision.get("presentation"))
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
                            "payload": _plan_progress_payload(current_question, steps, payload.get("presentation", {})),
                        }
                    )
                emits.append({"event": "task.plan", "payload": payload})
                return {"emits": emits}
            emits = _sql_plan_emits(question, allowed_events, current_question)
            if emits is not None:
                return {"emits": emits}
            if "task.result" in allowed_events:
                reason = decision["reason"] or "No matching capability found."
                return {"emits": [{"event": "task.result", "payload": {"detail": reason}}]}
            return {"emits": []}
        _debug_log("LLM planner decision was invalid.")
    except Exception as exc:
        _debug_log(f"LLM planning failed. Error: {type(exc).__name__}: {exc}")

    emits = _sql_plan_emits(question, allowed_events, current_question)
    if emits is not None:
        return {"emits": emits}

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
