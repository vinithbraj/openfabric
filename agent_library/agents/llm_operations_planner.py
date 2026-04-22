import json
import os
import re
import shlex
from typing import Any, Dict, List

import requests
from web_compat import FastAPI

from agent_library.common import EventRequest, EventResponse, shared_llm_api_settings, with_node_envelope
from agent_library.contracts import api_emit_events, api_trigger_event
from agent_library.template import (
    agent_api,
    agent_descriptor,
    emit_sequence,
    failure_result,
    noop,
    task_result,
)
from runtime.console import log_debug

app = FastAPI()

AGENT_DESCRIPTOR = agent_descriptor(
    name="ops_planner",
    role="router",
    description="LLM planner that decides whether a request is processable by discovered system capabilities.",
    capability_domains=["planning", "routing", "operations"],
    action_verbs=["plan", "route", "assess"],
    side_effect_policy="read_only",
    safety_enforced_by_agent=True,
    routing_notes=[
        "Only decide if request is processable by discovered capabilities.",
        "If processable, emit task.plan with original task and the best target agent for focused execution.",
        "If not processable, emit task.result with reason.",
        "When an executing step fails or requests decomposition, emit planner.replan.result with a smaller replacement subplan.",
    ],
    apis=[
        agent_api(
            name="assess_processable_request",
            trigger_event="user.ask",
            emits=["plan.progress", "task.plan"],
            summary="Assesses whether a request can be handled by discovered system capabilities.",
            when="When request can be handled by at least one discovered agent capability, including multi-step chains.",
            intent_tags=["processable", "capability_match"],
            deterministic=False,
            side_effect_level="read_only",
        ),
        agent_api(
            name="reject_unprocessable_request",
            trigger_event="user.ask",
            emits=["task.result"],
            summary="Explains why a request cannot be handled by discovered capabilities.",
            when="When request cannot be handled by discovered capabilities.",
            intent_tags=["unprocessable"],
            deterministic=False,
            side_effect_level="read_only",
        ),
        agent_api(
            name="decompose_failed_step",
            trigger_event="planner.replan.request",
            emits=["planner.replan.result"],
            summary="Builds a smaller replacement subplan for a failed step.",
            when="When a running step requests further decomposition or clarification.",
            intent_tags=["replan", "decomposition"],
            deterministic=False,
            side_effect_level="read_only",
        ),
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR

SUPPORTED_EVENT_SCHEMAS = {
    "task.plan": '{"task":"...","task_shape":"lookup","steps":[{"id":"step1","target_agent":"shell_runner","task":"...","instruction":{"operation":"run_command","command":"docker ps","capture":{"mode":"stdout_first_line"}}}]}',
    "task.result": '{"detail":"..."}',
    "plan.progress": '{"stage":"planned","message":"...","steps":[...],"presentation":{...}}',
    "planner.replan.result": '{"replace_step_id":"step1","reason":"...","steps":[{"id":"step1_1","target_agent":"shell_runner","task":"..."}]}',
}
DEFAULT_ALLOWED_EVENTS = set(SUPPORTED_EVENT_SCHEMAS.keys())
CAPABILITIES = {"agents": [], "available_events": sorted(DEFAULT_ALLOWED_EVENTS), "execution_policy": {}}


def _debug_enabled() -> bool:
    return os.getenv("LLM_OPS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("LLM_OPS_DEBUG", message)


def _agent_api_specs(agent: dict) -> list[dict]:
    api_list = agent.get("apis")
    if isinstance(api_list, list):
        return [item for item in api_list if isinstance(item, dict)]
    method_list = agent.get("methods")
    if isinstance(method_list, list):
        return [item for item in method_list if isinstance(item, dict)]
    return []


def _agent_trigger_events(agent: dict) -> set[str]:
    triggers = {
        item
        for item in agent.get("subscribes_to", [])
        if isinstance(item, str) and item.strip()
    }
    for api in _agent_api_specs(agent):
        trigger_event = api_trigger_event(api)
        if trigger_event:
            triggers.add(trigger_event)
    return triggers


def _agent_emitted_events(agent: dict) -> set[str]:
    emitted = {
        item
        for item in agent.get("emits", [])
        if isinstance(item, str) and item.strip()
    }
    for api in _agent_api_specs(agent):
        emitted.update(api_emit_events(api))
    return emitted


def _agent_handles_trigger(agent: dict, event_name: str) -> bool:
    return event_name in _agent_trigger_events(agent)


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
        for method in _agent_api_specs(item):
            method_name = method.get("name")
            method_event = api_trigger_event(method)
            emitted_events = api_emit_events(method)
            if isinstance(method_name, str) and method_event:
                method_summary = f"{method_name} triggered by {method_event}"
                if emitted_events:
                    method_summary += f" -> {', '.join(emitted_events)}"
                methods.append(method_summary)
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
        execution_model = item.get("execution_model")
        if isinstance(execution_model, str) and execution_model.strip():
            metadata.append(f"execution_model={execution_model.strip()}")
        catalog_version = item.get("deterministic_catalog_version")
        if isinstance(catalog_version, str) and catalog_version.strip():
            metadata.append(f"deterministic_catalog_version={catalog_version.strip()}")
        catalog_size = item.get("deterministic_catalog_size")
        if isinstance(catalog_size, (int, float)):
            metadata.append(f"deterministic_catalog_size={int(catalog_size)}")
        catalog_families = item.get("deterministic_catalog_families")
        if isinstance(catalog_families, list):
            families = ", ".join(entry for entry in catalog_families if isinstance(entry, str) and entry.strip())
            if families:
                metadata.append(f"deterministic_catalog_families=[{families}]")
        cluster_name = item.get("cluster_name")
        if isinstance(cluster_name, str) and cluster_name.strip():
            metadata.append(f"cluster_name={cluster_name.strip()}")
        cluster_aliases = item.get("cluster_aliases")
        if isinstance(cluster_aliases, list):
            aliases = ", ".join(alias for alias in cluster_aliases if isinstance(alias, str) and alias.strip())
            if aliases:
                metadata.append(f"cluster_aliases=[{aliases}]")
        metadata_text = f" Metadata[{'; '.join(metadata)}]" if metadata else ""
        lines.append(
            f"- {name}: {description} Domains[{domain_text or 'none'}] "
            f"Verbs[{verb_text or 'none'}] Methods[{method_text}]{metadata_text}"
        )
    return "\n".join(lines) if lines else "- unknown: No discovered agents"


def _format_execution_policy(capabilities: dict) -> str:
    policy = capabilities.get("execution_policy")
    if not isinstance(policy, dict) or not policy:
        return "- none"
    lines = []
    if "allow_python_package_installs" in policy:
        lines.append(f"- allow_python_package_installs={bool(policy.get('allow_python_package_installs'))}")
    tool = policy.get("python_package_install_tool")
    if isinstance(tool, str) and tool.strip():
        lines.append(f"- python_package_install_tool={tool.strip()}")
    scope = policy.get("install_scope")
    if isinstance(scope, str) and scope.strip():
        lines.append(f"- install_scope={scope.strip()}")
    return "\n".join(lines) if lines else "- none"


def _build_prompt(question: str, capabilities: dict) -> str:
    return (
        "You are the routing planner for an operations assistant.\n"
        "Task: decide whether the request is processable by at least one discovered agent capability and, when processable, "
        "produce the best execution plan and presentation intent.\n"
        "Output JSON only with exact keys: "
        '{"processable":true|false,"reason":"short reason","task_shape":"count|boolean_check|save_artifact|schema_summary|compare|list|summarize_dataset|lookup|command_execution","steps":[{"id":"step1","target_agent":"agent_name","task":"step task","instruction":{"operation":"agent_specific_operation"},"depends_on":["optional_step_id"],"when":{"$from":"optional.path","equals":"optional value"},"steps":[{"id":"step1_1","task":"optional nested substep"}]}],"presentation":{"task":"how to present the result","format":"markdown|markdown_table|json|bullets|plain","audience":"openwebui","include_context":true|false,"include_internal_steps":true|false}}\n'
        "No markdown, no extra keys, no prose outside JSON.\n"
        "Decision policy:\n"
        "1) processable=true if any discovered agent can attempt the request.\n"
        "2) For processable=true, steps MUST be a non-empty ordered list.\n"
        "2a) Produce one strong primary workflow in steps. Do not spend output budget inventing fallback workflows up front.\n"
        "2b) task_shape MUST describe the user goal at a high level using one of the allowed values.\n"
        "3) Each step MUST have id and task. Leaf steps MUST have target_agent. Group steps may omit target_agent when they contain nested steps.\n"
        "4) Every leaf step MUST include an instruction object with an agent-native operation and inputs.\n"
        "5) For shell_runner use instruction.operation=run_command with fields such as command, input, and capture.\n"
        "5a) For shell_runner tasks that need structured parsing, aggregation, generation, or file creation from prior results, prefer a single python3 heredoc snippet over brittle shell pipelines.\n"
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
        "18a) If execution requires a missing Python dependency and runtime policy allows it, shell_runner may install the required Python package into the current Python environment to complete the task.\n"
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
        "Runtime execution policy:\n"
        f"{_format_execution_policy(capabilities)}\n"
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
        "8a) If runtime policy allows Python package installation, you may replan through shell_runner steps that install a missing Python package when that is necessary to complete the task.\n"
        "9) Tolerate minor typos and informal phrasing.\n"
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        "Runtime execution policy:\n"
        f"{_format_execution_policy(capabilities)}\n"
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
    plan_options = raw.get("plan_options", raw.get("options"))
    presentation = raw.get("presentation")
    task_shape = raw.get("task_shape")
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
        "plan_options": _parse_plan_options(plan_options),
        "task_shape": task_shape.strip().lower() if isinstance(task_shape, str) and task_shape.strip() else None,
        "presentation": _parse_presentation(presentation),
    }


def _parse_plan_options(raw: Any):
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    parsed = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            return None
        steps = item.get("steps", [])
        if not isinstance(steps, list):
            return None
        normalized_steps = _parse_steps(steps)
        if normalized_steps is None or not normalized_steps:
            return None
        option_id = item.get("id") or f"option{index}"
        label = item.get("label") or f"Option {index}"
        reason = item.get("reason") or ""
        if not isinstance(option_id, str) or not option_id.strip():
            return None
        if not isinstance(label, str) or not label.strip():
            label = f"Option {index}"
        if not isinstance(reason, str):
            reason = ""
        parsed.append(
            {
                "id": option_id.strip(),
                "label": label.strip(),
                "reason": reason.strip(),
                "steps": normalized_steps,
            }
        )
    return parsed


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


TASK_SHAPES = {
    "count",
    "boolean_check",
    "save_artifact",
    "schema_summary",
    "compare",
    "list",
    "summarize_dataset",
    "lookup",
    "command_execution",
}


def _normalize_task_shape(question: str, raw: Any = None, *, target_agent: str | None = None, presentation: dict | None = None) -> str:
    inferred = _infer_task_shape(question, target_agent=target_agent, presentation=presentation)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in TASK_SHAPES:
            if normalized in {"count", "boolean_check"} and inferred != normalized:
                return inferred
            if normalized == "count" and _looks_like_mixed_count_and_detail_request(question):
                return inferred
            if inferred == "save_artifact" and normalized != "save_artifact":
                return inferred
            return normalized
    return inferred


def _infer_task_shape(question: str, *, target_agent: str | None = None, presentation: dict | None = None) -> str:
    text = str(question or "").strip().lower()
    family = _agent_family(target_agent or "") if isinstance(target_agent, str) else ""
    fmt = presentation.get("format") if isinstance(presentation, dict) else None
    compound_parts = _split_compound_request(text)
    count_part_count = sum(1 for part in compound_parts if any(marker in part for marker in COUNT_REQUEST_MARKERS))
    boolean_part_count = sum(
        1
        for part in compound_parts
        if (
            any(token in part for token in ("whether", "check if", "check whether", "exists", "exist", "is there", "are there"))
            or re.match(r"^(?:is|are|does|do|did|can|has|have|was|were)\b", part)
        )
    )

    if _extract_save_output_path(text) or (
        any(token in text for token in ("save", "write", "export", "store", "create"))
        and any(token in text for token in ("file", "path", "location", ".txt", ".csv", ".json", ".md"))
    ):
        return "save_artifact"
    if _planner_is_schema_request(text):
        return "schema_summary"
    if _looks_like_slurm_elapsed_summary_question(text):
        return "lookup"
    if _looks_like_slurm_node_inventory_summary_question(text):
        return "lookup"
    if _looks_like_mixed_count_and_detail_request(text):
        return "list" if fmt == "markdown_table" or family == "sql_runner" else "lookup"
    if len(compound_parts) > 1 and (count_part_count > 1 or (count_part_count and boolean_part_count)):
        return "lookup"
    if len(compound_parts) > 1 and boolean_part_count and boolean_part_count < len(compound_parts):
        return "lookup"
    if any(token in text for token in ("how many", "count ", "number of", "total number", "total count")):
        return "count"
    if (
        " and whether " not in text
        and " and check whether " not in text
        and any(token in text for token in ("whether", "check if", "check whether", "exists", "exist", "is there", "are there"))
    ):
        return "boolean_check"
    if " and " not in text and re.match(r"^(?:is|are|does|do|did|can|has|have|was|were)\b", text):
        return "boolean_check"
    if any(token in text for token in ("compare", "difference", "different", "versus", " vs ")):
        return "compare"
    if any(token in text for token in ("summarize", "summary", "overview", "describe the dataset", "summarise")):
        return "summarize_dataset"
    if fmt == "markdown_table" or any(token in text for token in ("list", "rows", "all ", "show me all", "table of", "tables", "columns")):
        return "list"
    if family in {"shell_runner", "filesystem", "notifier"} and re.match(
        r"^(?:run|execute|open|read|commit|push|checkout|create|delete|rename|move|copy|install)\b",
        text,
    ):
        return "command_execution"
    return "lookup"


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
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")

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
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")

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
        for event in _agent_trigger_events(agent):
            if event in SUPPORTED_EVENT_SCHEMAS and event != "user.ask":
                available.add(event)
        for event in _agent_emitted_events(agent):
            if event in SUPPORTED_EVENT_SCHEMAS:
                available.add(event)
    return sorted(available or DEFAULT_ALLOWED_EVENTS)


def _agent_names_with_task_plan(capabilities: dict) -> set[str]:
    names = set()
    for agent in capabilities.get("agents", []):
        name = agent.get("name")
        if isinstance(name, str) and _agent_handles_trigger(agent, "task.plan"):
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
    for key in ("routing_hints", "domain_hints", "schema_hints", "entity_hints"):
        value = agent.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.append(" ".join(entry for entry in value if isinstance(entry, str)))
    for method in _agent_api_specs(agent):
        parts.extend(str(method.get(key, "")) for key in ("name", "event", "trigger_event", "when", "summary"))
        parts.extend(_agent_emitted_events({"apis": [method]}))
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


def _first_agent_in_family(capabilities: dict, family: str) -> str | None:
    for agent in capabilities.get("agents", []):
        if not isinstance(agent, dict):
            continue
        name = agent.get("name")
        if (
            isinstance(name, str)
            and _agent_family(name) == family
            and _agent_handles_trigger(agent, "task.plan")
        ):
            return name
    return None


def _has_agent_family(capabilities: dict, family: str) -> bool:
    return any(
        isinstance(agent, dict)
        and isinstance(agent.get("name"), str)
        and _agent_family(agent["name"]) == family
        and _agent_handles_trigger(agent, "task.plan")
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


def _agent_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _sql_agent_hint_tokens(agent: dict) -> set[str]:
    tokens: set[str] = set()
    for key in ("routing_hints", "domain_hints", "schema_hints", "entity_hints"):
        value = agent.get(key)
        if isinstance(value, str):
            tokens.update(_tokenize(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    tokens.update(_tokenize(item))
    return tokens


def _sql_agent_priority(agent: dict) -> int:
    return _agent_int(agent.get("routing_priority"), 0)


def _looks_like_sql_question(question: str, capabilities: dict) -> bool:
    tokens = _tokenize(question)
    text = str(question or "").strip().lower()
    if _looks_like_repo_file_scan(question):
        strong_sql_tokens = {
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
        data_entity_tokens = {
            "patient",
            "patients",
            "study",
            "studies",
            "user",
            "users",
            "customer",
            "customers",
            "order",
            "orders",
            "record",
            "records",
            "row",
            "rows",
            "detail",
            "details",
        }
        if not (tokens & strong_sql_tokens) and not (tokens & data_entity_tokens) and not (tokens & _configured_database_tokens(capabilities)):
            return False
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
    if _has_agent_family(capabilities, "sql_runner"):
        entity_tokens = {
            "patient",
            "patients",
            "study",
            "studies",
            "user",
            "users",
            "customer",
            "customers",
            "order",
            "orders",
            "record",
            "records",
            "row",
            "rows",
            "detail",
            "details",
        }
        filter_tokens = {
            "count",
            "counts",
            "list",
            "show",
            "having",
            "where",
            "group",
            "filter",
            "greater",
            "less",
            "over",
            "under",
        }
        if (tokens & entity_tokens) and (
            (tokens & filter_tokens)
            or "more than" in text
            or "less than" in text
            or "all their details" in text
        ):
            return True
    return False


def _looks_like_workspace_file_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    tokens = _tokenize(question)
    pathlike = bool(re.search(r"\.(?:log|txt|md|py|sh|json|yaml|yml|csv|ini|cfg)\b", text))
    file_tokens = {
        "file",
        "files",
        "path",
        "paths",
        "directory",
        "directories",
        "folder",
        "folders",
        "repo",
        "repository",
        "workspace",
        "log",
        "logs",
    }
    action_tokens = {
        "find",
        "tail",
        "head",
        "grep",
        "search",
        "show",
        "open",
        "read",
        "newest",
        "latest",
        "current",
    }
    return bool(pathlike or (tokens & file_tokens)) and bool(
        (tokens & action_tokens) or "current directory" in text or "this repo" in text
    )


def _looks_like_repo_file_scan(question: str) -> bool:
    text = str(question or "").strip().lower()
    tokens = _tokenize(question)
    repo_scope_tokens = {"repo", "repository", "workspace", "directory", "directories", "folder", "folders"}
    file_scope_tokens = {"file", "files", "path", "paths", "log", "logs"}
    file_action_tokens = {"find", "list", "show", "search", "open", "read"}
    return bool(tokens & file_scope_tokens) and (
        "this repo" in text
        or "current directory" in text
        or bool(tokens & repo_scope_tokens)
    ) and bool(tokens & file_action_tokens)


def _looks_like_notification_request(question: str) -> bool:
    text = str(question or "").strip().lower()
    return bool(
        re.search(r"\b(notify|notification|alert|remind|reminder)\b", text)
        or re.search(r"\b(send|message|ping|tell)\s+(?:me|us|team|channel)\b", text)
    )


UNSAFE_MACHINE_REQUEST_PATTERNS = [
    r"\bshutdown\b",
    r"\breboot\b",
    r"\brm\s+-rf\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bdelete all docker containers\b",
    r"\bremove all docker containers\b",
    r"\bdestroy all docker containers\b",
    r"\bkill all docker containers\b",
    r"\bstop all docker containers\b",
    r"\bdocker\s+(?:rm|rmi|kill|stop|system prune)\b",
]


def _is_unsafe_machine_request(question: str) -> bool:
    text = str(question or "").strip().lower()
    return any(re.search(pattern, text) for pattern in UNSAFE_MACHINE_REQUEST_PATTERNS)


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


def _looks_like_slurm_elapsed_summary_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not any(token in text for token in ("slurm", "sacct", "job", "jobs", "partition", "cluster", "scheduler")):
        return False
    return any(
        marker in text
        for marker in (
            "how long",
            "took to complete",
            "take to complete",
            "total elapsed",
            "elapsed time",
            "duration",
            "average time",
            "avg time",
            "mean time",
            "median time",
            "longest",
            "shortest",
        )
    )


def _looks_like_slurm_node_inventory_summary_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not any(token in text for token in ("slurm", "cluster", "sinfo", "scheduler")):
        return False
    has_nodes = any(token in text for token in ("node", "nodes", "nodelist", "compute node", "compute nodes"))
    asks_for_count = any(token in text for token in ("how many", "count", "number of", "total nodes", "total number"))
    asks_for_state = any(token in text for token in ("state", "states", "status", "statuses"))
    return has_nodes and asks_for_count and asks_for_state


def _looks_like_local_hardware_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    tokens = _tokenize(question)
    if "nvidia-smi" in text:
        return True
    local_scope_tokens = {
        "machine",
        "system",
        "host",
        "local",
        "installed",
        "hardware",
        "driver",
        "drivers",
        "cuda",
        "memory",
        "vram",
        "pcie",
        "pci",
        "spec",
        "specs",
        "detail",
        "details",
    }
    gpu_tokens = {
        "gpu",
        "gpus",
        "nvidia",
        "geforce",
        "rtx",
        "tesla",
        "quadro",
    }
    slurm_scope_tokens = {
        "slurm",
        "cluster",
        "clusters",
        "partition",
        "partitions",
        "node",
        "nodes",
        "job",
        "jobs",
        "queue",
        "queued",
        "reservation",
        "reservations",
        "sinfo",
        "squeue",
        "hpc",
    }
    if tokens & slurm_scope_tokens:
        return False
    if "this machine" in text or "this system" in text or "on this machine" in text or "on this system" in text:
        return True
    return bool(tokens & gpu_tokens) and bool(tokens & local_scope_tokens)


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
        if not _agent_handles_trigger(agent, "task.plan"):
            continue
        candidates.append(agent)

    if not candidates:
        return None

    if _looks_like_slurm_elapsed_summary_question(question) or _looks_like_slurm_node_inventory_summary_question(question):
        native_slurm = _first_agent_in_family(capabilities, "slurm_runner")
        if native_slurm:
            return native_slurm

    priority_boosts = {
        "shell_runner": 0,
        "sql_runner": 0,
        "slurm_runner": 0,
        "notifier": 0,
    }
    looks_like_sql = _looks_like_sql_question(question, capabilities)
    looks_like_sql_export = _looks_like_sql_export_request(question, capabilities)

    if any(token in question_tokens for token in {"docker", "container", "containers", "image", "images", "installed", "available", "git", "grep", "rg", "find", "list", "logs", "restart", "stop", "start", "ps", "ports", "tail", "head", "line", "lines"}):
        priority_boosts["shell_runner"] += 10
    if _looks_like_workspace_file_question(question) and not looks_like_sql:
        priority_boosts["shell_runner"] += 15
    if _looks_like_repo_file_scan(question):
        priority_boosts["shell_runner"] += 20
    if _looks_like_local_hardware_question(question):
        priority_boosts["shell_runner"] += 24
    if looks_like_sql:
        priority_boosts["sql_runner"] += 12
    if looks_like_sql_export:
        priority_boosts["sql_runner"] += 18
    if _looks_like_slurm_question(question):
        priority_boosts["slurm_runner"] += 12
    if re.search(r"\b(add|subtract|multiply|divide|calculate|compute|sum)\b", question_lc) or len(re.findall(r"\b\d+(?:\.\d+)?\b", question_lc)) >= 2:
        priority_boosts["shell_runner"] += 10
    if _looks_like_notification_request(question):
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


def _select_sql_target_agent(question: str, capabilities: dict) -> str | None:
    sql_agents = [
        agent
        for agent in capabilities.get("agents", [])
        if isinstance(agent, dict)
        and isinstance(agent.get("name"), str)
        and _agent_family(agent["name"]) == "sql_runner"
        and _agent_handles_trigger(agent, "task.plan")
    ]
    if not sql_agents:
        return None
    if len(sql_agents) == 1:
        return str(sql_agents[0]["name"])

    question_tokens = _tokenize(question)
    question_lc = str(question or "").strip().lower()
    best_name = None
    best_score = float("-inf")
    for agent in sql_agents:
        name = str(agent["name"])
        alias_tokens = set()
        for key in ("database_name", "argument_name"):
            value = agent.get(key)
            if isinstance(value, str):
                alias_tokens.update(_tokenize(value))
        aliases = agent.get("database_aliases")
        if isinstance(aliases, list):
            for item in aliases:
                if isinstance(item, str):
                    alias_tokens.update(_tokenize(item))
        hint_tokens = _sql_agent_hint_tokens(agent)
        score = len(question_tokens & alias_tokens) * 25
        score += len(question_tokens & hint_tokens) * 12
        score += len(question_tokens & _tokenize(_agent_search_blob(agent)))
        score += _sql_agent_priority(agent)
        database_name = str(agent.get("database_name") or "").strip().lower()
        if database_name and database_name in question_lc:
            score += 100
        if database_name == "dicom_mock" and question_tokens & {"mrn", "dicom", "rtplan", "rtstruct", "rtdose", "series", "instances"}:
            score += 20
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def _derive_shell_command(task: str, step_index: int = 1) -> str | None:
    task_lc = task.lower().strip()
    save_path = _extract_save_output_path(task)

    if "current git branch" in task_lc and "working tree" in task_lc and "clean" in task_lc:
        return 'printf "Branch: %s\\nWorking tree clean: %s\\n" "$(git branch --show-current)" "$(git diff --quiet && echo true || echo false)"'

    if ("newest log file" in task_lc or "latest log file" in task_lc) and (
        ("last" in task_lc or "tail" in task_lc) and ("line" in task_lc or "lines" in task_lc)
    ):
        tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
        tail = tail_match.group(1) if tail_match else "20"
        return (
            'file=$(find . -type f -name "*.log" -printf "%T@ %p\\n" 2>/dev/null | sort -nr | head -n 1 | cut -d" " -f2-)'
            f' && if [ -n "$file" ]; then printf "%s\\n" "$file"; tail -n {tail} "$file";'
            ' else printf "%s\\n" "No matching log file found."; fi'
        )

    create_copy_match = re.search(
        r"file named\s+([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)\s+with\s+(.+?)\s*,\s*copy it to\s+([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)",
        task,
        flags=re.IGNORECASE,
    )
    if create_copy_match and any(token in task_lc for token in ("path", "paths", "location")):
        source_path = create_copy_match.group(1)
        file_content = create_copy_match.group(2).strip().strip("\"'")
        target_path = create_copy_match.group(3)
        return (
            f"printf \"%s\\n\" {json.dumps(file_content)} > {shlex.quote(source_path)}"
            f" && cp {shlex.quote(source_path)} {shlex.quote(target_path)}"
            f' && printf "%s\\n%s\\n" "$(realpath {shlex.quote(source_path)})" "$(realpath {shlex.quote(target_path)})"'
        )

    if save_path and any(token in task_lc for token in {"list", "save", "write", "create"}):
        return _build_save_rows_command(save_path)

    if "{{prev}}" in task_lc:
        if "log" in task_lc:
            tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
            tail = tail_match.group(1) if tail_match else "50"
            return f"docker logs --tail {tail} {{{{prev}}}}"
        if "tail" in task_lc or "line" in task_lc or "lines" in task_lc:
            tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
            tail = tail_match.group(1) if tail_match else "20"
            return f'if [ -n "{{{{prev}}}}" ]; then tail -n {tail} "{{{{prev}}}}"; else printf "%s\\n" "No matching log file found."; fi'
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
        if "count" in task_lc and ("container" in task_lc or "containers" in task_lc):
            return "docker ps -aq | wc -l"
        if "count" in task_lc and ("image" in task_lc or "images" in task_lc):
            return "docker image ls -q | wc -l"
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
        return 'find . -type f -name "*.log" -printf "%T@ %p\\n" 2>/dev/null | sort -nr | head -n 1 | cut -d" " -f2-'

    if "show listening ports" in task_lc or "listening ports" in task_lc:
        return "ss -ltnp"

    if "nvidia-smi" in task_lc and any(token in task_lc for token in ("installed", "available", "exists", "check if", "check whether", "whether")):
        return "command -v nvidia-smi"

    if (
        "nvidia-smi" in task_lc
        or (
            any(token in task_lc for token in ("gpu", "gpus", "nvidia"))
            and any(token in task_lc for token in ("spec", "specs", "detail", "details", "driver", "drivers", "cuda", "memory", "vram"))
        )
    ):
        return "nvidia-smi -q"

    if "top-level directories" in task_lc or "top level directories" in task_lc:
        return 'find . -mindepth 1 -maxdepth 1 -type d -printf "%f\\n" | sort'

    if "current branch" in task_lc:
        return "git branch --show-current"

    if "working tree clean" in task_lc:
        return 'if git diff --quiet && git diff --cached --quiet; then echo true; else echo false; fi'

    if "last commit message" in task_lc:
        return "git log --format=%s -n 1"

    if "last commit" in task_lc and "message" not in task_lc:
        return "git log --oneline -n 1"

    if "git status" in task_lc or (task_lc == "status" and step_index == 1):
        return "git status --short"

    if "git log" in task_lc or "recent commits" in task_lc:
        return "git log --oneline -n 10"

    if "docker images" in task_lc or (task_lc == "images"):
        return "docker images"

    if ("docker" in task_lc) and any(token in task_lc for token in {"installed", "available"}):
        return 'if command -v docker >/dev/null 2>&1; then echo true; else echo false; fi'

    if "docker compose" in task_lc and "ps" in task_lc:
        return "docker compose ps"

    if "database files" in task_lc or ("database" in task_lc and "files" in task_lc):
        return 'find . -type f \\( -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" -o -name "*.sql" \\) | sort'

    if "count" in task_lc and "python" in task_lc and "file" in task_lc:
        if "repository root" in task_lc:
            return 'find . -maxdepth 1 -type f -name "*.py" | wc -l'
        code = "\n".join(
            [
                "import pathlib",
                "count = sum(1 for path in pathlib.Path('.').rglob('*.py') if path.is_file())",
                "print(count)",
            ]
        )
        return "python3 - <<'PY'\n" + code + "\nPY"

    if (
        any(token in task_lc for token in ("list", "show"))
        and "python" in task_lc
        and "file" in task_lc
        and any(token in task_lc for token in ("alphabetical", "alphabetically", "sorted"))
        and any(token in task_lc for token in ("first", "top"))
    ):
        limit_match = re.search(r"\b(?:first|top)\s+(\d+)\b", task_lc)
        limit = limit_match.group(1) if limit_match else "5"
        if "repository root" in task_lc:
            return f'find . -maxdepth 1 -type f -name "*.py" -printf "%f\\n" | sort | head -n {limit}'
        return f'find . -type f -name "*.py" | sort | head -n {limit}'

    if "largest" in task_lc and "python" in task_lc and "file" in task_lc:
        limit_match = re.search(r"\b(?:top|largest)\s+(\d+)\b", task_lc)
        limit = limit_match.group(1) if limit_match else "3"
        code = "\n".join(
            [
                "import pathlib",
                f"limit = int({limit})",
                "rows = []",
                "for path in pathlib.Path('.').rglob('*.py'):",
                "    if path.is_file():",
                "        rows.append((path.stat().st_size, str(path)))",
                "for size, path in sorted(rows, reverse=True)[:limit]:",
                "    print(f\"{size}\\t{path}\")",
            ]
        )
        return "python3 - <<'PY'\n" + code + "\nPY"

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


EXPORT_REQUEST_MARKERS = (
    "save",
    "write",
    "create",
    "export",
    "download",
)


EXPORT_TARGET_MARKERS = (
    "file",
    "report",
    "output",
    "artifact",
)


EXPORT_LOCATION_MARKERS = (
    "location",
    "path",
    "where",
    "review",
)


def _slugify_export_task(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    if not slug:
        return "query_results"
    return slug[:80].rstrip("_")


def _looks_like_sql_export_request(question: str, capabilities: dict) -> bool:
    if not _looks_like_sql_question(question, capabilities):
        return False
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    has_export_verb = any(marker in text for marker in EXPORT_REQUEST_MARKERS)
    has_target = any(marker in text for marker in EXPORT_TARGET_MARKERS)
    has_location = any(marker in text for marker in EXPORT_LOCATION_MARKERS)
    return bool(_extract_save_output_path(text)) or (has_export_verb and has_target) or (has_export_verb and has_location)


def _default_export_output_path(question: str) -> str:
    base_question = re.split(
        r"\b(?:and\s+(?:save|write|create|export|download)|then\s+(?:save|write|create|export|download))\b",
        str(question or ""),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    slug = _slugify_export_task(base_question or question)
    return f"artifacts/exports/{slug}.json"


def _resolve_save_output_path(question: str, capabilities: dict) -> str | None:
    explicit = _extract_save_output_path(question)
    if explicit:
        return explicit
    if _looks_like_sql_export_request(question, capabilities):
        return _default_export_output_path(question)
    return None


def _build_save_rows_command(path: str) -> str:
    script = "\n".join(
        [
            "import csv",
            "import json",
            "import pathlib",
            "import sys",
            "",
            "input_path = pathlib.Path(sys.argv[1])",
            "raw = input_path.read_text(encoding='utf-8').strip()",
            "rows = json.loads(raw) if raw else []",
            f"path = pathlib.Path({json.dumps(path)})",
            "path.parent.mkdir(parents=True, exist_ok=True)",
            "suffix = path.suffix.lower()",
            "normalized = rows if isinstance(rows, list) else [rows]",
            "normalized = [row if isinstance(row, dict) else {'value': row} for row in normalized]",
            "if suffix == '.csv':",
            "    fieldnames = []",
            "    for row in normalized:",
            "        for key in row.keys():",
            "            if key not in fieldnames:",
            "                fieldnames.append(key)",
            "    with path.open('w', newline='', encoding='utf-8') as handle:",
            "        writer = csv.DictWriter(handle, fieldnames=fieldnames or ['value'])",
            "        writer.writeheader()",
            "        writer.writerows(normalized)",
            "else:",
            "    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True, default=str), encoding='utf-8')",
            "print(path.resolve())",
        ]
    )
    return (
        "tmp_json=$(mktemp) && "
        "cat > \"$tmp_json\" && "
        "python3 - \"$tmp_json\" <<'PY'\n"
        f"{script}\n"
        "PY\n"
        "status=$?; rm -f \"$tmp_json\"; exit $status"
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


SHELL_COMMAND_OVERRIDE_MARKERS = (
    "top-level directories",
    "top level directories",
    "database files",
    "docker is installed",
    "docker installed",
    "docker available",
    "newest log file",
    "latest log file",
    "current branch",
    "working tree",
    "file path",
    "file paths",
    "location of the file",
    "free space",
    "usable free space",
    "physical drives",
    "non physical drives",
    "nvidia-smi",
    "gpu specs",
    "gpu spec",
    "gpu specifications",
)


def _should_override_shell_command(task_lc: str) -> bool:
    return any(marker in task_lc for marker in SHELL_COMMAND_OVERRIDE_MARKERS)


def _looks_like_storage_free_space_request(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    has_space_intent = ("free space" in text) or ("disk space" in text) or ("storage" in text and "free" in text)
    has_drive_scope = any(token in text for token in ("drive", "drives", "disk", "disks", "mount", "mounts", "physical drives", "physical disks"))
    wants_total = any(token in text for token in ("total", "across all", "all drives", "usable free space", "how much"))
    return has_space_intent and has_drive_scope and wants_total


def _build_physical_mount_discovery_command() -> str:
    code = """
import json
import sys

data = json.load(sys.stdin)
mounts = []

def walk(devices, physical_root=None):
    for dev in devices or []:
        dtype = str(dev.get("type") or "")
        name = str(dev.get("name") or "")
        current = name if dtype == "disk" else physical_root
        if current:
            for mount in dev.get("mountpoints") or []:
                if isinstance(mount, str):
                    mount = mount.strip()
                    if mount and mount != "[SWAP]":
                        mounts.append(mount)
        walk(dev.get("children") or [], current)

walk(data.get("blockdevices") or [])
print(json.dumps(sorted(set(mounts))))
""".strip()
    return f"lsblk -J -o NAME,TYPE,MOUNTPOINTS | python3 -c {shlex.quote(code)}"


def _build_total_free_space_command_gb() -> str:
    code = """
import json
import os
import sys

mounts = json.load(sys.stdin)
total = 0
seen = set()

for mount in mounts:
    if not isinstance(mount, str):
        continue
    mount = mount.strip()
    if not mount or mount in seen:
        continue
    seen.add(mount)
    try:
        stat = os.statvfs(mount)
    except OSError:
        continue
    total += stat.f_frsize * stat.f_bavail

print(f"{total / (1024 ** 3):.2f}")
""".strip()
    return f"python3 -c {shlex.quote(code)}"


def _build_dependency_results_difference_command() -> str:
    code = """
import json
import re
import sys

payload = json.load(sys.stdin)
items = payload.get("dependency_results") if isinstance(payload, dict) else []
numbers = []

for item in items:
    if not isinstance(item, dict):
        continue
    candidates = [item.get("value"), item.get("result"), item.get("summary")]
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        candidates.append(evidence.get("summary_text"))
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text or "\\n" in text:
            continue
        match = re.fullmatch(r"-?\\d+(?:\\.\\d+)?", text)
        if not match:
            match = re.fullmatch(r"[^:]+:\\s*(-?\\d+(?:\\.\\d+)?)", text)
        if not match:
            continue
        value = float(match.group(1) if match.lastindex else match.group(0))
        numbers.append(value)
        break

if len(numbers) < 2:
    raise SystemExit("Unable to compute the difference from dependency_results.")

diff = abs(numbers[0] - numbers[1])
print(int(diff) if diff.is_integer() else diff)
""".strip()
    return f"python3 -c {shlex.quote(code)}"


def _looks_like_root_python_inventory_request(question: str) -> bool:
    text = str(question or "").strip().lower()
    return (
        "repository root" in text
        and "python" in text
        and "file" in text
        and any(token in text for token in ("list", "show"))
        and any(token in text for token in ("alphabetical", "alphabetically", "sorted"))
        and any(token in text for token in ("first", "top"))
        and any(token in text for token in ("count", "total count", "total number", "how many"))
    )


def _root_python_inventory_fallback_steps() -> list[dict]:
    return [
        {
            "id": "step1",
            "target_agent": "shell_runner",
            "task": "list the first five Python files alphabetically in the repository root",
            "instruction": {
                "operation": "run_command",
                "command": 'find . -maxdepth 1 -type f -name "*.py" -printf "%f\\n" | sort | head -n 5',
                "capture": {"mode": "stdout_stripped"},
            },
        },
        {
            "id": "step2",
            "target_agent": "shell_runner",
            "task": "count Python files in the repository root",
            "instruction": {
                "operation": "run_command",
                "command": 'find . -maxdepth 1 -type f -name "*.py" | wc -l',
                "capture": {"mode": "stdout_stripped"},
            },
        },
    ]


def _storage_fallback_steps(question: str, capabilities: dict) -> list[dict]:
    if not _looks_like_storage_free_space_request(question):
        return []
    return [
        {
            "id": "step1",
            "target_agent": "shell_runner",
            "task": "list mounted filesystems backed by physical drives",
            "instruction": {
                "operation": "run_command",
                "command": _build_physical_mount_discovery_command(),
                "capture": {"mode": "stdout_stripped"},
            },
        },
        {
            "id": "step2",
            "target_agent": "shell_runner",
            "task": "total usable free space across physical drives (GB)",
            "instruction": {
                "operation": "run_command",
                "command": _build_total_free_space_command_gb(),
                "input": {"$from": "step1.value"},
                "capture": {"mode": "stdout_stripped"},
            },
            "depends_on": ["step1"],
        },
    ]


def _derive_shell_instruction(task: str, index: int) -> dict | None:
    command = _derive_shell_command(task, index)
    if not isinstance(command, str) or not command.strip():
        return None
    instruction: dict[str, Any] = {"operation": "run_command", "command": command.strip()}
    derived_mode = _derive_shell_result_mode(task, instruction["command"])
    if derived_mode:
        instruction["capture"] = {"mode": derived_mode}
    env_name = _extract_conda_env_name(task, instruction.get("command"))
    if env_name and _is_conda_env_existence_task(task, instruction.get("command")):
        instruction["command"] = _build_conda_env_exists_command(env_name)
        instruction["capture"] = {"mode": "json"}
        instruction["allow_returncodes"] = [0, 1]
    return instruction


def _normalize_shell_instruction(task: str, instruction: dict | None, command: str | None, index: int):
    task_lc = task.lower()
    derived_instruction = _derive_shell_instruction(task, index)
    shell_instruction = instruction if isinstance(instruction, dict) and instruction.get("operation") == "run_command" else None

    if (
        shell_instruction is not None
        and isinstance(shell_instruction.get("command"), str)
        and derived_instruction
        and _should_override_shell_command(task_lc)
    ):
        shell_instruction = {
            **shell_instruction,
            "command": derived_instruction["command"],
        }
        if "capture" in derived_instruction and not isinstance(shell_instruction.get("capture"), dict):
            shell_instruction["capture"] = derived_instruction["capture"]
        if "allow_returncodes" in derived_instruction:
            shell_instruction["allow_returncodes"] = derived_instruction["allow_returncodes"]

    if shell_instruction is None:
        if isinstance(command, str) and command.strip():
            stripped = command.strip()
            if derived_instruction and _should_override_shell_command(task_lc):
                stripped = str(derived_instruction["command"])
            if _looks_like_metadata_command(stripped):
                return None
            shell_instruction = {"operation": "run_command", "command": stripped}
        else:
            shell_instruction = dict(derived_instruction) if isinstance(derived_instruction, dict) else None

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


def _derive_presentation(question: str) -> dict:
    question_lc = question.lower()
    presentation = {
        "task": "Answer the user request directly using clean Markdown.",
        "format": "markdown",
        "audience": "openwebui",
        "include_context": True,
        "include_internal_steps": False,
    }
    docker_listing_request = (
        "docker" in question_lc
        and any(token in question_lc for token in ("container", "containers"))
        and any(token in question_lc for token in ("list", "show"))
        and "table" not in question_lc
    )
    if _looks_like_slurm_elapsed_summary_question(question_lc):
        presentation["format"] = "markdown"
        presentation["task"] = "Report the computed Slurm timing summary clearly in Markdown."
    elif _looks_like_slurm_node_inventory_summary_question(question_lc):
        presentation["format"] = "markdown"
        presentation["task"] = "Report the total Slurm node count and state summary clearly in Markdown."
    elif docker_listing_request:
        presentation["format"] = "markdown_table"
        presentation["task"] = "Show the Docker results in a clean Markdown table."
    elif any(token in question_lc for token in ("table", "tabulate", "columns", "rows", "compare", "comparison", "status", "list")):
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

    if any(
        token in question_lc
        for token in (
            "show command",
            "commands used",
            "how did",
            "debug",
            "workflow",
            "steps",
            "stage",
            "stages",
            "raw output",
            "raw outputs",
            "stdout",
            "stderr",
            "log output",
            "logs",
        )
    ):
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
        return _normalize_shell_instruction(task, instruction, command, index)

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


FOLLOWUP_PLACEHOLDER_RE = re.compile(
    r"\{(file_path|filepath|path|file|filename|value|result|prev|previous|container_id|container_name|branch)\}",
    flags=re.IGNORECASE,
)


def _normalize_followup_shell_instruction(instruction: dict, prior_step_id: str | None = None) -> dict:
    if not isinstance(instruction, dict):
        return instruction
    normalized = dict(instruction)
    command = normalized.get("command")
    if isinstance(command, str) and command.strip():
        rewritten = FOLLOWUP_PLACEHOLDER_RE.sub("{{prev}}", command)
        if rewritten != command:
            normalized["command"] = rewritten
    return normalized


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
            "trigger_events": sorted(_agent_trigger_events(agent)),
            "emits": sorted(_agent_emitted_events(agent)),
            "request_contract": agent.get("request_contract", ""),
            "result_contract": agent.get("result_contract", ""),
            "apis": [
                {
                    "name": method.get("name", ""),
                    "trigger_event": api_trigger_event(method),
                    "emits": api_emit_events(method),
                    "request_contract": method.get("request_contract", ""),
                    "result_contract": method.get("result_contract", ""),
                }
                for method in _agent_api_specs(agent)
                if isinstance(method, dict)
            ],
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
        trigger_events = agent.get("trigger_events", [])
        emitted_events = agent.get("emits", [])
        domain_text = ", ".join(item for item in domains if isinstance(item, str)) if isinstance(domains, list) else ""
        action_text = ", ".join(item for item in actions if isinstance(item, str)) if isinstance(actions, list) else ""
        trigger_text = ", ".join(item for item in trigger_events if isinstance(item, str)) if isinstance(trigger_events, list) else ""
        emit_text = ", ".join(item for item in emitted_events if isinstance(item, str)) if isinstance(emitted_events, list) else ""
        detail = description
        if domain_text:
            detail += f" Domains: {domain_text}."
        if action_text:
            detail += f" Actions: {action_text}."
        if trigger_text:
            detail += f" Triggered by: {trigger_text}."
        if emit_text:
            detail += f" Emits: {emit_text}."
        lines.append(f"- {name}: {detail}")
    return "\n".join(lines)


def _fallback_steps(question: str, capabilities: dict):
    if _looks_like_slurm_elapsed_summary_question(question) or _looks_like_slurm_node_inventory_summary_question(question):
        target_agent = _first_agent_in_family(capabilities, "slurm_runner")
        if target_agent:
            return [
                {
                    "id": "step1",
                    "target_agent": target_agent,
                    "task": _sanitize_task_text(question) or question.strip(),
                    "instruction": {
                        "operation": "query_from_request",
                        "question": question.strip(),
                    },
                }
            ]
    if _looks_like_root_python_inventory_request(question):
        return _root_python_inventory_fallback_steps()
    mixed_parts = _mixed_count_and_detail_parts(question)
    if mixed_parts is not None and _looks_like_sql_question(question, capabilities):
        target_agent = _select_sql_target_agent(question, capabilities)
        if target_agent and _agent_family(target_agent) == "sql_runner":
            return [
                {
                    "id": "step1",
                    "target_agent": target_agent,
                    "task": "List the qualifying rows with requested details",
                    "instruction": {
                        "operation": "query_from_request",
                        "question": _mixed_count_detail_sql_question(question),
                    },
                },
            ]
    storage_steps = _storage_fallback_steps(question, capabilities)
    if storage_steps:
        return storage_steps
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
    if _looks_like_multi_domain_count_workflow(task_question, capabilities) or _looks_like_multi_domain_count_workflow(
        planning_question,
        capabilities,
    ):
        return []
    if not _looks_like_sql_question(planning_question, capabilities):
        return []
    target_agent = _select_sql_target_agent(planning_question, capabilities)
    if not target_agent or _agent_family(target_agent) != "sql_runner":
        return []
    export_path = _resolve_save_output_path(task_question, capabilities)
    if _planner_is_schema_request(task_question):
        schema_step = {
            "id": "step1",
            "target_agent": target_agent,
            "task": task_question.strip(),
            "instruction": {
                "operation": "inspect_schema",
                "focus": task_question.strip(),
            },
        }
        if export_path and _looks_like_save_list_sql_request(task_question, capabilities):
            return [schema_step, _build_save_rows_step("step1", export_path, 2)]
        return [schema_step]
    if export_path and _looks_like_save_list_sql_request(task_question, capabilities):
        return [
            {
                "id": "step1",
                "target_agent": target_agent,
                "task": "List the qualifying rows with requested details",
                "instruction": {
                    "operation": "query_from_request",
                    "question": _list_query_from_request(task_question),
                },
            },
            _build_save_rows_step("step1", export_path, 2),
        ]
    mixed_parts = _mixed_count_and_detail_parts(task_question)
    if mixed_parts is not None:
        count_question, detail_question = mixed_parts
        return [
            {
                "id": "step1",
                "target_agent": target_agent,
                "task": "Count the qualifying rows",
                "instruction": {
                    "operation": "query_from_request",
                    "question": count_question,
                },
            },
            {
                "id": "step2",
                "target_agent": target_agent,
                "task": "List the qualifying rows with requested details",
                "instruction": {
                    "operation": "query_from_request",
                    "question": detail_question,
                },
                "depends_on": ["step1"],
            },
        ]
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


def _split_compound_request(question: str) -> list[str]:
    text = re.sub(r"\s+", " ", question).strip()
    if not text:
        return []
    separator_re = re.compile(
        r"\s*(?:;|,?\s+and\s+|,?\s+then\s+|,\s+also\s+)(?=(?:what\b|how\b|count\b|list\b|show\b|find\b|which\b|is\b|are\b|does\b|do\b|did\b|can\b|has\b|have\b|current\b|last\b|latest\b|get\b|provide\b|check\b|create\b|copy\b|save\b|write\b|tail\b|open\b|read\b|tell\b|report\b|give\b))",
        flags=re.IGNORECASE,
    )
    parts = [
        _sanitize_task_text(part)
        for part in separator_re.split(text)
        if part and _sanitize_task_text(part)
    ]
    return parts or [text]


def _sanitize_task_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip(" ,;:."))
    if not compact:
        return ""
    compact = re.sub(r"^(?:and|then|also)\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+(?:and|then|also)\s*$", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip(" ,;:.")
    return compact


def _normalized_sql_query_question(task: str, question: str) -> str:
    task_text = _sanitize_task_text(task)
    question_text = _sanitize_task_text(question)
    if not question_text:
        return task_text or question_text
    if not task_text:
        return question_text

    task_lc = task_text.lower()
    question_lc = question_text.lower()
    count_like_task = any(marker in task_lc for marker in ("count ", "how many", "number of", "total count", "total number"))
    if count_like_task and "distinct" in question_lc and "distinct" not in task_lc:
        return task_text
    return question_text


def _normalize_compound_shell_part(part: str, question: str) -> str:
    compact = _sanitize_task_text(part)
    if not compact:
        return compact
    compact_lc = compact.lower()
    question_lc = str(question or "").lower()
    if any(marker in compact_lc for marker in COUNT_REQUEST_MARKERS):
        if not _looks_like_slurm_question(question):
            entity = _strip_count_request_prefix(compact)
            entity_lc = entity.lower()
            if "docker" in question_lc and "docker" not in entity_lc:
                if re.search(r"\bcontainers?\b", entity_lc):
                    entity = f"docker {entity}"
                elif re.search(r"\bimages?\b", entity_lc):
                    entity = f"docker {entity}"
            compact = f"count {entity}".strip()
    elif "docker" in question_lc and "docker" not in compact_lc:
        if re.search(r"\bcontainers?\b", compact_lc):
            compact = re.sub(r"\bcontainers?\b", lambda m: f"docker {m.group(0)}", compact, count=1, flags=re.IGNORECASE)
        elif re.search(r"\bimages?\b", compact_lc):
            compact = re.sub(r"\bimages?\b", lambda m: f"docker {m.group(0)}", compact, count=1, flags=re.IGNORECASE)
    if "git" in question_lc and "git" not in compact_lc:
        if any(token in compact_lc for token in ("branch", "commit", "commits", "working tree", "status", "log")):
            compact = f"git {compact}".strip()
    if "nvidia-smi" in question_lc and "nvidia-smi" not in compact_lc:
        if any(
            token in compact_lc
            for token in ("gpu", "gpus", "nvidia", "cuda", "driver", "drivers", "spec", "specs", "detail", "details", "memory", "vram")
        ):
            compact = f"{compact} using nvidia-smi".strip()
    return compact


def _contextualize_shell_followup_task(task: str, question: str) -> str:
    compact = _sanitize_task_text(task)
    if not compact:
        return compact
    compact_lc = compact.lower()
    question_lc = str(question or "").lower()
    wants_count = any(marker in compact_lc for marker in ("count", "total count", "number of", "how many"))
    if (
        wants_count
        and "python" not in compact_lc
        and "file" not in compact_lc
        and "python" in question_lc
        and "file" in question_lc
    ):
        location = " in the repository root" if "repository root" in question_lc else ""
        return f"count Python files{location}"
    return compact


SLURM_STATE_ALIASES = (
    ("pending", "pending"),
    ("queued", "pending"),
    ("running", "running"),
    ("completed", "completed"),
    ("failed", "failed"),
    ("cancelled", "cancelled"),
    ("suspended", "suspended"),
    ("held", "held"),
)


def _slurm_request_context(question: str) -> dict[str, str]:
    text = str(question or "").strip()
    lowered = text.lower()
    context = {
        "user": "",
        "state": "",
        "partition": "",
        "cluster_scope": "yes" if any(token in lowered for token in ("slurm", "cluster", "scheduler")) else "",
    }
    for pattern in (
        r"\buser\s+([a-zA-Z0-9_.-]+)\b",
        r"\bjobs?\s+for\s+([a-zA-Z0-9_.-]+)\b",
        r"\bfor\s+user\s+([a-zA-Z0-9_.-]+)\b",
        r"\bdoes\s+([a-zA-Z0-9_.-]+)\s+have\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            context["user"] = match.group(1)
            break
    partition_match = re.search(r"\bpartition\s+([a-zA-Z0-9_.-]+)\b", text, flags=re.IGNORECASE)
    if partition_match:
        context["partition"] = partition_match.group(1)
    for token, normalized in SLURM_STATE_ALIASES:
        if token in lowered:
            context["state"] = normalized
            break
    return context


def _contextualize_slurm_followup_task(task: str, question: str) -> str:
    compact = _sanitize_task_text(task)
    if not compact or not _looks_like_slurm_question(question):
        return compact
    lowered = compact.lower()
    if any(token in lowered for token in ("pending", "running", "completed", "failed", "cancelled", "suspended", "held", "queued")):
        return compact
    if "slurm" in lowered or "cluster" in lowered or "scheduler" in lowered:
        return compact
    context = _slurm_request_context(question)
    needs_user_context = bool(context["user"]) and "user" not in lowered and context["user"].lower() not in lowered
    needs_state_context = bool(context["state"]) and context["state"] not in lowered
    if not needs_user_context and not needs_state_context:
        return compact
    if "job id" in lowered or re.search(r"\bids\b", lowered):
        rewritten = "list the"
        if context["state"]:
            rewritten += f" {context['state']}"
        rewritten += " job IDs"
    elif re.search(r"\bjobs?\b", lowered):
        rewritten = "list the"
        if context["state"]:
            rewritten += f" {context['state']}"
        rewritten += " jobs"
    else:
        return compact
    if context["user"]:
        rewritten += f" for user {context['user']}"
    if context["partition"]:
        rewritten += f" in partition {context['partition']}"
    elif context["cluster_scope"]:
        rewritten += " in the Slurm cluster"
    return rewritten


def _looks_like_multi_domain_count_workflow(question: str, capabilities: dict) -> bool:
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    if not text:
        return False
    count_marker_count = sum(text.count(marker) for marker in COUNT_REQUEST_MARKERS)
    if count_marker_count < 2:
        return False
    domain_count = 0
    if _looks_like_sql_question(question, capabilities):
        domain_count += 1
    if _looks_like_slurm_question(question):
        domain_count += 1
    if (
        ("python" in text and "file" in text)
        or _looks_like_workspace_file_question(question)
        or _looks_like_repo_file_scan(question)
    ):
        domain_count += 1
    return domain_count >= 2


def _split_multi_domain_count_request(question: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not text:
        return []
    separator_re = re.compile(
        r"\s*(?:;|,\s*|,\s+and\s+|,\s+then\s+|\.\s+|(?:and|then)\s+)(?=(?:(?:\S+\s+){0,6}(?:what|how|count|list|show|find|report|tell|give|provide)\b))",
        flags=re.IGNORECASE,
    )
    raw_parts = separator_re.split(text)
    parts: list[str] = []
    for raw_part in raw_parts:
        part = _sanitize_task_text(raw_part)
        if not part:
            continue
        part_lc = part.lower()
        if parts and part_lc in {"difference", "the difference"}:
            parts[-1] = f"{parts[-1]} and the difference"
            continue
        parts.append(part)
    return parts or [text]


def _select_multi_domain_count_target_agent(part: str, question: str, capabilities: dict) -> str | None:
    normalized_part = _normalize_compound_shell_part(part, question)
    if _looks_like_slurm_question(normalized_part):
        return _first_agent_in_family(capabilities, "slurm_runner")
    if _looks_like_sql_question(normalized_part, capabilities):
        return _select_sql_target_agent(normalized_part, capabilities)
    normalized_lc = normalized_part.lower()
    if "python" in normalized_lc and "file" in normalized_lc:
        return "shell_runner"
    if _looks_like_workspace_file_question(normalized_part) or _looks_like_repo_file_scan(normalized_part):
        return "shell_runner"
    return _select_target_agent(normalized_part, capabilities) or _select_target_agent(part, capabilities)


def _multi_domain_count_steps(question: str, capabilities: dict):
    if not _looks_like_multi_domain_count_workflow(question, capabilities):
        return []

    parts = _split_multi_domain_count_request(question)
    if len(parts) < 2:
        return []

    steps = []
    executable_step_ids: list[str] = []
    wants_difference = any(token in question.lower() for token in ("difference", "compare", "versus", " vs "))

    for part in parts:
        part_lc = part.lower()
        presentation_clause = bool(
            re.match(r"^(?:report|tell|give|provide|show)\b", part_lc)
            and any(token in part_lc for token in ("answer", "count", "counts", "difference", "summary"))
        )
        if _is_presentation_only_task(part) or presentation_clause:
            continue
        target_agent = _select_multi_domain_count_target_agent(part, question, capabilities)
        if not target_agent or target_agent in {"ops_planner", "synthesizer"}:
            return []

        effective_task = _sanitize_task_text(part)
        if _agent_family(target_agent) == "slurm_runner":
            effective_task = _contextualize_slurm_followup_task(effective_task, question)
        elif target_agent == "shell_runner":
            effective_task = _contextualize_shell_followup_task(effective_task, question)

        instruction = _normalize_agent_instruction(target_agent, effective_task, None, None, len(steps) + 1)
        if not isinstance(instruction, dict) or not instruction:
            return []

        step_id = f"step{len(steps) + 1}"
        step = {
            "id": step_id,
            "target_agent": target_agent,
            "task": effective_task,
            "instruction": instruction,
        }
        steps.append(step)
        executable_step_ids.append(step_id)

    if wants_difference and len(executable_step_ids) == 2:
        steps.append(
            {
                "id": f"step{len(steps) + 1}",
                "target_agent": "shell_runner",
                "task": "compute the absolute difference between the previous counts",
                "instruction": {
                    "operation": "run_command",
                    "command": _build_dependency_results_difference_command(),
                    "capture": {"mode": "stdout_stripped"},
                },
                "depends_on": executable_step_ids[:2],
            }
        )

    return steps if len(executable_step_ids) >= 2 else []


def _compound_fallback_steps(question: str, capabilities: dict):
    multi_domain_steps = _multi_domain_count_steps(question, capabilities)
    if multi_domain_steps:
        return multi_domain_steps

    parts = _split_compound_request(question)
    if len(parts) < 2:
        return []

    steps = []
    for index, part in enumerate(parts, start=1):
        previous_step = steps[-1] if steps else None
        previous_target = str(previous_step.get("target_agent") or "") if isinstance(previous_step, dict) else ""
        previous_step_id = str(previous_step.get("id") or "") if isinstance(previous_step, dict) else ""
        followup_task = None
        part_lc = part.lower()
        if previous_target == "shell_runner":
            if ("last" in part_lc or "tail" in part_lc) and ("line" in part_lc or "lines" in part_lc):
                tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", part_lc)
                tail = tail_match.group(1) if tail_match else "20"
                followup_task = f"show the last {tail} lines of {{prev}}"
        normalized_part = _normalize_compound_shell_part(part, question)
        normalized_part = _contextualize_shell_followup_task(normalized_part, question)
        target_agent = _select_target_agent(normalized_part, capabilities) or _select_target_agent(part, capabilities)
        normalized_lc = normalized_part.lower()
        if _looks_like_workspace_file_question(normalized_part) or _looks_like_repo_file_scan(normalized_part):
            target_agent = "shell_runner"
        if "python" in normalized_lc and "file" in normalized_lc:
            target_agent = "shell_runner"
        if not target_agent and any(
            token in normalized_lc
            for token in ("docker", "container", "containers", "image", "images", "installed", "available", "git", "file", "files", "port", "ports", "log", "logs")
        ):
            target_agent = "shell_runner"
        if not target_agent and any(
            token in normalized_lc
            for token in ("branch", "commit", "commits", "working tree", "status")
        ):
            target_agent = "shell_runner"
        if not target_agent or target_agent in {"ops_planner", "synthesizer"}:
            return []
        effective_task = followup_task or normalized_part
        if _agent_family(target_agent) == "slurm_runner":
            effective_task = _contextualize_slurm_followup_task(effective_task, question)
        if target_agent == "shell_runner":
            effective_task = _contextualize_shell_followup_task(effective_task, question)
        if followup_task:
            target_agent = "shell_runner"
        instruction = _normalize_agent_instruction(target_agent, effective_task, None, None, index)
        if not isinstance(instruction, dict) or not instruction:
            return []
        step = {
            "id": f"step{index}",
            "target_agent": target_agent,
            "task": effective_task,
            "instruction": instruction,
        }
        if followup_task and previous_step_id:
            step["depends_on"] = [previous_step_id]
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
    if original_task == question.strip().lower():
        return compound_steps
    if original_task and original_task in split_tasks:
        return compound_steps
    if original_task != question.strip().lower():
        return compound_steps
    return steps


def _looks_like_save_list_sql_request(question: str, capabilities: dict) -> bool:
    question_lc = question.lower()
    return (
        _looks_like_sql_export_request(question, capabilities)
        and bool(re.search(r"\b(?:list|llist|users|patients|rows|results|tables|schema)\b", question_lc))
    )


DETAIL_REQUEST_MARKERS = (
    "provide",
    "show",
    "list",
    "include",
    "return",
    "with their",
    "including",
    "details",
    "detail",
    "mrn",
    "name",
    "names",
    "rows",
    "columns",
    "fields",
    "attributes",
)


COUNT_REQUEST_MARKERS = (
    "how many",
    "count ",
    "count of",
    "number of",
    "total number",
    "total count",
)


DETAIL_SPLIT_MARKERS = (
    " and provide ",
    " and show ",
    " and list ",
    " and include ",
    " and return ",
    " including ",
    " with their ",
)


def _looks_like_mixed_count_and_detail_request(question: str) -> bool:
    text = re.sub(r"\s+", " ", str(question or "").strip().lower())
    if not text:
        return False
    has_count = any(marker in text for marker in COUNT_REQUEST_MARKERS)
    if not has_count:
        return False
    return any(marker in text for marker in DETAIL_REQUEST_MARKERS)


def _strip_count_request_prefix(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip(" ,")
    patterns = (
        r"^what\s+is\s+the\s+count\s+of\s+",
        r"^what\s+is\s+count\s+of\s+",
        r"^what\s+is\s+the\s+number\s+of\s+",
        r"^what\s+is\s+the\s+total\s+count\s+of\s+",
        r"^count of\s+",
        r"^count\s+of\s+",
        r"^count\s+",
        r"^how many\s+",
        r"^number of\s+",
        r"^total number of\s+",
        r"^total count of\s+",
    )
    for pattern in patterns:
        rewritten = re.sub(pattern, "", compact, flags=re.IGNORECASE).strip(" ,")
        if rewritten != compact:
            return rewritten
    return compact


def _mixed_count_and_detail_parts(question: str) -> tuple[str, str] | None:
    text = re.sub(r"\s+", " ", str(question or "").strip())
    if not _looks_like_mixed_count_and_detail_request(text):
        return None
    lower_text = text.lower()
    split_index = -1
    split_marker = ""
    for marker in DETAIL_SPLIT_MARKERS:
        index = lower_text.find(marker)
        if index != -1 and (split_index == -1 or index < split_index):
            split_index = index
            split_marker = marker
    if split_index != -1:
        count_part = text[:split_index].strip(" ,")
        detail_part = text[split_index + len(split_marker) :].strip(" ,")
    else:
        count_part = text
        detail_part = ""
    count_question = count_part
    if not any(marker in count_question.lower() for marker in COUNT_REQUEST_MARKERS):
        count_question = f"count {count_question}".strip()
    entity_part = _strip_count_request_prefix(count_part) or count_part
    detail_clause = detail_part.strip()
    if detail_clause:
        detail_clause = re.sub(r"^(?:me\s+)?their\s+", "their ", detail_clause, flags=re.IGNORECASE)
        detail_clause = re.sub(r"^(?:the\s+)?", "", detail_clause, flags=re.IGNORECASE)
        detail_question = (
            f"list {entity_part} and include {detail_clause}. "
            "Include the qualifying aggregate count used for filtering as a returned column. "
            "Return one row per matching result and do not return only an aggregate count."
        )
    else:
        detail_question = (
            f"list {entity_part}. "
            "Include the qualifying aggregate count used for filtering as a returned column. "
            "Return one row per matching result and include identifying and relevant detail columns. "
            "Do not return only an aggregate count."
        )
    return count_question.strip(), detail_question.strip()


def _mixed_count_detail_sql_question(question: str) -> str:
    compact = re.sub(r"\s+", " ", str(question or "").strip()).strip(" ,")
    return (
        f"{compact}. "
        "Return one row per matching result with identifying and requested detail columns. "
        "Preserve the exact filtering criteria from the request, including aggregate filters such as counts or thresholds. "
        "Include the qualifying aggregate used for filtering as a returned column when relevant. "
        "Do not broaden the result set and do not return only an aggregate count."
    ).strip()


def _expand_mixed_sql_steps(question: str, steps: list[dict], capabilities: dict) -> list[dict]:
    if not steps or not _looks_like_sql_question(question, capabilities):
        return steps
    parts = _mixed_count_and_detail_parts(question)
    if parts is None:
        return steps
    if len(steps) == 1 and _agent_family(str(steps[0].get("target_agent"))) == "sql_runner":
        count_question, detail_question = parts
        sql_step = steps[0]
        step1_id = str(sql_step.get("id") or "step1")
        target_agent = sql_step.get("target_agent")
        return [
            {
                "id": step1_id,
                "target_agent": target_agent,
                "task": "Count the qualifying rows",
                "instruction": {
                    "operation": "query_from_request",
                    "question": count_question,
                },
            },
            {
                "id": f"{step1_id}_details" if step1_id != "step1" else "step2",
                "target_agent": target_agent,
                "task": "List the qualifying rows with requested details",
                "instruction": {
                    "operation": "query_from_request",
                    "question": detail_question,
                },
                "depends_on": [step1_id],
            },
        ]
    if not all(_agent_family(str(step.get("target_agent"))) == "sql_runner" for step in steps):
        return steps
    count_question, detail_question = parts
    sql_step = steps[0]
    target_agent = sql_step.get("target_agent")
    step1_id = str(sql_step.get("id") or "step1")
    return [
        {
            "id": step1_id,
            "target_agent": target_agent,
            "task": "Count the qualifying rows",
            "instruction": {
                "operation": "query_from_request",
                "question": count_question,
            },
        },
        {
            "id": f"{step1_id}_details" if step1_id != "step1" else "step2",
            "target_agent": target_agent,
            "task": "List the qualifying rows with requested details",
            "instruction": {
                "operation": "query_from_request",
                "question": detail_question,
            },
            "depends_on": [step1_id],
        },
    ]


def _list_query_from_request(question: str) -> str:
    compact = re.sub(r"\s+", " ", question).strip()
    compact = re.split(r"\s*,\s*(?:create|save|write|export|download)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    compact = re.split(r"\b(?:and then|then)\b\s+(?:create|save|write|export|download)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    compact = re.split(r"\s+and\s+(?:create|save|write|export|download)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    compact = re.split(r"\s+and\s+provide\s+(?:me\s+)?(?:the\s+)?(?:location|path)\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    compact = re.split(r"\s+so\s+i\s+can\s+review\b", compact, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,.")
    return (
        f"{compact}. Return one row per matching result and include all relevant detail columns. "
        "Do not return only an aggregate count."
    )


def _expand_sql_export_steps(question: str, steps: list[dict], capabilities: dict) -> list[dict]:
    if not _looks_like_save_list_sql_request(question, capabilities):
        return steps

    save_path = _resolve_save_output_path(question, capabilities)
    if not save_path:
        return steps

    sql_steps = [step for step in steps if _agent_family(str(step.get("target_agent"))) == "sql_runner"]
    if not sql_steps:
        target_agent = _select_sql_target_agent(question, capabilities)
        if not target_agent or _agent_family(target_agent) != "sql_runner":
            return steps
        return [
            {
                "id": "step1",
                "target_agent": target_agent,
                "task": "List the qualifying rows with requested details",
                "instruction": {
                    "operation": "query_from_request",
                    "question": _list_query_from_request(question),
                },
            },
            _build_save_rows_step("step1", save_path, 2),
        ]

    export_source_step_id = str(sql_steps[-1].get("id") or "step1")
    aligned = []
    saw_save_step = False
    for index, step in enumerate(steps):
        updated = dict(step)
        target_agent = updated.get("target_agent")
        instruction = updated.get("instruction")
        if (
            str(updated.get("id") or "") == export_source_step_id
            and _agent_family(str(target_agent)) == "sql_runner"
            and isinstance(instruction, dict)
            and instruction.get("operation") == "query_from_request"
        ):
            updated_instruction = dict(instruction)
            updated_instruction["question"] = _list_query_from_request(question)
            updated["instruction"] = updated_instruction
            updated["task"] = "List the qualifying rows with requested details for export"

        if str(updated.get("target_agent")) == "shell_runner" and any(
            marker in str(updated.get("task") or "").lower()
            for marker in EXPORT_REQUEST_MARKERS + EXPORT_TARGET_MARKERS
        ):
            saw_save_step = True
            updated["task"] = f"Save the exported rows to {save_path} and print the absolute file path"
            updated["instruction"] = {
                "operation": "run_command",
                "command": _build_save_rows_command(save_path),
                "input": {"$from": f"{export_source_step_id}.rows"},
                "capture": {"mode": "stdout_stripped"},
            }
            updated["depends_on"] = [export_source_step_id]
        aligned.append(updated)
    if not saw_save_step:
        aligned.append(_build_save_rows_step(export_source_step_id, save_path, len(aligned) + 1))
    return aligned


def _normalize_steps(question: str, steps: List[Dict[str, str]], capabilities: dict):
    if _looks_like_slurm_elapsed_summary_question(question) or _looks_like_slurm_node_inventory_summary_question(question):
        slurm_steps = _fallback_steps(question, capabilities)
        if slurm_steps:
            return slurm_steps
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
                        "task": _sanitize_task_text(step.get("task", "").strip()) or question,
                        "steps": normalized_nested,
                    }
                )
                available_step_ids.add(group_id)
            continue

        target_agent = step.get("target_agent")
        task = _sanitize_task_text(step.get("task", "").strip())
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
        if _agent_family(target_agent) == "notifier" and not _looks_like_notification_request(task or question):
            target_agent = "shell_runner" if _looks_like_workspace_file_question(task or question) or "git" in (task or question).lower() else _select_target_agent(question, capabilities)
            if not target_agent or _agent_family(target_agent) == "notifier":
                target_agent = "shell_runner"
        if (
            _agent_family(target_agent) == "sql_runner"
            and (_looks_like_repo_file_scan(task or question) or (_looks_like_workspace_file_question(task or question) and not _looks_like_sql_question(task or question, capabilities)))
        ):
            target_agent = "shell_runner"
        if _agent_family(target_agent) == "slurm_runner" and (
            _looks_like_repo_file_scan(task or question)
            or _looks_like_workspace_file_question(task or question)
            or ("python" in (task or question).lower() and "file" in (task or question).lower())
        ):
            target_agent = "shell_runner"
        if _agent_family(target_agent) == "sql_runner" and (
            _looks_like_slurm_elapsed_summary_question(task or question)
            or _looks_like_slurm_node_inventory_summary_question(task or question)
            or (_looks_like_slurm_question(task or question) and not _looks_like_sql_question(task or question, capabilities))
        ):
            target_agent = _first_agent_in_family(capabilities, "slurm_runner") or target_agent
        if _is_presentation_only_task(task or question):
            continue
        normalized_step = {
            "id": step.get("id") or f"step{index}",
            "target_agent": target_agent,
            "task": task or question,
        }
        if target_agent == "shell_runner":
            normalized_step["task"] = _contextualize_shell_followup_task(normalized_step["task"], question)
        if _agent_family(target_agent) == "slurm_runner":
            normalized_step["task"] = _contextualize_slurm_followup_task(normalized_step["task"], question)
        clean_depends_on = []
        depends_on = step.get("depends_on")
        if isinstance(depends_on, list):
            clean_depends_on = [item for item in depends_on if isinstance(item, str) and item.strip()]
        if (
            target_agent == "shell_runner"
            and len(clean_depends_on) >= 2
            and any(token in question.lower() for token in ("difference", "compare", "versus", " vs "))
        ):
            normalized_step["task"] = "compute the absolute difference between the previous counts"
            instruction = {
                "operation": "run_command",
                "command": _build_dependency_results_difference_command(),
                "capture": {"mode": "stdout_stripped"},
            }
        if (
            target_agent == "shell_runner"
            and clean_depends_on
            and "{{prev}}" not in normalized_step["task"].lower()
        ):
            task_lc = normalized_step["task"].lower()
            if ("last" in task_lc or "tail" in task_lc) and ("line" in task_lc or "lines" in task_lc):
                tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\b", task_lc)
                tail = tail_match.group(1) if tail_match else "20"
                normalized_step["task"] = f"show the last {tail} lines of {{prev}}"
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
        if (
            _agent_family(target_agent) == "slurm_runner"
            and normalized_instruction.get("operation") == "query_from_request"
            and isinstance(normalized_instruction.get("question"), str)
        ):
            normalized_instruction = dict(normalized_instruction)
            normalized_instruction["question"] = normalized_step["task"]
        if (
            _agent_family(target_agent) == "sql_runner"
            and normalized_instruction.get("operation") == "query_from_request"
            and isinstance(normalized_instruction.get("question"), str)
        ):
            normalized_instruction = dict(normalized_instruction)
            normalized_instruction["question"] = _normalized_sql_query_question(
                normalized_step["task"],
                normalized_instruction["question"],
            )
        if target_agent == "shell_runner":
            previous_step_id = normalized[-1]["id"] if normalized else None
            normalized_instruction = _normalize_followup_shell_instruction(normalized_instruction, previous_step_id)
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
        if clean_depends_on:
            normalized_step["depends_on"] = clean_depends_on
        when = step.get("when")
        if isinstance(when, dict) and when:
            normalized_step["when"] = when
        normalized.append(normalized_step)
        available_step_ids.add(normalized_step["id"])
    if not _looks_like_save_list_sql_request(question, capabilities):
        normalized = _recover_compound_steps(question, normalized, capabilities)
    normalized = _expand_mixed_sql_steps(question, normalized, capabilities)
    return _expand_sql_export_steps(question, normalized, capabilities)


def _workflow_replan_allowed_families(payload: dict, capabilities: dict) -> set[str]:
    if str(payload.get("step_id") or "").strip() != "__workflow__":
        return set()
    question = str(payload.get("task") or "").strip()
    task_families = set()
    if _looks_like_slurm_question(question):
        task_families.add("slurm_runner")
    if _looks_like_sql_question(question, capabilities):
        task_families.add("sql_runner")
    available_context = payload.get("available_context") if isinstance(payload.get("available_context"), dict) else {}
    last_steps = available_context.get("last_steps")
    if isinstance(last_steps, list):
        step_families = {
            _agent_family(str(step.get("target_agent") or ""))
            for step in last_steps
            if isinstance(step, dict)
        }
        step_families.discard("")
        if len(step_families) == 1:
            task_families |= step_families
    return task_families if len(task_families) == 1 else set()


def _replan_steps_respect_workflow_context(payload: dict, steps: list[dict], capabilities: dict) -> bool:
    allowed_families = _workflow_replan_allowed_families(payload, capabilities)
    if not allowed_families:
        return True
    candidate_families = {
        _agent_family(str(step.get("target_agent") or ""))
        for step in _flatten_plan_steps(steps)
        if isinstance(step, dict)
    }
    candidate_families.discard("")
    if not candidate_families:
        return False
    return candidate_families.issubset(allowed_families)


def _normalize_presentation(question: str, presentation: dict | None):
    normalized = _derive_presentation(question)
    question_lc = question.lower()
    docker_listing_request = (
        "docker" in question_lc
        and any(token in question_lc for token in ("container", "containers"))
        and any(token in question_lc for token in ("list", "show"))
        and "table" not in question_lc
    )
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
            normalized[key] = normalized[key] or value if key == "include_internal_steps" else value
    if (
        _looks_like_sql_question(question, CAPABILITIES)
        and not _planner_is_schema_request(question)
        and normalized.get("format") == "markdown_table"
    ):
        normalized["task"] = "Show grounded SQL results in a clean Markdown table."
    if _looks_like_slurm_elapsed_summary_question(question):
        normalized["format"] = "markdown"
        normalized["task"] = "Report the computed Slurm timing summary clearly in Markdown."
    if _looks_like_slurm_node_inventory_summary_question(question):
        normalized["format"] = "markdown"
        normalized["task"] = "Report the total Slurm node count and state summary clearly in Markdown."
    if docker_listing_request:
        normalized["format"] = "markdown"
        normalized["task"] = "Show the raw Docker command output clearly in a preformatted block."
    return normalized


def _steps_signature(steps: list[dict]) -> str:
    return json.dumps(steps, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _build_primary_plan(question: str, decision: dict, capabilities: dict):
    if _looks_like_slurm_elapsed_summary_question(question) or _looks_like_slurm_node_inventory_summary_question(question):
        slurm_steps = _fallback_steps(question, capabilities)
        if slurm_steps:
            return slurm_steps
    if _looks_like_sql_export_request(question, capabilities):
        sql_export_steps = _sql_fallback_steps(question, capabilities)
        if sql_export_steps:
            return sql_export_steps
    if _should_override_shell_command(question.lower()):
        deterministic_shell_steps = _fallback_steps(question, capabilities)
        if deterministic_shell_steps:
            return deterministic_shell_steps
    raw_primary = decision.get("steps", [])
    primary_steps = _normalize_steps(question, raw_primary, capabilities) if raw_primary else []
    if primary_steps:
        return primary_steps
    fallback_steps = _fallback_steps(question, capabilities)
    if fallback_steps:
        return fallback_steps
    sql_fallback = _sql_fallback_steps_for_task(question, question, capabilities)
    if sql_fallback:
        return sql_fallback
    return []


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
    payload = {
        "stage": "planned",
        "message": message,
        "steps": flat_steps,
        "presentation": presentation,
    }
    payload["task_shape"] = _infer_task_shape(question, presentation=presentation)
    return payload


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
        "task_shape": _infer_task_shape(task_question, target_agent=sql_steps[0]["target_agent"], presentation=presentation),
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


def _fallback_plan_emits(question: str, allowed_events: set[str], task_question: str | None = None):
    task_question = task_question or _current_request_from_context(question)
    if "task.plan" not in allowed_events:
        return None
    steps = _fallback_steps(task_question, CAPABILITIES)
    if not steps:
        return None
    presentation = _normalize_presentation(task_question, None)
    payload = {
        "task": task_question,
        "task_shape": _infer_task_shape(
            task_question,
            target_agent=steps[0]["target_agent"] if len(steps) == 1 else None,
            presentation=presentation,
        ),
        "steps": steps,
        "presentation": presentation,
    }
    if len(steps) == 1:
        payload["target_agent"] = steps[0]["target_agent"]
    emits = []
    if "plan.progress" in allowed_events:
        emits.append(
            {
                "event": "plan.progress",
                "payload": _plan_progress_payload(task_question, steps, presentation),
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
@with_node_envelope("ops_planner", "router")
def handle_event(req: EventRequest):
    if req.event == "system.capabilities":
        agents = req.payload.get("agents", [])
        if isinstance(agents, list):
            CAPABILITIES["agents"] = agents
            CAPABILITIES["available_events"] = _derive_available_events(agents)
        execution_policy = req.payload.get("execution_policy")
        CAPABILITIES["execution_policy"] = execution_policy if isinstance(execution_policy, dict) else {}
        return noop()

    if req.event == "planner.replan.request":
        allowed_events = set(CAPABILITIES.get("available_events", DEFAULT_ALLOWED_EVENTS))
        if "planner.replan.result" not in allowed_events:
            return noop()
        replace_step_id = str(req.payload.get("step_id") or "").strip()
        if not replace_step_id:
            return noop()
        steps = []
        try:
            decision = _llm_replan(req.payload, CAPABILITIES)
            if decision is not None and decision["replace_step_id"] == replace_step_id:
                steps = _normalize_steps(str(req.payload.get("task") or ""), decision.get("steps", []), CAPABILITIES)
                if steps and not _replan_steps_respect_workflow_context(req.payload, steps, CAPABILITIES):
                    _debug_log(
                        "Discarding workflow replan candidate that drifted to the wrong agent family: "
                        + json.dumps({"task": req.payload.get("task"), "steps": steps}, ensure_ascii=True)
                    )
                    steps = []
        except Exception as exc:
            _debug_log(f"Planner replan failed. Error: {type(exc).__name__}: {exc}")
        if not steps:
            steps = _fallback_replan_steps(req.payload, CAPABILITIES)
        if not steps:
            return failure_result(
                str(req.payload.get("reason") or "Planner could not further decompose the failed step."),
                error=req.payload.get("error") or req.payload.get("reason"),
            )
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
        return emit_sequence(emits)

    if req.event != "user.ask":
        return noop()

    question = req.payload["question"]
    current_question = _current_request_from_context(question)
    allowed_events = set(CAPABILITIES.get("available_events", DEFAULT_ALLOWED_EVENTS))
    if _is_unsafe_machine_request(current_question) and "task.result" in allowed_events:
        return task_result(
            "I won’t autonomously plan high-risk destructive machine operations. Please use a safer, more specific alternative."
        )
    if _is_capability_question(current_question) and "task.result" in allowed_events:
        return task_result(_format_capability_answer(CAPABILITIES), result=_capability_summary(CAPABILITIES))
    try:
        decision = _llm_decide(question, CAPABILITIES)
        if decision is not None:
            if decision["processable"] and "task.plan" in allowed_events:
                steps = _build_primary_plan(current_question, decision, CAPABILITIES)
                payload = {"task": current_question}
                if steps:
                    payload["steps"] = steps
                    presentation = _normalize_presentation(current_question, decision.get("presentation"))
                    payload["presentation"] = presentation
                    primary_target = steps[0]["target_agent"] if len(steps) == 1 else None
                    payload["task_shape"] = _normalize_task_shape(
                        current_question,
                        decision.get("task_shape"),
                        target_agent=primary_target,
                        presentation=presentation,
                    )
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
                return emit_sequence(emits)
            emits = _sql_plan_emits(question, allowed_events, current_question)
            if emits is not None:
                return emit_sequence(emits)
            if "task.result" in allowed_events:
                reason = decision["reason"] or "No matching capability found."
                return task_result(reason)
            return noop()
        _debug_log("LLM planner decision was invalid.")
    except Exception as exc:
        _debug_log(f"LLM planning failed. Error: {type(exc).__name__}: {exc}")

    emits = _sql_plan_emits(question, allowed_events, current_question)
    if emits is not None:
        return emit_sequence(emits)

    emits = _fallback_plan_emits(question, allowed_events, current_question)
    if emits is not None:
        return emit_sequence(emits)

    if "task.result" in allowed_events:
        return task_result(
            "Planner could not determine if the request is processable. Check LLM connectivity/response format and retry."
        )
    return noop()
