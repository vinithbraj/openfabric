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


def _planner_llm_max_attempts() -> int:
    raw = os.getenv("PLANNER_LLM_MAX_ATTEMPTS", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 3
    return max(1, value)

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


def _planning_hint_dict(item: dict) -> dict[str, Any]:
    raw = item.get("planning_hints")
    return raw if isinstance(raw, dict) else {}


def _planning_hint_list(hints: dict[str, Any], key: str) -> list[str]:
    value = hints.get(key)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _planning_hint_int(hints: dict[str, Any], key: str, default: int = 0) -> int:
    value = hints.get(key)
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


def _planning_hint_bool(hints: dict[str, Any], key: str) -> bool:
    value = hints.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _agent_planning_hints(agent: dict) -> dict[str, Any]:
    merged = {
        "keywords": [],
        "anti_keywords": [],
        "preferred_task_shapes": [],
        "instruction_operations": [],
        "routing_priority": 0,
        "structured_followup": False,
        "native_count_preferred": False,
    }
    hint_sources: list[dict[str, Any]] = []
    descriptor_hints = _planning_hint_dict(agent)
    if descriptor_hints:
        hint_sources.append(descriptor_hints)
    for api in _agent_api_specs(agent):
        api_hints = _planning_hint_dict(api)
        if api_hints:
            hint_sources.append(api_hints)

    for hints in hint_sources:
        for key in ("keywords", "anti_keywords", "preferred_task_shapes", "instruction_operations"):
            merged[key].extend(_planning_hint_list(hints, key))
        merged["routing_priority"] += _planning_hint_int(hints, "routing_priority")
        merged["structured_followup"] = merged["structured_followup"] or _planning_hint_bool(hints, "structured_followup")
        merged["native_count_preferred"] = merged["native_count_preferred"] or _planning_hint_bool(hints, "native_count_preferred")

    for key in ("routing_hints", "domain_hints", "schema_hints", "entity_hints"):
        value = agent.get(key)
        if isinstance(value, str) and value.strip():
            merged["keywords"].append(value.strip())
        elif isinstance(value, list):
            merged["keywords"].extend(str(item).strip() for item in value if isinstance(item, str) and str(item).strip())

    for key in ("database_name", "argument_name", "template_agent", "cluster_name"):
        value = agent.get(key)
        if isinstance(value, str) and value.strip():
            merged["keywords"].append(value.strip())
    for key in ("database_aliases", "cluster_aliases"):
        value = agent.get(key)
        if isinstance(value, list):
            merged["keywords"].extend(str(item).strip() for item in value if isinstance(item, str) and str(item).strip())

    merged["keywords"] = sorted({item for item in merged["keywords"] if item})
    merged["anti_keywords"] = sorted({item for item in merged["anti_keywords"] if item})
    merged["preferred_task_shapes"] = sorted({item for item in merged["preferred_task_shapes"] if item})
    merged["instruction_operations"] = sorted({item for item in merged["instruction_operations"] if item})
    return merged


def _format_planning_hints(hints: dict[str, Any]) -> str:
    parts = []
    keywords = _planning_hint_list(hints, "keywords")
    if keywords:
        parts.append(f"keywords=[{', '.join(keywords[:10])}]")
    preferred_task_shapes = _planning_hint_list(hints, "preferred_task_shapes")
    if preferred_task_shapes:
        parts.append(f"preferred_task_shapes=[{', '.join(preferred_task_shapes)}]")
    instruction_operations = _planning_hint_list(hints, "instruction_operations")
    if instruction_operations:
        parts.append(f"instruction_operations=[{', '.join(instruction_operations)}]")
    if _planning_hint_bool(hints, "structured_followup"):
        parts.append("structured_followup=true")
    if _planning_hint_bool(hints, "native_count_preferred"):
        parts.append("native_count_preferred=true")
    routing_priority = _planning_hint_int(hints, "routing_priority")
    if routing_priority:
        parts.append(f"routing_priority={routing_priority}")
    return "; ".join(parts)


def _format_runtime_agent_planning_guide(capabilities: dict) -> str:
    lines = []
    for agent in capabilities.get("agents", []):
        if not isinstance(agent, dict):
            continue
        name = agent.get("name")
        if not isinstance(name, str) or name in {"ops_planner", "synthesizer"}:
            continue
        if not _agent_handles_trigger(agent, "task.plan"):
            continue
        hints = _agent_planning_hints(agent)
        hint_text = _format_planning_hints(hints)
        api_summaries = []
        for api in _agent_api_specs(agent):
            if not isinstance(api, dict) or api_trigger_event(api) != "task.plan":
                continue
            api_name = str(api.get("name") or "").strip()
            summary = str(api.get("summary") or api.get("when") or "").strip()
            api_hint_text = _format_planning_hints(_planning_hint_dict(api))
            parts = [part for part in (api_name, summary, api_hint_text) if part]
            if parts:
                api_summaries.append(" | ".join(parts))
        detail = f"- {name}: {str(agent.get('description') or '').strip() or 'No description provided.'}"
        if hint_text:
            detail += f" Planning[{hint_text}]"
        if api_summaries:
            detail += f" APIs[{'; '.join(api_summaries)}]"
        lines.append(detail)
    return "\n".join(lines) if lines else "- none"


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
        planning_hints = _agent_planning_hints(item)
        planning_hint_text = _format_planning_hints(planning_hints)
        if planning_hint_text:
            metadata.append(f"planning_hints[{planning_hint_text}]")
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
        "5) Use discovered descriptor planning hints and API summaries to choose target_agent and instruction.operation.\n"
        "5a) target_agent MUST be one discovered runtime agent name that can handle task.plan; never use ops_planner or synthesizer.\n"
        "5b) instruction.operation MUST come from the selected agent's advertised planning hints or API guidance below.\n"
        "5c) For deterministic shell checks, prefer machine-readable output and capture hints such as stdout_first_line, stdout_stripped, json, or json_field when the selected agent advertises them.\n"
        "11) Break chained requests into multiple atomic steps. Do not combine unrelated actions into one step. Use nested steps for grouped subtasks.\n"
        "11a) If the user asks for multiple executable sub-goals joined by words like and, then, or check whether, preserve those sub-goals as separate leaf steps unless a clause is purely presentation-only.\n"
        "11b) Do not satisfy a request like 'list/show X and check whether Y' with one broad step. The inspection or boolean follow-up must be its own explicit step.\n"
        "12) Rewrite each step as a direct, concrete instruction for the target agent. Do not simply copy long stretches of the user's wording.\n"
        "13) Each step should be self-contained, operational, and phrased so another LLM can execute it without guessing.\n"
        "14) If a later step depends on an earlier step result, set depends_on and use a when condition or instruction inputs that reference prior results.\n"
        "15) References to previous step data must use objects of the form {\"$from\":\"step_id.field.path\"}. Do not splice raw prose into commands.\n"
        "15a) When a later step needs to inspect or transform rows returned by an earlier step, prefer shell_runner with instruction.input referencing prior rows or the full prior step result.\n"
        "15b) For tasks like 'in this list', 'from those rows', 'check the previous result', or 'search the output', do not issue a fresh SQL query unless the user explicitly asks to query again.\n"
        "16) when is optional. Use it for conditional execution such as {\"$from\":\"step1.returncode\",\"not_equals\":0}.\n"
        "17) Prefer agents that advertise native_count_preferred for count, stats, or filtering tasks instead of decomposing into shell pipes.\n"
        "18) Prefer agents that advertise structured_followup when a later step needs to inspect prior rows or structured outputs.\n"
        "18a) If execution requires a missing Python dependency and runtime policy allows it, only use that path when the selected agent explicitly advertises local command execution.\n"
        "19) For a single-goal database or scheduler question, one native step is often enough; do not collapse a multi-clause request into one step just because one command might answer several clauses implicitly.\n"
        "20) Do not invent agent names, instruction operations, or capabilities beyond the discovered guide below.\n"
        "22) When you emit multiple SQL steps, each later SQL step should clearly state what it is trying to learn or produce, and it must declare depends_on when it uses earlier findings.\n"
        "23) When using shell_runner, prefer explicit machine-readable commands for deterministic checks.\n"
        "24) For existence checks, avoid substring grep against broad human-readable output. Use exact or JSON-based checks.\n"
        "25) Do not invent shell commands or fake CLI syntax for non-shell agents.\n"
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
        '- "list all conda environments and check whether there is an environment named vinith" -> {"processable":true,"reason":"list then inspect environment presence","steps":[{"id":"step1","target_agent":"shell_runner","task":"list all conda environments","instruction":{"operation":"run_command","command":"conda env list","capture":{"mode":"stdout_stripped"}}},{"id":"step2","target_agent":"shell_runner","task":"check whether the conda environment named vinith exists","instruction":{"operation":"run_command","command":"conda env list --json | python3 -c \\"import json,sys,os; name=\\\\\\"vinith\\\\\\"; envs=json.load(sys.stdin).get(\\\\\\"envs\\\\\\", []); exists=any(os.path.basename(path.rstrip(\\\\\\"/\\\\\\")) == name for path in envs); print(json.dumps({\\\\\\"exists\\\\\\": exists, \\\\\\"name\\\\\\": name}))\\"","capture":{"mode":"json"}},"depends_on":["step1"]}],"presentation":{"task":"List the conda environments and clearly report whether vinith exists.","format":"markdown","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "list patients with more than 20 studies and then check whether any listed patient is named Test case-insensitively" -> {"processable":true,"reason":"query then inspect prior rows","steps":[{"id":"step1","target_agent":"sql_runner","task":"list patients with more than 20 studies","instruction":{"operation":"query_from_request","question":"list patients with more than 20 studies and include patient names in a neat table"}},{"id":"step2","target_agent":"shell_runner","task":"check whether any patient in the previous SQL result is named Test using a case-insensitive exact match","instruction":{"operation":"run_command","command":"python3 -c \\"import json,sys; rows=json.load(sys.stdin); print(any(str(row.get(\\\\\\\"PatientName\\\\\\\", \\\\\\\"\\\\\\\")).strip().lower() == \\\\\\\"test\\\\\\\" for row in rows))\\"","input":{"$from":"step1.rows"},"capture":{"mode":"stdout_stripped"}},"depends_on":["step1"]}],"presentation":{"task":"Show the patient table and clearly report whether any listed patient name matches Test case-insensitively.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "show database schema and relationships" -> {"processable":true,"reason":"SQL schema introspection","steps":[{"id":"step1","target_agent":"sql_runner","task":"inspect the database schema and relationships","instruction":{"operation":"inspect_schema","focus":"tables, columns, and relationships"}}],"presentation":{"task":"Summarize schemas, tables, columns, and relationships clearly.","format":"markdown","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "which customers spent the most money last month" -> {"processable":true,"reason":"SQL database question","steps":[{"id":"step1","target_agent":"sql_runner","task":"query which customers spent the most money last month","instruction":{"operation":"query_from_request","question":"which customers spent the most money last month"}}],"presentation":{"task":"Show the query results as a readable Markdown table.","format":"markdown_table","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "show queued slurm jobs for user vinith" -> {"processable":true,"reason":"Slurm queue query","steps":[{"id":"step1","target_agent":"slurm_runner","task":"show queued Slurm jobs for user vinith","instruction":{"operation":"query_from_request","question":"show queued Slurm jobs for user vinith"}}],"presentation":{"task":"Show the Slurm results clearly in Markdown.","format":"markdown","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "how many jobs are in a non running state" -> {"processable":true,"reason":"Slurm state count","steps":[{"id":"step1","target_agent":"slurm_runner","task":"how many jobs are in a non running state","instruction":{"operation":"query_from_request","question":"how many jobs are in a non running state"}}],"presentation":{"task":"Report the count of non-running jobs clearly.","format":"plain","audience":"openwebui","include_context":true,"include_internal_steps":false}}\n'
        '- "open Readme.md" -> {"processable":true,"reason":"file read","steps":[{"id":"step1","target_agent":"filesystem","task":"read Readme.md","instruction":{"operation":"read_file","path":"Readme.md"}}]}\n'
        '- "book a flight to NYC" -> {"processable":false,"reason":"no travel booking capability","steps":[]}\n'
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        "Runtime planning guide:\n"
        f"{_format_runtime_agent_planning_guide(capabilities)}\n"
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
        "7) Use discovered descriptor planning hints and API summaries below to choose replacement agents and instruction operations.\n"
        "8) If runtime policy allows Python package installation, only use that path when the selected agent explicitly advertises local command execution.\n"
        "9) Tolerate minor typos and informal phrasing.\n"
        "Discovered runtime agents:\n"
        f"{_format_discovered_agents(capabilities)}\n"
        "Runtime planning guide:\n"
        f"{_format_runtime_agent_planning_guide(capabilities)}\n"
        "Runtime execution policy:\n"
        f"{_format_execution_policy(capabilities)}\n"
        "Replan request JSON:\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _build_plan_retry_prompt(
    question: str,
    capabilities: dict,
    rejection_reason: str,
    previous_decision: dict | None,
    attempt: int,
) -> str:
    previous_json = json.dumps(previous_decision or {}, indent=2, ensure_ascii=True)
    return (
        _build_prompt(question, capabilities)
        + "\n\nPlanner self-repair loop:\n"
        + f"- Attempt {attempt} of {_planner_llm_max_attempts()}.\n"
        + f"- The previous candidate plan was rejected for this reason: {rejection_reason}\n"
        + "- Produce a corrected plan that directly fixes the rejection reason.\n"
        + "- Do not repeat the same invalid step structure, target-agent drift, or collapsed workflow shape.\n"
        + "- If the rejection mentions collapsed clause coverage, split the user request into distinct executable leaf steps.\n"
        + "- If the request is processable, steps must be executable and dependency-aware.\n"
        + "Rejected candidate JSON:\n"
        + previous_json
    )


def _build_replan_retry_prompt(
    payload: dict,
    capabilities: dict,
    rejection_reason: str,
    previous_decision: dict | None,
    attempt: int,
) -> str:
    previous_json = json.dumps(previous_decision or {}, indent=2, ensure_ascii=True)
    return (
        _build_replan_prompt(payload, capabilities)
        + "\n\nPlanner self-repair loop:\n"
        + f"- Attempt {attempt} of {_planner_llm_max_attempts()}.\n"
        + f"- The previous replacement plan was rejected for this reason: {rejection_reason}\n"
        + "- Produce a corrected replacement subplan that fixes the rejection reason.\n"
        + "- Do not repeat the same invalid replacement structure, agent-family drift, or collapsed subplan.\n"
        + "Rejected candidate JSON:\n"
        + previous_json
    )


def _build_plan_validation_prompt(
    question: str,
    steps: list[dict],
    capabilities: dict,
    *,
    replan_payload: dict | None = None,
) -> str:
    context_lines = []
    if isinstance(replan_payload, dict) and replan_payload:
        context_lines.append("Replan context JSON:")
        context_lines.append(json.dumps(replan_payload, indent=2, ensure_ascii=True))

    return (
        "You are the semantic plan validator for an operations assistant.\n"
        "Task: judge whether the candidate plan actually satisfies the user's intent, preserves the necessary decomposition, "
        "and keeps only execution-relevant user actions.\n"
        "Output JSON only with exact keys: "
        '{"valid":true|false,"reason":"short reason","goal_coverage":"complete|partial|incorrect","decomposition":"good|collapsed|over_decomposed","user_action_alignment":"strong|weak|meta","issues":["issue"],"rewarded_paths":["good path"],"disallowed_paths":["bad path"]}\n'
        "Validation rules:\n"
        "1) valid=true only if the plan preserves the user's executable goals and would reasonably produce the information or artifact the user asked for.\n"
        "2) Reward plans that decompose the request into concrete user-action-oriented steps that causally advance the task.\n"
        "3) Reject plans that collapse multiple executable user goals into one broad step when the user asked for distinct actions, checks, comparisons, or follow-ups.\n"
        "4) Reject meta steps, presentation-only steps, synthesis-only steps, or steps that do not materially advance execution.\n"
        "5) Reject steps that drift semantically away from the user's request or from the failed-step scope in a replan context.\n"
        "6) Prefer plans where later steps explicitly inspect or transform earlier results instead of silently assuming those checks are covered.\n"
        "7) Do not reject only because execution might fail at runtime; judge planning quality and goal coverage.\n"
        "8) If the request has multiple user-visible actions joined by words like and, then, compare, check whether, confirm, verify, or save, expect those actions to be represented explicitly unless one clause is clearly presentation-only.\n"
        "9) For exact existence or membership checks, disallow broad human-readable grep or substring heuristics when the plan could use an exact or machine-readable inspection step instead.\n"
        "User request:\n"
        f"{question}\n"
        "Candidate plan JSON:\n"
        f"{json.dumps(steps, indent=2, ensure_ascii=True)}\n"
        + ("\n".join(context_lines) + "\n" if context_lines else "")
        + "Discovered runtime agents:\n"
        + f"{_format_discovered_agents(capabilities)}\n"
        + "Runtime planning guide:\n"
        + f"{_format_runtime_agent_planning_guide(capabilities)}\n"
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


def _llm_prompt_json(prompt: str, *, debug_label: str):
    api_key, base_url, timeout_seconds, model = shared_llm_api_settings("gpt-4o-mini")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You produce strict JSON only."},
        {"role": "user", "content": prompt},
    ]

    _debug_log(f"Constructed {debug_label} prompt:")
    _debug_log(prompt)
    _debug_log(f"Messages sent to {debug_label} LLM:")
    _debug_log(json.dumps(messages, indent=2))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if not response.ok:
        _debug_log(f"{debug_label} LLM HTTP error status: {response.status_code}")
        _debug_log(f"{debug_label} LLM HTTP error body:")
        _debug_log(response.text[:4000])
        response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    _debug_log(f"Raw {debug_label} LLM response content:")
    _debug_log(content)
    json_blob, parsed = _parse_planner_json(content)
    if not json_blob:
        _debug_log(f"No JSON object found in {debug_label} LLM response content.")
        return None
    _debug_log(f"Extracted JSON object from {debug_label} LLM response:")
    _debug_log(json_blob)
    if parsed is None:
        _debug_log(f"Could not parse {debug_label} JSON after repair attempts.")
        return None
    return parsed


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


def _parse_plan_validation(raw: Any):
    if not isinstance(raw, dict):
        return None
    valid = raw.get("valid")
    reason = raw.get("reason", "")
    goal_coverage = raw.get("goal_coverage", "")
    decomposition = raw.get("decomposition", "")
    user_action_alignment = raw.get("user_action_alignment", "")
    issues = raw.get("issues", [])
    rewarded_paths = raw.get("rewarded_paths", [])
    disallowed_paths = raw.get("disallowed_paths", [])
    if not isinstance(valid, bool):
        return None
    if not isinstance(reason, str):
        reason = ""
    if not isinstance(goal_coverage, str):
        goal_coverage = ""
    if not isinstance(decomposition, str):
        decomposition = ""
    if not isinstance(user_action_alignment, str):
        user_action_alignment = ""
    if not isinstance(issues, list):
        issues = []
    if not isinstance(rewarded_paths, list):
        rewarded_paths = []
    if not isinstance(disallowed_paths, list):
        disallowed_paths = []
    return {
        "valid": valid,
        "reason": reason.strip(),
        "goal_coverage": goal_coverage.strip().lower(),
        "decomposition": decomposition.strip().lower(),
        "user_action_alignment": user_action_alignment.strip().lower(),
        "issues": [str(item).strip() for item in issues if str(item).strip()],
        "rewarded_paths": [str(item).strip() for item in rewarded_paths if str(item).strip()],
        "disallowed_paths": [str(item).strip() for item in disallowed_paths if str(item).strip()],
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
    parsed = _llm_prompt_json(_build_prompt(question, capabilities), debug_label="planner")
    if parsed is None:
        return None
    decision = _parse_decision(parsed)
    if decision is None:
        _debug_log("Parsed planner JSON missing valid processable/reason fields.")
        return None
    _debug_log("Planner decision:")
    _debug_log(json.dumps(decision, indent=2))
    return decision


def _llm_replan(payload: dict, capabilities: dict):
    parsed = _llm_prompt_json(_build_replan_prompt(payload, capabilities), debug_label="planner replan")
    if parsed is None:
        return None
    decision = _parse_replan_decision(parsed)
    if decision is None:
        return None
    _debug_log("Planner replan decision:")
    _debug_log(json.dumps(decision, indent=2))
    return decision


def _llm_retry_decide(
    question: str,
    capabilities: dict,
    rejection_reason: str,
    previous_decision: dict | None,
    attempt: int,
):
    parsed = _llm_prompt_json(
        _build_plan_retry_prompt(question, capabilities, rejection_reason, previous_decision, attempt),
        debug_label="planner retry",
    )
    if parsed is None:
        return None
    decision = _parse_decision(parsed)
    if decision is None:
        _debug_log("Parsed planner retry JSON missing valid processable/reason fields.")
        return None
    _debug_log("Planner retry decision:")
    _debug_log(json.dumps(decision, indent=2))
    return decision


def _llm_retry_replan(
    payload: dict,
    capabilities: dict,
    rejection_reason: str,
    previous_decision: dict | None,
    attempt: int,
):
    parsed = _llm_prompt_json(
        _build_replan_retry_prompt(payload, capabilities, rejection_reason, previous_decision, attempt),
        debug_label="planner replan retry",
    )
    if parsed is None:
        return None
    decision = _parse_replan_decision(parsed)
    if decision is None:
        _debug_log("Parsed planner replan retry JSON missing valid fields.")
        return None
    _debug_log("Planner replan retry decision:")
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

def _agent_family(name: str) -> str:
    if name.startswith("sql_runner"):
        return "sql_runner"
    if name.startswith("slurm_runner"):
        return "slurm_runner"
    return name


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


SHELL_COMMAND_OVERRIDE_MARKERS = (
    "top-level directories",
    "top level directories",
    "database files",
    "how many lines are in",
    "how many times does the string",
    "how many times does the word",
    "yaml spec filenames",
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


def _is_compound_presentation_clause(task: str) -> bool:
    task_lc = str(task or "").lower().strip()
    if not task_lc:
        return False
    return bool(
        re.match(r"^(?:report|tell|give|provide|show)\b", task_lc)
        and any(token in task_lc for token in ("answer", "count", "counts", "difference", "summary", "include"))
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
            "planning_hints": _agent_planning_hints(agent),
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
                    "planning_hints": _planning_hint_dict(method),
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


def _current_request_from_context(question: str) -> str:
    match = re.search(r"(?:^|\n)Current user request:\s*(.+)\s*$", question, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return question
    current = match.group(1).strip()
    return current or question


def _sanitize_task_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip(" ,;:."))
    if not compact:
        return ""
    compact = re.sub(r"^(?:and|then|also)\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+(?:and|then|also)\s*$", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip(" ,;:.")
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


def _coerce_llm_presentation(presentation: dict | None) -> dict:
    normalized = {
        "task": "Answer the user request directly using clean Markdown.",
        "format": "markdown",
        "audience": "openwebui",
        "include_context": True,
        "include_internal_steps": False,
    }
    if not isinstance(presentation, dict):
        return normalized

    allowed_formats = {"markdown", "markdown_table", "json", "bullets", "plain"}
    task = presentation.get("task")
    if isinstance(task, str) and task.strip():
        normalized["task"] = task.strip()
    fmt = presentation.get("format")
    if isinstance(fmt, str) and fmt.strip().lower() in allowed_formats:
        normalized["format"] = fmt.strip().lower()
    audience = presentation.get("audience")
    if isinstance(audience, str) and audience.strip():
        normalized["audience"] = audience.strip()
    for key in ("include_context", "include_internal_steps"):
        value = presentation.get(key)
        if isinstance(value, bool):
            normalized[key] = value
    return normalized


def _coerce_llm_task_shape(raw: Any) -> str:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in TASK_SHAPES:
            return normalized
    return "lookup"


def _leaf_plan_steps(steps: list[dict]) -> list[dict]:
    leaves: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        nested = step.get("steps")
        if isinstance(nested, list) and nested:
            leaves.extend(_leaf_plan_steps(nested))
        else:
            leaves.append(step)
    return leaves


def _llm_plan_with_retries(question: str, capabilities: dict) -> dict[str, Any] | None:
    max_attempts = _planner_llm_max_attempts()
    previous_decision: dict[str, Any] | None = None
    last_reason = "Planner LLM response could not be parsed into a valid decision."

    for attempt in range(1, max_attempts + 1):
        decision = (
            _llm_decide(question, capabilities)
            if attempt == 1
            else _llm_retry_decide(question, capabilities, last_reason, previous_decision, attempt)
        )
        if decision is None:
            previous_decision = None
            continue
        if not decision.get("processable"):
            return {
                "decision": decision,
                "steps": [],
                "attempt": attempt,
                "validation_reason": decision.get("reason") or "",
            }

        normalized, validation = _assess_llm_candidate_plan(question, decision.get("steps", []), capabilities)
        if validation.get("valid"):
            accepted = dict(decision)
            accepted["steps"] = normalized
            return {
                "decision": accepted,
                "steps": normalized,
                "attempt": attempt,
                "validation_reason": validation.get("reason") or "",
            }

        last_reason = str(validation.get("reason") or "Planner steps failed structural validation.")
        previous_decision = dict(decision)
        if normalized:
            previous_decision["steps"] = normalized
        _debug_log(
            "Planner rejected iterative LLM candidate plan: "
            + json.dumps(
                {
                    "task": question,
                    "attempt": attempt,
                    "reason": last_reason,
                    "steps": previous_decision.get("steps", []),
                },
                ensure_ascii=True,
            )
        )

    return {
        "decision": previous_decision,
        "steps": [],
        "attempt": max_attempts,
        "validation_reason": last_reason,
    }


def _llm_replan_with_retries(payload: dict, capabilities: dict) -> dict[str, Any] | None:
    max_attempts = _planner_llm_max_attempts()
    previous_decision: dict[str, Any] | None = None
    last_reason = "Planner replan response could not be parsed into a valid decision."
    task = str(payload.get("task") or "")

    for attempt in range(1, max_attempts + 1):
        decision = (
            _llm_replan(payload, capabilities)
            if attempt == 1
            else _llm_retry_replan(payload, capabilities, last_reason, previous_decision, attempt)
        )
        if decision is None:
            previous_decision = None
            continue

        normalized, validation = _assess_llm_candidate_plan(
            task,
            decision.get("steps", []),
            capabilities,
            replan_payload=payload,
        )
        if validation.get("valid"):
            accepted = dict(decision)
            accepted["steps"] = normalized
            return {
                "decision": accepted,
                "steps": normalized,
                "attempt": attempt,
                "validation_reason": validation.get("reason") or "",
            }

        last_reason = str(validation.get("reason") or "Planner replan failed structural validation.")
        previous_decision = dict(decision)
        if normalized:
            previous_decision["steps"] = normalized
        _debug_log(
            "Planner rejected iterative LLM replan candidate: "
            + json.dumps(
                {
                    "task": task,
                    "attempt": attempt,
                    "reason": last_reason,
                    "steps": previous_decision.get("steps", []),
                },
                ensure_ascii=True,
            )
        )

    return {
        "decision": previous_decision,
        "steps": [],
        "attempt": max_attempts,
        "validation_reason": last_reason,
    }


def _llm_validate_plan_semantics(
    question: str,
    steps: list[dict],
    capabilities: dict,
    *,
    replan_payload: dict | None = None,
) -> dict[str, Any]:
    parsed = _llm_prompt_json(
        _build_plan_validation_prompt(
            question,
            steps,
            capabilities,
            replan_payload=replan_payload,
        ),
        debug_label="planner validator",
    )
    validation = _parse_plan_validation(parsed)
    if validation is None:
        return {
            "valid": False,
            "reason": "Semantic plan validator could not produce a valid judgment.",
            "goal_coverage": "incorrect",
            "decomposition": "collapsed",
            "user_action_alignment": "weak",
            "issues": ["validator_parse_failure"],
            "rewarded_paths": [],
            "disallowed_paths": [],
        }
    return validation


def _normalize_llm_candidate_steps(steps: list[dict], capabilities: dict) -> list[dict]:
    valid_names = _agent_names_with_task_plan(capabilities)
    normalized: list[dict] = []

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        nested_steps = step.get("steps")
        if isinstance(nested_steps, list) and nested_steps:
            normalized_nested = _normalize_llm_candidate_steps(nested_steps, capabilities)
            if not normalized_nested:
                continue
            group_id = str(step.get("id") or f"step{index}").strip()
            task = _sanitize_task_text(str(step.get("task") or "").strip())
            normalized.append(
                {
                    "id": group_id or f"step{index}",
                    "task": task or f"group {index}",
                    "steps": normalized_nested,
                }
            )
            continue

        step_id = str(step.get("id") or f"step{index}").strip()
        target_agent = str(step.get("target_agent") or "").strip()
        task = _sanitize_task_text(str(step.get("task") or "").strip())
        instruction = step.get("instruction") if isinstance(step.get("instruction"), dict) else None
        if not step_id or not target_agent or not task or not instruction:
            continue
        if target_agent in {"ops_planner", "synthesizer"} or target_agent not in valid_names:
            continue
        normalized_instruction = dict(instruction)
        if target_agent == "shell_runner":
            normalized_instruction = _normalize_followup_shell_instruction(normalized_instruction)
        normalized_step: dict[str, Any] = {
            "id": step_id,
            "target_agent": target_agent,
            "task": task,
            "instruction": normalized_instruction,
        }
        depends_on = step.get("depends_on")
        if isinstance(depends_on, list):
            cleaned = [str(item).strip() for item in depends_on if str(item).strip()]
            if cleaned:
                normalized_step["depends_on"] = cleaned
        when = step.get("when")
        if isinstance(when, dict) and when:
            normalized_step["when"] = when
        normalized.append(normalized_step)
    return normalized


def _validate_plan_structure(question: str, steps: list[dict], capabilities: dict) -> dict[str, Any]:
    del question
    if not isinstance(steps, list) or not steps:
        return {"valid": False, "reason": "Planner produced no executable steps."}

    leaf_steps = _leaf_plan_steps(steps)
    if not leaf_steps:
        return {"valid": False, "reason": "Planner produced no leaf steps to execute."}

    valid_names = _agent_names_with_task_plan(capabilities)
    seen_ids: set[str] = set()
    available_ids: set[str] = set()

    for step in leaf_steps:
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            return {"valid": False, "reason": "Planner produced a step without an id."}
        if step_id in seen_ids:
            return {"valid": False, "reason": f"Planner produced duplicate step id '{step_id}'."}
        seen_ids.add(step_id)
        available_ids.add(step_id)

        target_agent = str(step.get("target_agent") or "").strip()
        if not target_agent:
            return {"valid": False, "reason": f"Planner step '{step_id}' is missing a target agent."}
        if target_agent in {"ops_planner", "synthesizer"}:
            return {"valid": False, "reason": f"Planner step '{step_id}' targeted a non-executor agent."}
        if target_agent not in valid_names:
            return {"valid": False, "reason": f"Planner step '{step_id}' targeted unknown agent '{target_agent}'."}

        task_text = str(step.get("task") or "").strip()
        if not task_text:
            return {"valid": False, "reason": f"Planner step '{step_id}' is missing a task."}
        if _is_presentation_only_task(task_text) or _is_compound_presentation_clause(task_text):
            return {"valid": False, "reason": f"Planner step '{step_id}' is presentation-only instead of executable."}

        instruction = step.get("instruction") if isinstance(step.get("instruction"), dict) else {}
        operation = str(instruction.get("operation") or "").strip().lower()
        if not operation:
            return {"valid": False, "reason": f"Planner step '{step_id}' is missing an instruction.operation."}

        family = _agent_family(target_agent)
        if family == "sql_runner":
            if operation not in {"inspect_schema", "query_from_request", "execute_sql", "sample_rows"}:
                return {"valid": False, "reason": f"Planner step '{step_id}' used unsupported SQL operation '{operation}'."}
            if operation == "query_from_request" and not isinstance(instruction.get("question"), str):
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing SQL question text."}
            if operation == "inspect_schema" and not isinstance(instruction.get("focus"), str):
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing SQL schema focus text."}
            if operation == "execute_sql" and not isinstance(instruction.get("sql"), str):
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing SQL text."}
        elif family == "slurm_runner":
            if operation not in {"query_from_request", "execute_command", "cluster_status", "list_jobs", "job_details"}:
                return {"valid": False, "reason": f"Planner step '{step_id}' used unsupported Slurm operation '{operation}'."}
            if operation in {"query_from_request", "job_details"} and not isinstance(instruction.get("question"), str):
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing Slurm question text."}
            if operation == "execute_command" and "command" not in instruction:
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing Slurm command text."}
        elif target_agent == "shell_runner":
            if operation != "run_command":
                return {"valid": False, "reason": f"Planner step '{step_id}' used unsupported shell operation '{operation}'."}
            if "command" not in instruction:
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing shell command text."}
        elif target_agent == "filesystem":
            if operation != "read_file" or not isinstance(instruction.get("path"), str):
                return {"valid": False, "reason": f"Planner step '{step_id}' is missing a concrete file read instruction."}
        elif target_agent == "notifier":
            if operation != "send_notification":
                return {"valid": False, "reason": f"Planner step '{step_id}' used unsupported notifier operation '{operation}'."}

    for step in leaf_steps:
        step_id = str(step.get("id") or "").strip()
        depends_on = step.get("depends_on")
        if not isinstance(depends_on, list):
            continue
        for dependency in depends_on:
            dependency_id = str(dependency or "").strip()
            if not dependency_id:
                continue
            if dependency_id == step_id:
                return {"valid": False, "reason": f"Planner step '{step_id}' depends on itself."}
            if dependency_id not in available_ids:
                return {"valid": False, "reason": f"Planner step '{step_id}' depends on unknown step '{dependency_id}'."}

    return {"valid": True, "reason": "Plan passed structural validation."}


def _assess_llm_candidate_plan(
    question: str,
    steps: list[dict],
    capabilities: dict,
    *,
    replan_payload: dict | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    if not isinstance(steps, list) or not steps:
        return [], {"valid": False, "reason": "Planner produced no executable steps."}

    normalized = _normalize_llm_candidate_steps(steps, capabilities)
    if not normalized:
        return [], {"valid": False, "reason": "Planner produced no structurally usable steps."}

    structure_validation = _validate_plan_structure(question, normalized, capabilities)
    if not structure_validation.get("valid"):
        return normalized, structure_validation

    semantic_validation = _llm_validate_plan_semantics(
        question,
        normalized,
        capabilities,
        replan_payload=replan_payload,
    )
    return normalized, semantic_validation


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


def _plan_progress_payload(steps: list, presentation: dict, *, task_shape: str):
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
        "task_shape": _coerce_llm_task_shape(task_shape),
    }
    return payload


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
        planning = None
        try:
            planning = _llm_replan_with_retries(req.payload, CAPABILITIES)
            decision = planning.get("decision") if isinstance(planning, dict) else None
            if decision is not None and decision["replace_step_id"] == replace_step_id:
                steps = planning.get("steps", []) if isinstance(planning, dict) else []
        except Exception as exc:
            _debug_log(f"Planner replan failed. Error: {type(exc).__name__}: {exc}")
        if not steps:
            return failure_result(
                (
                    planning.get("validation_reason")
                    if isinstance(planning, dict) and planning.get("validation_reason")
                    else str(req.payload.get("reason") or "Planner could not further decompose the failed step.")
                ),
                error=req.payload.get("error") or req.payload.get("reason"),
            )
        emits = []
        if "plan.progress" in allowed_events:
            emits.append(
                {
                    "event": "plan.progress",
                    "payload": _plan_progress_payload(
                        steps,
                        req.payload.get("presentation", {}),
                        task_shape=str(req.payload.get("task_shape") or "lookup"),
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
        planning = _llm_plan_with_retries(question, CAPABILITIES)
        decision = planning.get("decision") if isinstance(planning, dict) else None
        if decision is not None:
            if decision["processable"]:
                steps = planning.get("steps", []) if isinstance(planning, dict) else []
                if not steps:
                    _debug_log(
                        "Planner exhausted iterative LLM attempts without a valid executable plan: "
                        + json.dumps(
                            {
                                "task": current_question,
                                "attempt": planning.get("attempt") if isinstance(planning, dict) else None,
                                "reason": planning.get("validation_reason") if isinstance(planning, dict) else None,
                            },
                            ensure_ascii=True,
                        )
                    )
                if steps and "task.plan" in allowed_events:
                    payload = {"task": current_question}
                    payload["steps"] = steps
                    presentation = _coerce_llm_presentation(decision.get("presentation"))
                    payload["presentation"] = presentation
                    payload["task_shape"] = _coerce_llm_task_shape(decision.get("task_shape"))
                    if len(steps) == 1:
                        payload["target_agent"] = steps[0]["target_agent"]
                    _debug_log("Planner steps:")
                    _debug_log(json.dumps(steps, indent=2))
                    emits = []
                    if "plan.progress" in allowed_events:
                        emits.append(
                            {
                                "event": "plan.progress",
                                "payload": _plan_progress_payload(
                                    steps,
                                    payload.get("presentation", {}),
                                    task_shape=payload.get("task_shape", "lookup"),
                                ),
                            }
                        )
                    emits.append({"event": "task.plan", "payload": payload})
                    return emit_sequence(emits)
                if "task.result" in allowed_events:
                    return task_result(
                        planning.get("validation_reason") if isinstance(planning, dict) and planning.get("validation_reason")
                        else "Planner could not produce a valid executable step sequence."
                    )
                return noop()
            if "task.result" in allowed_events:
                reason = decision["reason"] or "No matching capability found."
                return task_result(reason)
            return noop()
        _debug_log("LLM planner decision was invalid.")
    except Exception as exc:
        _debug_log(f"LLM planning failed. Error: {type(exc).__name__}: {exc}")

    if "task.result" in allowed_events:
        return task_result(
            "Planner could not determine if the request is processable. Check LLM connectivity/response format and retry."
        )
    return noop()
