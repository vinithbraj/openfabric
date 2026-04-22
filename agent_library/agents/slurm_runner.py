import json
import os
import re
import shlex
import subprocess
import time
from typing import Any

import requests
from web_compat import FastAPI

from agent_library.common import EventRequest, EventResponse, shared_llm_api_settings, task_plan_context, with_node_envelope
from agent_library.reduction import (
    build_slurm_reduction_request,
    deterministic_elapsed_reducer_command as shared_deterministic_elapsed_reducer_command,
    deterministic_line_count_reducer_command as shared_deterministic_line_count_reducer_command,
    deterministic_node_inventory_reducer_command as shared_deterministic_node_inventory_reducer_command,
    deterministic_state_breakdown_reducer_command as shared_deterministic_state_breakdown_reducer_command,
    execute_reduction_request,
    generate_shell_reduction_command,
    looks_like_elapsed_summary_task,
    looks_like_node_inventory_summary_task,
)
from agent_library.template import agent_api, agent_descriptor, emit, failure_result, noop
from runtime.console import log_debug, log_raw

app = FastAPI()

SLURM_EXECUTION_MODEL = "deterministic_first_with_llm_fallback"
SLURM_DETERMINISTIC_CATALOG_VERSION = "v4-initial"
SLURM_DETERMINISTIC_PRIMITIVES = [
    {
        "primitive_id": "slurm.cluster.node_inventory_summary",
        "family": "cluster",
        "summary": "Count nodes and summarize node states, optionally within one partition.",
        "required_params": [],
        "optional_params": ["partition_name"],
        "intent_tags": ["nodes", "count", "state", "cluster"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.cluster.node_list",
        "family": "cluster",
        "summary": "List nodes and node states, optionally filtered by partition or state.",
        "required_params": [],
        "optional_params": ["partition_name", "node_states"],
        "intent_tags": ["nodes", "list", "state", "cluster"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.cluster.partition_summary",
        "family": "cluster",
        "summary": "List partitions and their availability summary.",
        "required_params": [],
        "optional_params": ["partition_name"],
        "intent_tags": ["partition", "availability", "cluster"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.queue_list",
        "family": "queue",
        "summary": "List current queued or running jobs with optional user, partition, and state filters.",
        "required_params": [],
        "optional_params": ["user", "partition_name", "job_states"],
        "intent_tags": ["jobs", "queue", "list", "running", "pending"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.queue_count",
        "family": "queue",
        "summary": "Count current queued or running jobs with optional user, partition, and state filters.",
        "required_params": [],
        "optional_params": ["user", "partition_name", "job_states"],
        "intent_tags": ["jobs", "count", "queue", "running", "pending"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.queue_state_breakdown",
        "family": "queue",
        "summary": "Summarize current jobs grouped by state.",
        "required_params": [],
        "optional_params": ["user", "partition_name"],
        "intent_tags": ["jobs", "state", "breakdown", "queue"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.history_list",
        "family": "history",
        "summary": "List historical jobs from sacct with optional user, partition, and state filters.",
        "required_params": [],
        "optional_params": ["user", "partition_name", "job_states"],
        "intent_tags": ["jobs", "history", "failed", "completed"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.history_count",
        "family": "history",
        "summary": "Count historical jobs from sacct with optional user, partition, and state filters.",
        "required_params": [],
        "optional_params": ["user", "partition_name", "job_states"],
        "intent_tags": ["jobs", "history", "count", "failed", "completed"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.elapsed_summary",
        "family": "history",
        "summary": "Summarize elapsed durations for historical jobs from sacct.",
        "required_params": [],
        "optional_params": ["partition_name", "job_states"],
        "intent_tags": ["jobs", "elapsed", "duration", "history"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.details",
        "family": "job",
        "summary": "Show details for one Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "details", "inspect"],
        "read_only": True,
    },
    {
        "primitive_id": "slurm.jobs.cancel",
        "family": "control",
        "summary": "Cancel one Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "cancel", "control"],
        "read_only": False,
    },
    {
        "primitive_id": "slurm.jobs.hold",
        "family": "control",
        "summary": "Hold one Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "hold", "control"],
        "read_only": False,
    },
    {
        "primitive_id": "slurm.jobs.release",
        "family": "control",
        "summary": "Release one held Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "release", "control"],
        "read_only": False,
    },
    {
        "primitive_id": "slurm.jobs.requeue",
        "family": "control",
        "summary": "Requeue one Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "requeue", "control"],
        "read_only": False,
    },
    {
        "primitive_id": "slurm.jobs.resume",
        "family": "control",
        "summary": "Resume one suspended Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "resume", "control"],
        "read_only": False,
    },
    {
        "primitive_id": "slurm.jobs.suspend",
        "family": "control",
        "summary": "Suspend one Slurm job.",
        "required_params": ["job_id"],
        "optional_params": [],
        "intent_tags": ["job", "suspend", "control"],
        "read_only": False,
    },
]
SLURM_DETERMINISTIC_PRIMITIVE_IDS = {
    item["primitive_id"] for item in SLURM_DETERMINISTIC_PRIMITIVES if isinstance(item, dict) and item.get("primitive_id")
}
SLURM_DETERMINISTIC_CATALOG_FAMILIES = sorted(
    {
        str(item.get("family")).strip()
        for item in SLURM_DETERMINISTIC_PRIMITIVES
        if isinstance(item, dict) and str(item.get("family") or "").strip()
    }
)

AGENT_DESCRIPTOR = agent_descriptor(
    name="slurm_runner",
    role="executor",
    description=(
        "Connects to a remote Slurm gateway, executes deterministic Slurm primitives first for "
        "cluster and job operations, and falls back to generated Slurm commands only when the "
        "deterministic path cannot answer cleanly."
    ),
    capability_domains=[
        "slurm",
        "hpc",
        "cluster_operations",
        "job_queue_inspection",
        "scheduler_control",
        "accounting",
        "deterministic_primitives",
    ],
    action_verbs=[
        "inspect",
        "query",
        "list",
        "show",
        "describe",
        "cancel",
        "hold",
        "release",
        "requeue",
        "resume",
        "suspend",
    ],
    execution_model=SLURM_EXECUTION_MODEL,
    deterministic_catalog_version=SLURM_DETERMINISTIC_CATALOG_VERSION,
    deterministic_catalog_families=SLURM_DETERMINISTIC_CATALOG_FAMILIES,
    deterministic_catalog_size=len(SLURM_DETERMINISTIC_PRIMITIVES),
    deterministic_primitives=[item["primitive_id"] for item in SLURM_DETERMINISTIC_PRIMITIVES],
    deterministic_catalog_reference="VERSION_4_PRIMITIVE_CATALOG.md",
    fallback_policy=(
        "Run deterministic Slurm primitive first. If the primitive cannot answer cleanly, "
        "run the selector-provided fallback command or legacy generated Slurm command."
    ),
    side_effect_policy="remote_slurm_operations_via_gateway",
    safety_enforced_by_agent=True,
    planning_hints={
        "keywords": [
            "slurm",
            "scheduler",
            "cluster",
            "hpc",
            "job",
            "jobs",
            "queue",
            "queued",
            "running",
            "pending",
            "partition",
            "partitions",
            "node",
            "nodes",
            "reservation",
            "reservations",
            "fairshare",
            "sacct",
            "squeue",
            "sinfo",
            "scancel",
        ],
        "anti_keywords": ["database schema", "sql table", "repo file search", "docker container"],
        "preferred_task_shapes": ["count", "boolean_check", "list", "compare", "lookup"],
        "instruction_operations": ["query_from_request", "execute_command", "cluster_status", "list_jobs", "job_details"],
        "structured_followup": True,
        "native_count_preferred": True,
        "routing_priority": 45,
    },
    routing_notes=[
        "Use for Slurm, HPC cluster, node, partition, queue, job, reservation, and accounting requests.",
        "This agent talks to a remote Slurm gateway over HTTP and does not run Slurm CLIs locally.",
        "Prefer query_from_request for natural-language requests and execute_command only for explicit Slurm commands.",
        "Natural-language Slurm requests now use deterministic primitives first and only fall back to free-form command generation when needed.",
        "Security is intentionally minimal for now; use only configured gateway endpoints.",
    ],
    apis=[
        agent_api(
            name="select_deterministic_slurm_primitive",
            trigger_event="task.plan",
            emits=["slurm.result", "task.result"],
            summary="Selects one deterministic Slurm primitive plus a fallback command plan for a natural-language scheduler request.",
            when="Selects one deterministic Slurm primitive plus a fallback command plan for a natural-language scheduler request.",
            intent_tags=["slurm_primitive_selection", "cluster_status", "job_queue", "accounting"],
            examples=[
                "show cluster node status",
                "list queued jobs for user vinith",
                "show partition availability",
                "give me failed jobs from yesterday",
            ],
            deterministic=True,
            side_effect_level="read_or_control",
            planning_hints={
                "keywords": ["slurm", "scheduler", "cluster", "partition", "queue", "accounting"],
                "preferred_task_shapes": ["count", "list", "lookup"],
                "instruction_operations": ["query_from_request", "cluster_status", "list_jobs"],
                "native_count_preferred": True,
            },
        ),
        agent_api(
            name="execute_deterministic_slurm_primitive",
            trigger_event="task.plan",
            emits=["slurm.result", "task.result"],
            summary="Executes deterministic Slurm primitives via the gateway before any fallback command generation.",
            when="Executes deterministic Slurm primitives via the gateway before any fallback command generation.",
            intent_tags=["slurm_deterministic_execution", "cluster_status", "job_queue", "accounting"],
            examples=[
                "how many nodes are in my slurm cluster and what is their state",
                "how many pending jobs are there",
                "cancel job 12345",
            ],
            deterministic=True,
            side_effect_level="read_or_control",
            planning_hints={
                "keywords": ["node inventory", "job count", "pending jobs", "failed jobs", "job control"],
                "preferred_task_shapes": ["count", "list", "lookup", "compare"],
                "instruction_operations": ["query_from_request", "cluster_status", "job_details"],
                "native_count_preferred": True,
            },
        ),
        agent_api(
            name="execute_explicit_slurm_command",
            trigger_event="slurm.query",
            emits=["slurm.result", "task.result"],
            summary="Executes an explicit Slurm command via the configured gateway.",
            when="Executes an explicit Slurm command via the configured gateway.",
            intent_tags=["slurm_command", "scheduler_control"],
            examples=[
                "sinfo -Nel",
                "squeue -u vinith",
                "scancel 12345",
            ],
            deterministic=True,
            side_effect_level="read_or_control",
            planning_hints={
                "keywords": ["explicit slurm command", "squeue", "sinfo", "sacct", "scancel"],
                "instruction_operations": ["execute_command"],
            },
        ),
    ],
)
AGENT_METADATA = AGENT_DESCRIPTOR

SLURM_COMMAND_CATALOG = {
    "sinfo": "cluster, partition, and node state",
    "squeue": "queued and running jobs",
    "sacct": "historical accounting data via slurmdbd",
    "scontrol": "show, hold, release, requeue, update, suspend, and resume objects",
    "scancel": "cancel or signal Slurm jobs",
    "sreport": "aggregated accounting reports via slurmdbd",
    "sshare": "fairshare and association usage",
    "sacctmgr": "accounting associations, users, and accounts",
    "sdiag": "scheduler diagnostics",
    "sprio": "job priority inspection",
    "sstat": "live job step statistics",
}

SLURM_QUERY_PREFIXES = ("sinfo", "squeue", "sacct", "sreport", "sshare", "sdiag", "sprio", "sstat")
SLURM_CONTROL_PREFIXES = ("scancel", "scontrol", "sacctmgr")
SLURM_QUEUE_STATE_ALIASES = {
    "pending": "PENDING",
    "queued": "PENDING",
    "waiting": "PENDING",
    "running": "RUNNING",
    "active": "RUNNING",
    "completing": "COMPLETING",
    "configuring": "CONFIGURING",
    "failed": "FAILED",
    "failure": "FAILED",
    "completed": "COMPLETED",
    "complete": "COMPLETED",
    "finished": "COMPLETED",
    "cancelled": "CANCELLED",
    "canceled": "CANCELLED",
    "timeout": "TIMEOUT",
    "timed_out": "TIMEOUT",
    "suspended": "SUSPENDED",
    "suspend": "SUSPENDED",
}
SLURM_NODE_STATE_ALIASES = {
    "idle": "idle",
    "mixed": "mixed",
    "mix": "mixed",
    "allocated": "allocated",
    "alloc": "allocated",
    "down": "down",
    "drain": "drain",
    "drained": "drain",
    "draining": "drain",
    "unknown": "unknown",
}


def _debug_enabled() -> bool:
    return os.getenv("SLURM_AGENT_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str):
    if _debug_enabled():
        log_debug("SLURM_AGENT_DEBUG", message)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _gateway_host() -> str:
    return os.getenv("SLURM_GATEWAY_HOST", "127.0.0.1")


def _gateway_port() -> int:
    raw = os.getenv("SLURM_GATEWAY_PORT", "8312")
    try:
        return int(raw)
    except ValueError:
        return 8312


def _gateway_scheme() -> str:
    scheme = os.getenv("SLURM_GATEWAY_SCHEME", "http").strip().lower()
    return scheme if scheme in {"http", "https"} else "http"


def _gateway_base_url() -> str:
    explicit = os.getenv("SLURM_GATEWAY_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return f"{_gateway_scheme()}://{_gateway_host()}:{_gateway_port()}"


def _gateway_timeout_seconds() -> float:
    raw = os.getenv("SLURM_GATEWAY_TIMEOUT_SECONDS", "120")
    try:
        return max(1.0, min(float(raw), 600.0))
    except ValueError:
        return 120.0


def _llm_timeout_seconds() -> float:
    raw = os.getenv("LLM_OPS_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, min(float(raw), 600.0))
    except ValueError:
        return 300.0


def _is_slurm_task(task: str) -> bool:
    task_lc = task.lower()
    if any(
        token in task_lc
        for token in (
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
            "cluster",
            "reservation",
            "reservations",
            "fairshare",
            "qos",
            "accounting",
            "scancel",
            "hpc",
            "scheduler",
        )
    ):
        return True
    if "gpu" in task_lc and any(token in task_lc for token in ("partition", "node", "nodes", "cluster", "slurm", "sinfo", "squeue", "job", "jobs", "hpc")):
        return True
    if "gpu" in task_lc and any(token in task_lc for token in ("availability", "available", "capacity", "allocat", "utilization", "free")):
        return True
    if "worker" in task_lc and any(token in task_lc for token in ("cluster", "slurm", "node", "nodes", "hpc")):
        return True
    if "compute" in task_lc and any(token in task_lc for token in ("cluster", "slurm", "node", "nodes", "partition", "hpc")):
        return True
    return False


def _slurm_gateway_ready() -> bool:
    try:
        response = requests.get(f"{_gateway_base_url()}/health", timeout=min(_gateway_timeout_seconds(), 5.0))
        response.raise_for_status()
        payload = response.json()
        return bool(payload.get("ok"))
    except Exception as exc:
        _debug_log(f"Gateway health check failed: {type(exc).__name__}: {exc}")
        return False


def _parse_command_input(command: Any, args: Any = None) -> tuple[str, list[str]]:
    if isinstance(command, str) and command.strip():
        parts = shlex.split(command.strip())
        if not parts:
            raise RuntimeError("No Slurm command provided.")
        parsed_args = parts[1:]
        if isinstance(args, list) and args:
            parsed_args.extend(str(item) for item in args)
        return parts[0], parsed_args
    if isinstance(command, dict):
        raw_command = command.get("command")
        raw_args = command.get("args")
        if isinstance(raw_command, str) and raw_command.strip():
            parts = shlex.split(raw_command.strip())
            if not parts:
                raise RuntimeError("No Slurm command provided.")
            parsed_args = parts[1:]
            if isinstance(raw_args, list) and raw_args:
                parsed_args.extend(str(item) for item in raw_args)
            if isinstance(args, list) and args:
                parsed_args.extend(str(item) for item in args)
            return parts[0], parsed_args
    if isinstance(command, list) and command:
        parsed_args = [str(item) for item in command[1:]]
        if isinstance(args, list) and args:
            parsed_args.extend(str(item) for item in args)
        return str(command[0]), parsed_args
    if isinstance(command, str):
        raise RuntimeError("No Slurm command provided.")
    if isinstance(args, list) and args:
        raise RuntimeError("Slurm command args were provided without a command.")
    raise RuntimeError("No Slurm command provided.")


def _command_kind(command: str) -> str:
    if command in SLURM_QUERY_PREFIXES:
        return "query"
    if command in SLURM_CONTROL_PREFIXES:
        return "control"
    return "unknown"


def _gateway_execute(command: str, args: list[str]) -> dict[str, Any]:
    response = requests.post(
        f"{_gateway_base_url()}/execute",
        headers={"Content-Type": "application/json"},
        json={"command": command, "args": args},
        timeout=_gateway_timeout_seconds(),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Slurm gateway returned an invalid response.")
    return payload


def _slurm_command_catalog_text() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in SLURM_COMMAND_CATALOG.items())


def _slurm_catalog_prompt_text() -> str:
    lines = []
    for primitive in SLURM_DETERMINISTIC_PRIMITIVES:
        primitive_id = str(primitive.get("primitive_id") or "").strip()
        if not primitive_id:
            continue
        summary = str(primitive.get("summary") or "").strip()
        required_params = primitive.get("required_params") if isinstance(primitive.get("required_params"), list) else []
        optional_params = primitive.get("optional_params") if isinstance(primitive.get("optional_params"), list) else []
        lines.append(
            f"- {primitive_id}: {summary} "
            f"required=[{', '.join(str(item) for item in required_params) or 'none'}] "
            f"optional=[{', '.join(str(item) for item in optional_params) or 'none'}]"
        )
    return "\n".join(lines)


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _extract_user_name(task: str) -> str:
    text = str(task or "").strip()
    patterns = (
        r"\bfor\s+user\s+['\"]?([A-Za-z0-9._-]+)['\"]?\b",
        r"\buser\s+['\"]?([A-Za-z0-9._-]+)['\"]?\b",
        r"\bowned\s+by\s+['\"]?([A-Za-z0-9._-]+)['\"]?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_job_id(task: str) -> str:
    text = str(task or "").strip()
    patterns = (
        r"\bjob(?:\s+id)?\s*[:#]?\s*([0-9]+)\b",
        r"\b(?:scancel|requeue|resume|suspend|hold|release)\s+([0-9]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _normalize_job_state(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    return SLURM_QUEUE_STATE_ALIASES.get(text, text.upper())


def _normalize_job_states(value: Any) -> list[str]:
    items = value if isinstance(value, list) else []
    states: list[str] = []
    seen: set[str] = set()
    for item in items:
        state = _normalize_job_state(item)
        if not state or state in seen:
            continue
        seen.add(state)
        states.append(state)
    return states


def _normalize_node_state(value: Any) -> str:
    text = str(value or "").strip().lower().rstrip("*")
    if not text:
        return ""
    return SLURM_NODE_STATE_ALIASES.get(text, text)


def _normalize_node_states(value: Any) -> list[str]:
    items = value if isinstance(value, list) else []
    states: list[str] = []
    seen: set[str] = set()
    for item in items:
        state = _normalize_node_state(item)
        if not state or state in seen:
            continue
        seen.add(state)
        states.append(state)
    return states


def _extract_requested_job_states(task: str) -> list[str]:
    text = str(task or "").lower()
    matches: list[str] = []
    for token, normalized in SLURM_QUEUE_STATE_ALIASES.items():
        if token.replace("_", " ") in text:
            matches.append(normalized)
    seen: set[str] = set()
    result: list[str] = []
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_requested_node_states(task: str) -> list[str]:
    text = str(task or "").lower()
    matches: list[str] = []
    for token, normalized in SLURM_NODE_STATE_ALIASES.items():
        if token in text:
            matches.append(normalized)
    seen: set[str] = set()
    result: list[str] = []
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_partition_name(partition_name: Any, context: dict[str, Any]) -> str:
    raw = str(partition_name or "").strip()
    if not raw:
        return ""
    partitions = context.get("partitions") if isinstance(context.get("partitions"), list) else []
    for item in partitions:
        if str(item).strip() == raw:
            return str(item).strip()
    for item in partitions:
        if str(item).strip().lower() == raw.lower():
            return str(item).strip()
    return raw


def _looks_like_job_count_task(task: str) -> bool:
    text = str(task or "").lower()
    asks_for_count = any(token in text for token in ("how many", "count", "number of", "total jobs"))
    mentions_jobs = any(token in text for token in ("job", "jobs", "queue", "queued", "pending", "running", "failed", "completed"))
    return asks_for_count and mentions_jobs


def _looks_like_job_state_breakdown_task(task: str) -> bool:
    text = str(task or "").lower()
    mentions_jobs = any(token in text for token in ("job", "jobs", "queue"))
    asks_state = any(token in text for token in ("state", "states", "status", "statuses", "breakdown"))
    return mentions_jobs and asks_state and not _looks_like_node_inventory_summary_task(task)


def _looks_like_job_history_task(task: str) -> bool:
    text = str(task or "").lower()
    return any(token in text for token in ("history", "historical", "yesterday", "failed", "completed", "finished", "past"))


def _heuristic_slurm_selection(task: str, context: dict[str, Any]) -> dict[str, Any] | None:
    text = str(task or "").strip().lower()
    if not text:
        return None

    partition_name = _normalize_partition_name(_extract_partition_name(task), context)
    user = _extract_user_name(task)
    job_id = _extract_job_id(task)
    job_states = _extract_requested_job_states(task)
    node_states = _extract_requested_node_states(task)

    if _looks_like_node_inventory_summary_task(task):
        return {
            "primitive_id": "slurm.cluster.node_inventory_summary",
            "selection_reason": "Heuristic node inventory summary match.",
            "parameters": {"partition_name": partition_name} if partition_name else {},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if any(token in text for token in ("list nodes", "show nodes", "node names", "node list")):
        parameters: dict[str, Any] = {}
        if partition_name:
            parameters["partition_name"] = partition_name
        if node_states:
            parameters["node_states"] = node_states
        return {
            "primitive_id": "slurm.cluster.node_list",
            "selection_reason": "Heuristic node list match.",
            "parameters": parameters,
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if any(token in text for token in ("partition", "partitions", "availability", "available resources")) and not any(
        token in text for token in ("job", "jobs")
    ):
        return {
            "primitive_id": "slurm.cluster.partition_summary",
            "selection_reason": "Heuristic partition summary match.",
            "parameters": {"partition_name": partition_name} if partition_name else {},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and any(token in text for token in ("cancel", "scancel")):
        return {
            "primitive_id": "slurm.jobs.cancel",
            "selection_reason": "Heuristic job cancel match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and "hold" in text:
        return {
            "primitive_id": "slurm.jobs.hold",
            "selection_reason": "Heuristic job hold match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and "release" in text:
        return {
            "primitive_id": "slurm.jobs.release",
            "selection_reason": "Heuristic job release match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and "requeue" in text:
        return {
            "primitive_id": "slurm.jobs.requeue",
            "selection_reason": "Heuristic job requeue match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and "resume" in text:
        return {
            "primitive_id": "slurm.jobs.resume",
            "selection_reason": "Heuristic job resume match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and "suspend" in text:
        return {
            "primitive_id": "slurm.jobs.suspend",
            "selection_reason": "Heuristic job suspend match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if job_id and any(token in text for token in ("details", "detail", "inspect", "show job")):
        return {
            "primitive_id": "slurm.jobs.details",
            "selection_reason": "Heuristic job details match.",
            "parameters": {"job_id": job_id},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if _looks_like_elapsed_summary_task(task):
        parameters = {}
        if partition_name:
            parameters["partition_name"] = partition_name
        if job_states:
            parameters["job_states"] = job_states
        return {
            "primitive_id": "slurm.jobs.elapsed_summary",
            "selection_reason": "Heuristic elapsed summary match.",
            "parameters": parameters,
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if _looks_like_job_count_task(task):
        parameters = {}
        if user:
            parameters["user"] = user
        if partition_name:
            parameters["partition_name"] = partition_name
        if job_states:
            parameters["job_states"] = job_states
        primitive_id = "slurm.jobs.history_count" if _looks_like_job_history_task(task) else "slurm.jobs.queue_count"
        return {
            "primitive_id": primitive_id,
            "selection_reason": "Heuristic job count match.",
            "parameters": parameters,
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if _looks_like_job_state_breakdown_task(task):
        parameters = {}
        if user:
            parameters["user"] = user
        if partition_name:
            parameters["partition_name"] = partition_name
        return {
            "primitive_id": "slurm.jobs.queue_state_breakdown",
            "selection_reason": "Heuristic job state breakdown match.",
            "parameters": parameters,
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    if any(token in text for token in ("job", "jobs", "queue", "queued", "pending", "running")):
        parameters = {}
        if user:
            parameters["user"] = user
        if partition_name:
            parameters["partition_name"] = partition_name
        if job_states:
            parameters["job_states"] = job_states
        primitive_id = "slurm.jobs.history_list" if _looks_like_job_history_task(task) else "slurm.jobs.queue_list"
        return {
            "primitive_id": primitive_id,
            "selection_reason": "Heuristic job list match.",
            "parameters": parameters,
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    return None


def _llm_select_slurm_strategy(task: str, context: dict[str, Any]) -> dict[str, Any]:
    api_key, base_url, timeout_seconds, model = _llm_api_settings()
    prompt = (
        "You are the deterministic Slurm primitive selector for a scheduler agent.\n"
        "Choose exactly one deterministic primitive from the executable primitive catalog below.\n"
        "Also provide a fallback Slurm command that would answer the same user request if deterministic execution cannot answer cleanly.\n"
        "Return STRICT JSON only with exactly these top-level keys:\n"
        "{\"primitive_id\":\"...\",\"selection_reason\":\"...\",\"parameters\":{...},\"fallback_command\":\"...\",\"fallback_args\":[...],\"fallback_reason\":\"...\"}\n"
        "Rules:\n"
        "- primitive_id MUST be one of the executable primitive IDs listed below, or fallback_only.\n"
        "- Prefer the simplest primitive that fully answers the request.\n"
        "- If the request is outside the executable primitives, use primitive_id=fallback_only.\n"
        "- fallback_command must be a single Slurm binary or an empty string if no fallback can be proposed.\n"
        "- fallback_args must be a JSON array of strings.\n"
        "- parameters must only include fields that help execute the chosen primitive, such as partition_name, user, job_id, job_states, or node_states.\n"
        "Executable deterministic primitives:\n"
        f"{_slurm_catalog_prompt_text()}\n"
        "Cluster context:\n"
        f"- Partitions: {context.get('partitions', [])}\n"
        f"- Node states: {context.get('node_states', {})}\n"
        f"- Allowed commands: {context.get('allowed_commands', [])}\n"
        "Fallback command catalog:\n"
        f"{_slurm_command_catalog_text()}\n"
        f"User request: {task}"
    )
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SLURM_SELECTOR_LLM_RAW", content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_slurm_selection(selection: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(selection, dict):
        return {
            "primitive_id": "fallback_only",
            "selection_reason": "",
            "parameters": {},
            "fallback_command": "",
            "fallback_args": [],
            "fallback_reason": "",
        }
    primitive_id = str(selection.get("primitive_id") or "").strip()
    if primitive_id not in SLURM_DETERMINISTIC_PRIMITIVE_IDS:
        primitive_id = "fallback_only"
    raw_parameters = selection.get("parameters") if isinstance(selection.get("parameters"), dict) else {}
    parameters: dict[str, Any] = {}
    job_id = str(raw_parameters.get("job_id") or "").strip()
    if job_id:
        parameters["job_id"] = job_id
    user = str(raw_parameters.get("user") or "").strip()
    if user:
        parameters["user"] = user
    partition_name = _normalize_partition_name(raw_parameters.get("partition_name"), context)
    if partition_name:
        parameters["partition_name"] = partition_name
    job_states = _normalize_job_states(raw_parameters.get("job_states"))
    if job_states:
        parameters["job_states"] = job_states
    node_states = _normalize_node_states(raw_parameters.get("node_states"))
    if node_states:
        parameters["node_states"] = node_states
    fallback_command = str(selection.get("fallback_command") or "").strip()
    fallback_args = [str(item) for item in selection.get("fallback_args", [])] if isinstance(selection.get("fallback_args"), list) else []
    return {
        "primitive_id": primitive_id,
        "selection_reason": str(selection.get("selection_reason") or "").strip(),
        "parameters": parameters,
        "fallback_command": fallback_command,
        "fallback_args": fallback_args,
        "fallback_reason": str(selection.get("fallback_reason") or "").strip(),
    }


def _fallback_command_from_selection(selection: dict[str, Any]) -> tuple[str, list[str]] | None:
    command = str(selection.get("fallback_command") or "").strip()
    args = selection.get("fallback_args") if isinstance(selection.get("fallback_args"), list) else []
    if not command:
        return None
    return command, [str(item) for item in args]


def _llm_api_settings() -> tuple[str, str, float, str]:
    return shared_llm_api_settings("gpt-4o-mini", timeout_seconds=_llm_timeout_seconds())


def _extract_partition_name(task: str) -> str:
    text = str(task or "").strip()
    patterns = (
        r"\bin\s+['\"]?([A-Za-z0-9._-]+)['\"]?\s+partition\b",
        r"\bfor\s+['\"]?([A-Za-z0-9._-]+)['\"]?\s+partition\b",
        r"\bpartition\s+named\s+['\"]?([A-Za-z0-9._-]+)['\"]?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _looks_like_elapsed_summary_task(task: str) -> bool:
    return looks_like_elapsed_summary_task(task)


def _looks_like_node_inventory_summary_task(task: str) -> bool:
    return looks_like_node_inventory_summary_task(task)


def _deterministic_slurm_command(task: str) -> dict[str, Any] | None:
    if _looks_like_node_inventory_summary_task(task):
        args = ["-N", "-h", "-o", "%N|%T"]
        partition = _extract_partition_name(task)
        if partition:
            args = ["-p", partition, *args]
        return {
            "command": "sinfo",
            "args": args,
            "reason": "Using deterministic sinfo query for Slurm node count and state summary.",
        }
    if not _looks_like_elapsed_summary_task(task):
        return None
    args = ["-X", "-P", "--format=JobID,JobName,State,Elapsed,End"]
    partition = _extract_partition_name(task)
    if partition:
        args.append(f"--partition={partition}")
    task_lc = str(task or "").lower()
    if any(token in task_lc for token in ("complete", "completed")):
        args.append("--state=COMPLETED")
    elif "failed" in task_lc:
        args.append("--state=FAILED")
    elif "cancel" in task_lc:
        args.append("--state=CANCELLED")
    return {
        "command": "sacct",
        "args": args,
        "reason": "Using deterministic sacct accounting query for an elapsed-time summary request.",
    }


def _is_usage_error(stderr: str) -> bool:
    if not isinstance(stderr, str):
        return False
    stderr_lc = stderr.lower()
    return any(
        term in stderr_lc
        for term in (
            "unrecognized option",
            "invalid option",
            "usage:",
            "invalid argument",
            "illegal option",
            "not a valid",
            "invalid value",
        )
    )


def _llm_repair_slurm_command(task: str, failed_command: str, error_message: str, help_text: str = "") -> dict[str, Any]:
    api_key, base_url, timeout_seconds, model = _llm_api_settings()

    prompt = (
        "You are an expert Slurm administrator. A previously generated Slurm command failed due to a syntax or usage error.\n"
        "Your task is to fix the command based on the error message and (if available) the command's help documentation.\n\n"

        f"User Task: {task}\n"
        f"Failed Command: {failed_command}\n"
        f"Error Message: {error_message}\n\n"
    )

    if help_text:
        prompt += f"Command Help Output:\n```\n{help_text[:5000]}\n```\n\n"

    prompt += (
        "Instructions:\n"
        "- Generate a corrected version of the command that achieves the user's task and resolves the error.\n"
        "- Return STRICT JSON only: {\"command\":\"...\",\"args\":[...],\"reason\":\"...\"}.\n"
        "- Ensure all flags are valid for the current Slurm version based on the help output.\n"
        "Corrected Command JSON:"
    )

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a Slurm command repair expert. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SLURM_REPAIR_LLM_RAW", content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("Could not parse Slurm repair JSON.")
    return json.loads(content[start : end + 1])

# NEW: Slurm context cache
_SLURM_CONTEXT_CACHE: dict[str, Any] = {}
_SLURM_CONTEXT_TS: float = 0.0
_SLURM_CONTEXT_TTL = 60


def _get_slurm_context() -> dict[str, Any]:
    global _SLURM_CONTEXT_CACHE, _SLURM_CONTEXT_TS

    now = time.time()
    if _SLURM_CONTEXT_CACHE and (now - _SLURM_CONTEXT_TS) < _SLURM_CONTEXT_TTL:
        return _SLURM_CONTEXT_CACHE

    context: dict[str, Any] = {}

    try:
        res = _gateway_execute("sinfo", ["-h", "-o", "%P|%t|%D|%G"])
        parts = set()
        for line in res.get("stdout", "").splitlines():
            cols = line.split("|")
            if cols:
                parts.add(cols[0])
        context["partitions"] = sorted(parts)
    except Exception:
        context["partitions"] = []

    try:
        res = _gateway_execute("sinfo", ["-h", "-o", "%t|%D"])
        states = {}
        for line in res.get("stdout", "").splitlines():
            cols = line.split("|")
            if len(cols) == 2:
                state, count = cols
                try:
                    states[state] = states.get(state, 0) + int(count)
                except:
                    pass
        context["node_states"] = states
    except Exception:
        context["node_states"] = {}

    try:
        r = requests.get(f"{_gateway_base_url()}/health", timeout=5)
        if r.ok:
            context["allowed_commands"] = r.json().get("allowed_commands", [])
        else:
            context["allowed_commands"] = []
    except Exception:
        context["allowed_commands"] = []

    _SLURM_CONTEXT_CACHE = context
    _SLURM_CONTEXT_TS = now

    return context


def _build_squeue_args(parameters: dict[str, Any], *, include_partition_column: bool = True) -> list[str]:
    format_string = "%i|%u|%T|%P|%j" if include_partition_column else "%i|%u|%T|%j"
    args = ["-h", "-o", format_string]
    user = str(parameters.get("user") or "").strip()
    if user:
        args.extend(["-u", user])
    partition_name = str(parameters.get("partition_name") or "").strip()
    if partition_name:
        args.extend(["-p", partition_name])
    job_states = parameters.get("job_states") if isinstance(parameters.get("job_states"), list) else []
    if job_states:
        args.extend(["-t", ",".join(str(item) for item in job_states)])
    return args


def _build_sacct_args(parameters: dict[str, Any]) -> list[str]:
    args = ["-X", "-P", "-n", "--format=JobID,JobName,User,State,Partition,Elapsed,End"]
    user = str(parameters.get("user") or "").strip()
    if user:
        args.extend(["-u", user])
    partition_name = str(parameters.get("partition_name") or "").strip()
    if partition_name:
        args.append(f"--partition={partition_name}")
    job_states = parameters.get("job_states") if isinstance(parameters.get("job_states"), list) else []
    if job_states:
        args.append(f"--state={','.join(str(item) for item in job_states)}")
    return args


def _build_deterministic_slurm_plan(selection: dict[str, Any], task: str, context: dict[str, Any]) -> dict[str, Any] | None:
    primitive_id = str(selection.get("primitive_id") or "").strip()
    params = selection.get("parameters") if isinstance(selection.get("parameters"), dict) else {}
    if primitive_id == "slurm.cluster.node_inventory_summary":
        command = _deterministic_slurm_command(task)
        if command is None:
            args = ["-N", "-h", "-o", "%N|%T"]
            partition_name = _normalize_partition_name(params.get("partition_name"), context)
            if partition_name:
                args = ["-p", partition_name, *args]
            command = {
                "command": "sinfo",
                "args": args,
                "reason": "Using deterministic node inventory summary primitive.",
            }
        return {"primitive_id": primitive_id, **command}
    if primitive_id == "slurm.cluster.node_list":
        args = ["-N", "-h", "-o", "%N|%T|%P"]
        partition_name = _normalize_partition_name(params.get("partition_name"), context)
        if partition_name:
            args = ["-p", partition_name, *args]
        node_states = params.get("node_states") if isinstance(params.get("node_states"), list) else []
        if node_states:
            args.extend(["-t", ",".join(str(item) for item in node_states)])
        return {
            "primitive_id": primitive_id,
            "command": "sinfo",
            "args": args,
            "reason": "Using deterministic node list primitive.",
        }
    if primitive_id == "slurm.cluster.partition_summary":
        args = ["-h", "-o", "%P|%a|%l|%D|%t|%N"]
        partition_name = _normalize_partition_name(params.get("partition_name"), context)
        if partition_name:
            args = ["-p", partition_name, *args]
        return {
            "primitive_id": primitive_id,
            "command": "sinfo",
            "args": args,
            "reason": "Using deterministic partition summary primitive.",
        }
    if primitive_id == "slurm.jobs.queue_list":
        return {
            "primitive_id": primitive_id,
            "command": "squeue",
            "args": _build_squeue_args(params),
            "reason": "Using deterministic current queue listing primitive.",
        }
    if primitive_id == "slurm.jobs.queue_count":
        return {
            "primitive_id": primitive_id,
            "command": "squeue",
            "args": _build_squeue_args(params),
            "reason": "Using deterministic current queue count primitive.",
        }
    if primitive_id == "slurm.jobs.queue_state_breakdown":
        return {
            "primitive_id": primitive_id,
            "command": "squeue",
            "args": _build_squeue_args(params),
            "reason": "Using deterministic current queue state breakdown primitive.",
        }
    if primitive_id == "slurm.jobs.history_list":
        return {
            "primitive_id": primitive_id,
            "command": "sacct",
            "args": _build_sacct_args(params),
            "reason": "Using deterministic historical job listing primitive.",
        }
    if primitive_id == "slurm.jobs.history_count":
        return {
            "primitive_id": primitive_id,
            "command": "sacct",
            "args": _build_sacct_args(params),
            "reason": "Using deterministic historical job count primitive.",
        }
    if primitive_id == "slurm.jobs.elapsed_summary":
        command = _deterministic_slurm_command(task)
        if command is None:
            args = _build_sacct_args(params)
            if "--format=JobID,JobName,User,State,Partition,Elapsed,End" in args:
                args[args.index("--format=JobID,JobName,User,State,Partition,Elapsed,End")] = (
                    "--format=JobID,JobName,State,Elapsed,End"
                )
            return {
                "primitive_id": primitive_id,
                "command": "sacct",
                "args": args,
                "reason": "Using deterministic elapsed summary primitive.",
            }
        return {"primitive_id": primitive_id, **command}
    if primitive_id == "slurm.jobs.details":
        job_id = str(params.get("job_id") or "").strip()
        if not job_id:
            return None
        return {
            "primitive_id": primitive_id,
            "command": "scontrol",
            "args": ["show", "job", job_id],
            "reason": "Using deterministic job details primitive.",
        }
    if primitive_id == "slurm.jobs.cancel":
        job_id = str(params.get("job_id") or "").strip()
        if not job_id:
            return None
        return {
            "primitive_id": primitive_id,
            "command": "scancel",
            "args": [job_id],
            "reason": "Using deterministic job cancel primitive.",
        }
    if primitive_id in {"slurm.jobs.hold", "slurm.jobs.release", "slurm.jobs.requeue", "slurm.jobs.resume", "slurm.jobs.suspend"}:
        job_id = str(params.get("job_id") or "").strip()
        if not job_id:
            return None
        action = primitive_id.rsplit(".", 1)[-1]
        return {
            "primitive_id": primitive_id,
            "command": "scontrol",
            "args": [action, job_id],
            "reason": f"Using deterministic job {action} primitive.",
        }
    return None


def _deterministic_line_count_reducer_command(label: str) -> str:
    return shared_deterministic_line_count_reducer_command(label)


def _deterministic_state_breakdown_reducer_command(state_index: int) -> str:
    return shared_deterministic_state_breakdown_reducer_command(state_index)


def _selection_requires_refined_answer(primitive_id: str) -> bool:
    return primitive_id in {
        "slurm.cluster.node_inventory_summary",
        "slurm.jobs.queue_count",
        "slurm.jobs.history_count",
        "slurm.jobs.queue_state_breakdown",
        "slurm.jobs.elapsed_summary",
    }


def _refine_deterministic_slurm_result(
    task: str,
    primitive_id: str,
    command: str,
    stdout: str,
    stderr: str,
) -> tuple[str, str]:
    reduction_request = build_slurm_reduction_request(
        task,
        command,
        stdout,
        stderr,
        primitive_id=primitive_id,
        natural_language_request=True,
    )
    if not isinstance(reduction_request, dict):
        return "", ""
    reduction = execute_reduction_request(reduction_request, stdout)
    reduced_result = reduction.reduced_result if isinstance(reduction.reduced_result, str) else ""
    return reduced_result, reduction.local_reduction_command


def _llm_slurm_command(task: str, *, allow_deterministic: bool = True) -> dict[str, Any]:
    if allow_deterministic:
        deterministic = _deterministic_slurm_command(task)
        if deterministic is not None:
            return deterministic
    api_key, base_url, timeout_seconds, model = _llm_api_settings()

    context = _get_slurm_context()

    prompt = (
        "You are an expert Slurm administrator. Your task is to generate exactly one valid Slurm CLI command "
        "for a remote execution gateway based on a natural-language request.\n\n"

        "Return STRICT JSON only in the form {\"command\":\"...\",\"args\":[...],\"reason\":\"...\"}.\n\n"

        "Cluster Context (Current State):\n"
        f"- Partitions: {context.get('partitions')}\n"
        f"- Node states: {context.get('node_states')}\n"
        f"- Allowed commands: {context.get('allowed_commands')}\n\n"

        "====================\n"
        "COMMAND GENERATION RULES\n"
        "====================\n"
        "Follow these rules precisely to ensure command compatibility across Slurm versions:\n\n"

        "1. CURRENT QUEUE (squeue):\n"
        "- Use for 'queued', 'pending', 'running', 'active', or 'current' jobs.\n"
        "- Flags:\n"
        "  * -h: Suppress headers (preferred for data extraction).\n"
        "  * -t <STATE>: Filter by state (PENDING, RUNNING, COMPLETING, etc.).\n"
        "  * -u <USER>: Filter by specific user.\n"
        "  * -p <PARTITION>: Filter by partition.\n"
        "- If querying all jobs across all users, DO NOT include -u.\n\n"

        "2. HISTORICAL DATA (sacct):\n"
        "- Use for 'finished', 'completed', 'failed', 'yesterday', 'past', 'history', or 'how long' requests.\n"
        "- ALWAYS include -X (show only one line per job) and -P (pipe-separated / parsable).\n"
        "- ALWAYS use --partition=<NAME> for partition filtering in sacct. DO NOT USE -p (it is ambiguous for parsable).\n"
        "- Use --format=JobID,JobName,State,Elapsed,End for duration/timing questions.\n"
        "- Use --state=COMPLETED|FAILED|TIMEOUT to filter results.\n\n"

        "3. AGGREGATION & COUNTS:\n"
        "- For 'how many' or 'count' questions, generate the command that returns exactly the list of matching objects.\n"
        "- Example: 'how many non-running jobs' -> squeue -h -t PENDING,CONFIGURING,STOPPED,SUSPENDED\n"
        "- Example: 'how many failed jobs' -> sacct -X -P --state=FAILED\n"
        "- The synthesizer will perform the final count on the lines returned.\n\n"

        "4. CLUSTER & NODE STATE (sinfo):\n"
        "- Use for 'node status', 'partition info', 'available resources', or 'cluster health'.\n"
        "- Use -Nel for detailed node/partition output.\n"
        "- Use -p <PARTITION> for partition info.\n\n"

        "5. JOB CONTROL (scancel, scontrol):\n"
        "- Use scancel <job_id> to cancel jobs.\n"
        "- Use scontrol show job <job_id> for deep inspection of a specific job.\n\n"

        "====================\n"
        "CRITICAL CONSTRAINTS\n"
        "====================\n"
        "- NEVER use shell operators (|, >, &, ;, etc.). The gateway only executes single binaries.\n"
        "- NEVER use '*' or 'all' as argument values.\n"
        "- If the request is for 'non-running' jobs in squeue, use -t with all states EXCEPT RUNNING (e.g., -t PD,CF,S,ST).\n"
        "- ALWAYS USE LONG FORM FLAGS (e.g. --partition=) if there is any doubt about short flag meaning.\n"

        "====================\n"
        "EXAMPLES\n"
        "====================\n"
        "Q: how many pending jobs are there\n"
        "A: {\"command\":\"squeue\",\"args\":[\"-h\",\"-t\",\"PENDING\"],\"reason\":\"Listing pending jobs to be counted.\"}\n\n"

        "Q: how long did the totalseg jobs take\n"
        "A: {\"command\":\"sacct\",\"args\":[\"-X\",\"-P\",\"--state=COMPLETED\",\"--partition=totalseg\",\"--format=JobID,JobName,State,Elapsed,End\"],\"reason\":\"Querying historical job database for completed jobs in the totalseg partition.\"}\n\n"

        "Q: show failed jobs from yesterday\n"
        "A: {\"command\":\"sacct\",\"args\":[\"-X\",\"-P\",\"--state=FAILED\",\"--starttime=yesterday\"],\"reason\":\"Querying historical job database for failed jobs.\"}\n\n"
        
        "Q: status of nodes in the 'gpu' partition\n"
        "A: {\"command\":\"sinfo\",\"args\":[\"-p\",\"gpu\"],\"reason\":\"Checking status of gpu partition.\"}\n\n"

        "Q: cancel job 5521\n"
        "A: {\"command\":\"scancel\",\"args\":[\"5521\"],\"reason\":\"Cancelling specific job ID.\"}\n\n"

        f"User request: {task}"
    )

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    log_raw("SLURM_LLM_RAW", content)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Could not parse Slurm command JSON.")
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Could not parse Slurm command JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Could not parse Slurm command JSON.")
    return parsed


def _instruction_to_command(instruction: dict[str, Any], task: str) -> tuple[str, list[str]]:
    operation = str(instruction.get("operation") or "").strip()
    if operation == "execute_command":
        return _parse_command_input(instruction.get("command"), instruction.get("args"))
    if operation == "cluster_status":
        return "sinfo", ["-Nel"]
    if operation == "list_jobs":
        args = ["-a"]
        user = instruction.get("user")
        if isinstance(user, str) and user.strip():
            args.extend(["-u", user.strip()])
        return "squeue", args
    if operation == "job_details":
        job_id = str(instruction.get("job_id") or "").strip()
        if not job_id:
            raise RuntimeError("job_details requires job_id.")
        return "scontrol", ["show", "job", job_id]
    if operation == "query_from_request":
        raw = _llm_slurm_command(str(instruction.get("question") or task))
        command = raw.get("command")
        args = raw.get("args", [])
        if not isinstance(command, str) or not command.strip():
            raise RuntimeError("Could not generate a Slurm command.")
        if not isinstance(args, list):
            args = []
        return command.strip(), [str(item) for item in args]
    raise RuntimeError(f"Unsupported Slurm operation: {operation or 'unknown'}")


def _format_detail(command: str, returncode: int) -> str:
    kind = _command_kind(command)
    if returncode == 0:
        if kind == "control":
            return "Slurm control command executed."
        return "Slurm query executed."
    return "Slurm command failed."


def _llm_generate_local_command(
    task: str,
    command: str,
    sample_stdout: str,
    previous_command: str = "",
    previous_error: str = "",
) -> str:
    try:
        return generate_shell_reduction_command(
            task,
            command,
            sample_stdout,
            previous_command,
            previous_error,
        )
    except Exception as exc:
        _debug_log(f"Local command generation failed: {exc}")
        return ""


def _deterministic_elapsed_reducer_command(task: str) -> str:
    return shared_deterministic_elapsed_reducer_command(task)


def _deterministic_node_inventory_reducer_command(task: str) -> str:
    return shared_deterministic_node_inventory_reducer_command(task)


def _deterministic_reduce_slurm_result(task: str, command: str, stdout: str) -> tuple[str, str]:
    primitive_id = ""
    if _looks_like_node_inventory_summary_task(task):
        primitive_id = "slurm.cluster.node_inventory_summary"
    elif _looks_like_elapsed_summary_task(task):
        primitive_id = "slurm.jobs.elapsed_summary"
    reduction_request = build_slurm_reduction_request(
        task,
        command,
        stdout,
        "",
        primitive_id=primitive_id,
        natural_language_request=True,
    )
    if not isinstance(reduction_request, dict):
        return "", ""
    reduction = execute_reduction_request(reduction_request, stdout)
    reduced_result = reduction.reduced_result if isinstance(reduction.reduced_result, str) else ""
    return reduced_result, reduction.local_reduction_command


def _llm_process_result(task: str, command: str, stdout: str, stderr: str) -> tuple[str, str]:
    reduction_request = build_slurm_reduction_request(
        task,
        command,
        stdout,
        stderr,
        natural_language_request=True,
    )
    if not isinstance(reduction_request, dict):
        return "", ""
    input_data = stdout if reduction_request.get("source_field") != "stderr" else stderr
    reduction = execute_reduction_request(reduction_request, input_data)
    reduced_result = reduction.reduced_result if isinstance(reduction.reduced_result, str) else ""
    return reduced_result, reduction.local_reduction_command


def _result_payload(
    command: str,
    args: list[str],
    gateway_result: dict[str, Any],
    stats: dict[str, float],
    refined_answer: str = "",
    local_reduction_command: str = "",
    reduction_request: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    *,
    execution_strategy: str = "",
) -> dict[str, Any]:
    returncode = int(gateway_result.get("returncode", 1))
    stdout = str(gateway_result.get("stdout", "") or "")
    stderr = str(gateway_result.get("stderr", "") or "")
    duration_ms = gateway_result.get("duration_ms")
    if isinstance(duration_ms, (int, float)):
        stats["gateway_command_ms"] = round(float(duration_ms), 2)
    result = {
        "command": command,
        "args": args,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "ok": returncode == 0,
        "kind": _command_kind(command),
    }
    if refined_answer:
        result["refined_answer"] = refined_answer
        result["reduced_result"] = refined_answer

    payload = {
        "detail": refined_answer or _format_detail(command, returncode),
        "command": " ".join([command, *args]).strip(),
        "local_reduction_command": local_reduction_command or None,
        "reduced_result": refined_answer or None,
        "stats": stats,
        "result": result,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "execution_strategy": execution_strategy or None,
        "deterministic_primitive": selection.get("primitive_id") if isinstance(selection, dict) else None,
        "deterministic_selection_reason": selection.get("selection_reason") if isinstance(selection, dict) else None,
        "fallback_command": (
            " ".join([str(selection.get("fallback_command") or "").strip(), *selection.get("fallback_args", [])]).strip()
            if isinstance(selection, dict) and str(selection.get("fallback_command") or "").strip()
            else None
        ),
        "fallback_used": execution_strategy in {"selector_fallback_command", "legacy_llm_fallback_command"},
    }
    if isinstance(reduction_request, dict):
        payload["reduction_request"] = reduction_request
    return payload


@app.post("/handle", response_model=EventResponse)
@with_node_envelope("slurm_runner", "executor")
def handle_event(req: EventRequest):
    natural_language_request = False
    if req.event == "slurm.query":
        task = str(req.payload.get("question") or req.payload.get("query") or req.payload.get("command") or "")
        instruction = {
            "operation": "execute_command" if req.payload.get("command") else "query_from_request",
            "command": req.payload.get("command"),
            "args": req.payload.get("args"),
            "question": task,
        }
        classification_task = task
        natural_language_request = not bool(req.payload.get("command"))
    elif req.event == "task.plan":
        plan_context = task_plan_context(req.payload)
        classification_task = plan_context.classification_task
        explicitly_targeted = plan_context.targets("slurm_runner")
        if not classification_task or (not explicitly_targeted and not _is_slurm_task(classification_task)):
            return noop()
        instruction = req.payload.get("instruction") if isinstance(req.payload.get("instruction"), dict) else {}
        task = instruction.get("question") if isinstance(instruction.get("question"), str) else plan_context.execution_task
        natural_language_request = True
    else:
        return noop()

    if not _slurm_gateway_ready():
        return failure_result(
            f"Slurm gateway is unavailable at {_gateway_base_url()}.",
            error=f"Slurm gateway is unavailable at {_gateway_base_url()}.",
        )

    started = time.perf_counter()
    stats: dict[str, float] = {}
    selection: dict[str, Any] | None = None
    execution_strategy = "explicit_command" if str(instruction.get("operation") or "").strip() == "execute_command" else "agent_operation"

    if str(instruction.get("operation") or "").strip() == "query_from_request":
        context = _get_slurm_context()
        selector_started = time.perf_counter()
        raw_selection = _heuristic_slurm_selection(task, context)
        if raw_selection is None:
            raw_selection = _llm_select_slurm_strategy(task, context)
        stats["deterministic_selection_ms"] = _elapsed_ms(selector_started)
        selection = _normalize_slurm_selection(raw_selection, context)

        deterministic_plan = None
        if selection.get("primitive_id") and selection.get("primitive_id") != "fallback_only":
            deterministic_plan = _build_deterministic_slurm_plan(selection, task, context)

        if isinstance(deterministic_plan, dict):
            try:
                command = str(deterministic_plan.get("command") or "").strip()
                args = [str(item) for item in deterministic_plan.get("args", [])] if isinstance(deterministic_plan.get("args"), list) else []
                if command:
                    gateway_started = time.perf_counter()
                    gateway_result = _gateway_execute(command, args)
                    stats["deterministic_gateway_roundtrip_ms"] = _elapsed_ms(gateway_started)
                    reduction_request = None
                    if gateway_result.get("returncode") == 0:
                        reduction_request = build_slurm_reduction_request(
                            task,
                            " ".join([command, *args]).strip(),
                            str(gateway_result.get("stdout", "") or ""),
                            str(gateway_result.get("stderr", "") or ""),
                            primitive_id=str(selection.get("primitive_id") or ""),
                            natural_language_request=natural_language_request,
                        )
                    if gateway_result.get("returncode") == 0:
                        stats["total_ms"] = _elapsed_ms(started)
                        payload = _result_payload(
                            command,
                            args,
                            gateway_result,
                            stats,
                            reduction_request=reduction_request,
                            selection=selection,
                            execution_strategy="deterministic",
                        )
                        return emit("slurm.result", payload)
            except Exception as exc:
                _debug_log(f"Deterministic Slurm primitive failed: {type(exc).__name__}: {exc}")

        fallback = _fallback_command_from_selection(selection)
        if fallback is not None:
            execution_strategy = "selector_fallback_command"
            instruction = {
                "operation": "execute_command",
                "command": fallback[0],
                "args": fallback[1],
                "question": task,
            }
        else:
            fallback_started = time.perf_counter()
            raw_fallback = _llm_slurm_command(task, allow_deterministic=False)
            stats["fallback_command_planning_ms"] = _elapsed_ms(fallback_started)
            execution_strategy = "legacy_llm_fallback_command"
            instruction = {
                "operation": "execute_command",
                "command": raw_fallback.get("command"),
                "args": raw_fallback.get("args"),
                "question": task,
            }

    for attempt in range(2):
        try:
            command_started = time.perf_counter()
            command, args = _instruction_to_command(instruction, task)
            stats[f"command_planning_ms_try{attempt+1}"] = _elapsed_ms(command_started)
            
            gateway_started = time.perf_counter()
            gateway_result = _gateway_execute(command, args)
            stats[f"gateway_roundtrip_ms_try{attempt+1}"] = _elapsed_ms(gateway_started)
            
            # Check for usage errors that might be repairable
            if gateway_result.get("returncode", 0) != 0:
                stderr = gateway_result.get("stderr", "")
                if _is_usage_error(stderr) and attempt == 0:
                    _debug_log(f"Detected usage error, attempting repair: {stderr.strip()}")
                    
                    # Try to fetch help context for repair
                    help_text = ""
                    try:
                        help_res = _gateway_execute(command, ["--help"])
                        help_text = help_res.get("stdout", "") or help_res.get("stderr", "")
                    except Exception as e:
                        _debug_log(f"Could not fetch help text for repair: {e}")
                    
                    # Repair the command via LLM
                    repaired = _llm_repair_slurm_command(task, f"{command} {' '.join(args)}", stderr, help_text)
                    
                    # Update instruction for next attempt
                    instruction = {
                        "operation": "execute_command",
                        "command": repaired.get("command"),
                        "args": repaired.get("args"),
                        "question": task
                    }
                    continue

            # If we reach here, we either succeeded or failed in a non-repairable way (or out of retries)
            stats["total_ms"] = _elapsed_ms(started)

            reduction_request = None
            if gateway_result.get("returncode") == 0 and natural_language_request and task:
                reduction_request = build_slurm_reduction_request(
                    task,
                    " ".join([command, *args]),
                    str(gateway_result.get("stdout", "") or ""),
                    str(gateway_result.get("stderr", "") or ""),
                    primitive_id=str(selection.get("primitive_id") or "") if isinstance(selection, dict) else "",
                    natural_language_request=natural_language_request,
                )

            payload = _result_payload(
                command,
                args,
                gateway_result,
                stats,
                reduction_request=reduction_request,
                selection=selection,
                execution_strategy=execution_strategy,
            )
            if payload["returncode"] != 0:
                return failure_result(
                    f"Slurm command failed: {payload['stderr'] or payload['stdout'] or payload['command']}",
                    error=payload["stderr"] or payload["stdout"] or payload["command"],
                    result={"ok": False, "stats": stats, "slurm": payload["result"]},
                )
            return emit("slurm.result", payload)

        except Exception as exc:
            if attempt == 0:
                _debug_log(f"Slurm attempt 1 failed: {exc}. Retrying...")
                continue
            
            stats["total_ms"] = _elapsed_ms(started)
            _debug_log(f"Slurm task failed after retries: {type(exc).__name__}: {exc}")
            return failure_result(
                f"Slurm task failed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
                result={"ok": False, "stats": stats} if stats else None,
            )
