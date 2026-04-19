import json
import os
import shlex
import time
from typing import Any

import requests
from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse, task_plan_context
from runtime.console import log_debug, log_raw

app = FastAPI()

AGENT_METADATA = {
    "description": (
        "Connects to a remote Slurm gateway, inspects cluster and job state, runs read-only "
        "Slurm CLI queries, and performs explicit Slurm control operations on user request."
    ),
    "capability_domains": [
        "slurm",
        "hpc",
        "cluster_operations",
        "job_queue_inspection",
        "scheduler_control",
        "accounting",
    ],
    "action_verbs": [
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
    "side_effect_policy": "remote_slurm_operations_via_gateway",
    "safety_enforced_by_agent": True,
    "routing_notes": [
        "Use for Slurm, HPC cluster, node, partition, queue, job, reservation, and accounting requests.",
        "This agent talks to a remote Slurm gateway over HTTP and does not run Slurm CLIs locally.",
        "Prefer query_from_request for natural-language requests and execute_command only for explicit Slurm commands.",
        "Security is intentionally minimal for now; use only configured gateway endpoints.",
    ],
    "methods": [
        {
            "name": "query_slurm_from_request",
            "event": "task.plan",
            "when": "Generates one Slurm CLI command from a natural-language scheduler request and executes it via the gateway.",
            "intent_tags": ["slurm_query", "cluster_status", "job_queue", "accounting"],
            "examples": [
                "show cluster node status",
                "list queued jobs for user vinith",
                "show partition availability",
                "give me failed jobs from yesterday",
            ],
        },
        {
            "name": "execute_explicit_slurm_command",
            "event": "slurm.query",
            "when": "Executes an explicit Slurm command via the configured gateway.",
            "intent_tags": ["slurm_command", "scheduler_control"],
            "examples": [
                "sinfo -Nel",
                "squeue -u vinith",
                "scancel 12345",
            ],
        },
    ],
}

SLURM_COMMAND_CATALOG = {
    "sinfo": "cluster, partition, and node state",
    "squeue": "queued and running jobs",
    "sacct": "historical accounting data via slurmdbd",
    "scontrol": "show, hold, release, requeue, update, suspend, and resume objects",
    "sreport": "aggregated accounting reports via slurmdbd",
    "sshare": "fairshare and association usage",
    "sacctmgr": "accounting associations, users, and accounts",
    "sdiag": "scheduler diagnostics",
    "sprio": "job priority inspection",
    "sstat": "live job step statistics",
}

SLURM_QUERY_PREFIXES = ("sinfo", "squeue", "sacct", "sreport", "sshare", "sdiag", "sprio", "sstat")
SLURM_CONTROL_PREFIXES = ("scancel", "scontrol", "sacctmgr")


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
    raw = os.getenv("SLURM_GATEWAY_PORT", "8310")
    try:
        return int(raw)
    except ValueError:
        return 8310


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
    return any(
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
        )
    )


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
        return parts[0], parts[1:]
    if isinstance(command, dict):
        raw_command = command.get("command")
        raw_args = command.get("args")
        if isinstance(raw_command, str) and raw_command.strip():
            parsed_args = [str(item) for item in raw_args] if isinstance(raw_args, list) else []
            return raw_command.strip(), parsed_args
    if isinstance(command, list) and command:
        return str(command[0]), [str(item) for item in command[1:]]
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


def _llm_api_settings() -> tuple[str, str, float, str]:
    api_key = os.getenv("LLM_OPS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_OPS_BASE_URL")
    model = os.getenv("LLM_OPS_MODEL")
    timeout_seconds = _llm_timeout_seconds()
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
    return api_key, base_url.rstrip("/"), timeout_seconds, model

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


def _llm_slurm_command(task: str) -> dict[str, Any]:
    api_key, base_url, timeout_seconds, model = _llm_api_settings()

    context = _get_slurm_context()

    prompt = (
        "You generate exactly one valid Slurm CLI command for a remote execution gateway.\n"
        "Return STRICT JSON only in the form {\"command\":\"...\",\"args\":[...],\"reason\":\"...\"}.\n\n"

        "Cluster context:\n"
        f"- Partitions: {context.get('partitions')}\n"
        f"- Node states: {context.get('node_states')}\n"
        f"- Allowed commands: {context.get('allowed_commands')}\n\n"

        "====================\n"
        "INTENT → COMMAND RULES\n"
        "====================\n"

        "1. CURRENT QUEUE / LIVE JOBS:\n"
        "- Use squeue\n"
        "- Examples: queued jobs, pending jobs, running jobs, jobs right now\n"
        "- Use filters:\n"
        "  * pending → -t PENDING\n"
        "  * running → -t RUNNING\n"
        "  * specific user → -u <user>\n"
        "- If querying ALL users → DO NOT include -u\n\n"

        "2. HISTORICAL / COMPLETED JOBS:\n"
        "- Use sacct\n"
        "- ALWAYS include -X and -P\n"
        "- Examples: completed jobs, failed jobs, past runs\n"
        "- For time/duration questions → use:\n"
        "  --format=JobID,Elapsed,State\n\n"

        "3. CLUSTER / NODE / PARTITION STATE:\n"
        "- Use sinfo\n"
        "- Examples: node status, partition availability\n\n"

        "====================\n"
        "CRITICAL RULES\n"
        "====================\n"
        "- NEVER use '*' or 'all' as argument values\n"
        "- NEVER invent flags\n"
        "- NEVER use shell syntax (pipes, redirects, etc.)\n"
        "- Prefer minimal correct command\n\n"

        "====================\n"
        "SPECIAL HANDLING\n"
        "====================\n"

        "COUNT / HOW MANY:\n"
        "- Generate the command that RETURNS the matching rows\n"
        "- DO NOT attempt to count in the command\n"
        "- The system will count the rows\n\n"

        "TIME / DURATION QUESTIONS:\n"
        "- Use sacct (NOT squeue)\n"
        "- Include Elapsed field\n"
        "- Example fields:\n"
        "  --format=JobID,JobName,Elapsed,State\n\n"

        "====================\n"
        "EXAMPLES (FOLLOW EXACTLY)\n"
        "====================\n"

        "Q: how many jobs are pending\n"
        "A:\n"
        "{\"command\":\"squeue\",\"args\":[\"-h\",\"-t\",\"PENDING\"]}\n\n"

        "Q: how many jobs are running\n"
        "A:\n"
        "{\"command\":\"squeue\",\"args\":[\"-h\",\"-t\",\"RUNNING\"]}\n\n"

        "Q: show all queued jobs\n"
        "A:\n"
        "{\"command\":\"squeue\",\"args\":[\"-h\"]}\n\n"

        "Q: how long did jobs take to complete\n"
        "A:\n"
        "{\"command\":\"sacct\",\"args\":[\"-X\",\"-P\",\"--format=JobID,JobName,Elapsed,State\"]}\n\n"

        "Q: show completed jobs\n"
        "A:\n"
        "{\"command\":\"sacct\",\"args\":[\"-X\",\"-P\",\"--state=COMPLETED\"]}\n\n"

        "Q: show failed jobs\n"
        "A:\n"
        "{\"command\":\"sacct\",\"args\":[\"-X\",\"-P\",\"--state=FAILED\"]}\n\n"

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


def _result_payload(command: str, args: list[str], gateway_result: dict[str, Any], stats: dict[str, float]) -> dict[str, Any]:
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
    return {
        "detail": _format_detail(command, returncode),
        "command": " ".join([command, *args]).strip(),
        "stats": stats,
        "result": result,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
    }


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "slurm.query":
        task = str(req.payload.get("question") or req.payload.get("query") or req.payload.get("command") or "")
        instruction = {
            "operation": "execute_command" if req.payload.get("command") else "query_from_request",
            "command": req.payload.get("command"),
            "args": req.payload.get("args"),
            "question": task,
        }
        classification_task = task
    elif req.event == "task.plan":
        plan_context = task_plan_context(req.payload)
        classification_task = plan_context.classification_task
        explicitly_targeted = plan_context.targets("slurm_runner")
        if not classification_task or (not explicitly_targeted and not _is_slurm_task(classification_task)):
            return {"emits": []}
        instruction = req.payload.get("instruction") if isinstance(req.payload.get("instruction"), dict) else {}
        task = instruction.get("question") if isinstance(instruction.get("question"), str) else plan_context.execution_task
    else:
        return {"emits": []}

    if not _slurm_gateway_ready():
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": f"Slurm gateway is unavailable at {_gateway_base_url()}.",
                        "status": "failed",
                        "error": f"Slurm gateway is unavailable at {_gateway_base_url()}.",
                        "result": None,
                    },
                }
            ]
        }

    started = time.perf_counter()
    stats: dict[str, float] = {}
    try:
        command_started = time.perf_counter()
        command, args = _instruction_to_command(instruction, task)
        stats["command_planning_ms"] = _elapsed_ms(command_started)
        gateway_started = time.perf_counter()
        gateway_result = _gateway_execute(command, args)
        stats["gateway_roundtrip_ms"] = _elapsed_ms(gateway_started)
        stats["total_ms"] = _elapsed_ms(started)
        payload = _result_payload(command, args, gateway_result, stats)
        if payload["returncode"] != 0:
            return {
                "emits": [
                    {
                        "event": "task.result",
                        "payload": {
                            "detail": f"Slurm command failed: {payload['stderr'] or payload['stdout'] or payload['command']}",
                            "status": "failed",
                            "error": payload["stderr"] or payload["stdout"] or payload["command"],
                            "result": {"ok": False, "stats": stats, "slurm": payload["result"]},
                        },
                    }
                ]
            }
        return {"emits": [{"event": "slurm.result", "payload": payload}]}
    except Exception as exc:
        stats["total_ms"] = _elapsed_ms(started)
        _debug_log(f"Slurm task failed: {type(exc).__name__}: {exc}")
        return {
            "emits": [
                {
                    "event": "task.result",
                    "payload": {
                        "detail": f"Slurm task failed: {type(exc).__name__}: {exc}",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "result": {"ok": False, "stats": stats} if stats else None,
                    },
                }
            ]
        }
