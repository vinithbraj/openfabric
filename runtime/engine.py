import atexit
import copy
import json
import os
import re
import socket
import subprocess
import sys
import time
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any
from importlib import import_module
from urllib.parse import urlparse

import requests

from agent_library.contracts import metadata_api_emit_events, metadata_api_trigger_events, normalize_agent_metadata

from .console import log_boot, log_event, log_event_handler
from .contracts import ContractRegistry
from .event_bus import EventBus
from .graph import build_agent_graph_node, build_capability_graph, build_workflow_graph
from .run_inspector import render_workflow_graph_mermaid
from .registry import ADAPTER_REGISTRY
from .run_store import RunStore


class Engine:

    DEFAULT_WORKFLOW_LIMITS = {
        "max_attempts": 3,
        "max_replans": 1,
        "max_step_replans": 1,
        "max_uncertain_attempts": 1,
        "max_validation_llm_calls": 1,
        "max_step_validation_llm_calls": 1,
    }
    DEFAULT_EXECUTION_POLICY = {
        "allow_python_package_installs": True,
        "python_package_install_tool": "python -m pip",
        "install_scope": "current_python_environment",
    }

    def __init__(self, spec: dict, global_timeout_seconds: float | None = None):
        self.spec = copy.deepcopy(spec)
        self.global_timeout_seconds = global_timeout_seconds
        self.contracts = ContractRegistry(spec["contracts"])
        self.bus = EventBus()
        self.agents = {}
        self._managed_processes = []
        self._shutdown_registered = False
        self.run_store = RunStore()

    def setup(self):
        self._autostart_http_services()

        # Instantiate agents
        for name, config in self.spec["agents"].items():
            runtime_cfg = self._effective_runtime_config(config.get("runtime", {}))
            adapter_type = runtime_cfg["adapter"]

            if adapter_type not in ADAPTER_REGISTRY:
                raise ValueError(f"Unknown adapter type: {adapter_type}")

            adapter_cls = ADAPTER_REGISTRY[adapter_type]
            adapter = adapter_cls(runtime_cfg)

            self.agents[name] = {
                "adapter": adapter,
                "subscribes_to": config.get("subscribes_to", [])
            }

        # Register subscriptions
        for agent_name, agent in self.agents.items():
            for event in agent["subscribes_to"]:
                self.bus.subscribe(event, agent_name)

        self._emit_system_capabilities()

    def _effective_runtime_config(self, runtime_cfg: dict):
        effective = copy.deepcopy(runtime_cfg)
        if self.global_timeout_seconds is not None and effective.get("adapter") == "http":
            effective["timeout_seconds"] = self.global_timeout_seconds
            autostart_cfg = effective.get("autostart")
            if isinstance(autostart_cfg, dict):
                autostart_cfg["timeout_seconds"] = self.global_timeout_seconds
        return effective

    def _emit_system_capabilities(self):
        if "system.capabilities" not in self.spec.get("events", {}):
            return
        agent_catalog = self._build_agent_catalog()
        payload = {
            "agents": agent_catalog,
            "execution_policy": self._build_execution_policy(),
            "graph": build_capability_graph(agent_catalog),
        }
        self.emit("system.capabilities", payload)

    def _build_execution_policy(self):
        allow_python_installs = os.getenv("OPENFABRIC_ALLOW_PYTHON_PACKAGE_INSTALLS", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        install_tool = os.getenv("OPENFABRIC_PYTHON_PACKAGE_INSTALL_TOOL", "python -m pip").strip() or "python -m pip"
        return {
            **self.DEFAULT_EXECUTION_POLICY,
            "allow_python_package_installs": allow_python_installs,
            "python_package_install_tool": install_tool,
        }

    def _build_agent_catalog(self):
        catalog = []
        for agent_name, config in self.spec["agents"].items():
            runtime_cfg = config.get("runtime", {})
            metadata = self._load_agent_metadata(agent_name, runtime_cfg)
            config_metadata = config.get("metadata", {})
            if isinstance(config_metadata, dict):
                for key, value in config_metadata.items():
                    if key == "routing_notes" and isinstance(value, list):
                        existing = metadata.get("routing_notes", [])
                        metadata["routing_notes"] = [
                            *(existing if isinstance(existing, list) else []),
                            *[item for item in value if isinstance(item, str)],
                        ]
                    else:
                        metadata[key] = value
            entry = {
                "subscribes_to": sorted(
                    {
                        *[item for item in config.get("subscribes_to", []) if isinstance(item, str)],
                        *metadata_api_trigger_events(metadata),
                    }
                ),
                "emits": sorted(
                    {
                        *[item for item in config.get("emits", []) if isinstance(item, str)],
                        *metadata_api_emit_events(metadata),
                    }
                ),
                "name": agent_name,
                "description": metadata.get("description", config.get("description", "")),
                "methods": metadata.get("methods", config.get("methods", [])),
                "apis": metadata.get("apis", metadata.get("methods", config.get("methods", []))),
                "routing_notes": metadata.get("routing_notes", []),
                "adapter": runtime_cfg.get("adapter"),
                "endpoint": runtime_cfg.get("endpoint"),
            }
            descriptor_name = metadata.get("name")
            if (
                isinstance(descriptor_name, str)
                and descriptor_name.strip()
                and descriptor_name.strip() != agent_name
                and not metadata.get("template_agent")
            ):
                entry["template_agent"] = descriptor_name.strip()
            for key, value in metadata.items():
                if key in {"description", "methods", "routing_notes", "name"}:
                    continue
                entry[key] = value
            entry["graph_node"] = build_agent_graph_node(agent_name, config, metadata)
            catalog.append(entry)
        return catalog

    def _load_agent_metadata(self, agent_name: str, runtime_cfg: dict):
        autostart_cfg = runtime_cfg.get("autostart", {})
        app_ref = autostart_cfg.get("app")
        if not isinstance(app_ref, str) or ":" not in app_ref:
            return {}

        module_name = app_ref.split(":", 1)[0]
        try:
            module = import_module(module_name)
        except Exception:
            return {}

        raw = getattr(module, "AGENT_DESCRIPTOR", None)
        if raw is None:
            raw = getattr(module, "AGENT_METADATA", None)
        if raw is None:
            return {}

        try:
            return normalize_agent_metadata(agent_name, raw)
        except Exception:
            return {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid.uuid4().hex[:12]}"

    def _resume_run_id(self, payload: dict | None) -> str:
        if not isinstance(payload, dict):
            return ""
        raw = payload.get("resume_run_id")
        return str(raw).strip() if isinstance(raw, str) else ""

    def _persistence_metadata(self, session: dict, *, resumable: bool, status: str | None = None) -> dict:
        metadata = {
            "run_id": session.get("run_id"),
            "resume_run_id": session.get("run_id"),
            "resumable": resumable,
            "status": status or session.get("status"),
        }
        state_path = session.get("state_path")
        if isinstance(state_path, str) and state_path.strip():
            metadata["state_path"] = state_path
        return metadata

    def _checkpoint_run_state(self, session: dict, stage: str) -> None:
        session["updated_at"] = self._now_iso()
        session["state_path"] = self.run_store.save(copy.deepcopy(session), stage=stage)

    def _node_request(
        self,
        *,
        agent: str,
        role: str,
        request_event: str,
        task: str = "",
        original_task: str = "",
        step_id: str = "",
        target_agent: str = "",
        run_id: str = "",
        attempt: int | None = None,
        scope: str = "",
        operation: str = "",
        status: str = "",
    ) -> dict:
        envelope = {
            "agent": agent,
            "role": role,
            "request_event": request_event,
            "task": task,
            "original_task": original_task,
            "step_id": step_id,
            "target_agent": target_agent,
            "run_id": run_id,
            "attempt": attempt,
            "scope": scope,
            "operation": operation,
            "status": status,
        }
        return {key: value for key, value in envelope.items() if value not in (None, "", [], {})}

    def _new_run_session(self, payload: dict, plan_options: list[dict], limits: dict) -> dict:
        run_id = str(payload.get("run_id") or "").strip() or self._new_run_id()
        session_payload = copy.deepcopy(payload)
        session_payload.pop("resume_run_id", None)
        now = self._now_iso()
        session = {
            "run_id": run_id,
            "status": "running",
            "task": session_payload.get("task", ""),
            "task_shape": session_payload.get("task_shape", ""),
            "created_at": now,
            "updated_at": now,
            "payload": session_payload,
            "plan_options": copy.deepcopy(plan_options),
            "limits": copy.deepcopy(limits),
            "attempts": [],
            "replan_count": 0,
            "uncertain_count": 0,
            "current_attempt_index": 0,
            "selected_attempt_index": None,
            "deferred_clarification": None,
            "terminal_event": None,
            "terminal_payload": None,
        }
        self._checkpoint_run_state(session, "run_created")
        return session

    def _load_or_create_run_session(self, payload: dict, plan_options: list[dict], limits: dict) -> tuple[dict, bool]:
        resume_run_id = self._resume_run_id(payload)
        if resume_run_id:
            session = self.run_store.load(resume_run_id)
            if not isinstance(session, dict):
                raise ValueError(f"Cannot resume run '{resume_run_id}': persisted state was not found.")
            session["state_path"] = str(self.run_store.state_path(resume_run_id))
            return session, True
        return self._new_run_session(payload, plan_options, limits), False

    def replay_run(self, run_id: str, depth: int = 0):
        session = self.run_store.load(run_id)
        if not isinstance(session, dict):
            raise ValueError(f"Cannot replay run '{run_id}': persisted state was not found.")
        terminal_event = session.get("terminal_event")
        terminal_payload = session.get("terminal_payload")
        if not isinstance(terminal_event, str) or not terminal_event.strip() or not isinstance(terminal_payload, dict):
            raise ValueError(f"Cannot replay run '{run_id}': the run has not reached a terminal state.")
        self.emit(terminal_event.strip(), copy.deepcopy(terminal_payload), depth)

    def resume_run(self, run_id: str, depth: int = 0):
        session = self.run_store.load(run_id)
        if not isinstance(session, dict):
            raise ValueError(f"Cannot resume run '{run_id}': persisted state was not found.")
        terminal_event = session.get("terminal_event")
        terminal_payload = session.get("terminal_payload")
        if isinstance(terminal_event, str) and terminal_event.strip() and isinstance(terminal_payload, dict):
            self.emit(terminal_event.strip(), copy.deepcopy(terminal_payload), depth)
            return
        payload = copy.deepcopy(session.get("payload") or {})
        payload["resume_run_id"] = run_id
        self._execute_task_plan(payload, depth)

    def list_runs(
        self,
        limit: int = 20,
        status: str | None = None,
        *,
        task_contains: str | None = None,
        agent: str | None = None,
        has_errors: bool | None = None,
        min_duration_ms: float | None = None,
        max_duration_ms: float | None = None,
        slow_step_ms: float | None = None,
        resumable: bool | None = None,
        replayable: bool | None = None,
    ) -> list[dict[str, Any]]:
        return self.run_store.list_runs(
            limit=limit,
            status=status,
            task_contains=task_contains,
            agent=agent,
            has_errors=has_errors,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            slow_step_ms=slow_step_ms,
            resumable=resumable,
            replayable=replayable,
        )

    def inspect_run(self, run_id: str, *, include_timeline: bool = True) -> dict[str, Any]:
        inspection = self.run_store.inspect(run_id, include_timeline=include_timeline)
        if not isinstance(inspection, dict):
            raise ValueError(f"Cannot inspect run '{run_id}': persisted state was not found.")
        return inspection

    def inspect_run_observability(self, run_id: str) -> dict[str, Any]:
        inspection = self.inspect_run(run_id, include_timeline=True)
        observability = inspection.get("observability")
        if not isinstance(observability, dict):
            raise ValueError(f"Cannot inspect observability for run '{run_id}': payload was not found.")
        return copy.deepcopy(observability)

    def render_run_graph(self, run_id: str, *, format: str = "mermaid") -> str | dict[str, Any]:
        inspection = self.inspect_run(run_id, include_timeline=False)
        target_format = str(format or "mermaid").strip().lower()
        if target_format == "json":
            return copy.deepcopy(inspection.get("graph") or {})
        if target_format == "mermaid":
            graph_mermaid = inspection.get("graph_mermaid")
            if isinstance(graph_mermaid, str) and graph_mermaid.strip():
                return graph_mermaid
            return render_workflow_graph_mermaid(inspection.get("graph") or {})
        raise ValueError(f"Unsupported graph format '{format}'.")

    def _autostart_http_services(self):
        for agent_name, config in self.spec["agents"].items():
            runtime_cfg = self._effective_runtime_config(config.get("runtime", {}))
            if runtime_cfg.get("adapter") != "http":
                continue

            autostart_cfg = runtime_cfg.get("autostart")
            if not autostart_cfg:
                continue

            endpoint = runtime_cfg.get("endpoint")
            parsed = urlparse(endpoint)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port
            if port is None:
                raise ValueError(
                    f"HTTP agent '{agent_name}' endpoint must include an explicit port"
                )

            if self._is_port_open(host, port):
                log_boot(f"HTTP agent '{agent_name}' already reachable at {endpoint}")
                continue

            app = autostart_cfg.get("app")
            if not app:
                raise ValueError(
                    f"HTTP agent '{agent_name}' autostart requires runtime.autostart.app"
                )

            module = autostart_cfg.get("module", "uvicorn")
            bind_host = autostart_cfg.get("host", "127.0.0.1")
            bind_port = int(autostart_cfg.get("port", port))
            timeout = float(autostart_cfg.get("timeout_seconds", 300))

            command = [
                sys.executable,
                "-m",
                module,
                app,
                "--host",
                bind_host,
                "--port",
                str(bind_port),
            ]
            log_boot(f"starting HTTP agent '{agent_name}': {' '.join(command)}")
            process_env = os.environ.copy()
            process_env.setdefault(
                "OPENFABRIC_ALLOW_PYTHON_PACKAGE_INSTALLS",
                "1" if self._build_execution_policy().get("allow_python_package_installs") else "0",
            )
            process_env.setdefault(
                "OPENFABRIC_PYTHON_PACKAGE_INSTALL_TOOL",
                str(self._build_execution_policy().get("python_package_install_tool") or "python -m pip"),
            )
            env_cfg = autostart_cfg.get("env")
            if isinstance(env_cfg, dict):
                for key, value in env_cfg.items():
                    if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                        process_env[key] = str(value)
            process = subprocess.Popen(command, env=process_env)
            self._managed_processes.append(process)

            if not self._wait_for_port(host, port, timeout, process):
                exit_code = process.poll()
                self.shutdown()
                if exit_code is not None:
                    raise RuntimeError(
                        f"HTTP agent '{agent_name}' exited before startup "
                        f"(code {exit_code}); endpoint {endpoint} is unavailable"
                    )
                raise RuntimeError(
                    f"HTTP agent '{agent_name}' did not become reachable at {endpoint} "
                    f"within {timeout:.1f}s"
                )
            if not self._wait_for_http_ready(endpoint, timeout, process):
                exit_code = process.poll()
                self.shutdown()
                if exit_code is not None:
                    raise RuntimeError(
                        f"HTTP agent '{agent_name}' exited before HTTP readiness "
                        f"(code {exit_code}); endpoint {endpoint} is unavailable"
                    )
                raise RuntimeError(
                    f"HTTP agent '{agent_name}' did not become HTTP-ready at {endpoint} "
                    f"within {timeout:.1f}s"
                )

        if self._managed_processes and not self._shutdown_registered:
            atexit.register(self.shutdown)
            self._shutdown_registered = True

    def _is_port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            return False

    def _wait_for_port(
        self,
        host: str,
        port: int,
        timeout: float,
        process: subprocess.Popen,
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                return False
            if self._is_port_open(host, port):
                return True
            time.sleep(0.1)
        return False

    def _wait_for_http_ready(
        self,
        endpoint: str,
        timeout: float,
        process: subprocess.Popen,
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                return False
            try:
                response = requests.get(endpoint, timeout=0.5)
                if response.status_code < 500:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.1)
        return False

    def emit(self, event_name: str, payload: dict, depth: int = 0):
        log_event(event_name, payload, depth)

        contract_name = self.spec["events"][event_name]["contract"]
        self.contracts.validate_payload(contract_name, payload)

        if event_name == "task.plan" and isinstance(payload, dict) and (
            isinstance(payload.get("steps"), list) or self._resume_run_id(payload)
        ):
            self._execute_task_plan(payload, depth)
            return

        results = self._invoke_subscribers(event_name, payload, depth)
        for new_event, new_payload in results:
            self.emit(new_event, new_payload, depth + 1)

    def _select_subscribers(self, event_name: str, payload: dict):
        subscribers = self.bus.get_subscribers(event_name)
        target_agent = payload.get("target_agent") if isinstance(payload, dict) else None
        if event_name == "task.plan" and isinstance(target_agent, str) and target_agent:
            subscribers = self._resolve_target_subscribers(target_agent, subscribers, payload)
        return subscribers

    def _agent_aliases(self, agent_name: str) -> set[str]:
        aliases = {agent_name}
        config = self.spec.get("agents", {}).get(agent_name, {})
        metadata = config.get("metadata", {}) if isinstance(config.get("metadata"), dict) else {}
        for key in ("argument_name", "template_agent"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                aliases.add(value.strip())
        return aliases

    def _target_resolution_text(self, payload: dict) -> str:
        parts: list[str] = []
        for key in ("task", "original_task"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        instruction = payload.get("instruction")
        if isinstance(instruction, dict):
            for key in ("question", "command"):
                value = instruction.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        return " ".join(parts).lower()

    def _target_match_score(self, agent_name: str, resolution_text: str) -> int:
        config = self.spec.get("agents", {}).get(agent_name, {})
        metadata = config.get("metadata", {}) if isinstance(config.get("metadata"), dict) else {}
        text_tokens = set(re.findall(r"[a-z0-9_.-]+", resolution_text))
        score = 0

        routing_priority = metadata.get("routing_priority")
        if isinstance(routing_priority, (int, float)):
            score += int(routing_priority)

        for key in ("database_name", "cluster_name", "argument_name", "template_agent"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = value.strip().lower()
            if normalized and normalized in resolution_text:
                score += 100
            score += len(text_tokens & set(re.findall(r"[a-z0-9_.-]+", normalized))) * 25

        for key in ("database_aliases", "cluster_aliases", "routing_hints"):
            values = metadata.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, str) or not item.strip():
                    continue
                normalized = item.strip().lower()
                if normalized and normalized in resolution_text:
                    score += 60
                score += len(text_tokens & set(re.findall(r"[a-z0-9_.-]+", normalized))) * 15

        return score

    def _resolve_target_subscribers(self, target_agent: str, subscribers: list[str], payload: dict) -> list[str]:
        if target_agent in subscribers:
            return [target_agent]

        alias_matches = [agent_name for agent_name in subscribers if target_agent in self._agent_aliases(agent_name)]
        if len(alias_matches) <= 1:
            return alias_matches

        resolution_text = self._target_resolution_text(payload)
        ranked = sorted(
            alias_matches,
            key=lambda agent_name: (self._target_match_score(agent_name, resolution_text), agent_name),
            reverse=True,
        )
        return ranked[:1]

    def _invoke_subscribers(self, event_name: str, payload: dict, depth: int):
        results = []
        for agent_name in self._select_subscribers(event_name, payload):
            log_event_handler(agent_name, depth)
            agent = self.agents[agent_name]
            emitted = agent["adapter"].handle(event_name, payload)
            results.extend(emitted)
        return results

    def _extract_result_value(self, event_name: str, payload: dict, step_payload: dict | None = None):
        if not isinstance(payload, dict):
            return payload
        result_mode = step_payload.get("result_mode") if isinstance(step_payload, dict) else None
        instruction = step_payload.get("instruction") if isinstance(step_payload, dict) else None
        capture = instruction.get("capture") if isinstance(instruction, dict) else None
        if result_mode is None and isinstance(capture, dict):
            if capture.get("mode") == "json_field":
                field_name = capture.get("field")
                if isinstance(field_name, str) and field_name.strip():
                    result_mode = f"json_field:{field_name.strip()}"
            else:
                result_mode = capture.get("mode")
        if event_name == "task.result":
            return payload.get("result", payload.get("detail", ""))
        if event_name == "shell.result":
            stdout = payload.get("stdout", "")
            detail = payload.get("detail", "") if isinstance(payload.get("detail"), str) else ""
            normalized_result = payload.get("normalized_result")
            if not isinstance(normalized_result, str):
                normalized_result = ""
            if result_mode == "json":
                try:
                    return json.loads(stdout)
                except (json.JSONDecodeError, TypeError):
                    return stdout
            if result_mode == "stdout_first_line":
                if normalized_result.strip():
                    return normalized_result.strip()
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                if lines:
                    return lines[0]
                return detail.strip() if detail.strip() else ""
            if result_mode == "stdout_last_line":
                if normalized_result.strip():
                    return normalized_result.strip()
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                if lines:
                    return lines[-1]
                return detail.strip() if detail.strip() else ""
            if result_mode == "stdout_stripped":
                if normalized_result.strip():
                    return normalized_result.strip()
                compact_stdout = stdout.strip()
                if compact_stdout:
                    return compact_stdout
                return detail.strip() if detail.strip() else compact_stdout
            if isinstance(result_mode, str) and result_mode.startswith("json_field:"):
                field_name = result_mode.split(":", 1)[1].strip()
                if field_name:
                    try:
                        parsed = json.loads(stdout)
                        if isinstance(parsed, dict) and field_name in parsed:
                            return parsed[field_name]
                    except (json.JSONDecodeError, TypeError):
                        return stdout
            if normalized_result.strip():
                return normalized_result.strip()
            if isinstance(stdout, str) and stdout.strip():
                return stdout
            reduced = payload.get("reduced_result") or payload.get("refined_answer")
            if reduced not in (None, "", [], {}):
                return reduced
            if detail.strip():
                return detail.strip()
            return stdout
        if event_name == "file.content":
            return payload.get("content", "")
        if event_name == "notify.result":
            return payload.get("detail", "")
        if event_name == "answer.final":
            return payload.get("answer", "")
        if event_name == "sql.result":
            reduced = payload.get("reduced_result") or payload.get("refined_answer")
            if reduced not in (None, "", [], {}):
                return reduced
            result = payload.get("result")
            if isinstance(result, dict):
                rows = result.get("rows")
                columns = result.get("columns")
                if isinstance(rows, list) and len(rows) == 1 and rows and isinstance(rows[0], dict):
                    row = rows[0]
                    if isinstance(columns, list) and len(columns) == 1:
                        scalar = row.get(columns[0])
                        if scalar not in (None, "", [], {}):
                            return scalar
                    if len(row) == 1:
                        scalar = next(iter(row.values()))
                        if scalar not in (None, "", [], {}):
                            return scalar
            return payload
        return payload

    def _resolve_template_string(self, value: str, context: dict):
        pattern = re.compile(r"\{\{([a-zA-Z0-9_.-]+)\}\}")

        def replace(match: re.Match[str]):
            key = match.group(1)
            if key not in context:
                return match.group(0)
            replacement = context.get(key, "")
            if replacement is None:
                return ""
            return str(replacement)

        return pattern.sub(replace, value)

    def _unresolved_template_keys(self, value: Any):
        keys = []
        pattern = re.compile(r"\{\{([a-zA-Z0-9_.-]+)\}\}")
        if isinstance(value, str):
            keys.extend(pattern.findall(value))
        elif isinstance(value, list):
            for item in value:
                keys.extend(self._unresolved_template_keys(item))
        elif isinstance(value, dict):
            for item in value.values():
                keys.extend(self._unresolved_template_keys(item))
        return keys

    def _resolve_templates(self, value: Any, context: dict):
        if isinstance(value, str):
            return self._resolve_template_string(value, context)
        if isinstance(value, list):
            return [self._resolve_templates(item, context) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve_templates(item, context) for key, item in value.items()}
        return value

    def _walk_reference_parts(self, current: Any, parts: list[str]):
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        return current

    def _resolve_reference_path(self, path: str, context: dict):
        if not isinstance(path, str) or not path.strip():
            return None
        path = path.strip()
        if path in context:
            return context.get(path)
        parts = [part for part in path.split(".") if part]
        if not parts:
            return None
        for index in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:index])
            if prefix in context:
                return self._walk_reference_parts(context.get(prefix), parts[index:])
        step_results = context.get("__step_results_by_id__", {})
        head = parts[0]
        if isinstance(step_results, dict) and head in step_results:
            current = step_results[head]
        elif head in context:
            current = context[head]
        else:
            return None
        return self._walk_reference_parts(current, parts[1:])

    def _resolve_references(self, value: Any, context: dict):
        if isinstance(value, dict):
            if set(value.keys()) == {"$from"}:
                return self._resolve_reference_path(value.get("$from"), context)
            return {key: self._resolve_references(item, context) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_references(item, context) for item in value]
        return value

    def _evaluate_when(self, when: Any, context: dict):
        if when is None:
            return True
        if not isinstance(when, dict):
            return bool(when)
        left = self._resolve_references({"$from": when.get("$from")}, context) if "$from" in when else None
        if "equals" in when:
            return left == self._resolve_references(when.get("equals"), context)
        if "not_equals" in when:
            return left != self._resolve_references(when.get("not_equals"), context)
        if "contains" in when:
            right = self._resolve_references(when.get("contains"), context)
            if isinstance(left, (list, str)):
                return right in left
            return False
        if "not_contains" in when:
            right = self._resolve_references(when.get("not_contains"), context)
            if isinstance(left, (list, str)):
                return right not in left
            return True
        if "truthy" in when:
            return bool(left) is bool(when.get("truthy"))
        if "falsy" in when:
            return (not bool(left)) is bool(when.get("falsy"))
        if "gt" in when:
            return left > self._resolve_references(when.get("gt"), context)
        if "gte" in when:
            return left >= self._resolve_references(when.get("gte"), context)
        if "lt" in when:
            return left < self._resolve_references(when.get("lt"), context)
        if "lte" in when:
            return left <= self._resolve_references(when.get("lte"), context)
        return bool(left)

    def _execute_task_plan(self, payload: dict, depth: int):
        incoming_plan_options = self._plan_options(payload)
        limits = self._workflow_limits(payload)
        session, resumed = self._load_or_create_run_session(payload, incoming_plan_options, limits)
        if isinstance(session.get("terminal_event"), str) and session.get("terminal_event") and isinstance(session.get("terminal_payload"), dict):
            self.emit(session["terminal_event"], copy.deepcopy(session["terminal_payload"]), depth + 1)
            return

        active_payload = copy.deepcopy(session.get("payload") or payload)
        active_payload["run_id"] = session.get("run_id")
        plan_options = [
            option
            for option in (session.get("plan_options") or incoming_plan_options)
            if isinstance(option, dict)
        ]
        if not plan_options:
            return

        session["payload"] = active_payload
        session["plan_options"] = copy.deepcopy(plan_options)
        session["limits"] = copy.deepcopy(limits)
        session["status"] = "running"
        self._checkpoint_run_state(session, "run_resumed" if resumed else "run_started")

        attempts = [item for item in session.get("attempts", []) if isinstance(item, dict)]
        session["attempts"] = attempts
        deferred_clarification = session.get("deferred_clarification")
        selected_attempt = None
        selected_attempt_index = session.get("selected_attempt_index")
        if isinstance(selected_attempt_index, int) and 1 <= selected_attempt_index <= len(attempts):
            selected_attempt = attempts[selected_attempt_index - 1]
        replan_count = max(0, int(session.get("replan_count", 0)))
        uncertain_count = max(0, int(session.get("uncertain_count", 0)))

        next_attempt_index = len(attempts) + 1
        if attempts:
            last_attempt = attempts[-1]
            if last_attempt.get("status") == "running" or isinstance(last_attempt.get("workflow_state"), dict):
                next_attempt_index = max(1, int(last_attempt.get("attempt", len(attempts))))

        index = next_attempt_index
        while index <= len(plan_options):
            if index > limits["max_attempts"]:
                break
            option = plan_options[index - 1]
            is_resuming_attempt = index <= len(attempts)

            if is_resuming_attempt:
                attempt_record = attempts[index - 1]
                if attempt_record.get("status") not in {"running"} and not isinstance(attempt_record.get("workflow_state"), dict):
                    continue
            else:
                attempt_record = {
                    "attempt": index,
                    "option": {
                        "id": option.get("id"),
                        "label": option.get("label"),
                        "reason": option.get("reason"),
                    },
                    "status": "running",
                }
                if isinstance(option.get("origin"), dict):
                    attempt_record["option"]["origin"] = self._compact_value(option.get("origin"), text_limit=180, row_limit=3)
                if isinstance(option.get("derived_from_attempt"), int):
                    attempt_record["option"]["derived_from_attempt"] = option.get("derived_from_attempt")
                attempts.append(attempt_record)
                session["attempts"] = attempts
                session["current_attempt_index"] = index
                self._checkpoint_run_state(session, "attempt_started")

            option_payload = dict(active_payload)
            option_payload["steps"] = option.get("steps", [])
            option_payload["selected_option_id"] = option.get("id")
            option_payload["selected_option_label"] = option.get("label")
            option_payload["__attempts_so_far__"] = index
            option_payload["attempt"] = index
            workflow_state = attempt_record.get("workflow_state") if isinstance(attempt_record.get("workflow_state"), dict) else None
            if workflow_state is None:
                self._emit_validation_progress(
                    "option_started",
                    active_payload,
                    depth + 1,
                    option_id=option.get("id"),
                    option_label=option.get("label"),
                    attempt=index,
                    total_attempts=len(plan_options),
                    reason=option.get("reason"),
                    message=f"Trying workflow option {index} of {len(plan_options)}.",
                )
                context = {
                    "original_task": active_payload.get("task", ""),
                    "__step_results__": [],
                    "__step_results_by_id__": {},
                }
            else:
                self._emit_validation_progress(
                    "option_started",
                    active_payload,
                    depth + 1,
                    option_id=option.get("id"),
                    option_label=option.get("label"),
                    attempt=index,
                    total_attempts=len(plan_options),
                    reason=option.get("reason"),
                    message=f"Resuming workflow option {index} of {len(plan_options)}.",
                )
                context = copy.deepcopy(workflow_state.get("context")) if isinstance(workflow_state.get("context"), dict) else {
                    "original_task": active_payload.get("task", ""),
                    "__step_results__": [],
                    "__step_results_by_id__": {},
                }

            def checkpoint_attempt(workflow_snapshot: dict):
                attempt_record["workflow_state"] = workflow_snapshot
                attempt_record["status"] = workflow_snapshot.get("status", "running")
                session["attempts"] = attempts
                session["current_attempt_index"] = index
                session["status"] = "running"
                self._checkpoint_run_state(session, "attempt_checkpoint")

            workflow = self._execute_workflow_steps(
                option_payload.get("steps", []),
                option_payload,
                context,
                depth,
                resume_state=workflow_state,
                checkpoint=checkpoint_attempt,
            )
            attempt_record.pop("workflow_state", None)
            validation = self._validate_workflow_attempt(
                active_payload,
                option,
                workflow,
                context,
                index,
                len(plan_options),
                depth,
            )
            attempt_record["status"] = workflow.get("status")
            attempt_record["steps"] = workflow.get("steps", [])
            attempt_record["result"] = workflow.get("result")
            attempt_record["error"] = workflow.get("error")
            attempt_record["validation"] = validation
            clarification = workflow.get("clarification")
            if isinstance(clarification, dict) and clarification:
                attempt_record["clarification"] = clarification
            else:
                attempt_record.pop("clarification", None)
            session["attempts"] = attempts
            session["current_attempt_index"] = index
            self._checkpoint_run_state(session, "attempt_finished")

            routing = self._route_workflow_attempt(
                active_payload,
                option,
                workflow,
                validation,
                index,
                len(plan_options),
                limits,
                uncertain_count,
                replan_count,
                clarification=clarification if isinstance(clarification, dict) else None,
            )
            attempt_record["routing"] = self._compact_routing_record(routing)
            uncertain_count = max(uncertain_count, int(routing.get("uncertain_count_after", uncertain_count)))
            session["uncertain_count"] = uncertain_count
            session["attempts"] = attempts
            self._checkpoint_run_state(session, "attempt_routed")

            if routing.get("action") == "accept_attempt":
                selected_attempt = attempt_record
                session["selected_attempt_index"] = index
                break

            if routing.get("action") == "clarify":
                clarification_payload = clarification if isinstance(clarification, dict) else self._clarification_from_validation(
                    active_payload,
                    attempts,
                    attempt_record,
                    "Validation remained uncertain after repeated attempts.",
                )
                attempt_record["clarification"] = clarification_payload
                session["deferred_clarification"] = clarification_payload
                if deferred_clarification is None:
                    deferred_clarification = clarification_payload
                self._checkpoint_run_state(session, "clarification_deferred")
                break

            if routing.get("action") == "replan_workflow":
                derived = self._derive_fallback_option(active_payload, plan_options, attempt_record, context, len(attempts), depth)
                if isinstance(derived, dict):
                    if isinstance(derived.get("replan"), dict):
                        attempt_record["replan"] = derived.get("replan")
                    derived_option = derived.get("option")
                    if isinstance(derived_option, dict):
                        plan_options.append(derived_option)
                        session["plan_options"] = copy.deepcopy(plan_options)
                        replan_count += 1
                        session["replan_count"] = replan_count
                        if isinstance(attempt_record.get("routing"), dict):
                            attempt_record["routing"]["activated_option_id"] = derived_option.get("id")
                            attempt_record["routing"]["budget"] = self._compact_value(
                                self._budget_snapshot(
                                    limits,
                                    attempts_used=index,
                                    replan_count=replan_count,
                                    uncertain_count=uncertain_count,
                                    workflow_validation_requests=index,
                                ),
                                text_limit=220,
                                row_limit=3,
                            )
                        self._checkpoint_run_state(session, "workflow_replanned")
                    else:
                        fallback_reason = str(validation.get("reason") or workflow.get("error") or "Workflow fallback planning failed.").strip()
                        if validation.get("verdict") == "uncertain":
                            clarification_payload = self._clarification_from_validation(
                                active_payload,
                                attempts,
                                attempt_record,
                                fallback_reason,
                            )
                            attempt_record["clarification"] = clarification_payload
                            if isinstance(attempt_record.get("routing"), dict):
                                attempt_record["routing"]["action"] = "clarify"
                                attempt_record["routing"]["clarification_required"] = True
                            deferred_clarification = clarification_payload
                            session["deferred_clarification"] = clarification_payload
                            self._checkpoint_run_state(session, "clarification_deferred")
                            break
                        if isinstance(attempt_record.get("routing"), dict):
                            attempt_record["routing"]["action"] = "fail_run"
                            attempt_record["routing"]["reason"] = fallback_reason
                        self._checkpoint_run_state(session, "workflow_replan_failed")
                        break

            if routing.get("action") in {"try_next_option", "replan_workflow"} and index < len(plan_options):
                self._emit_validation_progress(
                    "retrying",
                    active_payload,
                    depth + 1,
                    option_id=option.get("id"),
                    option_label=option.get("label"),
                    attempt=index,
                    total_attempts=len(plan_options),
                    reason=routing.get("reason") or validation.get("reason"),
                    message="The workflow router rejected this attempt, so I am trying the next workflow option.",
                )
                session["current_attempt_index"] = index + 1
                self._checkpoint_run_state(session, "next_attempt_scheduled")
                index += 1
                continue

            if routing.get("action") == "fail_run":
                break

            index += 1

        if selected_attempt is not None:
            if "workflow.result" in self.spec.get("events", {}):
                presentation = active_payload.get("presentation")
                session["status"] = str(selected_attempt["status"] or "completed")
                session["selected_attempt_index"] = int(selected_attempt.get("attempt", 0) or 0)
                session["deferred_clarification"] = None
                result_payload = {
                    "task": active_payload.get("task", ""),
                    "task_shape": active_payload.get("task_shape"),
                    "status": selected_attempt["status"],
                    "steps": selected_attempt["steps"],
                    "result": selected_attempt.get("result"),
                    "attempts": attempts,
                    "selected_option": selected_attempt["option"],
                    "validation": selected_attempt["validation"],
                    "run_id": session.get("run_id"),
                    "persistence": self._persistence_metadata(session, resumable=False, status=session["status"]),
                }
                if isinstance(presentation, dict):
                    result_payload["presentation"] = presentation
                if selected_attempt.get("error"):
                    result_payload["error"] = selected_attempt["error"]
                result_payload["graph"] = build_workflow_graph(
                    task=active_payload.get("task", ""),
                    task_shape=str(active_payload.get("task_shape") or ""),
                    status=str(selected_attempt["status"] or ""),
                    attempts=attempts,
                    selected_option=selected_attempt["option"],
                    result=selected_attempt.get("result"),
                    presentation=presentation if isinstance(presentation, dict) else None,
                    run_id=str(session.get("run_id") or ""),
                )
                session["terminal_event"] = "workflow.result"
                session["terminal_payload"] = copy.deepcopy(result_payload)
                session["current_attempt_index"] = 0
                self._checkpoint_run_state(session, "run_completed")
                self.emit("workflow.result", result_payload, depth + 1)
                return
            return

        if deferred_clarification is not None and "clarification.required" in self.spec.get("events", {}):
            presentation = active_payload.get("presentation")
            session["status"] = "needs_clarification"
            terminal_attempts = copy.deepcopy(attempts)
            clarification_payload = copy.deepcopy(deferred_clarification)
            clarification_payload["attempts"] = terminal_attempts
            clarification_payload["run_id"] = session.get("run_id")
            clarification_payload["persistence"] = self._persistence_metadata(
                session,
                resumable=True,
                status=session["status"],
            )
            clarification_payload["graph"] = build_workflow_graph(
                task=active_payload.get("task", ""),
                task_shape=str(active_payload.get("task_shape") or ""),
                status="needs_clarification",
                attempts=terminal_attempts,
                selected_option=None,
                result=(terminal_attempts[-1] or {}).get("result") if terminal_attempts else None,
                presentation=presentation if isinstance(presentation, dict) else None,
                run_id=str(session.get("run_id") or ""),
            )
            session["terminal_event"] = "clarification.required"
            session["terminal_payload"] = copy.deepcopy(clarification_payload)
            session["current_attempt_index"] = 0
            self._checkpoint_run_state(session, "run_needs_clarification")
            self.emit("clarification.required", clarification_payload, depth + 1)
            return

        last_attempt = attempts[-1] if attempts else None
        if "workflow.result" in self.spec.get("events", {}):
            presentation = active_payload.get("presentation")
            session["status"] = "failed"
            result_payload = {
                "task": active_payload.get("task", ""),
                "task_shape": active_payload.get("task_shape"),
                "status": "failed",
                "steps": last_attempt.get("steps", []) if isinstance(last_attempt, dict) else [],
                "result": last_attempt.get("result") if isinstance(last_attempt, dict) else None,
                "attempts": attempts,
                "validation": last_attempt.get("validation") if isinstance(last_attempt, dict) else None,
                "run_id": session.get("run_id"),
                "persistence": self._persistence_metadata(session, resumable=False, status=session["status"]),
            }
            if isinstance(presentation, dict):
                result_payload["presentation"] = presentation
            if isinstance(last_attempt, dict):
                result_payload["selected_option"] = last_attempt.get("option")
                resolved_error = last_attempt.get("error") or (
                    last_attempt.get("validation", {}) or {}
                ).get("reason")
                if isinstance(resolved_error, str) and resolved_error.strip():
                    result_payload["error"] = resolved_error.strip()
            result_payload["graph"] = build_workflow_graph(
                task=active_payload.get("task", ""),
                task_shape=str(active_payload.get("task_shape") or ""),
                status="failed",
                attempts=attempts,
                selected_option=(last_attempt or {}).get("option") if isinstance(last_attempt, dict) else None,
                result=result_payload.get("result"),
                presentation=presentation if isinstance(presentation, dict) else None,
                run_id=str(session.get("run_id") or ""),
            )
            session["terminal_event"] = "workflow.result"
            session["terminal_payload"] = copy.deepcopy(result_payload)
            session["current_attempt_index"] = 0
            self._checkpoint_run_state(session, "run_failed")
            self.emit("workflow.result", result_payload, depth + 1)
            return

    def _workflow_limits(self, payload: dict) -> dict:
        limits = dict(self.DEFAULT_WORKFLOW_LIMITS)
        raw = payload.get("retry_budget")
        if isinstance(raw, dict):
            for key in limits:
                value = raw.get(key)
                if isinstance(value, (int, float)):
                    limits[key] = max(0, int(value))
        return limits

    def _clarification_from_validation(self, payload: dict, attempts: list[dict], last_attempt: dict, detail: str) -> dict:
        validation = last_attempt.get("validation") if isinstance(last_attempt, dict) else {}
        question = (
            (validation or {}).get("reason")
            or last_attempt.get("error")
            or detail
        )
        clarification = {
            "task": payload.get("task", ""),
            "task_shape": payload.get("task_shape", ""),
            "detail": detail,
            "question": f"I need either a clearer goal or a narrower target to continue: {question}",
            "available_context": {
                "attempts": self._compact_value(attempts, text_limit=180, row_limit=2),
                "last_attempt": self._compact_value(last_attempt, text_limit=180, row_limit=2),
            },
        }
        missing = (validation or {}).get("missing_requirements")
        if isinstance(missing, list) and missing:
            clarification["missing_information"] = [item for item in missing if isinstance(item, str) and item.strip()]
        return clarification

    def _budget_snapshot(
        self,
        limits: dict,
        *,
        attempts_used: int = 0,
        replan_count: int = 0,
        uncertain_count: int = 0,
        step_replans: dict[str, int] | None = None,
        workflow_validation_requests: int = 0,
        step_validation_requests: int = 0,
    ) -> dict:
        step_replans = step_replans or {}
        usage = {
            "attempts": max(0, attempts_used),
            "workflow_replans": max(0, replan_count),
            "uncertain_attempts": max(0, uncertain_count),
            "workflow_validation_requests": max(0, workflow_validation_requests),
            "step_validation_requests": max(0, step_validation_requests),
        }
        if step_replans:
            usage["step_replans"] = {
                key: max(0, int(value))
                for key, value in step_replans.items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }
        remaining = {
            "attempts": max(0, limits.get("max_attempts", 0) - usage["attempts"]),
            "workflow_replans": max(0, limits.get("max_replans", 0) - usage["workflow_replans"]),
            "uncertain_attempts": max(0, limits.get("max_uncertain_attempts", 0) - usage["uncertain_attempts"]),
            "workflow_validation_requests": max(0, limits.get("max_validation_llm_calls", 0) - usage["workflow_validation_requests"]),
            "step_validation_requests": max(0, limits.get("max_step_validation_llm_calls", 0) - usage["step_validation_requests"]),
        }
        if "step_replans" in usage:
            remaining["step_replans"] = {
                key: max(0, limits.get("max_step_replans", 0) - value)
                for key, value in usage["step_replans"].items()
            }
        return {
            "limits": copy.deepcopy(limits),
            "usage": usage,
            "remaining": remaining,
        }

    def _compact_routing_record(self, record: dict | None) -> dict | None:
        if not isinstance(record, dict) or not record:
            return None
        compact = {}
        for key in (
            "scope",
            "stage",
            "action",
            "reason",
            "verdict",
            "valid",
            "retry_recommended",
            "failure_class",
            "attempt",
            "total_attempts",
            "option_id",
            "option_label",
            "step_id",
            "target_agent",
            "event",
            "remaining_options",
            "activated_option_id",
            "clarification_required",
            "accepted_with_uncertainty",
            "uncertain_count_after",
        ):
            value = record.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value
        budget = record.get("budget")
        if isinstance(budget, dict) and budget:
            compact["budget"] = self._compact_value(budget, text_limit=220, row_limit=3)
        validation = record.get("validation")
        if isinstance(validation, dict) and validation:
            compact["validation"] = self._compact_value(validation, text_limit=180, row_limit=3)
        return compact or None

    def _compact_replan_record(self, record: dict | None) -> dict | None:
        if not isinstance(record, dict) or not record:
            return None
        compact = {}
        for key in (
            "scope",
            "status",
            "step_id",
            "replace_step_id",
            "reason",
            "failure_class",
            "event",
            "derived_option_id",
            "derived_from_attempt",
        ):
            value = record.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value
        request_payload = record.get("request")
        if isinstance(request_payload, dict) and request_payload:
            compact["request"] = self._compact_value(request_payload, text_limit=220, row_limit=3)
        result_payload = record.get("result")
        if isinstance(result_payload, dict) and result_payload:
            compact["result"] = self._compact_value(result_payload, text_limit=220, row_limit=3)
        steps = record.get("steps")
        if isinstance(steps, list) and steps:
            compact["steps"] = [self._compact_value(step, text_limit=180, row_limit=3) for step in steps[:3] if isinstance(step, dict)]
            if len(steps) > 3:
                compact["steps_note"] = f"Showing first 3 replanned steps out of {len(steps)}."
        return compact or None

    def _step_runtime_history(self, raw_step: dict, key: str) -> list[dict]:
        history = raw_step.get(key)
        if not isinstance(history, list):
            return []
        return [item for item in history if isinstance(item, dict)]

    def _attach_step_runtime_metadata(
        self,
        raw_step: dict,
        record: dict,
        *,
        routing: dict | None = None,
        replan: dict | None = None,
        clarification: dict | None = None,
    ) -> dict:
        routing_history = list(self._step_runtime_history(raw_step, "__routing_history__"))
        replan_history = list(self._step_runtime_history(raw_step, "__replan_history__"))
        compact_routing = self._compact_routing_record(routing)
        compact_replan = self._compact_replan_record(replan)
        if compact_routing:
            routing_history.append(compact_routing)
        if compact_replan:
            replan_history.append(compact_replan)
        if routing_history:
            record["routing"] = routing_history[-1]
            record["routing_history"] = routing_history
        if replan_history:
            record["replan"] = replan_history[-1]
            record["replan_history"] = replan_history
        if isinstance(clarification, dict) and clarification:
            record["clarification"] = clarification
        return record

    def _route_workflow_attempt(
        self,
        payload: dict,
        option: dict,
        workflow: dict,
        validation: dict,
        attempt: int,
        total_attempts: int,
        limits: dict,
        uncertain_count: int,
        replan_count: int,
        clarification: dict | None = None,
    ) -> dict:
        verdict = str(validation.get("verdict") or ("valid" if validation.get("valid") else "invalid")).strip() or "invalid"
        projected_uncertain = uncertain_count + (1 if verdict == "uncertain" else 0)
        remaining_options = max(0, total_attempts - attempt)
        can_try_next = remaining_options > 0
        can_replan = remaining_options == 0 and replan_count < limits["max_replans"] and attempt < limits["max_attempts"]
        action = "fail_run"
        reason = str(validation.get("reason") or workflow.get("error") or "Workflow output did not clearly satisfy the request.").strip()

        if workflow.get("status") == "needs_clarification" and isinstance(clarification, dict):
            action = "clarify"
            reason = str(clarification.get("detail") or clarification.get("question") or reason).strip()
        elif validation.get("valid"):
            action = "accept_attempt"
        elif verdict == "uncertain":
            if projected_uncertain > limits["max_uncertain_attempts"]:
                action = "clarify"
            elif can_try_next:
                action = "try_next_option"
            elif can_replan:
                action = "replan_workflow"
            else:
                action = "clarify"
        elif can_try_next:
            action = "try_next_option"
        elif can_replan:
            action = "replan_workflow"

        return {
            "scope": "workflow",
            "stage": "validation",
            "action": action,
            "reason": reason,
            "verdict": verdict,
            "valid": bool(validation.get("valid")),
            "retry_recommended": bool(validation.get("retry_recommended")),
            "attempt": attempt,
            "total_attempts": total_attempts,
            "option_id": option.get("id"),
            "option_label": option.get("label"),
            "remaining_options": remaining_options,
            "clarification_required": action == "clarify",
            "uncertain_count_after": projected_uncertain,
            "budget": self._budget_snapshot(
                limits,
                attempts_used=attempt,
                replan_count=replan_count,
                uncertain_count=projected_uncertain,
                workflow_validation_requests=attempt,
            ),
            "validation": self._compact_value(validation, text_limit=180, row_limit=3),
        }

    def _route_step_attempt(
        self,
        step_id: str,
        step_payload: dict,
        primary_event: str | None,
        validation_result: dict | None,
        error: str,
        stage: str,
        limits: dict,
        step_replans: dict[str, int],
        step_validation_requests: int,
        *,
        needs_clarification: bool = False,
        failure_class: str | None = None,
    ) -> dict:
        step_replan_count = max(0, int(step_replans.get(step_id, 0)))
        verdict = "failed"
        valid = False
        retry_recommended = False
        accepted_with_uncertainty = False

        if stage == "validation" and isinstance(validation_result, dict):
            verdict = str(validation_result.get("verdict") or ("valid" if validation_result.get("valid") else "invalid")).strip() or "invalid"
            valid = bool(validation_result.get("valid"))
            retry_recommended = bool(validation_result.get("retry_recommended"))
            accepted_with_uncertainty = verdict == "uncertain" and not retry_recommended
        elif needs_clarification:
            verdict = "needs_clarification"

        can_replan = step_replan_count < limits["max_step_replans"]
        action = "fail_run"
        resolved_failure_class = failure_class or ("needs_clarification" if needs_clarification else "execution_failed")
        if stage == "validation":
            resolved_failure_class = failure_class or ("validation_failed" if not (valid or accepted_with_uncertainty) else "")
            if valid or accepted_with_uncertainty:
                action = "accept_step"
            elif can_replan:
                action = "replan_step"
        else:
            if needs_clarification:
                action = "clarify"
            elif can_replan:
                action = "replan_step"

        return {
            "scope": "step",
            "stage": stage,
            "action": action,
            "reason": error,
            "verdict": verdict,
            "valid": valid,
            "retry_recommended": retry_recommended,
            "accepted_with_uncertainty": accepted_with_uncertainty,
            "failure_class": resolved_failure_class or None,
            "step_id": step_id,
            "target_agent": step_payload.get("target_agent", ""),
            "event": primary_event or "",
            "clarification_required": action == "clarify",
            "budget": self._budget_snapshot(
                limits,
                step_replans=step_replans,
                step_validation_requests=step_validation_requests,
            ),
            "validation": self._compact_value(validation_result, text_limit=180, row_limit=3) if isinstance(validation_result, dict) else None,
        }

    def _plan_options(self, payload: dict) -> list[dict]:
        raw_options = payload.get("plan_options")
        normalized = []
        if isinstance(raw_options, list):
            for index, option in enumerate(raw_options, start=1):
                if not isinstance(option, dict):
                    continue
                steps = option.get("steps")
                if not isinstance(steps, list) or not steps:
                    continue
                normalized.append(
                    {
                        "id": option.get("id") or f"option{index}",
                        "label": option.get("label") or f"Option {index}",
                        "reason": option.get("reason") or "",
                        "steps": steps,
                    }
                )
        if normalized:
            return normalized
        steps = payload.get("steps")
        if isinstance(steps, list) and steps:
            return [
                {
                    "id": "option1",
                    "label": "Primary plan",
                    "reason": "",
                    "steps": steps,
                }
            ]
        return []

    def _derive_fallback_option(self, payload: dict, current_options: list[dict], last_attempt: dict, context: dict, attempt_count: int, depth: int):
        existing = [option for option in current_options if isinstance(option, dict)]
        if attempt_count > 3:
            return None
        if "planner.replan.request" not in self.spec.get("events", {}):
            return None
        reason = (
            ((last_attempt.get("validation") or {}).get("reason"))
            or last_attempt.get("error")
            or "The previous workflow attempt did not satisfy the request."
        )
        replan_payload = {
            "task": payload.get("task", ""),
            "task_shape": payload.get("task_shape", ""),
            "step_id": "__workflow__",
            "original_task": payload.get("task", ""),
            "target_agent": "",
            "reason": reason,
            "failure_class": "execution_failed",
            "event": "workflow.result",
            "available_context": {
                "attempts": self._compact_value([attempt.get("validation") for attempt in [last_attempt]], text_limit=180, row_limit=2),
                "last_steps": self._compact_value(last_attempt.get("steps", []), text_limit=180, row_limit=3),
                "step_results": self._compact_value(context.get("__step_results__", []), text_limit=180, row_limit=3),
            },
            "partial_result": self._compact_value(last_attempt.get("result"), text_limit=180, row_limit=3),
            "presentation": payload.get("presentation", {}),
        }
        contract_name = self.spec["events"]["planner.replan.request"]["contract"]
        self.contracts.validate_payload(contract_name, replan_payload)
        log_event("planner.replan.request", replan_payload, depth + 1)
        emitted = self._invoke_subscribers("planner.replan.request", replan_payload, depth + 1)
        replan_result = None
        auxiliary = []
        for event_name, event_payload in emitted:
            if event_name == "planner.replan.result":
                replan_result = event_payload
            elif event_name == "plan.progress":
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        replan_trace = {
            "scope": "workflow",
            "status": "received" if isinstance(replan_result, dict) else "failed",
            "step_id": "__workflow__",
            "replace_step_id": replan_result.get("replace_step_id") if isinstance(replan_result, dict) else "__workflow__",
            "reason": str((replan_result or {}).get("reason") or reason),
            "failure_class": "execution_failed",
            "event": "workflow.result",
            "request": replan_payload,
            "result": replan_result if isinstance(replan_result, dict) else None,
            "derived_from_attempt": attempt_count,
        }
        if not isinstance(replan_result, dict):
            return {"option": None, "replan": self._compact_replan_record(replan_trace)}
        steps = replan_result.get("steps")
        if not isinstance(steps, list) or not steps:
            replan_trace["status"] = "empty"
            return {"option": None, "replan": self._compact_replan_record(replan_trace)}
        existing_signatures = {
            json.dumps(option.get("steps", []), sort_keys=True, ensure_ascii=True)
            for option in existing
            if isinstance(option, dict)
        }
        candidate_signature = json.dumps(steps, sort_keys=True, ensure_ascii=True)
        if candidate_signature in existing_signatures:
            replan_trace["status"] = "duplicate"
            replan_trace["steps"] = steps
            return {"option": None, "replan": self._compact_replan_record(replan_trace)}
        option_id = f"option{len(existing) + 1}"
        option = {
            "id": option_id,
            "label": "Recovered fallback",
            "reason": str(replan_result.get("reason") or reason),
            "steps": steps,
            "derived_from_attempt": attempt_count,
            "origin": {
                "kind": "workflow_replan",
                "derived_from_attempt": attempt_count,
                "reason": str(replan_result.get("reason") or reason),
            },
        }
        replan_trace["status"] = "derived"
        replan_trace["derived_option_id"] = option_id
        replan_trace["steps"] = steps
        return {
            "option": option,
            "replan": self._compact_replan_record(replan_trace),
        }

    def _structured_count_like(self, value: Any) -> bool:
        if self._looks_like_scalar(value):
            return True
        if isinstance(value, str):
            compact = value.strip().lower()
            return bool(re.search(r"\d", compact)) and any(token in compact for token in ("count", "total", "matching"))
        if isinstance(value, list) and len(value) == 1:
            row = value[0]
            if isinstance(row, dict):
                numeric_values = [item for item in row.values() if self._looks_like_scalar(item)]
                return len(numeric_values) == 1
        if isinstance(value, dict):
            for key in ("count", "total", "matching_jobs", "jobs_considered", "total_nodes"):
                if key in value and self._looks_like_scalar(value.get(key)):
                    return True
            rows = value.get("rows")
            if self._structured_count_like(rows):
                return True
            nested = value.get("result")
            if nested is not None and nested is not value and self._structured_count_like(nested):
                return True
            reduced = value.get("reduced_result") or value.get("refined_answer")
            if isinstance(reduced, str):
                reduced_lower = reduced.lower()
                return bool(re.search(r"\d", reduced_lower)) and any(token in reduced_lower for token in ("count", "total", "matching"))
        return False

    def _contains_state_evidence(self, value: Any) -> bool:
        try:
            text = json.dumps(value, ensure_ascii=True, default=str).lower()
        except TypeError:
            text = str(value).lower()
        return any(
            token in text
            for token in (
                "state",
                "status",
                "idle",
                "mixed",
                "allocated",
                "running",
                "pending",
                "failed",
                "completed",
                "down",
                "drain",
            )
        )

    def _task_requires_count_and_state(self, task_text: str) -> bool:
        text = str(task_text or "").lower()
        return any(token in text for token in ("how many", "count ", "number of", "total ")) and any(
            token in text for token in (" state", " states", "status", "breakdown")
        )

    def _task_requires_count_and_identifiers(self, task_text: str) -> bool:
        text = str(task_text or "").lower()
        has_count = any(token in text for token in ("how many", "count ", "number of", "total "))
        has_identifier_request = "job id" in text or bool(re.search(r"\bids\b", text))
        return has_count and has_identifier_request

    def _contains_identifier_evidence(self, value: Any) -> bool:
        if isinstance(value, str):
            lines = [line.strip() for line in value.splitlines() if line.strip()]
            if len(lines) >= 2 and all(re.fullmatch(r"[A-Za-z0-9_.:-]+", line) for line in lines):
                return True
            if any("job id" in line.lower() for line in lines):
                return True
            return False
        if isinstance(value, list):
            return any(self._contains_identifier_evidence(item) for item in value)
        if isinstance(value, dict):
            for key in ("job_id", "job_ids", "ids", "identifier", "identifiers"):
                candidate = value.get(key)
                if candidate not in (None, "", [], {}):
                    if self._contains_identifier_evidence(candidate):
                        return True
            excerpt = value.get("excerpt")
            if isinstance(excerpt, str) and self._contains_identifier_evidence(excerpt):
                return True
            rows = value.get("rows")
            if isinstance(rows, list) and rows:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    for key, candidate in row.items():
                        if "id" in str(key).lower() and candidate not in (None, "", [], {}):
                            return True
            nested = value.get("result")
            if nested not in (None, value) and self._contains_identifier_evidence(nested):
                return True
            reduced = value.get("reduced_result") or value.get("refined_answer")
            if isinstance(reduced, str) and self._contains_identifier_evidence(reduced):
                return True
        return False

    def _infer_task_shape(self, task_text: str, fallback_shape: str, primary_event: str | None = None) -> str:
        text = str(task_text or "").strip().lower()
        inherited_shape = str(fallback_shape or "").strip().lower()
        if any(token in text for token in ("save ", "write ", "create ")) and any(
            token in text for token in (" file", " path", ".txt", ".csv", ".json", ".md")
        ):
            return "save_artifact"
        if any(token in text for token in ("how many", "count ", "number of", "total ")) or text.startswith("count "):
            return "count"
        if any(token in text for token in ("confirm ", "verify ", "check ")) and any(
            token in text for token in (" removed", " absent", " missing", " exists", " present")
        ) and not any(token in text for token in ("list ", "show ", "display ")):
            return "boolean_check"
        if any(token in text for token in ("whether", "check whether", "if any", "is there", "exists", "does ", "do any", "has ")) and not any(
            token in text for token in ("list ", "show ", "display ")
        ):
            return "boolean_check"
        if any(token in text for token in ("schema", "schemas", "columns", "column ", "relationship", "relationships", "foreign key")):
            return "schema_summary"
        if any(token in text for token in ("compare", "difference", "versus", " vs ")):
            return "compare"
        if any(token in text for token in ("summarize", "summary", "overview")):
            return "summarize_dataset"
        if any(token in text for token in ("list ", "show ", "find ", "which ", "sample rows", "display ")):
            return "list"
        if inherited_shape:
            return inherited_shape
        if primary_event == "shell.result":
            return "command_execution"
        return "lookup"

    def _default_validation_result(
        self,
        workflow_payload: dict,
        option: dict,
        workflow: dict,
        context: dict,
        attempt: int,
        total_attempts: int,
    ) -> dict:
        status = workflow.get("status")
        error = workflow.get("error")
        result = workflow.get("result")
        task_text = str(workflow_payload.get("task") or "").strip()
        task_shape = self._infer_task_shape(task_text, str(workflow_payload.get("task_shape") or ""))
        completed = status == "completed"
        prior_results = self._prior_step_results(context)
        shape_assessment = self._assess_task_shape_result(task_shape, result, prior_results)
        if self._task_requires_count_and_state(task_text):
            has_count = self._structured_count_like(result)
            has_state = self._contains_state_evidence(result)
            shape_assessment = {
                "verdict": "valid" if has_count and has_state else "invalid",
                "reason": (
                    "Combined count/state output detected."
                    if has_count and has_state
                    else "Count/state task did not include both count and state evidence."
                ),
                "missing_requirements": [
                    requirement
                    for requirement, present in (("count", has_count), ("state breakdown", has_state))
                    if not present
                ],
                "trace": [
                    "Detected compound count/state intent from the workflow task.",
                    f"Count evidence detected: {has_count}.",
                    f"State evidence detected: {has_state}.",
                ],
            }
        elif self._task_requires_count_and_identifiers(task_text):
            evidence_values = [result]
            for item in prior_results:
                if not isinstance(item, dict):
                    continue
                evidence_values.extend(
                    candidate
                    for candidate in (item.get("value"), item.get("result"), item.get("summary"))
                    if candidate not in (None, "", [], {})
                )
            has_count = any(self._structured_count_like(item) for item in evidence_values)
            has_identifiers = any(self._contains_identifier_evidence(item) for item in evidence_values)
            shape_assessment = {
                "verdict": "valid" if has_count and has_identifiers else "invalid",
                "reason": (
                    "Combined count and identifier output detected."
                    if has_count and has_identifiers
                    else "Count-and-identifier task did not include both the requested count and identifier list."
                ),
                "missing_requirements": [
                    requirement
                    for requirement, present in (("count", has_count), ("identifier list", has_identifiers))
                    if not present
                ],
                "trace": [
                    "Detected compound count/identifier intent from the workflow task.",
                    f"Count evidence detected: {has_count}.",
                    f"Identifier evidence detected: {has_identifiers}.",
                ],
            }
        valid = completed and shape_assessment["verdict"] == "valid"
        trace = [
            f"Attempt {attempt} of {total_attempts} used option '{option.get('label') or option.get('id') or 'workflow'}'.",
            f"Workflow status was '{status or 'unknown'}'.",
            f"Task shape was '{task_shape}'.",
        ]
        if error:
            trace.append(f"Execution error observed: {error}")
        trace.extend(shape_assessment["trace"])
        if valid:
            trace.append("The workflow completed and satisfied the task-shape checks, so it is accepted.")
        else:
            trace.append("The workflow did not clearly satisfy the request, so another option should be tried if available.")
        return {
            "valid": valid,
            "verdict": shape_assessment["verdict"] if completed else "invalid",
            "reason": "Workflow satisfied the request." if valid else (str(error or shape_assessment["reason"] or "Workflow output did not clearly satisfy the request.")),
            "retry_recommended": not valid,
            "missing_requirements": shape_assessment.get("missing_requirements", []),
            "trace": trace,
        }

    def _infer_step_task_shape(self, step_payload: dict, primary_event: str | None, primary_payload: Any, primary_value: Any) -> str:
        return self._infer_task_shape(
            str(step_payload.get("task") or ""),
            str(step_payload.get("task_shape") or ""),
            primary_event,
        )

    def _default_step_validation_result(
        self,
        workflow_payload: dict,
        step_id: str,
        step_payload: dict,
        primary_event: str | None,
        primary_payload: Any,
        primary_value: Any,
        context: dict,
    ) -> dict:
        task_shape = self._infer_step_task_shape(step_payload, primary_event, primary_payload, primary_value)
        prior_results = self._prior_step_results(context)
        shape_assessment = self._assess_task_shape_result(task_shape, primary_value, prior_results)
        valid = shape_assessment["verdict"] == "valid"
        trace = [
            f"Validated step '{step_id}' for task '{step_payload.get('task', '')}'.",
            f"Inferred step task shape '{task_shape}'.",
            f"Primary event was '{primary_event or 'unknown'}'.",
            *shape_assessment["trace"],
        ]
        if valid:
            trace.append("The step output satisfies the inferred step intent.")
        else:
            trace.append("The step output did not clearly satisfy the inferred step intent.")
        return {
            "valid": valid,
            "verdict": shape_assessment["verdict"],
            "reason": "Step output satisfied the step intent." if valid else str(shape_assessment["reason"] or "Step output did not satisfy the step intent."),
            "retry_recommended": not valid,
            "missing_requirements": shape_assessment.get("missing_requirements", []),
            "trace": trace,
        }

    def _validate_step_attempt(
        self,
        workflow_payload: dict,
        step_id: str,
        step_payload: dict,
        primary_event: str | None,
        primary_payload: Any,
        primary_value: Any,
        context: dict,
        validation_request_index: int,
        depth: int,
    ) -> dict:
        limits = self._workflow_limits(workflow_payload)
        validation_llm_budget_remaining = max(0, limits["max_step_validation_llm_calls"] - max(0, validation_request_index - 1))
        step_task_shape = self._infer_step_task_shape(step_payload, primary_event, primary_payload, primary_value)
        if "validation.request" not in self.spec.get("events", {}):
            return self._default_step_validation_result(
                workflow_payload,
                step_id,
                step_payload,
                primary_event,
                primary_payload,
                primary_value,
                context,
            )

        request_payload = {
            "run_id": workflow_payload.get("run_id", ""),
            "validation_scope": "step",
            "task": workflow_payload.get("task", ""),
            "original_task": workflow_payload.get("task", ""),
            "task_shape": step_task_shape,
            "workflow_status": "completed",
            "validation_llm_budget_remaining": validation_llm_budget_remaining,
            "step_id": step_id,
            "step_task": step_payload.get("task", ""),
            "step_target_agent": step_payload.get("target_agent", ""),
            "step_event": primary_event or "",
            "steps": [
                {
                    "id": step_id,
                    "task": step_payload.get("task", ""),
                    "status": "completed",
                    "event": primary_event or "",
                    "payload": self._compact_event_payload(primary_event or "", primary_payload),
                }
            ],
            "result": self._compact_event_payload(primary_event or "", primary_payload)
            if isinstance(primary_payload, dict)
            else self._compact_value(primary_value, text_limit=240, row_limit=5),
            "step_value": self._compact_value(primary_value, text_limit=240, row_limit=5),
            "available_context": {
                key: self._compact_value(value, text_limit=180, row_limit=3)
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
            "node": self._node_request(
                agent="validator",
                role="validator",
                request_event="validation.request",
                task=str(step_payload.get("task") or ""),
                original_task=str(workflow_payload.get("task") or ""),
                step_id=step_id,
                target_agent=str(step_payload.get("target_agent") or ""),
                run_id=str(workflow_payload.get("run_id") or ""),
                attempt=workflow_payload.get("attempt") if isinstance(workflow_payload.get("attempt"), int) else None,
                scope="step",
                operation="validate_step",
                status="pending",
            ),
        }
        contract_name = self.spec["events"]["validation.request"]["contract"]
        self.contracts.validate_payload(contract_name, request_payload)
        log_event("validation.request", request_payload, depth + 1)
        emitted = self._invoke_subscribers("validation.request", request_payload, depth + 1)
        validation_result = None
        auxiliary = []
        for event_name, event_payload in emitted:
            if event_name == "validation.result":
                validation_result = event_payload
            else:
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        if not isinstance(validation_result, dict):
            return self._default_step_validation_result(
                workflow_payload,
                step_id,
                step_payload,
                primary_event,
                primary_payload,
                primary_value,
                context,
        )
        return validation_result

    def _reducer_input_data(self, primary_event: str | None, primary_payload: Any, step_payload: dict) -> Any:
        if not isinstance(primary_payload, dict):
            return None
        reduction_request = primary_payload.get("reduction_request")
        if isinstance(reduction_request, dict):
            source_field = str(reduction_request.get("source_field") or "").strip()
            if source_field == "stderr":
                stderr = primary_payload.get("stderr")
                if stderr not in (None, "", [], {}):
                    return stderr
            if source_field == "detail":
                detail = primary_payload.get("detail")
                if detail not in (None, "", [], {}):
                    return detail
        if primary_event in {"shell.result", "slurm.result"}:
            stdout = primary_payload.get("stdout")
            if isinstance(stdout, str):
                return stdout
        if primary_event == "sql.result":
            result = primary_payload.get("result")
            if isinstance(result, dict):
                rows = result.get("rows")
                if isinstance(rows, list):
                    row_count = result.get("total_matching_rows", result.get("row_count"))
                    return {
                        "task": step_payload.get("task", ""),
                        "sql": primary_payload.get("sql", ""),
                        "columns": result.get("columns", []),
                        "rows": rows,
                        "row_count": row_count if isinstance(row_count, int) else len(rows),
                        "returned_row_count": len(rows),
                    }
        if primary_event == "file.content":
            return primary_payload.get("content")
        if primary_event == "task.result":
            if "result" in primary_payload:
                return primary_payload.get("result")
            return primary_payload.get("detail")
        if "result" in primary_payload:
            return primary_payload.get("result")
        return primary_payload

    def _step_has_reduction_candidate(self, primary_event: str | None, primary_payload: Any) -> bool:
        if not primary_event or not isinstance(primary_payload, dict):
            return False
        if isinstance(primary_payload.get("reduction_request"), dict):
            return True
        if primary_payload.get("local_reduction_command"):
            return True
        return primary_payload.get("reduced_result") not in (None, "", [], {}) or primary_payload.get("refined_answer") not in (None, "", [], {})

    def _reduce_step_output(
        self,
        workflow_payload: dict,
        step_id: str,
        step_payload: dict,
        primary_event: str | None,
        primary_payload: Any,
        primary_value: Any,
        context: dict,
        depth: int,
    ) -> dict | None:
        if (
            "data.reduce" not in self.spec.get("events", {})
            or "data.reduced" not in self.spec.get("events", {})
            or not self.bus.get_subscribers("data.reduce")
        ):
            return None
        if not self._step_has_reduction_candidate(primary_event, primary_payload):
            return None
        reducer_payload = {
            "run_id": workflow_payload.get("run_id", ""),
            "attempt": workflow_payload.get("attempt"),
            "task": step_payload.get("task", ""),
            "original_task": workflow_payload.get("task", ""),
            "step_id": step_id,
            "target_agent": step_payload.get("target_agent", ""),
            "source_event": primary_event or "",
            "existing_reduced_result": (
                primary_payload.get("reduced_result") or primary_payload.get("refined_answer")
                if isinstance(primary_payload, dict)
                else None
            ),
            "input_data": self._reducer_input_data(primary_event, primary_payload, step_payload),
            "source_value": self._compact_value(primary_value, text_limit=240, row_limit=4),
            "available_context": {
                key: self._compact_value(value, text_limit=180, row_limit=3)
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
            "node": self._node_request(
                agent="data_reducer",
                role="reducer",
                request_event="data.reduce",
                task=str(step_payload.get("task") or ""),
                original_task=str(workflow_payload.get("task") or ""),
                step_id=step_id,
                target_agent=str(step_payload.get("target_agent") or ""),
                run_id=str(workflow_payload.get("run_id") or ""),
                attempt=workflow_payload.get("attempt") if isinstance(workflow_payload.get("attempt"), int) else None,
                scope="step",
                operation="reduce_step_output",
                status="pending",
            ),
        }
        if isinstance(primary_payload, dict):
            reduction_request = primary_payload.get("reduction_request")
            if isinstance(reduction_request, dict):
                reducer_payload["reduction_request"] = reduction_request
            local_reduction_command = primary_payload.get("local_reduction_command")
            if isinstance(local_reduction_command, str) and local_reduction_command.strip():
                reducer_payload["local_reduction_command"] = local_reduction_command.strip()
        if isinstance(primary_payload, dict):
            reducer_payload["source_payload"] = primary_payload
        contract_name = self.spec["events"]["data.reduce"]["contract"]
        self.contracts.validate_payload(contract_name, reducer_payload)
        log_event("data.reduce", reducer_payload, depth + 1)
        emitted = self._invoke_subscribers("data.reduce", reducer_payload, depth + 1)
        reduction_result = None
        auxiliary = []
        for event_name, event_payload in emitted:
            if event_name == "data.reduced":
                reduction_result = event_payload
            else:
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        return reduction_result if isinstance(reduction_result, dict) else None

    def _prefers_explicit_step_value(self, step_payload: dict) -> bool:
        if not isinstance(step_payload, dict):
            return False
        if isinstance(step_payload.get("result_mode"), str) and step_payload.get("result_mode").strip():
            return True
        instruction = step_payload.get("instruction")
        capture = instruction.get("capture") if isinstance(instruction, dict) else None
        return isinstance(capture, dict) and bool(capture)

    def _apply_reduction_result(self, primary_event: str | None, primary_payload: Any, reduction_result: dict | None):
        if not isinstance(primary_payload, dict) or not isinstance(reduction_result, dict):
            return primary_payload
        reduced_result = reduction_result.get("reduced_result")
        if reduced_result in (None, "", [], {}):
            return primary_payload
        updated_payload = copy.deepcopy(primary_payload)
        updated_payload["reduced_result"] = reduced_result
        updated_payload["refined_answer"] = reduced_result
        command = reduction_result.get("local_reduction_command")
        if isinstance(command, str) and command.strip():
            updated_payload["local_reduction_command"] = command.strip()
        strategy = reduction_result.get("strategy")
        if isinstance(strategy, str) and strategy.strip():
            updated_payload["reduction_strategy"] = strategy.strip()
        attempts = reduction_result.get("attempts")
        if isinstance(attempts, (int, float)):
            updated_payload["reduction_attempts"] = int(attempts)
        updated_payload["reduction"] = self._compact_value(reduction_result, text_limit=180, row_limit=3)
        nested_result = updated_payload.get("result")
        if isinstance(nested_result, dict):
            nested_result = copy.deepcopy(nested_result)
            nested_result["reduced_result"] = reduced_result
            nested_result["refined_answer"] = reduced_result
            updated_payload["result"] = nested_result
        if primary_event in {"shell.result", "slurm.result"} and not str(updated_payload.get("detail") or "").strip():
            updated_payload["detail"] = str(reduced_result)
        return updated_payload

    def _looks_like_scalar(self, value: Any) -> bool:
        if isinstance(value, (int, float, bool)):
            return True
        if isinstance(value, str):
            compact = value.strip()
            return bool(compact) and bool(re.fullmatch(r"-?\d+(?:\.\d+)?", compact))
        return False

    def _looks_like_boolean(self, value: Any) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, dict):
            for key in ("exists", "present", "removed", "missing", "ok", "success"):
                candidate = value.get(key)
                if isinstance(candidate, bool):
                    return True
                if isinstance(candidate, str) and candidate.strip().lower() in {
                    "true",
                    "false",
                    "yes",
                    "no",
                    "exists",
                    "missing",
                    "removed",
                    "present",
                }:
                    return True
        if isinstance(value, str):
            return value.strip().lower() in {"true", "false", "yes", "no", "exists", "missing", "0", "1"}
        return False

    def _looks_like_path(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        compact = value.strip()
        return bool(compact) and ("/" in compact or "\\" in compact or re.search(r"\.[A-Za-z0-9]+$", compact) is not None)

    def _has_nonempty_collection(self, value: Any) -> bool:
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, dict):
            if isinstance(value.get("rows"), list):
                return len(value["rows"]) > 0
            return len(value) > 0
        if isinstance(value, str):
            return len([line for line in value.splitlines() if line.strip()]) > 1
        return False

    def _assess_task_shape_result(self, task_shape: str, result: Any, prior_results: list[dict]) -> dict:
        if isinstance(result, dict) and "rows" in result and isinstance(result.get("rows"), list):
            rows = result.get("rows", [])
        else:
            rows = None

        if task_shape == "count":
            if self._structured_count_like(result):
                return {"verdict": "valid", "reason": "Count-like scalar detected.", "missing_requirements": [], "trace": ["Detected scalar output compatible with a count task."]}
            if isinstance(rows, list) and len(rows) == 1:
                return {"verdict": "uncertain", "reason": "Single-row structured output may still need local reduction into a scalar count.", "missing_requirements": ["scalar count"], "trace": ["Observed one structured row; count reduction may still be required."]}
            return {"verdict": "invalid", "reason": "Count task did not produce a scalar result.", "missing_requirements": ["scalar count"], "trace": ["Did not detect a scalar-like count output."]}

        if task_shape == "boolean_check":
            if self._looks_like_boolean(result):
                return {"verdict": "valid", "reason": "Boolean-like output detected.", "missing_requirements": [], "trace": ["Detected boolean-like output compatible with a boolean check."]}
            return {"verdict": "invalid", "reason": "Boolean check did not produce a boolean-like result.", "missing_requirements": ["boolean result"], "trace": ["Did not detect a clear yes/no or true/false style result."]}

        if task_shape == "save_artifact":
            if self._looks_like_path(result):
                return {"verdict": "valid", "reason": "Artifact path detected.", "missing_requirements": [], "trace": ["Detected a path-like output compatible with save_artifact."]}
            return {"verdict": "invalid", "reason": "Save task did not return an artifact path.", "missing_requirements": ["artifact path"], "trace": ["Did not detect a path-like result for the saved artifact."]}

        if task_shape == "schema_summary":
            for item in prior_results:
                if not isinstance(item, dict):
                    continue
                if item.get("event") == "sql.result":
                    summary = item.get("summary")
                    if isinstance(summary, dict) and ("sql" in summary or "rows" in summary or "row_count" in summary):
                        return {"verdict": "valid", "reason": "Schema-oriented SQL output detected.", "missing_requirements": [], "trace": ["Observed schema-oriented SQL output from a prior step."]}
            return {"verdict": "uncertain", "reason": "Schema workflow completed but the summary evidence is thin.", "missing_requirements": ["schema/table/column summary"], "trace": ["Workflow completed, but schema evidence was not strongly structured in the reduced output."]}

        if task_shape in {"list", "compare"}:
            if self._has_nonempty_collection(result):
                return {"verdict": "valid", "reason": "Collection-like output detected.", "missing_requirements": [], "trace": ["Detected non-empty collection output compatible with list/compare tasks."]}
            return {"verdict": "invalid", "reason": "List/compare task did not produce structured collection output.", "missing_requirements": ["non-empty collection"], "trace": ["Did not detect a non-empty collection-like result."]}

        if task_shape == "summarize_dataset":
            if isinstance(result, str) and result.strip():
                return {"verdict": "valid", "reason": "Summary text detected.", "missing_requirements": [], "trace": ["Detected non-empty summary text."]}
            return {"verdict": "uncertain", "reason": "Dataset summary task completed but summary text is not clearly reduced yet.", "missing_requirements": ["summary text"], "trace": ["Did not detect a clear reduced summary string."]}

        if task_shape in {"lookup", "command_execution"}:
            if result not in (None, "", [], {}):
                return {"verdict": "valid", "reason": "Material output detected.", "missing_requirements": [], "trace": ["Detected material output compatible with lookup/command execution."]}
            if any(item.get("status") == "completed" for item in prior_results):
                return {"verdict": "uncertain", "reason": "Steps completed but reduced output is sparse.", "missing_requirements": ["reduced final output"], "trace": ["Completed steps exist, but the reduced result is too sparse to accept confidently."]}
            return {"verdict": "invalid", "reason": "No usable output detected.", "missing_requirements": ["usable output"], "trace": ["Did not detect usable output for the completed workflow."]}

        return {"verdict": "uncertain", "reason": "Unknown task shape.", "missing_requirements": ["shape-specific output"], "trace": ["Task shape was not recognized by deterministic validation."]}

    def _emit_validation_progress(self, stage: str, workflow_payload: dict, depth: int, **extra):
        if "validation.progress" not in self.spec.get("events", {}):
            return
        payload = {
            "stage": stage,
            "task": workflow_payload.get("task", ""),
            "message": extra.pop("message", f"Validation stage: {stage}."),
        }
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        self.emit("validation.progress", payload, depth)

    def _validate_workflow_attempt(
        self,
        workflow_payload: dict,
        option: dict,
        workflow: dict,
        context: dict,
        attempt: int,
        total_attempts: int,
        depth: int,
    ) -> dict:
        limits = self._workflow_limits(workflow_payload)
        prior_attempts = workflow_payload.get("__attempts_so_far__", 0)
        validation_llm_budget_remaining = max(0, limits["max_validation_llm_calls"] - max(0, prior_attempts - 1))
        self._emit_validation_progress(
            "started",
            workflow_payload,
            depth + 1,
            option_id=option.get("id"),
            option_label=option.get("label"),
            attempt=attempt,
            total_attempts=total_attempts,
            message=f"Validating workflow option {attempt} of {total_attempts}.",
        )
        if "validation.request" not in self.spec.get("events", {}):
            result = self._default_validation_result(workflow_payload, option, workflow, context, attempt, total_attempts)
            self._emit_validation_progress(
                "passed" if result.get("valid") else "failed",
                workflow_payload,
                depth + 1,
                option_id=option.get("id"),
                option_label=option.get("label"),
                attempt=attempt,
                total_attempts=total_attempts,
                reason=result.get("reason"),
                trace=result.get("trace"),
                message=result.get("reason") or "Validation completed.",
            )
            return result

        request_payload = {
            "run_id": workflow_payload.get("run_id", ""),
            "task": workflow_payload.get("task", ""),
            "task_shape": self._infer_task_shape(str(workflow_payload.get("task") or ""), str(workflow_payload.get("task_shape") or "")),
            "attempt": attempt,
            "total_attempts": total_attempts,
            "option_id": option.get("id"),
            "option_label": option.get("label"),
            "option_reason": option.get("reason"),
            "workflow_status": workflow.get("status", ""),
            "validation_llm_budget_remaining": validation_llm_budget_remaining,
            "steps": [self._compact_value(step, text_limit=240, row_limit=4) for step in workflow.get("steps", []) if isinstance(step, dict)],
            "result": self._compact_value(workflow.get("result"), text_limit=240, row_limit=4),
            "presentation": workflow_payload.get("presentation", {}),
            "available_context": {
                key: self._compact_value(value, text_limit=180, row_limit=3)
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
            "node": self._node_request(
                agent="validator",
                role="validator",
                request_event="validation.request",
                task=str(workflow_payload.get("task") or ""),
                original_task=str(workflow_payload.get("task") or ""),
                run_id=str(workflow_payload.get("run_id") or ""),
                attempt=attempt,
                scope="workflow",
                operation="validate_workflow",
                status="pending",
            ),
        }
        if isinstance(workflow.get("error"), str) and workflow.get("error").strip():
            request_payload["error"] = workflow.get("error").strip()
        contract_name = self.spec["events"]["validation.request"]["contract"]
        self.contracts.validate_payload(contract_name, request_payload)
        log_event("validation.request", request_payload, depth + 1)
        emitted = self._invoke_subscribers("validation.request", request_payload, depth + 1)
        validation_result = None
        auxiliary = []
        for event_name, event_payload in emitted:
            if event_name == "validation.result":
                validation_result = event_payload
            else:
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        if not isinstance(validation_result, dict):
            validation_result = self._default_validation_result(workflow_payload, option, workflow, context, attempt, total_attempts)
        self._emit_validation_progress(
            "passed" if validation_result.get("valid") else "failed",
            workflow_payload,
            depth + 1,
            option_id=option.get("id"),
            option_label=option.get("label"),
            attempt=attempt,
            total_attempts=total_attempts,
            reason=validation_result.get("reason"),
            trace=validation_result.get("trace"),
            message=validation_result.get("reason") or "Validation completed.",
        )
        return validation_result

    def _emit_step_progress(self, stage: str, step_id: str, step_payload: dict, depth: int, **extra):
        if "step.progress" not in self.spec.get("events", {}):
            return
        payload = {
            "stage": stage,
            "step_id": step_id,
            "target_agent": step_payload.get("target_agent", ""),
            "task": step_payload.get("task", ""),
            "message": extra.pop("message", f"Step {step_id} {stage}."),
        }
        command = extra.pop("command", step_payload.get("command"))
        if isinstance(command, str) and command.strip():
            payload["command"] = command

        stdout = extra.pop("stdout", None)
        if isinstance(stdout, str) and stdout.strip():
            payload["stdout"] = self._compact_text(stdout, 1000)
            payload["stdout_excerpt"] = self._compact_text(stdout, 500)

        stderr = extra.pop("stderr", None)
        if isinstance(stderr, str) and stderr.strip():
            payload["stderr"] = self._compact_text(stderr, 600)
            payload["stderr_excerpt"] = self._compact_text(stderr, 300)
            
        local_reduction_command = extra.pop("local_reduction_command", None)
        if isinstance(local_reduction_command, str) and local_reduction_command.strip():
            payload["local_reduction_command"] = local_reduction_command
            
        sql = extra.pop("sql", step_payload.get("sql"))
        if isinstance(sql, str) and sql.strip():
            payload["sql"] = sql
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        self.emit("step.progress", payload, depth)

    def _compact_text(self, value: Any, limit: int = 800):
        if not isinstance(value, str):
            return value
        value = value.strip()
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    def _compact_rows(self, rows: Any, limit: int = 5):
        if not isinstance(rows, list):
            return rows
        compact = []
        for row in rows[:limit]:
            if isinstance(row, dict):
                compact.append({str(key): self._compact_value(val, text_limit=120) for key, val in row.items()})
            else:
                compact.append(self._compact_value(row, text_limit=120))
        return compact

    def _compact_schema(self, schema: Any):
        if not isinstance(schema, dict):
            return schema
        tables = schema.get("tables")
        if not isinstance(tables, dict):
            return self._compact_value(schema, text_limit=200)
        return {
            "dialect": schema.get("dialect"),
            "tables": {
                table_name: {
                    "column_count": len(table_info.get("columns", [])) if isinstance(table_info, dict) and isinstance(table_info.get("columns"), list) else None,
                    "foreign_key_count": len(table_info.get("foreign_keys", [])) if isinstance(table_info, dict) and isinstance(table_info.get("foreign_keys"), list) else None,
                }
                for table_name, table_info in list(tables.items())[:20]
                if isinstance(table_name, str)
            },
        }

    def _compact_value(self, value: Any, *, text_limit: int = 800, row_limit: int = 5):
        if isinstance(value, str):
            return self._compact_text(value, text_limit)
        if isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                return self._compact_rows(value, row_limit)
            return [self._compact_value(item, text_limit=text_limit, row_limit=row_limit) for item in value[:row_limit]]
        if isinstance(value, dict):
            compact = {}
            preferred_keys = [
                "detail",
                "reduced_result",
                "refined_answer",
                "sql",
                "command",
                "returncode",
                "ok",
                "kind",
                "row_count",
                "returned_row_count",
                "total_matching_rows",
                "truncated",
                "columns",
                "limit",
                "stats",
                "stdout",
                "stderr",
                "rows",
                "result",
                "schema",
                "queries",
                "local_reduction_command",
                "reduction_request",
                "reduction_strategy",
                "reduction",
                "node",
                "note",
                "error",
                "status",
            ]
            keys = [key for key in preferred_keys if key in value]
            if not keys:
                keys = list(value.keys())[:8]
            for key in keys:
                item = value.get(key)
                if key == "rows":
                    compact[key] = self._compact_rows(item, row_limit)
                    if isinstance(item, list) and len(item) > row_limit:
                        compact["rows_note"] = f"Showing first {row_limit} rows out of {len(item)}."
                elif key == "stdout":
                    compact["stdout_excerpt"] = self._compact_text(item, 500)
                elif key == "stderr":
                    compact["stderr_excerpt"] = self._compact_text(item, 300)
                elif key == "schema":
                    compact[key] = self._compact_schema(item)
                elif key == "queries" and isinstance(item, list):
                    compact[key] = [self._compact_value(entry, text_limit=240, row_limit=2) for entry in item[:5]]
                else:
                    compact[key] = self._compact_value(item, text_limit=text_limit, row_limit=row_limit)
            return compact
        return value

    def _compact_event_payload(self, event_name: str, payload: Any):
        if not isinstance(payload, dict):
            return payload
        compact = {"event": event_name}
        if event_name == "shell.result":
            compact.update(
                {
                    "detail": payload.get("detail"),
                    "command": payload.get("command"),
                    "returncode": payload.get("returncode"),
                    "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
                    "local_reduction_command": payload.get("local_reduction_command"),
                    "reduction_request": self._compact_value(payload.get("reduction_request"), text_limit=180, row_limit=3),
                    "reduction_strategy": payload.get("reduction_strategy"),
                    "reduction": self._compact_value(payload.get("reduction"), text_limit=180, row_limit=3),
                    "stdout": self._compact_text(payload.get("stdout"), 1000),
                    "stdout_excerpt": self._compact_text(payload.get("stdout"), 500),
                    "stderr": self._compact_text(payload.get("stderr"), 600),
                    "stderr_excerpt": self._compact_text(payload.get("stderr"), 300),
                    "stats": self._compact_value(payload.get("stats"), text_limit=120),
                    "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=3),
                    "node": self._compact_value(payload.get("node"), text_limit=160, row_limit=4),
                }
            )
            return compact
        if event_name == "sql.result":
            row_limit = 5
            detail = str(payload.get("detail") or "").strip().lower()
            result_payload = payload.get("result")
            if "database tables listed" in detail and isinstance(result_payload, dict):
                rows = result_payload.get("rows")
                if isinstance(rows, list) and rows and len(rows) <= 50:
                    row_limit = len(rows)
            compact.update(
                {
                    "detail": payload.get("detail"),
                    "sql": self._compact_text(payload.get("sql"), 500),
                    "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
                    "local_reduction_command": payload.get("local_reduction_command"),
                    "reduction_request": self._compact_value(payload.get("reduction_request"), text_limit=180, row_limit=3),
                    "reduction_strategy": payload.get("reduction_strategy"),
                    "reduction": self._compact_value(payload.get("reduction"), text_limit=180, row_limit=3),
                    "schema": self._compact_schema(payload.get("schema")),
                    "stats": self._compact_value(payload.get("stats"), text_limit=120),
                    "result": self._compact_value(result_payload, text_limit=240, row_limit=row_limit),
                    "node": self._compact_value(payload.get("node"), text_limit=160, row_limit=4),
                }
            )
            return compact
        if event_name == "slurm.result":
            compact.update(
                {
                    "detail": payload.get("detail"),
                    "command": payload.get("command"),
                    "returncode": payload.get("returncode"),
                    "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
                    "local_reduction_command": payload.get("local_reduction_command"),
                    "reduction_request": self._compact_value(payload.get("reduction_request"), text_limit=180, row_limit=3),
                    "reduction_strategy": payload.get("reduction_strategy"),
                    "reduction": self._compact_value(payload.get("reduction"), text_limit=180, row_limit=3),
                    "stdout": self._compact_text(payload.get("stdout"), 1000),
                    "stdout_excerpt": self._compact_text(payload.get("stdout"), 500),
                    "stderr": self._compact_text(payload.get("stderr"), 600),
                    "stderr_excerpt": self._compact_text(payload.get("stderr"), 300),
                    "stats": self._compact_value(payload.get("stats"), text_limit=120),
                    "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=3),
                    "node": self._compact_value(payload.get("node"), text_limit=160, row_limit=4),
                }
            )
            return compact
        if event_name == "task.result":
            compact.update(
                {
                    "detail": payload.get("detail"),
                    "status": payload.get("status"),
                    "error": payload.get("error"),
                    "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=3),
                    "replan_hint": self._compact_value(payload.get("replan_hint"), text_limit=180, row_limit=3),
                    "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
                    "local_reduction_command": payload.get("local_reduction_command"),
                    "reduction_request": self._compact_value(payload.get("reduction_request"), text_limit=180, row_limit=3),
                    "reduction_strategy": payload.get("reduction_strategy"),
                    "reduction": self._compact_value(payload.get("reduction"), text_limit=180, row_limit=3),
                    "node": self._compact_value(payload.get("node"), text_limit=160, row_limit=4),
                }
            )
            return compact
        return self._compact_value(payload, text_limit=240, row_limit=4)

    def _step_evidence(self, event_name: str | None, payload: Any, value: Any):
        compact_payload = self._compact_event_payload(event_name or "", payload) if event_name else self._compact_value(payload, text_limit=240, row_limit=4)
        evidence = {
            "event": event_name or "",
            "value_preview": self._compact_value(value, text_limit=240, row_limit=4),
            "payload": compact_payload,
        }
        if isinstance(compact_payload, dict):
            summary = (
                compact_payload.get("reduced_result")
                or compact_payload.get("detail")
                or compact_payload.get("stdout_excerpt")
                or compact_payload.get("result", {}).get("reduced_result") if isinstance(compact_payload.get("result"), dict) else None
            )
            if isinstance(summary, str) and summary.strip():
                evidence["summary_text"] = summary.strip()
        return evidence

    def _step_result_summary(self, event_name: str, payload: dict):
        if not isinstance(payload, dict):
            return None
        if event_name == "shell.result":
            return {
                "command": payload.get("command", ""),
                "returncode": payload.get("returncode"),
                "stdout_excerpt": self._compact_text(payload.get("stdout", ""), 500),
                "stderr_excerpt": self._compact_text(payload.get("stderr", ""), 300),
                "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
            }
        if event_name == "sql.result":
            result = payload.get("result")
            summary = {"sql": payload.get("sql", "")}
            stats = payload.get("stats")
            if isinstance(stats, dict):
                summary["stats"] = stats
            if isinstance(result, dict):
                summary["row_count"] = result.get("row_count")
                summary["returned_row_count"] = result.get("returned_row_count")
                summary["total_matching_rows"] = result.get("total_matching_rows")
                summary["truncated"] = result.get("truncated")
                summary["limit"] = result.get("limit")
                summary["columns"] = result.get("columns", [])
                summary["rows"] = self._compact_rows(result.get("rows", []), 5)
                summary["reduced_result"] = payload.get("reduced_result") or payload.get("refined_answer")
            return summary
        if event_name == "slurm.result":
            result = payload.get("result")
            summary = {"command": payload.get("command", "")}
            stats = payload.get("stats")
            if isinstance(stats, dict):
                summary["stats"] = stats
            if isinstance(result, dict):
                summary["returncode"] = result.get("returncode")
                summary["stdout_excerpt"] = self._compact_text(result.get("stdout", ""), 500)
                summary["stderr_excerpt"] = self._compact_text(result.get("stderr", ""), 300)
                summary["kind"] = result.get("kind", "")
                summary["reduced_result"] = payload.get("reduced_result") or payload.get("refined_answer")
            return summary
        if event_name == "task.result":
            return {"detail": payload.get("detail", ""), "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=3)}
        return payload

    def _step_failure_error(self, event_name: str, payload: dict, step_payload: dict | None = None):
        if not isinstance(payload, dict):
            return None
        if payload.get("status") == "failed":
            return payload.get("error") or payload.get("detail") or "Step failed."
        result = payload.get("result")
        if isinstance(result, dict) and result.get("ok") is False:
            return payload.get("error") or payload.get("detail") or "Step failed."
        if event_name == "shell.result" and payload.get("returncode") not in (0, None):
            instruction = step_payload.get("instruction") if isinstance(step_payload, dict) else None
            allowed = instruction.get("allow_returncodes") if isinstance(instruction, dict) else None
            if isinstance(allowed, list) and payload.get("returncode") in allowed:
                return None
            stderr = payload.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                return stderr.strip()
            return f"Command exited with status {payload.get('returncode')}."
        if event_name == "slurm.result" and payload.get("returncode") not in (0, None):
            stderr = payload.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                return stderr.strip()
            return f"Slurm command exited with status {payload.get('returncode')}."
        return None

    def _step_requests_replan(self, event_name: str, payload: dict):
        if not isinstance(payload, dict):
            return False
        status = payload.get("status")
        return event_name == "task.result" and status in {"needs_decomposition", "needs_clarification"}

    def _record_context_value(self, context: dict, step_id: str, primary_event: str, primary_payload: Any, primary_value: Any):
        context["prev"] = primary_value
        context[step_id] = primary_value
        context[f"{step_id}.value"] = primary_value
        context[f"{step_id}.event"] = primary_event
        context[f"{step_id}.detail"] = primary_payload.get("detail", "") if isinstance(primary_payload, dict) else ""
        if not isinstance(primary_payload, dict):
            context[f"{step_id}.result"] = primary_value
            return

        result = primary_payload.get("result", primary_value)
        context[f"{step_id}.result"] = result
        for key in (
            "stdout",
            "stderr",
            "sql",
            "command",
            "returncode",
            "schema",
            "stats",
            "reduced_result",
            "refined_answer",
            "local_reduction_command",
            "reduction_request",
            "error",
            "status",
        ):
            if key in primary_payload:
                context[f"{step_id}.{key}"] = primary_payload.get(key)
        if isinstance(result, dict):
            for key, value in result.items():
                context.setdefault(f"{step_id}.{key}", value)

    def _merge_child_context(self, context: dict, child_context: dict):
        for key, value in child_context.items():
            if key == "original_task":
                continue
            context[key] = value

    def _structured_step_result(
        self,
        step_id: str,
        step_payload: dict,
        primary_event: str | None,
        primary_payload: Any,
        primary_value: Any,
        status: str,
        duration_ms: float,
        validation_result: dict | None = None,
    ):
        detail = primary_payload.get("detail", "") if isinstance(primary_payload, dict) else ""
        envelope = {
            "step_id": step_id,
            "task": step_payload.get("task", ""),
            "target_agent": step_payload.get("target_agent", ""),
            "status": status,
            "event": primary_event or "",
            "detail": detail,
            "value": self._compact_value(primary_value, text_limit=240, row_limit=4),
            "result": self._compact_value(primary_payload.get("result", primary_value), text_limit=240, row_limit=4) if isinstance(primary_payload, dict) else self._compact_value(primary_value, text_limit=240, row_limit=4),
            "summary": self._step_result_summary(primary_event or "", primary_payload) if primary_event and isinstance(primary_payload, dict) else primary_value,
            "duration_ms": duration_ms,
            "evidence": self._step_evidence(primary_event, primary_payload, primary_value),
        }
        if isinstance(step_payload.get("instruction"), dict):
            envelope["instruction"] = self._compact_value(step_payload.get("instruction"), text_limit=240, row_limit=4)
        depends_on = step_payload.get("depends_on")
        if isinstance(depends_on, list) and depends_on:
            envelope["depends_on"] = [item for item in depends_on if isinstance(item, str)]
        when = step_payload.get("when")
        if isinstance(when, dict) and when:
            envelope["when"] = self._compact_value(when, text_limit=180, row_limit=3)
        if isinstance(validation_result, dict):
            envelope["validation"] = self._compact_value(validation_result, text_limit=180, row_limit=3)
        if isinstance(primary_payload, dict) and isinstance(primary_payload.get("node"), dict):
            envelope["node"] = self._compact_value(primary_payload.get("node"), text_limit=160, row_limit=4)
        if primary_event == "shell.result" and isinstance(primary_payload, dict):
            envelope["command"] = primary_payload.get("command")
            envelope["returncode"] = primary_payload.get("returncode")
            envelope["stdout"] = self._compact_text(primary_payload.get("stdout"), 1000)
            envelope["stderr"] = self._compact_text(primary_payload.get("stderr"), 600)
            envelope["stdout_excerpt"] = self._compact_text(primary_payload.get("stdout"), 500)
            envelope["stderr_excerpt"] = self._compact_text(primary_payload.get("stderr"), 300)
        if primary_event == "sql.result" and isinstance(primary_payload, dict):
            envelope["sql"] = primary_payload.get("sql")
            result = primary_payload.get("result")
            if isinstance(result, dict):
                envelope["row_count"] = result.get("row_count")
                envelope["returned_row_count"] = result.get("returned_row_count")
                envelope["total_matching_rows"] = result.get("total_matching_rows")
                envelope["truncated"] = result.get("truncated")
                envelope["limit"] = result.get("limit")
                envelope["columns"] = result.get("columns")
                envelope["rows"] = self._compact_rows(result.get("rows"), 5)
        if primary_event == "task.result" and isinstance(primary_payload, dict):
            result = primary_payload.get("result")
            if isinstance(result, dict):
                if isinstance(result.get("rows"), list):
                    envelope["rows"] = self._compact_rows(result.get("rows"), 5)
                if isinstance(result.get("columns"), list):
                    envelope["columns"] = result.get("columns")
                if result.get("row_count") is not None:
                    envelope["row_count"] = result.get("row_count")
                if result.get("exists") is not None:
                    envelope["exists"] = result.get("exists")
        if primary_event == "slurm.result" and isinstance(primary_payload, dict):
            envelope["command"] = primary_payload.get("command")
            envelope["returncode"] = primary_payload.get("returncode")
            envelope["stdout"] = self._compact_text(primary_payload.get("stdout"), 1000)
            envelope["stderr"] = self._compact_text(primary_payload.get("stderr"), 600)
            envelope["stdout_excerpt"] = self._compact_text(primary_payload.get("stdout"), 500)
            envelope["stderr_excerpt"] = self._compact_text(primary_payload.get("stderr"), 300)
        return envelope

    def _record_step_result(self, context: dict, step_result: dict):
        results = context.setdefault("__step_results__", [])
        if isinstance(results, list):
            results.append(step_result)
        index = context.setdefault("__step_results_by_id__", {})
        if isinstance(index, dict):
            index[step_result.get("step_id")] = step_result

    def _prior_step_results(self, context: dict):
        results = context.get("__step_results__", [])
        return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []

    def _dependency_results(self, context: dict, raw_step: dict):
        depends_on = raw_step.get("depends_on")
        if not isinstance(depends_on, list) or not depends_on:
            prior = self._prior_step_results(context)
            return [prior[-1]] if prior else []
        index = context.get("__step_results_by_id__", {})
        if not isinstance(index, dict):
            return []
        results = []
        for item in depends_on:
            if isinstance(item, str) and isinstance(index.get(item), dict):
                results.append(index[item])
        return results

    def _ready_to_run(self, raw_step: dict, completed_ids: set[str]):
        depends_on = raw_step.get("depends_on")
        if not isinstance(depends_on, list) or not depends_on:
            return True
        return all(isinstance(item, str) and item in completed_ids for item in depends_on)

    def _replan_signature(self, step_payload: dict, error: str):
        blob = json.dumps(
            {
                "task": step_payload.get("task"),
                "target_agent": step_payload.get("target_agent"),
                "command": step_payload.get("command"),
                "sql": step_payload.get("sql"),
                "error": error,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _request_replan(
        self,
        step_id: str,
        step_payload: dict,
        workflow_payload: dict,
        primary_event: str | None,
        primary_payload: Any,
        error: str,
        context: dict,
        depth: int,
        *,
        failure_class: str | None = None,
        validation_result: dict | None = None,
    ):
        if "planner.replan.request" not in self.spec.get("events", {}):
            return None
        replan_payload = {
            "task": step_payload.get("task", ""),
            "task_shape": workflow_payload.get("task_shape", ""),
            "step_id": step_id,
            "original_task": workflow_payload.get("task", context.get("original_task", "")),
            "target_agent": step_payload.get("target_agent", ""),
            "reason": error,
            "failure_class": failure_class
            or ("needs_decomposition" if self._step_requests_replan(primary_event or "", primary_payload) else "execution_failed"),
            "event": primary_event or "",
            "available_context": {
                key: value
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
            "presentation": workflow_payload.get("presentation", {}),
        }
        if isinstance(validation_result, dict):
            replan_payload["validation"] = self._compact_value(validation_result, text_limit=180, row_limit=3)
        if isinstance(primary_payload, dict):
            if primary_payload.get("error"):
                replan_payload["error"] = primary_payload.get("error")
            if primary_payload.get("replan_hint") is not None:
                replan_payload["partial_result"] = primary_payload.get("replan_hint")
            elif primary_payload.get("result") is not None:
                replan_payload["partial_result"] = primary_payload.get("result")
        contract_name = self.spec["events"]["planner.replan.request"]["contract"]
        self.contracts.validate_payload(contract_name, replan_payload)
        log_event("planner.replan.request", replan_payload, depth + 1)
        emitted = self._invoke_subscribers("planner.replan.request", replan_payload, depth + 1)
        replan_result = None
        auxiliary = []
        for event_name, event_payload in emitted:
            if event_name == "planner.replan.result":
                replan_result = event_payload
            elif event_name == "plan.progress":
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        trace = {
            "scope": "step",
            "status": "received" if isinstance(replan_result, dict) else "failed",
            "step_id": step_id,
            "replace_step_id": replan_result.get("replace_step_id") if isinstance(replan_result, dict) else step_id,
            "reason": str((replan_result or {}).get("reason") or error),
            "failure_class": replan_payload.get("failure_class"),
            "event": primary_event or "",
            "request": replan_payload,
            "result": replan_result if isinstance(replan_result, dict) else None,
        }
        if isinstance(replan_result, dict) and isinstance(replan_result.get("steps"), list):
            trace["steps"] = replan_result.get("steps")
        return {
            "request": replan_payload,
            "result": replan_result if isinstance(replan_result, dict) else None,
            "steps": replan_result.get("steps") if isinstance(replan_result, dict) else None,
            "trace": self._compact_replan_record(trace),
        }

    def _build_clarification_payload(
        self,
        step_id: str,
        step_payload: dict,
        workflow_payload: dict,
        primary_payload: Any,
        error: str,
        context: dict,
    ):
        detail = error or "More information is required to continue."
        question = None
        missing_information = []
        if isinstance(primary_payload, dict):
            question = primary_payload.get("question")
            replan_hint = primary_payload.get("replan_hint")
            if isinstance(replan_hint, dict):
                hint_question = replan_hint.get("question")
                if isinstance(hint_question, str) and hint_question.strip():
                    question = hint_question.strip()
                hint_missing = replan_hint.get("missing_information")
                if isinstance(hint_missing, list):
                    missing_information = [item for item in hint_missing if isinstance(item, str) and item.strip()]
        if not isinstance(question, str) or not question.strip():
            if missing_information:
                question = "I need a bit more information to continue: " + "; ".join(missing_information)
            else:
                question = detail
        clarification = {
            "task": workflow_payload.get("task", context.get("original_task", "")),
            "task_shape": workflow_payload.get("task_shape", ""),
            "step_id": step_id,
            "detail": detail,
            "question": question,
            "available_context": {
                key: value
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
        }
        if missing_information:
            clarification["missing_information"] = missing_information
        return clarification

    def _execute_single_workflow_step(
        self,
        raw_step: dict,
        workflow_payload: dict,
        context: dict,
        depth: int,
    ):
        step_id = raw_step.get("id") or "step"
        nested_steps = raw_step.get("steps")
        if isinstance(nested_steps, list) and nested_steps:
            nested_context = dict(context)
            nested_payload = {
                "task": raw_step.get("task", workflow_payload.get("task", "")),
                "step_id": step_id,
                "presentation": workflow_payload.get("presentation", {}),
                "task_shape": workflow_payload.get("task_shape", ""),
                "run_id": workflow_payload.get("run_id", ""),
                "attempt": workflow_payload.get("attempt"),
            }
            log_event("task.plan", nested_payload, depth + 1)
            nested = self._execute_workflow_steps(nested_steps, nested_payload, nested_context, depth + 1)
            return {
                "kind": "nested",
                "step_id": step_id,
                "task": raw_step.get("task", workflow_payload.get("task", "")),
                "target_agent": raw_step.get("target_agent"),
                "nested": nested,
                "nested_context": nested_context,
            }

        step_payload = {
            key: value
            for key, value in raw_step.items()
            if key not in {"id", "depends_on", "steps", "replan_budget", "when"} and not str(key).startswith("__")
        }
        step_payload = self._resolve_templates(step_payload, context)
        step_payload = self._resolve_references(step_payload, context)
        step_payload.setdefault("task", workflow_payload.get("task", ""))
        if isinstance(workflow_payload.get("task_shape"), str) and workflow_payload.get("task_shape"):
            step_payload.setdefault("task_shape", workflow_payload.get("task_shape"))
        if workflow_payload.get("run_id") not in (None, ""):
            step_payload.setdefault("run_id", workflow_payload.get("run_id"))
        if workflow_payload.get("attempt") not in (None, ""):
            step_payload.setdefault("attempt", workflow_payload.get("attempt"))
        step_payload["step_id"] = step_id
        step_payload["original_task"] = workflow_payload.get("task", context.get("original_task", ""))
        prior_step_results = self._prior_step_results(context)
        dependency_results = self._dependency_results(context, raw_step)
        if prior_step_results:
            step_payload["prior_step_results"] = prior_step_results
            step_payload["previous_step_result"] = prior_step_results[-1]
        if dependency_results:
            step_payload["dependency_results"] = dependency_results
        if isinstance(raw_step.get("depends_on"), list) and raw_step.get("depends_on"):
            step_payload["depends_on"] = [item for item in raw_step.get("depends_on") if isinstance(item, str)]
        if isinstance(raw_step.get("when"), dict) and raw_step.get("when"):
            step_payload["when"] = raw_step.get("when")
        resolved_targets = self._resolve_target_subscribers(
            str(step_payload.get("target_agent") or ""),
            self.bus.get_subscribers("task.plan"),
            step_payload,
        )
        if len(resolved_targets) == 1:
            step_payload["target_agent"] = resolved_targets[0]
        instruction = step_payload.get("instruction") if isinstance(step_payload.get("instruction"), dict) else {}
        step_payload["node"] = self._node_request(
            agent=str(step_payload.get("target_agent") or ""),
            role="executor",
            request_event="task.plan",
            task=str(step_payload.get("task") or ""),
            original_task=str(step_payload.get("original_task") or ""),
            step_id=step_id,
            target_agent=str(step_payload.get("target_agent") or ""),
            run_id=str(step_payload.get("run_id") or ""),
            attempt=step_payload.get("attempt") if isinstance(step_payload.get("attempt"), int) else None,
            scope="step",
            operation=str(instruction.get("operation") or ""),
            status="pending",
        )
        if not self._evaluate_when(raw_step.get("when"), context):
            return {
                "kind": "skipped",
                "step_id": step_id,
                "task": step_payload.get("task", ""),
                "target_agent": step_payload.get("target_agent"),
                "step_payload": step_payload,
                "status": "skipped",
                "duration_ms": 0,
            }
        unresolved_keys = self._unresolved_template_keys(step_payload)
        if unresolved_keys:
            error = (
                f"Plan step '{step_id}' referenced unavailable result(s): "
                f"{', '.join(sorted(set(unresolved_keys)))}."
            )
            return {
                "kind": "leaf",
                "step_id": step_id,
                "task": step_payload.get("task", ""),
                "target_agent": step_payload.get("target_agent"),
                "step_payload": step_payload,
                "status": "failed",
                "duration_ms": 0,
                "error": error,
                "unresolved": True,
            }

        log_event("task.plan", step_payload, depth + 1)
        contract_name = self.spec["events"]["task.plan"]["contract"]
        self.contracts.validate_payload(contract_name, step_payload)
        self._emit_step_progress(
            "started",
            step_id,
            step_payload,
            depth + 1,
            message=f"Starting step {step_id}: {step_payload.get('task', '')}",
        )

        step_started = time.perf_counter()
        step_results = self._invoke_subscribers("task.plan", step_payload, depth + 1)
        duration_ms = round((time.perf_counter() - step_started) * 1000, 2)
        if not step_results:
            primary_event = None
            primary_payload = None
            primary_value = None
            result_summary = None
            failure_error = f"Plan step '{step_id}' produced no result."
        else:
            primary_event, primary_payload = step_results[0]
            primary_value = self._extract_result_value(primary_event, primary_payload, step_payload)
            result_summary = self._step_result_summary(primary_event, primary_payload)
            failure_error = self._step_failure_error(primary_event, primary_payload, step_payload)

        return {
            "kind": "leaf",
            "step_id": step_id,
            "task": step_payload.get("task", ""),
            "target_agent": step_payload.get("target_agent"),
            "step_payload": step_payload,
            "status": "completed" if not failure_error and not self._step_requests_replan(primary_event or "", primary_payload) else "failed",
            "duration_ms": duration_ms,
            "primary_event": primary_event,
            "primary_payload": primary_payload,
            "primary_value": primary_value,
            "result_summary": result_summary,
            "failure_error": failure_error,
            "step_results": step_results,
        }

    def _execute_workflow_steps(
        self,
        steps: list,
        payload: dict,
        context: dict,
        depth: int,
        *,
        resume_state: dict | None = None,
        checkpoint=None,
    ):
        limits = self._workflow_limits(payload)
        if isinstance(resume_state, dict):
            records = [copy.deepcopy(item) for item in resume_state.get("records", []) if isinstance(item, dict)]
            final_results = copy.deepcopy(resume_state.get("final_results", [])) if isinstance(resume_state.get("final_results"), list) else []
            final_value = copy.deepcopy(resume_state.get("final_value"))
            pending = [copy.deepcopy(step) for step in resume_state.get("pending", []) if isinstance(step, dict)]
            inflight_step = resume_state.get("inflight_step")
            if isinstance(inflight_step, dict):
                pending.insert(0, copy.deepcopy(inflight_step))
            completed_ids = {
                item
                for item in resume_state.get("completed_ids", [])
                if isinstance(item, str)
            }
            step_replans = {
                key: max(0, int(value))
                for key, value in (resume_state.get("step_replans") or {}).items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }
            step_validation_requests = max(0, int(resume_state.get("step_validation_requests", 0)))
            restored_context = resume_state.get("context")
            if isinstance(restored_context, dict):
                context.clear()
                context.update(copy.deepcopy(restored_context))
        else:
            records = []
            final_results = []
            final_value = None
            pending = [step for step in steps if isinstance(step, dict)]
            completed_ids: set[str] = set()
            step_replans: dict[str, int] = {}
            step_validation_requests = 0

        def persist_workflow_state(
            status: str,
            *,
            inflight_step: dict | None = None,
            error: str | None = None,
            clarification: dict | None = None,
        ):
            if not callable(checkpoint):
                return
            snapshot = {
                "status": status,
                "records": copy.deepcopy(records),
                "final_results": copy.deepcopy(final_results),
                "final_value": copy.deepcopy(final_value),
                "pending": copy.deepcopy(pending),
                "completed_ids": sorted(completed_ids),
                "step_replans": copy.deepcopy(step_replans),
                "step_validation_requests": step_validation_requests,
                "context": copy.deepcopy(context),
            }
            if isinstance(inflight_step, dict):
                snapshot["inflight_step"] = copy.deepcopy(inflight_step)
            if isinstance(error, str) and error.strip():
                snapshot["error"] = error.strip()
            if isinstance(clarification, dict) and clarification:
                snapshot["clarification"] = copy.deepcopy(clarification)
            checkpoint(snapshot)

        while pending:
            ready_index = next(
                (index for index, item in enumerate(pending) if self._ready_to_run(item, completed_ids)),
                None,
            )
            if ready_index is None:
                error = "Workflow is blocked by unresolved or cyclic depends_on references."
                persist_workflow_state("failed", error=error)
                return {
                    "status": "failed",
                    "steps": records,
                    "result": final_value,
                    "error": error,
                    "final_results": final_results,
                }
            raw_step = pending.pop(ready_index)
            persist_workflow_state("running", inflight_step=raw_step)
            outcome = self._execute_single_workflow_step(raw_step, payload, context, depth)
            step_id = outcome["step_id"]
            if outcome["kind"] == "skipped":
                step_payload = outcome["step_payload"]
                self._record_step_result(
                    context,
                    {
                        "step_id": step_id,
                        "task": step_payload.get("task", ""),
                        "target_agent": step_payload.get("target_agent", ""),
                        "status": "skipped",
                        "event": "",
                        "detail": "Condition evaluated to false.",
                        "value": None,
                        "result": None,
                        "summary": None,
                        "duration_ms": 0,
                    },
                )
                completed_ids.add(step_id)
                records.append(
                    {
                        "id": step_id,
                        "task": step_payload.get("task", ""),
                        "target_agent": step_payload.get("target_agent"),
                        "status": "skipped",
                        "instruction": self._compact_value(step_payload.get("instruction"), text_limit=240, row_limit=4)
                        if isinstance(step_payload.get("instruction"), dict)
                        else None,
                        "depends_on": [item for item in step_payload.get("depends_on", []) if isinstance(item, str)],
                        "when": self._compact_value(step_payload.get("when"), text_limit=180, row_limit=3)
                        if isinstance(step_payload.get("when"), dict)
                        else None,
                        "result": None,
                    }
                )
                persist_workflow_state("running")
                continue
            if outcome["kind"] == "nested":
                nested = outcome["nested"]
                self._merge_child_context(context, outcome["nested_context"])
                final_results = nested.get("final_results", [])
                final_value = nested.get("result")
                self._record_context_value(context, step_id, "task.result", {"result": final_value}, final_value)
                completed_ids.add(step_id)
                nested_record = {
                    "id": step_id,
                    "task": outcome["task"],
                    "target_agent": outcome.get("target_agent"),
                    "status": nested["status"],
                    "depends_on": [item for item in raw_step.get("depends_on", []) if isinstance(item, str)],
                    "when": self._compact_value(raw_step.get("when"), text_limit=180, row_limit=3)
                    if isinstance(raw_step.get("when"), dict)
                    else None,
                    "steps": nested["steps"],
                    "result": final_value,
                }
                records.append(
                    self._attach_step_runtime_metadata(
                        raw_step,
                        nested_record,
                        clarification=nested.get("clarification") if isinstance(nested.get("clarification"), dict) else None,
                    )
                )
                if nested["status"] != "completed":
                    result = {
                        "status": nested["status"],
                        "steps": records,
                        "result": final_value,
                        "error": nested.get("error"),
                        "final_results": final_results,
                    }
                    if nested.get("clarification"):
                        result["clarification"] = nested["clarification"]
                    persist_workflow_state(
                        nested["status"],
                        error=nested.get("error"),
                        clarification=nested.get("clarification") if isinstance(nested.get("clarification"), dict) else None,
                    )
                    return result
                persist_workflow_state("running")
                continue

            step_payload = outcome["step_payload"]
            duration_ms = outcome.get("duration_ms", 0)
            if outcome.get("unresolved"):
                error = outcome["error"]
                self._emit_step_progress(
                    "failed",
                    step_id,
                    step_payload,
                    depth + 1,
                    message=error,
                    error=error,
                    duration_ms=duration_ms,
                )
                records.append(
                    {
                        "id": step_id,
                        "task": step_payload.get("task", ""),
                        "target_agent": step_payload.get("target_agent"),
                        "status": "failed",
                        "instruction": self._compact_value(step_payload.get("instruction"), text_limit=240, row_limit=4)
                        if isinstance(step_payload.get("instruction"), dict)
                        else None,
                        "depends_on": [item for item in step_payload.get("depends_on", []) if isinstance(item, str)],
                        "when": self._compact_value(step_payload.get("when"), text_limit=180, row_limit=3)
                        if isinstance(step_payload.get("when"), dict)
                        else None,
                        "error": error,
                        "duration_ms": duration_ms,
                    }
                )
                persist_workflow_state("failed", error=error)
                return {
                    "status": "failed",
                    "steps": records,
                    "result": final_value,
                    "error": error,
                    "final_results": [],
                }

            primary_event = outcome.get("primary_event")
            primary_payload = outcome.get("primary_payload")
            primary_value = outcome.get("primary_value")
            result_summary = outcome.get("result_summary")
            step_results = outcome.get("step_results") or []
            failure_error = outcome.get("failure_error")

            if failure_error or self._step_requests_replan(primary_event or "", primary_payload):
                error = failure_error or (
                    primary_payload.get("detail")
                    if isinstance(primary_payload, dict)
                    else None
                ) or "Step failed."
                needs_clarification = isinstance(primary_payload, dict) and primary_payload.get("status") == "needs_clarification"
                step_routing = self._route_step_attempt(
                    step_id,
                    step_payload,
                    primary_event,
                    None,
                    error,
                    "execution",
                    limits,
                    step_replans,
                    step_validation_requests,
                    needs_clarification=needs_clarification,
                    failure_class="needs_clarification" if needs_clarification else "execution_failed",
                )
                replan_envelope = None
                if step_routing.get("action") == "replan_step":
                    replan_envelope = self._request_replan(
                        step_id,
                        step_payload,
                        payload,
                        primary_event,
                        primary_payload,
                        error,
                        context,
                        depth,
                    )
                    replanned_steps = replan_envelope.get("steps") if isinstance(replan_envelope, dict) else None
                    if isinstance(replanned_steps, list) and replanned_steps:
                        step_replans[step_id] = step_replans.get(step_id, 0) + 1
                        self._emit_step_progress(
                            "retrying",
                            step_id,
                            step_payload,
                            depth + 1,
                            message=error,
                            event=primary_event,
                            duration_ms=duration_ms,
                            result=result_summary,
                            reason=error,
                        )
                        replanned_step = copy.deepcopy(raw_step)
                        routing_history = list(self._step_runtime_history(replanned_step, "__routing_history__"))
                        compact_routing = self._compact_routing_record(step_routing)
                        if compact_routing:
                            routing_history.append(compact_routing)
                            replanned_step["__routing_history__"] = routing_history
                        replan_history = list(self._step_runtime_history(replanned_step, "__replan_history__"))
                        compact_replan = replan_envelope.get("trace") if isinstance(replan_envelope, dict) else None
                        if isinstance(compact_replan, dict):
                            replan_history.append(compact_replan)
                            replanned_step["__replan_history__"] = replan_history
                        replanned_step["steps"] = replanned_steps
                        pending.insert(ready_index, replanned_step)
                        persist_workflow_state("running")
                        continue
                self._emit_step_progress(
                    "failed",
                    step_id,
                    step_payload,
                    depth + 1,
                    message=error,
                    event=primary_event,
                    duration_ms=duration_ms,
                    result=result_summary,
                    error=error,
                    sql=primary_payload.get("sql") if isinstance(primary_payload, dict) else None,
                    stdout=primary_payload.get("stdout") if isinstance(primary_payload, dict) else None,
                    stderr=primary_payload.get("stderr") if isinstance(primary_payload, dict) else None,
                )
                clarification = None
                if isinstance(primary_payload, dict) and primary_payload.get("status") == "needs_clarification":
                    clarification = self._build_clarification_payload(
                        step_id,
                        step_payload,
                        payload,
                        primary_payload,
                        error,
                        context,
                    )
                failed_record = {
                    "id": step_id,
                    "task": step_payload.get("task", ""),
                    "target_agent": step_payload.get("target_agent"),
                    "status": "needs_clarification" if self._step_requests_replan(primary_event or "", primary_payload) else "failed",
                    "instruction": self._compact_value(step_payload.get("instruction"), text_limit=240, row_limit=4)
                    if isinstance(step_payload.get("instruction"), dict)
                    else None,
                    "depends_on": [item for item in step_payload.get("depends_on", []) if isinstance(item, str)],
                    "when": self._compact_value(step_payload.get("when"), text_limit=180, row_limit=3)
                    if isinstance(step_payload.get("when"), dict)
                    else None,
                    "event": primary_event,
                    "duration_ms": duration_ms,
                    "payload": self._compact_event_payload(primary_event or "", primary_payload),
                    "result": self._compact_value(primary_value, text_limit=240, row_limit=4),
                    "evidence": self._step_evidence(primary_event, primary_payload, primary_value),
                    "error": error,
                    "emitted": [
                        {"event": event_name, "payload": self._compact_event_payload(event_name, event_payload)}
                        for event_name, event_payload in step_results
                    ],
                }
                records.append(
                    self._attach_step_runtime_metadata(
                        raw_step,
                        failed_record,
                        routing=step_routing,
                        replan=replan_envelope.get("trace") if isinstance(replan_envelope, dict) else None,
                        clarification=clarification,
                    )
                )
                if isinstance(clarification, dict):
                    persist_workflow_state(
                        "needs_clarification",
                        error=error,
                        clarification=clarification,
                    )
                    return {
                        "status": "needs_clarification",
                        "steps": records,
                        "result": primary_value,
                        "error": error,
                        "clarification": clarification,
                        "final_results": step_results,
                    }
                persist_workflow_state("failed", error=error)
                return {
                    "status": "failed",
                    "steps": records,
                    "result": primary_value,
                    "error": error,
                    "final_results": step_results,
                }

            reduction_result = self._reduce_step_output(
                payload,
                step_id,
                step_payload,
                primary_event,
                primary_payload,
                primary_value,
                context,
                depth,
            )
            if isinstance(reduction_result, dict) and reduction_result.get("reduced_result") not in (None, "", [], {}):
                primary_payload = self._apply_reduction_result(primary_event, primary_payload, reduction_result)
                if not self._prefers_explicit_step_value(step_payload):
                    primary_value = reduction_result.get("reduced_result")
                result_summary = self._step_result_summary(primary_event or "", primary_payload) if primary_event and isinstance(primary_payload, dict) else result_summary
                self._emit_step_progress(
                    "reduced",
                    step_id,
                    step_payload,
                    depth + 1,
                    message=f"Reduced step {step_id} output through data_reducer.",
                    event="data.reduced",
                    duration_ms=duration_ms,
                    result=self._compact_value(reduction_result.get("reduced_result"), text_limit=180, row_limit=3),
                    local_reduction_command=reduction_result.get("local_reduction_command"),
                    reduction=self._compact_value(reduction_result, text_limit=180, row_limit=3),
                )

            step_validation_requests += 1
            step_validation = self._validate_step_attempt(
                payload,
                step_id,
                step_payload,
                primary_event,
                primary_payload,
                primary_value,
                context,
                step_validation_requests,
                depth,
            )
            step_routing = self._route_step_attempt(
                step_id,
                step_payload,
                primary_event,
                step_validation,
                str(step_validation.get("reason") or "Step output diverged from the requested intent."),
                "validation",
                limits,
                step_replans,
                step_validation_requests,
                failure_class="validation_failed",
            )
            if step_routing.get("action") != "accept_step":
                error = str(step_validation.get("reason") or "Step output diverged from the requested intent.")
                replan_envelope = None
                if step_routing.get("action") == "replan_step":
                    replan_envelope = self._request_replan(
                        step_id,
                        step_payload,
                        payload,
                        primary_event,
                        primary_payload,
                        error,
                        context,
                        depth,
                        failure_class="validation_failed",
                        validation_result=step_validation,
                    )
                    replanned_steps = replan_envelope.get("steps") if isinstance(replan_envelope, dict) else None
                    if isinstance(replanned_steps, list) and replanned_steps:
                        step_replans[step_id] = step_replans.get(step_id, 0) + 1
                        self._emit_step_progress(
                            "retrying",
                            step_id,
                            step_payload,
                            depth + 1,
                            message=error,
                            event=primary_event,
                            duration_ms=duration_ms,
                            result=result_summary,
                            reason=error,
                            validation=self._compact_value(step_validation, text_limit=180, row_limit=3),
                        )
                        replanned_step = copy.deepcopy(raw_step)
                        routing_history = list(self._step_runtime_history(replanned_step, "__routing_history__"))
                        compact_routing = self._compact_routing_record(step_routing)
                        if compact_routing:
                            routing_history.append(compact_routing)
                            replanned_step["__routing_history__"] = routing_history
                        replan_history = list(self._step_runtime_history(replanned_step, "__replan_history__"))
                        compact_replan = replan_envelope.get("trace") if isinstance(replan_envelope, dict) else None
                        if isinstance(compact_replan, dict):
                            replan_history.append(compact_replan)
                            replanned_step["__replan_history__"] = replan_history
                        replanned_step["steps"] = replanned_steps
                        pending.insert(ready_index, replanned_step)
                        persist_workflow_state("running")
                        continue

                self._emit_step_progress(
                    "failed",
                    step_id,
                    step_payload,
                    depth + 1,
                    message=error,
                    event=primary_event,
                    duration_ms=duration_ms,
                    result=result_summary,
                    error=error,
                    validation=self._compact_value(step_validation, text_limit=180, row_limit=3),
                    sql=primary_payload.get("sql") if isinstance(primary_payload, dict) else None,
                    stdout=primary_payload.get("stdout") if isinstance(primary_payload, dict) else None,
                    stderr=primary_payload.get("stderr") if isinstance(primary_payload, dict) else None,
                )
                failed_record = {
                    "id": step_id,
                    "task": step_payload.get("task", ""),
                    "target_agent": step_payload.get("target_agent"),
                    "status": "failed",
                    "instruction": self._compact_value(step_payload.get("instruction"), text_limit=240, row_limit=4)
                    if isinstance(step_payload.get("instruction"), dict)
                    else None,
                    "depends_on": [item for item in step_payload.get("depends_on", []) if isinstance(item, str)],
                    "when": self._compact_value(step_payload.get("when"), text_limit=180, row_limit=3)
                    if isinstance(step_payload.get("when"), dict)
                    else None,
                    "event": primary_event,
                    "duration_ms": duration_ms,
                    "payload": self._compact_event_payload(primary_event or "", primary_payload),
                    "result": self._compact_value(primary_value, text_limit=240, row_limit=4),
                    "evidence": self._step_evidence(primary_event, primary_payload, primary_value),
                    "validation": self._compact_value(step_validation, text_limit=180, row_limit=3),
                    "error": error,
                    "emitted": [
                        {"event": event_name, "payload": self._compact_event_payload(event_name, event_payload)}
                        for event_name, event_payload in step_results
                    ],
                }
                records.append(
                    self._attach_step_runtime_metadata(
                        raw_step,
                        failed_record,
                        routing=step_routing,
                        replan=replan_envelope.get("trace") if isinstance(replan_envelope, dict) else None,
                    )
                )
                persist_workflow_state("failed", error=error)
                return {
                    "status": "failed",
                    "steps": records,
                    "result": primary_value,
                    "error": error,
                    "final_results": step_results,
                }

            self._emit_step_progress(
                "completed",
                step_id,
                step_payload,
                depth + 1,
                message=f"Completed step {step_id}.",
                event=primary_event,
                duration_ms=duration_ms,
                result=result_summary,
                sql=primary_payload.get("sql") if isinstance(primary_payload, dict) else None,
                command=primary_payload.get("command") if isinstance(primary_payload, dict) else None,
                local_reduction_command=primary_payload.get("local_reduction_command") if isinstance(primary_payload, dict) else None,
                stdout=primary_payload.get("stdout") if isinstance(primary_payload, dict) else None,
                stderr=primary_payload.get("stderr") if isinstance(primary_payload, dict) else None,
                validation=self._compact_value(step_validation, text_limit=180, row_limit=3),
            )
            self._record_context_value(context, step_id, primary_event, primary_payload, primary_value)
            self._record_step_result(
                context,
                self._structured_step_result(
                    step_id,
                    step_payload,
                    primary_event,
                    primary_payload,
                    primary_value,
                    "completed",
                    duration_ms,
                    validation_result=step_validation,
                ),
            )
            completed_ids.add(step_id)
            final_results = step_results
            final_value = primary_value
            completed_record = {
                "id": step_id,
                "task": step_payload.get("task", ""),
                "target_agent": step_payload.get("target_agent"),
                "status": "completed",
                "instruction": self._compact_value(step_payload.get("instruction"), text_limit=240, row_limit=4)
                if isinstance(step_payload.get("instruction"), dict)
                else None,
                "depends_on": [item for item in step_payload.get("depends_on", []) if isinstance(item, str)],
                "when": self._compact_value(step_payload.get("when"), text_limit=180, row_limit=3)
                if isinstance(step_payload.get("when"), dict)
                else None,
                "event": primary_event,
                "duration_ms": duration_ms,
                "payload": self._compact_event_payload(primary_event or "", primary_payload),
                "result": self._compact_value(primary_value, text_limit=240, row_limit=4),
                "evidence": self._step_evidence(primary_event, primary_payload, primary_value),
                "validation": self._compact_value(step_validation, text_limit=180, row_limit=3),
                "emitted": [
                    {"event": event_name, "payload": self._compact_event_payload(event_name, event_payload)}
                    for event_name, event_payload in step_results
                ],
            }
            records.append(self._attach_step_runtime_metadata(raw_step, completed_record, routing=step_routing))
            persist_workflow_state("running")

        persist_workflow_state("completed")
        return {
            "status": "completed",
            "steps": records,
            "result": final_value,
            "final_results": final_results,
        }

    def shutdown(self):
        for process in self._managed_processes:
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        self._managed_processes.clear()
