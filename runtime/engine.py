import atexit
import copy
import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any
from importlib import import_module
from urllib.parse import urlparse

import requests

from .console import log_boot, log_event, log_event_handler
from .contracts import ContractRegistry
from .event_bus import EventBus
from .registry import ADAPTER_REGISTRY


class Engine:

    def __init__(self, spec: dict, global_timeout_seconds: float | None = None):
        self.spec = copy.deepcopy(spec)
        self.global_timeout_seconds = global_timeout_seconds
        self.contracts = ContractRegistry(spec["contracts"])
        self.bus = EventBus()
        self.agents = {}
        self._managed_processes = []
        self._shutdown_registered = False

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
        payload = {"agents": self._build_agent_catalog()}
        self.emit("system.capabilities", payload)

    def _build_agent_catalog(self):
        catalog = []
        for agent_name, config in self.spec["agents"].items():
            runtime_cfg = config.get("runtime", {})
            metadata = self._load_agent_metadata(runtime_cfg)
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
                "name": agent_name,
                "description": metadata.get("description", config.get("description", "")),
                "methods": metadata.get("methods", config.get("methods", [])),
                "routing_notes": metadata.get("routing_notes", []),
                "adapter": runtime_cfg.get("adapter"),
                "endpoint": runtime_cfg.get("endpoint"),
                "subscribes_to": config.get("subscribes_to", []),
                "emits": config.get("emits", []),
            }
            for key, value in metadata.items():
                if key in {"description", "methods", "routing_notes"}:
                    continue
                entry[key] = value
            catalog.append(entry)
        return catalog

    def _load_agent_metadata(self, runtime_cfg: dict):
        autostart_cfg = runtime_cfg.get("autostart", {})
        app_ref = autostart_cfg.get("app")
        if not isinstance(app_ref, str) or ":" not in app_ref:
            return {}

        module_name = app_ref.split(":", 1)[0]
        try:
            module = import_module(module_name)
        except Exception:
            return {}

        raw = getattr(module, "AGENT_METADATA", None)
        if not isinstance(raw, dict):
            return {}

        def _sanitize_metadata_value(value: Any):
            if isinstance(value, (str, bool, int, float)):
                return value
            if isinstance(value, list):
                sanitized = [item for item in value if isinstance(item, (str, bool, int, float))]
                return sanitized if sanitized else None
            if isinstance(value, dict):
                sanitized = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        continue
                    nested = _sanitize_metadata_value(item)
                    if nested is not None:
                        sanitized[key] = nested
                return sanitized if sanitized else None
            return None

        metadata = {}
        description = raw.get("description")
        if isinstance(description, str):
            metadata["description"] = description

        routing_notes = raw.get("routing_notes")
        if isinstance(routing_notes, list):
            safe_notes = [item for item in routing_notes if isinstance(item, str)]
            if safe_notes:
                metadata["routing_notes"] = safe_notes

        methods = raw.get("methods")
        if isinstance(methods, list):
            valid_methods = []
            for method in methods:
                if not isinstance(method, dict):
                    continue
                name = method.get("name")
                event = method.get("event")
                when = method.get("when")
                if not isinstance(name, str) or not isinstance(event, str):
                    continue
                entry = {"name": name, "event": event}
                if isinstance(when, str):
                    entry["when"] = when
                for key, value in method.items():
                    if key in {"name", "event", "when"}:
                        continue
                    if isinstance(value, str):
                        entry[key] = value
                    elif isinstance(value, list):
                        safe_list = [item for item in value if isinstance(item, str)]
                        if safe_list:
                            entry[key] = safe_list
                valid_methods.append(entry)
            metadata["methods"] = valid_methods

        for key, value in raw.items():
            if key in {"description", "routing_notes", "methods"}:
                continue
            sanitized = _sanitize_metadata_value(value)
            if sanitized is not None:
                metadata[key] = sanitized

        return metadata

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

        if event_name == "task.plan" and isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            self._execute_task_plan(payload, depth)
            return

        results = self._invoke_subscribers(event_name, payload, depth)
        for new_event, new_payload in results:
            self.emit(new_event, new_payload, depth + 1)

    def _select_subscribers(self, event_name: str, payload: dict):
        subscribers = self.bus.get_subscribers(event_name)
        target_agent = payload.get("target_agent") if isinstance(payload, dict) else None
        if isinstance(target_agent, str) and target_agent:
            subscribers = [agent_name for agent_name in subscribers if agent_name == target_agent]
        return subscribers

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
        if event_name == "task.result":
            return payload.get("result", payload.get("detail", ""))
        if event_name == "shell.result":
            stdout = payload.get("stdout", "")
            if result_mode == "stdout_first_line":
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                return lines[0] if lines else ""
            if result_mode == "stdout_last_line":
                lines = [line.strip() for line in stdout.splitlines() if line.strip()]
                return lines[-1] if lines else ""
            if result_mode == "stdout_stripped":
                return stdout.strip()
            if isinstance(result_mode, str) and result_mode.startswith("json_field:"):
                field_name = result_mode.split(":", 1)[1].strip()
                if field_name:
                    try:
                        parsed = json.loads(stdout)
                        if isinstance(parsed, dict) and field_name in parsed:
                            value = parsed[field_name]
                            if value is None:
                                return ""
                            return str(value)
                    except (json.JSONDecodeError, TypeError):
                        return stdout
            return stdout
        if event_name == "file.content":
            return payload.get("content", "")
        if event_name == "notify.result":
            return payload.get("detail", "")
        if event_name == "answer.final":
            return payload.get("answer", "")
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

    def _execute_task_plan(self, payload: dict, depth: int):
        steps = payload.get("steps", [])
        if not steps:
            return

        context = {"original_task": payload.get("task", "")}
        workflow = self._execute_workflow_steps(steps, payload, context, depth)
        if "workflow.result" in self.spec.get("events", {}):
            result_payload = {
                "task": payload.get("task", ""),
                "status": workflow["status"],
                "steps": workflow["steps"],
                "result": workflow.get("result"),
            }
            presentation = payload.get("presentation")
            if isinstance(presentation, dict):
                result_payload["presentation"] = presentation
            if workflow.get("error"):
                result_payload["error"] = workflow["error"]
            self.emit("workflow.result", result_payload, depth + 1)
            return

        for new_event, new_payload in workflow.get("final_results", []):
            self.emit(new_event, new_payload, depth + 1)

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
        command = step_payload.get("command")
        if isinstance(command, str) and command.strip():
            payload["command"] = command
        sql = step_payload.get("sql")
        if isinstance(sql, str) and sql.strip():
            payload["sql"] = sql
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        self.emit("step.progress", payload, depth)

    def _step_result_summary(self, event_name: str, payload: dict):
        if not isinstance(payload, dict):
            return None
        if event_name == "shell.result":
            return {
                "command": payload.get("command", ""),
                "returncode": payload.get("returncode"),
                "stdout": payload.get("stdout", ""),
                "stderr": payload.get("stderr", ""),
            }
        if event_name == "sql.result":
            result = payload.get("result")
            summary = {"sql": payload.get("sql", "")}
            stats = payload.get("stats")
            if isinstance(stats, dict):
                summary["stats"] = stats
            if isinstance(result, dict):
                summary["row_count"] = result.get("row_count")
                summary["rows"] = result.get("rows", [])
            return summary
        if event_name == "task.result":
            return {"detail": payload.get("detail", ""), "result": payload.get("result")}
        return payload

    def _execute_workflow_steps(self, steps: list, payload: dict, context: dict, depth: int):
        records = []
        final_results = []
        final_value = None

        for index, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                continue

            step_id = raw_step.get("id") or f"step{index}"
            nested_steps = raw_step.get("steps")
            if isinstance(nested_steps, list) and nested_steps:
                nested_context = dict(context)
                nested_payload = {
                    "task": raw_step.get("task", payload.get("task", "")),
                    "step_id": step_id,
                }
                log_event("task.plan", nested_payload, depth + 1)
                nested = self._execute_workflow_steps(
                    nested_steps,
                    nested_payload,
                    nested_context,
                    depth + 1,
                )
                context.update(
                    {
                        key: value
                        for key, value in nested_context.items()
                        if key not in context or key not in {"original_task"}
                    }
                )
                final_results = nested.get("final_results", [])
                final_value = nested.get("result")
                context["prev"] = final_value
                context[step_id] = final_value
                records.append(
                    {
                        "id": step_id,
                        "task": raw_step.get("task", payload.get("task", "")),
                        "status": nested["status"],
                        "steps": nested["steps"],
                        "result": final_value,
                    }
                )
                if nested["status"] != "completed":
                    return {
                        "status": nested["status"],
                        "steps": records,
                        "result": final_value,
                        "error": nested.get("error"),
                        "final_results": final_results,
                    }
                continue

            step_payload = {
                key: value
                for key, value in raw_step.items()
                if key not in {"id", "depends_on", "steps"}
            }
            step_payload = self._resolve_templates(step_payload, context)
            step_payload.setdefault("task", payload.get("task", ""))
            step_payload["step_id"] = step_id
            step_payload["original_task"] = payload.get("task", context.get("original_task", ""))
            unresolved_keys = self._unresolved_template_keys(step_payload)
            if unresolved_keys:
                error = (
                    f"Plan step '{step_id}' referenced unavailable result(s): "
                    f"{', '.join(sorted(set(unresolved_keys)))}."
                )
                self._emit_step_progress(
                    "failed",
                    step_id,
                    step_payload,
                    depth + 1,
                    message=error,
                    error=error,
                    duration_ms=0,
                )
                records.append(
                    {
                        "id": step_id,
                        "task": step_payload.get("task", ""),
                        "target_agent": step_payload.get("target_agent"),
                        "status": "failed",
                        "error": error,
                        "duration_ms": 0,
                        "context": {
                            "available_results": sorted(
                                key for key in context.keys() if key != "original_task"
                            )
                        },
                    }
                )
                return {
                    "status": "failed",
                    "steps": records,
                    "result": final_value,
                    "error": error,
                    "final_results": [],
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
                error = f"Plan step '{step_id}' produced no result."
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
                        "error": error,
                        "duration_ms": duration_ms,
                    }
                )
                return {
                    "status": "failed",
                    "steps": records,
                    "result": final_value,
                    "error": error,
                    "final_results": [],
                }

            primary_event, primary_payload = step_results[0]
            primary_value = self._extract_result_value(primary_event, primary_payload, step_payload)
            result_summary = self._step_result_summary(primary_event, primary_payload)
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
            )
            context["prev"] = primary_value
            context[step_id] = primary_value
            context[f"{step_id}.event"] = primary_event
            context[f"{step_id}.detail"] = primary_payload.get("detail", "") if isinstance(primary_payload, dict) else ""
            context[f"{step_id}.result"] = primary_payload.get("result", primary_value) if isinstance(primary_payload, dict) else primary_value
            final_results = step_results
            final_value = primary_value
            records.append(
                {
                    "id": step_id,
                    "task": step_payload.get("task", ""),
                    "target_agent": step_payload.get("target_agent"),
                    "status": "completed",
                    "event": primary_event,
                    "duration_ms": duration_ms,
                    "payload": primary_payload,
                    "result": primary_value,
                    "emitted": [
                        {"event": event_name, "payload": event_payload}
                        for event_name, event_payload in step_results
                    ],
                }
            )

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
