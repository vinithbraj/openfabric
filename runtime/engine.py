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
from typing import Any
from importlib import import_module
from urllib.parse import urlparse

import requests

from .console import log_boot, log_event, log_event_handler
from .contracts import ContractRegistry
from .event_bus import EventBus
from .registry import ADAPTER_REGISTRY


class Engine:

    DEFAULT_WORKFLOW_LIMITS = {
        "max_attempts": 3,
        "max_replans": 1,
        "max_uncertain_attempts": 1,
        "max_validation_llm_calls": 1,
    }

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
            if result_mode == "json":
                try:
                    return json.loads(stdout)
                except (json.JSONDecodeError, TypeError):
                    return stdout
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
                            return parsed[field_name]
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

    def _resolve_reference_path(self, path: str, context: dict):
        if not isinstance(path, str) or not path.strip():
            return None
        parts = [part for part in path.split(".") if part]
        if not parts:
            return None
        step_results = context.get("__step_results_by_id__", {})
        head = parts[0]
        if isinstance(step_results, dict) and head in step_results:
            current = step_results[head]
        elif head in context:
            current = context[head]
        else:
            return None
        for part in parts[1:]:
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
        plan_options = self._plan_options(payload)
        if not plan_options:
            return
        limits = self._workflow_limits(payload)
        attempts = []
        deferred_clarification = None
        selected_attempt = None
        replan_count = 0
        uncertain_count = 0

        for index, option in enumerate(plan_options, start=1):
            if index > limits["max_attempts"]:
                break
            option_payload = dict(payload)
            option_payload["steps"] = option.get("steps", [])
            option_payload["selected_option_id"] = option.get("id")
            option_payload["selected_option_label"] = option.get("label")
            option_payload["__attempts_so_far__"] = index
            self._emit_validation_progress(
                "option_started",
                payload,
                depth + 1,
                option_id=option.get("id"),
                option_label=option.get("label"),
                attempt=index,
                total_attempts=len(plan_options),
                reason=option.get("reason"),
                message=f"Trying workflow option {index} of {len(plan_options)}.",
            )
            context = {
                "original_task": payload.get("task", ""),
                "__step_results__": [],
                "__step_results_by_id__": {},
            }
            workflow = self._execute_workflow_steps(option_payload.get("steps", []), option_payload, context, depth)
            validation = self._validate_workflow_attempt(
                payload,
                option,
                workflow,
                context,
                index,
                len(plan_options),
                depth,
            )
            attempt_record = {
                "attempt": index,
                "option": {
                    "id": option.get("id"),
                    "label": option.get("label"),
                    "reason": option.get("reason"),
                },
                "status": workflow.get("status"),
                "steps": workflow.get("steps", []),
                "result": workflow.get("result"),
                "error": workflow.get("error"),
                "validation": validation,
            }
            attempts.append(attempt_record)
            clarification = workflow.get("clarification")
            if workflow.get("status") == "needs_clarification" and isinstance(clarification, dict) and deferred_clarification is None:
                deferred_clarification = clarification
            if validation.get("verdict") == "uncertain":
                uncertain_count += 1
            if validation.get("valid"):
                selected_attempt = attempt_record
                break
            if validation.get("verdict") == "uncertain" and uncertain_count > limits["max_uncertain_attempts"]:
                deferred_clarification = self._clarification_from_validation(payload, attempts, attempt_record, "Validation remained uncertain after repeated attempts.")
                break
            if index >= len(plan_options) and replan_count < limits["max_replans"] and len(attempts) < limits["max_attempts"]:
                derived_option = self._derive_fallback_option(payload, plan_options, attempt_record, context, len(attempts), depth)
                if derived_option is not None:
                    plan_options.append(derived_option)
                    replan_count += 1
            if index < len(plan_options):
                self._emit_validation_progress(
                    "retrying",
                    payload,
                    depth + 1,
                    option_id=option.get("id"),
                    option_label=option.get("label"),
                    attempt=index,
                    total_attempts=len(plan_options),
                    reason=validation.get("reason"),
                    message="The validator rejected this attempt, so I am trying the next workflow option.",
                )

        if selected_attempt is not None:
            if "workflow.result" in self.spec.get("events", {}):
                result_payload = {
                    "task": payload.get("task", ""),
                    "status": selected_attempt["status"],
                    "steps": selected_attempt["steps"],
                    "result": selected_attempt.get("result"),
                    "attempts": attempts,
                    "selected_option": selected_attempt["option"],
                    "validation": selected_attempt["validation"],
                    "task_shape": payload.get("task_shape"),
                }
                presentation = payload.get("presentation")
                if isinstance(presentation, dict):
                    result_payload["presentation"] = presentation
                if selected_attempt.get("error"):
                    result_payload["error"] = selected_attempt["error"]
                self.emit("workflow.result", result_payload, depth + 1)
                return
            return

        if deferred_clarification is not None and "clarification.required" in self.spec.get("events", {}):
            deferred_clarification["attempts"] = attempts
            self.emit("clarification.required", deferred_clarification, depth + 1)
            return

        last_attempt = attempts[-1] if attempts else None
        if "workflow.result" in self.spec.get("events", {}):
            result_payload = {
                "task": payload.get("task", ""),
                "status": "failed",
                "steps": last_attempt.get("steps", []) if isinstance(last_attempt, dict) else [],
                "result": last_attempt.get("result") if isinstance(last_attempt, dict) else None,
                "attempts": attempts,
                "validation": last_attempt.get("validation") if isinstance(last_attempt, dict) else None,
                "task_shape": payload.get("task_shape"),
            }
            presentation = payload.get("presentation")
            if isinstance(presentation, dict):
                result_payload["presentation"] = presentation
            if isinstance(last_attempt, dict):
                result_payload["selected_option"] = last_attempt.get("option")
                result_payload["error"] = last_attempt.get("error") or (
                    last_attempt.get("validation", {}) or {}
                ).get("reason")
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
            else:
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        if not isinstance(replan_result, dict):
            return None
        steps = replan_result.get("steps")
        if not isinstance(steps, list) or not steps:
            return None
        existing_signatures = {
            json.dumps(option.get("steps", []), sort_keys=True, ensure_ascii=True)
            for option in existing
            if isinstance(option, dict)
        }
        candidate_signature = json.dumps(steps, sort_keys=True, ensure_ascii=True)
        if candidate_signature in existing_signatures:
            return None
        return {
            "id": f"option{len(existing) + 1}",
            "label": "Recovered fallback",
            "reason": str(replan_result.get("reason") or reason),
            "steps": steps,
        }

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
        task_shape = str(workflow_payload.get("task_shape") or "").strip().lower() or "lookup"
        completed = status == "completed"
        prior_results = self._prior_step_results(context)
        shape_assessment = self._assess_task_shape_result(task_shape, result, prior_results)
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
            if self._looks_like_scalar(result):
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
            "task": workflow_payload.get("task", ""),
            "task_shape": workflow_payload.get("task_shape", ""),
            "attempt": attempt,
            "total_attempts": total_attempts,
            "option_id": option.get("id"),
            "option_label": option.get("label"),
            "option_reason": option.get("reason"),
            "workflow_status": workflow.get("status", ""),
            "validation_llm_budget_remaining": validation_llm_budget_remaining,
            "steps": [self._compact_value(step, text_limit=240, row_limit=4) for step in workflow.get("steps", []) if isinstance(step, dict)],
            "result": self._compact_value(workflow.get("result"), text_limit=240, row_limit=4),
            "error": workflow.get("error"),
            "presentation": workflow_payload.get("presentation", {}),
            "available_context": {
                key: self._compact_value(value, text_limit=180, row_limit=3)
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
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
                    "stdout_excerpt": self._compact_text(payload.get("stdout"), 500),
                    "stderr_excerpt": self._compact_text(payload.get("stderr"), 300),
                    "stats": self._compact_value(payload.get("stats"), text_limit=120),
                    "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=3),
                }
            )
            return compact
        if event_name == "sql.result":
            compact.update(
                {
                    "detail": payload.get("detail"),
                    "sql": self._compact_text(payload.get("sql"), 500),
                    "reduced_result": payload.get("reduced_result") or payload.get("refined_answer"),
                    "local_reduction_command": payload.get("local_reduction_command"),
                    "schema": self._compact_schema(payload.get("schema")),
                    "stats": self._compact_value(payload.get("stats"), text_limit=120),
                    "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=5),
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
                    "stdout_excerpt": self._compact_text(payload.get("stdout"), 500),
                    "stderr_excerpt": self._compact_text(payload.get("stderr"), 300),
                    "stats": self._compact_value(payload.get("stats"), text_limit=120),
                    "result": self._compact_value(payload.get("result"), text_limit=240, row_limit=3),
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
        context[f"{step_id}.event"] = primary_event
        context[f"{step_id}.detail"] = primary_payload.get("detail", "") if isinstance(primary_payload, dict) else ""
        context[f"{step_id}.result"] = primary_payload.get("result", primary_value) if isinstance(primary_payload, dict) else primary_value

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
        if primary_event == "shell.result" and isinstance(primary_payload, dict):
            envelope["command"] = primary_payload.get("command")
            envelope["returncode"] = primary_payload.get("returncode")
            envelope["stdout_excerpt"] = self._compact_text(primary_payload.get("stdout"), 500)
            envelope["stderr_excerpt"] = self._compact_text(primary_payload.get("stderr"), 300)
        if primary_event == "sql.result" and isinstance(primary_payload, dict):
            envelope["sql"] = primary_payload.get("sql")
            result = primary_payload.get("result")
            if isinstance(result, dict):
                envelope["row_count"] = result.get("row_count")
                envelope["columns"] = result.get("columns")
                envelope["rows"] = self._compact_rows(result.get("rows"), 5)
        if primary_event == "slurm.result" and isinstance(primary_payload, dict):
            envelope["command"] = primary_payload.get("command")
            envelope["returncode"] = primary_payload.get("returncode")
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
            "failure_class": "needs_decomposition" if self._step_requests_replan(primary_event or "", primary_payload) else "execution_failed",
            "event": primary_event or "",
            "available_context": {
                key: value
                for key, value in context.items()
                if key != "original_task" and not key.endswith(".event")
            },
            "presentation": workflow_payload.get("presentation", {}),
        }
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
            else:
                auxiliary.append((event_name, event_payload))
        for event_name, event_payload in auxiliary:
            self.emit(event_name, event_payload, depth + 1)
        return replan_result

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
            if key not in {"id", "depends_on", "steps", "replan_budget", "when"}
        }
        step_payload = self._resolve_templates(step_payload, context)
        step_payload = self._resolve_references(step_payload, context)
        step_payload.setdefault("task", workflow_payload.get("task", ""))
        if isinstance(workflow_payload.get("task_shape"), str) and workflow_payload.get("task_shape"):
            step_payload.setdefault("task_shape", workflow_payload.get("task_shape"))
        step_payload["step_id"] = step_id
        step_payload["original_task"] = workflow_payload.get("task", context.get("original_task", ""))
        prior_step_results = self._prior_step_results(context)
        dependency_results = self._dependency_results(context, raw_step)
        if prior_step_results:
            step_payload["prior_step_results"] = prior_step_results
            step_payload["previous_step_result"] = prior_step_results[-1]
        if dependency_results:
            step_payload["dependency_results"] = dependency_results
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

    def _execute_workflow_steps(self, steps: list, payload: dict, context: dict, depth: int):
        records = []
        final_results = []
        final_value = None
        pending = [step for step in steps if isinstance(step, dict)]
        completed_ids: set[str] = set()

        while pending:
            ready_index = next(
                (index for index, item in enumerate(pending) if self._ready_to_run(item, completed_ids)),
                None,
            )
            if ready_index is None:
                error = "Workflow is blocked by unresolved or cyclic depends_on references."
                return {
                    "status": "failed",
                    "steps": records,
                    "result": final_value,
                    "error": error,
                    "final_results": final_results,
                }
            raw_step = pending.pop(ready_index)
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
                        "result": None,
                    }
                )
                continue
            if outcome["kind"] == "nested":
                nested = outcome["nested"]
                self._merge_child_context(context, outcome["nested_context"])
                final_results = nested.get("final_results", [])
                final_value = nested.get("result")
                self._record_context_value(context, step_id, "task.result", {"result": final_value}, final_value)
                completed_ids.add(step_id)
                records.append(
                    {
                        "id": step_id,
                        "task": outcome["task"],
                        "target_agent": outcome.get("target_agent"),
                        "status": nested["status"],
                        "steps": nested["steps"],
                        "result": final_value,
                    }
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
                    return result
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
                )
                records.append(
                    {
                        "id": step_id,
                        "task": step_payload.get("task", ""),
                        "target_agent": step_payload.get("target_agent"),
                        "status": "needs_clarification" if self._step_requests_replan(primary_event or "", primary_payload) else "failed",
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
                )
                if isinstance(primary_payload, dict) and primary_payload.get("status") == "needs_clarification":
                    clarification = self._build_clarification_payload(
                        step_id,
                        step_payload,
                        payload,
                        primary_payload,
                        error,
                        context,
                    )
                    return {
                        "status": "needs_clarification",
                        "steps": records,
                        "result": primary_value,
                        "error": error,
                        "clarification": clarification,
                        "final_results": step_results,
                    }
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
                ),
            )
            completed_ids.add(step_id)
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
                    "payload": self._compact_event_payload(primary_event or "", primary_payload),
                    "result": self._compact_value(primary_value, text_limit=240, row_limit=4),
                    "evidence": self._step_evidence(primary_event, primary_payload, primary_value),
                    "emitted": [
                        {"event": event_name, "payload": self._compact_event_payload(event_name, event_payload)}
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
