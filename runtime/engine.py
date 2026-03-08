import atexit
import socket
import subprocess
import sys
import time
from typing import Any
from importlib import import_module
from urllib.parse import urlparse

from .contracts import ContractRegistry
from .event_bus import EventBus
from .registry import ADAPTER_REGISTRY


class Engine:

    def __init__(self, spec: dict):
        self.spec = spec
        self.contracts = ContractRegistry(spec["contracts"])
        self.bus = EventBus()
        self.agents = {}
        self._managed_processes = []
        self._shutdown_registered = False

    def setup(self):
        self._autostart_http_services()

        # Instantiate agents
        for name, config in self.spec["agents"].items():
            adapter_type = config["runtime"]["adapter"]

            if adapter_type not in ADAPTER_REGISTRY:
                raise ValueError(f"Unknown adapter type: {adapter_type}")

            adapter_cls = ADAPTER_REGISTRY[adapter_type]
            adapter = adapter_cls(config["runtime"])

            self.agents[name] = {
                "adapter": adapter,
                "subscribes_to": config.get("subscribes_to", [])
            }

        # Register subscriptions
        for agent_name, agent in self.agents.items():
            for event in agent["subscribes_to"]:
                self.bus.subscribe(event, agent_name)

        self._emit_system_capabilities()

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
            runtime_cfg = config.get("runtime", {})
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
                print(
                    f"[BOOT] HTTP agent '{agent_name}' already reachable at {endpoint}"
                )
                continue

            app = autostart_cfg.get("app")
            if not app:
                raise ValueError(
                    f"HTTP agent '{agent_name}' autostart requires runtime.autostart.app"
                )

            module = autostart_cfg.get("module", "uvicorn")
            bind_host = autostart_cfg.get("host", "127.0.0.1")
            bind_port = int(autostart_cfg.get("port", port))
            timeout = float(autostart_cfg.get("timeout_seconds", 8))

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
            print(f"[BOOT] starting HTTP agent '{agent_name}': {' '.join(command)}")
            process = subprocess.Popen(command)
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

    def emit(self, event_name: str, payload: dict, depth: int = 0):

        indent = "  " * depth
        print(f"{indent}[EVENT] {event_name} -> {payload}")

        contract_name = self.spec["events"][event_name]["contract"]
        self.contracts.validate_payload(contract_name, payload)

        subscribers = self.bus.get_subscribers(event_name)

        for agent_name in subscribers:
            print(f"{indent}  ↳ handled by: {agent_name}")

            agent = self.agents[agent_name]
            results = agent["adapter"].handle(event_name, payload)

            for new_event, new_payload in results:
                self.emit(new_event, new_payload, depth + 1)

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
