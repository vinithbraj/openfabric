import copy
import os
import re
from urllib.parse import urlparse, urlunparse

import yaml

def load_spec(path: str) -> dict:
    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    return _expand_agent_arguments(spec)


def _safe_agent_suffix(name: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return suffix or "default"


def _endpoint_with_port(endpoint: str, port: int) -> str:
    parsed = urlparse(endpoint)
    netloc = parsed.hostname or "127.0.0.1"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        netloc = f"{auth}@{netloc}"
    netloc = f"{netloc}:{port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _resolve_env_placeholders(value):
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

        def replace(match):
            name = match.group(1)
            default = match.group(2)
            return os.getenv(name, default or "")

        return pattern.sub(replace, value)
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}
    return value


def _expand_agent_arguments(spec: dict) -> dict:
    arguments = spec.get("arguments", {})
    agent_arguments = arguments.get("agents", {}) if isinstance(arguments, dict) else {}
    if not isinstance(agent_arguments, dict) or not agent_arguments:
        return spec

    expanded = copy.deepcopy(spec)
    expanded_agents = expanded["agents"]
    for template_name, instances in agent_arguments.items():
        if not isinstance(instances, list) or not instances:
            continue
        template = expanded_agents.get(template_name)
        if not isinstance(template, dict):
            continue
        expanded_agents.pop(template_name)
        endpoint = template.get("runtime", {}).get("endpoint", "http://127.0.0.1:0/handle")
        parsed = urlparse(endpoint)
        base_port = parsed.port or 0

        for index, raw_instance in enumerate(instances):
            if not isinstance(raw_instance, dict):
                continue
            instance = _resolve_env_placeholders(raw_instance)
            instance_name = str(instance.get("name") or f"{template_name}_{index + 1}")
            suffix = _safe_agent_suffix(instance_name)
            agent_name = str(instance.get("agent_name") or f"{template_name}_{suffix}")
            agent = copy.deepcopy(template)

            if isinstance(instance.get("description"), str):
                agent["description"] = instance["description"]

            runtime_cfg = agent.setdefault("runtime", {})
            if isinstance(instance.get("endpoint"), str):
                runtime_cfg["endpoint"] = instance["endpoint"]
            elif base_port:
                runtime_cfg["endpoint"] = _endpoint_with_port(endpoint, base_port + index)

            if "port" in instance:
                port = int(instance["port"])
                runtime_cfg["endpoint"] = _endpoint_with_port(runtime_cfg["endpoint"], port)
                autostart_cfg = runtime_cfg.setdefault("autostart", {})
                autostart_cfg["port"] = port

            autostart_cfg = runtime_cfg.setdefault("autostart", {})
            env_cfg = autostart_cfg.setdefault("env", {})
            if isinstance(instance.get("env"), dict):
                env_cfg.update(instance["env"])

            metadata = copy.deepcopy(instance.get("metadata", {})) if isinstance(instance.get("metadata"), dict) else {}
            metadata.setdefault("argument_name", instance_name)
            metadata.setdefault("template_agent", template_name)
            agent["metadata"] = metadata
            expanded_agents[agent_name] = agent

    return expanded
