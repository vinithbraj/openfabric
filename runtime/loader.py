import copy
import os
import re
from urllib.parse import urlparse, urlunparse

import yaml


def load_spec(path: str, selected_agents: list[str] | None = None) -> dict:
    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    expanded = _expand_agent_arguments(spec)
    return _filter_agents(expanded, selected_agents)


def list_spec_agents(path: str) -> list[dict]:
    spec = load_spec(path)
    agents = []
    for agent_name, agent in spec.get("agents", {}).items():
        metadata = agent.get("metadata", {}) if isinstance(agent.get("metadata"), dict) else {}
        aliases = []
        argument_name = metadata.get("argument_name")
        template_agent = metadata.get("template_agent")
        if isinstance(argument_name, str) and argument_name.strip():
            aliases.append(argument_name.strip())
        if isinstance(template_agent, str) and template_agent.strip() and template_agent.strip() != agent_name:
            aliases.append(template_agent.strip())
        agents.append(
            {
                "name": agent_name,
                "aliases": aliases,
                "description": agent.get("description", ""),
            }
        )
    return agents


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


def _normalize_selected_agents(selected_agents: list[str] | None) -> list[str]:
    if not selected_agents:
        return []
    normalized = []
    for item in selected_agents:
        if not isinstance(item, str):
            continue
        parts = [part.strip() for part in item.split(",")]
        normalized.extend(part for part in parts if part)
    return normalized


def _agent_aliases(agent_name: str, agent_cfg: dict) -> set[str]:
    aliases = {agent_name}
    metadata = agent_cfg.get("metadata", {}) if isinstance(agent_cfg.get("metadata"), dict) else {}
    for key in ("argument_name", "template_agent"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            aliases.add(value.strip())
    return aliases


def _filter_agents(spec: dict, selected_agents: list[str] | None) -> dict:
    selectors = _normalize_selected_agents(selected_agents)
    if not selectors:
        return spec

    agents = spec.get("agents", {})
    if not isinstance(agents, dict):
        return spec

    matched_names: set[str] = set()
    unmatched: list[str] = []
    for selector in selectors:
        selector_matches = [
            agent_name
            for agent_name, agent_cfg in agents.items()
            if selector in _agent_aliases(agent_name, agent_cfg)
        ]
        if not selector_matches:
            unmatched.append(selector)
            continue
        matched_names.update(selector_matches)

    if unmatched:
        available = sorted(
            {
                alias
                for agent_name, agent_cfg in agents.items()
                for alias in _agent_aliases(agent_name, agent_cfg)
            }
        )
        raise ValueError(
            "Unknown agent selector(s): "
            + ", ".join(unmatched)
            + ". Available selectors: "
            + ", ".join(available)
        )

    filtered = copy.deepcopy(spec)
    filtered["agents"] = {
        agent_name: copy.deepcopy(agent_cfg)
        for agent_name, agent_cfg in agents.items()
        if agent_name in matched_names
    }
    return filtered


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
