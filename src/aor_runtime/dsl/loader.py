from __future__ import annotations

from pathlib import Path

import yaml

from aor_runtime.dsl.models import RuntimeSpec


def load_runtime_spec(path: str | Path) -> RuntimeSpec:
    source = Path(path)
    payload = yaml.safe_load(source.read_text()) or {}
    spec = RuntimeSpec.model_validate(payload)
    for agent in spec.agents.values():
        if agent.prompt:
            agent.prompt = str((source.parent / agent.prompt).resolve())
    for node in spec.graph.nodes.values():
        if node.prompt:
            node.prompt = str((source.parent / node.prompt).resolve())
    return spec
