from __future__ import annotations

from pathlib import Path

import yaml

from aor_runtime.dsl.models import RuntimeSpec


def load_runtime_spec(path: str | Path) -> RuntimeSpec:
    source = Path(path)
    payload = yaml.safe_load(source.read_text()) or {}
    spec = RuntimeSpec.model_validate(payload)
    if spec.planner.prompt:
        spec.planner.prompt = str((source.parent / spec.planner.prompt).resolve())
    return spec
