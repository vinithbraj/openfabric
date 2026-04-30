"""OpenFABRIC Runtime Module: aor_runtime.dsl.loader

Purpose:
    Load runtime DSL YAML files and validate them into models.

Responsibilities:
    Parse assistant specs into typed configuration objects before compilation and execution.

Data flow / Interfaces:
    Receives YAML/runtime spec data and returns validated DSL models for the compiler.

Boundaries:
    Keeps configuration parsing separate from request-time LLM planning and tool execution.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aor_runtime.dsl.models import RuntimeSpec


def load_runtime_spec(path: str | Path) -> RuntimeSpec:
    """Load runtime spec for the surrounding runtime workflow.

    Inputs:
        Receives path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by runtime spec loading code paths that import or call aor_runtime.dsl.loader.load_runtime_spec.
    """
    source = Path(path)
    payload = yaml.safe_load(source.read_text()) or {}
    spec = RuntimeSpec.model_validate(payload)
    if spec.planner.prompt:
        spec.planner.prompt = str((source.parent / spec.planner.prompt).resolve())
    if spec.planner.decomposer_prompt:
        spec.planner.decomposer_prompt = str((source.parent / spec.planner.decomposer_prompt).resolve())
    return spec
