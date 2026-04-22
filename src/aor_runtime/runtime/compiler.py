from __future__ import annotations

from aor_runtime.dsl.models import CompiledRuntimeSpec, RuntimeSpec


class GraphCompiler:
    def compile(self, spec: RuntimeSpec) -> CompiledRuntimeSpec:
        tools = list(dict.fromkeys(spec.tools))
        if not tools:
            raise ValueError("Runtime spec must define at least one tool.")
        return CompiledRuntimeSpec(
            name=spec.name,
            description=spec.description,
            planner=spec.planner,
            runtime=spec.runtime,
            tools=tools,
        )
