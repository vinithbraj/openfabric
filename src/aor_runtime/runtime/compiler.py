"""OpenFABRIC Runtime Module: aor_runtime.runtime.compiler

Purpose:
    Compile loaded runtime specifications into executable runtime configuration.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from aor_runtime.dsl.models import CompiledRuntimeSpec, RuntimeSpec


class GraphCompiler:
    """Represent graph compiler within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by GraphCompiler.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.compiler.GraphCompiler and related tests.
    """
    def compile(self, spec: RuntimeSpec) -> CompiledRuntimeSpec:
        """Compile for GraphCompiler instances.

        Inputs:
            Receives spec for this GraphCompiler method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through GraphCompiler.compile calls and related tests.
        """
        tools = list(dict.fromkeys(spec.tools))
        if not tools:
            raise ValueError("Runtime spec must define at least one tool.")
        return CompiledRuntimeSpec(
            name=spec.name,
            description=spec.description,
            planner=spec.planner,
            runtime=spec.runtime,
            tools=tools,
            nodes=spec.nodes,
        )
