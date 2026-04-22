from __future__ import annotations

from aor_runtime.dsl.models import CompiledGraphSpec, CompiledNode, RuntimeSpec


class GraphCompiler:
    def compile(self, spec: RuntimeSpec) -> CompiledGraphSpec:
        if spec.graph.start not in spec.graph.nodes:
            raise ValueError(f"Unknown start node: {spec.graph.start}")

        compiled_nodes: dict[str, CompiledNode] = {}
        for node_name, node in spec.graph.nodes.items():
            if node.agent and node.agent not in spec.agents:
                raise ValueError(f"Node '{node_name}' references unknown agent '{node.agent}'.")

            next_nodes: list[str] = []
            if isinstance(node.next, str):
                next_nodes = [node.next]
            elif isinstance(node.next, list):
                next_nodes = list(node.next)
            if node.condition:
                next_nodes.extend([node.condition.then, node.condition.else_])

            compiled_nodes[node_name] = CompiledNode(
                name=node_name,
                kind=node.type,
                agent_name=node.agent,
                next_nodes=list(dict.fromkeys(next_nodes)),
                condition=node.condition,
                prompt=node.prompt,
                metadata=node.metadata,
            )

        known_nodes = set(compiled_nodes)
        for compiled in compiled_nodes.values():
            for next_node in compiled.next_nodes:
                if next_node != "end" and next_node not in known_nodes:
                    raise ValueError(f"Node '{compiled.name}' references unknown next node '{next_node}'.")

        return CompiledGraphSpec(
            name=spec.name,
            description=spec.description,
            start_node=spec.graph.start,
            nodes=compiled_nodes,
            agents=spec.agents,
        )
