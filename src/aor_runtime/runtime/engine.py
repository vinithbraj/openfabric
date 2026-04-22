from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from aor_runtime.agents.base import AgentExecutor
from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import AgentRunResult, RouterDecision
from aor_runtime.core.utils import dumps_json, safe_eval_condition
from aor_runtime.dsl.loader import load_runtime_spec
from aor_runtime.dsl.models import CompiledGraphSpec, CompiledNode
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.compiler import GraphCompiler
from aor_runtime.runtime.state import RuntimeState, initial_runtime_state
from aor_runtime.runtime.store import SQLiteRunStore
from aor_runtime.tools.factory import build_tool_registry


DEFAULT_ROUTER_PROMPT = """You are the routing layer of a local multi-agent runtime.

Choose exactly one next node from the provided options.
Return valid JSON:
{
  "selected": "node_name",
  "rationale": "short explanation",
  "confidence": 0.0
}

Guidance:
- Route based on the task and current state.
- Prefer paths that directly advance execution.
- Do not invent node names.
"""


class LLMRouter:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def choose(
        self,
        *,
        node_name: str,
        task: str,
        state: dict[str, Any],
        options: list[str],
        prompt_path: str | None,
    ) -> RouterDecision:
        if len(options) == 1:
            return RouterDecision(selected=options[0], rationale="Only one route available.", confidence=1.0)
        system_prompt = self.llm.load_prompt(prompt_path, DEFAULT_ROUTER_PROMPT)
        payload = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=dumps_json(
                {
                    "node": node_name,
                    "task": task,
                    "options": options,
                    "state": state,
                },
                indent=2,
            ),
        )
        decision = RouterDecision.model_validate(payload)
        if decision.selected not in options:
            decision.selected = options[0]
            decision.rationale = f"Model returned an invalid option. Fell back to {options[0]!r}."
        return decision


class ExecutionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = SQLiteRunStore(self.settings.run_store_path)
        self.llm = LLMClient(self.settings)
        self.router = LLMRouter(self.llm)
        self.tool_registry = build_tool_registry(self.settings)
        self.agent_executor = AgentExecutor(llm=self.llm, tool_registry=self.tool_registry, settings=self.settings)
        self.compiler = GraphCompiler()

    def run_spec(self, spec_path: str, input_payload: dict[str, Any]) -> dict[str, Any]:
        spec = load_runtime_spec(spec_path)
        compiled = self.compiler.compile(spec)
        run_id = self._new_run_id()
        self.store.create_run(run_id=run_id, spec_name=compiled.name, input_payload=input_payload)
        self.store.append_event(run_id=run_id, node_name="runtime", event_type="run.created", payload={"input": input_payload, "spec": compiled.name})
        graph = self._build_graph(compiled)
        initial_state = initial_runtime_state(run_id=run_id, spec_name=compiled.name, input_payload=input_payload)
        final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})
        status = final_state.get("status", "completed")
        self.store.finalize_run(run_id=run_id, status=status, final_state=final_state)
        self.store.append_event(run_id=run_id, node_name="runtime", event_type="run.completed", payload={"status": status})
        return final_state

    def validate_spec(self, spec_path: str) -> CompiledGraphSpec:
        spec = load_runtime_spec(spec_path)
        return self.compiler.compile(spec)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        summary = self.store.get_run(run_id)
        if summary is None:
            return None
        return {"run": summary.model_dump(), "events": self.store.get_events(run_id)}

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.store.list_runs(limit=limit)]

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid.uuid4().hex[:10]}"

    def _build_graph(self, compiled: CompiledGraphSpec):
        builder = StateGraph(RuntimeState)
        for node_name, node in compiled.nodes.items():
            if node.kind == "router":
                builder.add_node(node_name, self._router_node(compiled, node))
            elif node.kind == "agent":
                builder.add_node(node_name, self._agent_node(compiled, node))
            else:
                builder.add_node(node_name, self._end_node(compiled, node))

        builder.add_edge(START, compiled.start_node)

        for node_name, node in compiled.nodes.items():
            if node.kind == "end":
                builder.add_edge(node_name, END)
                continue
            if len(node.next_nodes) > 1 or node.kind == "router" or node.condition is not None:
                path_map = {next_name: next_name for next_name in node.next_nodes}
                path_map[END] = END
                builder.add_conditional_edges(
                    node_name,
                    lambda state, _node_name=node_name: state.get("next_node", END),
                    path_map,
                )
            elif node.next_nodes:
                builder.add_edge(node_name, node.next_nodes[0])
            else:
                builder.add_edge(node_name, END)

        return builder.compile(checkpointer=InMemorySaver())

    def _router_node(self, compiled: CompiledGraphSpec, node: CompiledNode):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            self.store.append_event(run_id=run_id, node_name=node.name, event_type="node.started", payload={"kind": "router"})
            decision = self.router.choose(
                node_name=node.name,
                task=state.get("task", ""),
                state=state,
                options=node.next_nodes,
                prompt_path=node.prompt,
            )
            update = {
                "current_node": node.name,
                "next_node": decision.selected,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_result": {"type": "router", **decision.model_dump()},
                "history": [{"node": node.name, "kind": "router", **decision.model_dump()}],
                "metadata": {**state.get("metadata", {}), f"route:{node.name}": decision.model_dump()},
            }
            self.store.append_event(run_id=run_id, node_name=node.name, event_type="router.selected", payload=decision.model_dump())
            self.store.save_snapshot(run_id=run_id, node_name=node.name, state={**state, **update})
            return update

        return run

    def _agent_node(self, compiled: CompiledGraphSpec, node: CompiledNode):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            self.store.append_event(run_id=run_id, node_name=node.name, event_type="node.started", payload={"kind": "agent", "agent": node.agent_name})
            agent_definition = compiled.agents[node.agent_name or ""]

            result = self._run_agent_with_retries(node=node, state=state, agent_definition=agent_definition)
            merged_outputs = dict(state.get("intermediate_outputs", {}))
            merged_outputs[node.name] = result.model_dump()

            next_node = self._determine_agent_next_node(node=node, state=state, result=result, merged_outputs=merged_outputs)
            status = "failed" if result.status == "failed" else "running"
            if node.kind == "agent" and not next_node:
                status = "completed"

            update = {
                "current_node": node.name,
                "next_node": next_node,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "intermediate_outputs": merged_outputs,
                "last_result": result.model_dump(),
                "history": [{"node": node.name, "kind": "agent", "status": result.status, "summary": result.summary, "next": next_node}],
                "status": state.get("status", "running") if result.status != "failed" else "failed",
                "error": {"node": node.name, "message": result.error} if result.error else None,
            }
            self.store.append_event(run_id=run_id, node_name=node.name, event_type="agent.completed", payload=result.model_dump())
            self.store.save_snapshot(run_id=run_id, node_name=node.name, state={**state, **update})
            return update

        return run

    def _end_node(self, compiled: CompiledGraphSpec, node: CompiledNode):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            self.store.append_event(run_id=run_id, node_name=node.name, event_type="node.started", payload={"kind": "end"})
            terminal_status = "completed" if state.get("status") != "failed" else "failed"
            update = {
                "current_node": node.name,
                "next_node": END,
                "status": terminal_status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "history": [{"node": node.name, "kind": "end", "status": terminal_status}],
            }
            self.store.save_snapshot(run_id=run_id, node_name=node.name, state={**state, **update})
            return update

        return run

    def _run_agent_with_retries(self, *, node: CompiledNode, state: RuntimeState, agent_definition) -> AgentRunResult:
        task = state.get("task", "")
        last_error: str | None = None
        for attempt in range(1, self.settings.max_node_retries + 2):
            try:
                result = self.agent_executor.run(
                    agent_name=node.agent_name or node.name,
                    agent_definition=agent_definition,
                    task=task,
                    state=state,
                    node_name=node.name,
                )
                if result.status != "failed":
                    return result
                last_error = result.error
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            self.store.append_event(
                run_id=state["run_id"],
                node_name=node.name,
                event_type="agent.retry",
                payload={"attempt": attempt, "error": last_error},
            )
        return AgentRunResult(
            agent_name=node.agent_name or node.name,
            status="failed",
            summary=f"Agent failed after retries at node {node.name}.",
            error=last_error or "Unknown agent failure.",
        )

    def _determine_agent_next_node(
        self,
        *,
        node: CompiledNode,
        state: RuntimeState,
        result: AgentRunResult,
        merged_outputs: dict[str, Any],
    ) -> str:
        if result.status == "failed":
            return "end" if "end" in node.next_nodes else END
        if node.condition:
            context = {
                "input": state.get("input", {}),
                "task": state.get("task", ""),
                "last_result": result.model_dump(),
                "outputs": merged_outputs,
                "metadata": state.get("metadata", {}),
            }
            return node.condition.then if safe_eval_condition(node.condition.if_, context) else node.condition.else_
        if node.next_nodes:
            return node.next_nodes[0]
        return END
