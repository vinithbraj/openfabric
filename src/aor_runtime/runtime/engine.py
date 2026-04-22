from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ExecutionPlan, FinalOutput, ValidationReport
from aor_runtime.core.utils import ensure_jsonable
from aor_runtime.dsl.loader import load_runtime_spec
from aor_runtime.dsl.models import CompiledRuntimeSpec
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.compiler import GraphCompiler
from aor_runtime.runtime.executor import PlanExecutor, summarize_final_output
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.runtime.state import RuntimeState, initial_runtime_state
from aor_runtime.runtime.store import SQLiteRunStore
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry


class ExecutionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = SQLiteRunStore(self.settings.run_store_path)
        self.llm = LLMClient(self.settings)
        self.tool_registry = build_tool_registry(self.settings)
        self.compiler = GraphCompiler()
        self.planner = TaskPlanner(llm=self.llm, tools=self.tool_registry)
        self.executor = PlanExecutor(self.tool_registry)
        self.validator = RuntimeValidator(self.settings)

    def run_spec(self, spec_path: str, input_payload: dict[str, Any]) -> dict[str, Any]:
        spec = load_runtime_spec(spec_path)
        compiled = self.compiler.compile(spec)
        run_id = self._new_run_id()
        self.store.create_run(run_id=run_id, spec_name=compiled.name, input_payload=input_payload)
        self.store.append_event(
            run_id=run_id,
            node_name="runtime",
            event_type="run.created",
            payload={"input": input_payload, "spec": compiled.name},
        )
        graph = self._build_graph(compiled)
        initial_state = initial_runtime_state(run_id=run_id, spec_name=compiled.name, input_payload=input_payload)
        final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})
        final_status = str(final_state.get("status", "failed"))
        self.store.finalize_run(run_id=run_id, status=final_status, final_state=final_state)
        self.store.append_event(
            run_id=run_id,
            node_name="runtime",
            event_type="run.completed",
            payload={"status": final_status, "retries": final_state.get("retries", 0)},
        )
        return ensure_jsonable(final_state)

    def validate_spec(self, spec_path: str) -> CompiledRuntimeSpec:
        spec = load_runtime_spec(spec_path)
        compiled = self.compiler.compile(spec)
        for tool_name in compiled.tools:
            self.tool_registry.get(tool_name)
        return compiled

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

    def _build_graph(self, compiled: CompiledRuntimeSpec):
        builder = StateGraph(RuntimeState)
        builder.add_node("planner", self._planner_node(compiled))
        builder.add_node("executor", self._executor_node(compiled))
        builder.add_node("validator", self._validator_node(compiled))
        builder.add_node("finalize", self._finalize_node())
        builder.add_edge(START, "planner")
        for node_name in ("planner", "executor", "validator"):
            builder.add_conditional_edges(
                node_name,
                lambda state: state.get("next_node", END),
                {"planner": "planner", "executor": "executor", "validator": "validator", "finalize": "finalize", END: END},
            )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=InMemorySaver())

    def _planner_node(self, compiled: CompiledRuntimeSpec):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            self.store.append_event(
                run_id=run_id,
                node_name="planner",
                event_type="planner.started",
                payload={"retry": state.get("retries", 0)},
            )
            try:
                plan = self.planner.build_plan(
                    goal=state.get("goal", ""),
                    planner=compiled.planner,
                    allowed_tools=compiled.tools,
                    input_payload=state.get("input", {}),
                    failure_context=state.get("failure_context"),
                )
                payload = plan.model_dump()
                self.store.append_event(run_id=run_id, node_name="planner", event_type="planner.completed", payload=payload)
                update = {
                    "current_node": "planner",
                    "next_node": "executor",
                    "status": "executing",
                    "plan": payload,
                    "error": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:  # noqa: BLE001
                self.store.append_event(
                    run_id=run_id,
                    node_name="planner",
                    event_type="planner.failed",
                    payload={"error": str(exc)},
                )
                update = {
                    "current_node": "planner",
                    "next_node": "finalize",
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "final_output": {"content": "", "artifacts": [], "metadata": {}},
                }
            self.store.save_snapshot(run_id=run_id, node_name="planner", state={**state, **update})
            return update

        return run

    def _executor_node(self, compiled: CompiledRuntimeSpec):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            plan = ExecutionPlan.model_validate(state.get("plan", {}))
            self.store.append_event(
                run_id=run_id,
                node_name="executor",
                event_type="executor.started",
                payload={"step_count": len(plan.steps)},
            )
            history_models, failure = self.executor.execute(plan)
            history_payload = [item.model_dump() for item in history_models]
            for item in history_payload:
                self.store.append_event(
                    run_id=run_id,
                    node_name="executor",
                    event_type="executor.step",
                    payload=item,
                )

            if failure:
                retries = int(state.get("retries", 0))
                should_retry = retries < min(compiled.runtime.max_retries, self.settings.max_plan_retries)
                update = {
                    "current_node": "executor",
                    "next_node": "planner" if should_retry else "finalize",
                    "status": "retrying" if should_retry else "failed",
                    "history": history_payload,
                    "failure_context": failure,
                    "error": failure["error"],
                    "retries": retries + 1 if should_retry else retries,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                final_output = summarize_final_output(state.get("goal", ""), history_models)
                update = {
                    "current_node": "executor",
                    "next_node": "validator",
                    "status": "validating",
                    "history": history_payload,
                    "failure_context": None,
                    "error": None,
                    "final_output": final_output,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            self.store.save_snapshot(run_id=run_id, node_name="executor", state={**state, **update})
            return update

        return run

    def _validator_node(self, compiled: CompiledRuntimeSpec):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            history = state.get("history", [])
            validation = self.validator.validate([self._step_log_from_dict(item) for item in history])
            payload = validation.model_dump()
            self.store.append_event(run_id=run_id, node_name="validator", event_type="validator.completed", payload=payload)
            if validation.success:
                final_output = state.get("final_output", {})
                if not final_output:
                    final_output = FinalOutput(content="", artifacts=[], metadata={}).model_dump()
                update = {
                    "current_node": "validator",
                    "next_node": "finalize",
                    "status": "completed",
                    "validation": payload,
                    "error": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "final_output": final_output,
                }
            else:
                retries = int(state.get("retries", 0))
                should_retry = retries < min(compiled.runtime.max_retries, self.settings.max_plan_retries)
                failure_context = {
                    "reason": "validation_failed",
                    "validation": payload,
                    "plan": state.get("plan", {}),
                    "history": history,
                }
                update = {
                    "current_node": "validator",
                    "next_node": "planner" if should_retry else "finalize",
                    "status": "retrying" if should_retry else "failed",
                    "validation": payload,
                    "failure_context": failure_context,
                    "error": validation.detail,
                    "retries": retries + 1 if should_retry else retries,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            self.store.save_snapshot(run_id=run_id, node_name="validator", state={**state, **update})
            return update

        return run

    def _finalize_node(self):
        def run(state: RuntimeState) -> dict[str, Any]:
            run_id = state["run_id"]
            final_output = state.get("final_output", {"content": "", "artifacts": [], "metadata": {}})
            status = state.get("status", "failed")
            if status not in {"completed", "failed"}:
                status = "failed"
            if status == "failed" and not str(final_output.get("content", "")).strip():
                final_output = {
                    "content": str(state.get("error") or "Task failed."),
                    "artifacts": final_output.get("artifacts", []),
                    "metadata": final_output.get("metadata", {}),
                }
            update = {
                "current_node": "finalize",
                "next_node": END,
                "status": status,
                "final_output": final_output,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.store.append_event(
                run_id=run_id,
                node_name="finalize",
                event_type="finalize.completed",
                payload={"status": status, "final_output": final_output},
            )
            self.store.save_snapshot(run_id=run_id, node_name="finalize", state={**state, **update})
            return update

        return run

    @staticmethod
    def _step_log_from_dict(payload: dict[str, Any]):
        from aor_runtime.core.contracts import StepLog

        return StepLog.model_validate(payload)
