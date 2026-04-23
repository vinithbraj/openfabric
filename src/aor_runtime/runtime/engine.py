from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import AgentSession, ExecutionPlan, FinalOutput, RunMetrics
from aor_runtime.core.utils import ensure_jsonable
from aor_runtime.dsl.loader import load_runtime_spec
from aor_runtime.dsl.models import CompiledRuntimeSpec
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.compiler import GraphCompiler
from aor_runtime.runtime.executor import PlanExecutor, summarize_final_output
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.runtime.sessions import SessionManager
from aor_runtime.runtime.state import RuntimeState
from aor_runtime.runtime.store import SQLiteRunStore
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry


TERMINAL_STATUSES = {"completed", "failed"}


class ExecutionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = SQLiteRunStore(self.settings.run_store_path)
        self.session_manager = SessionManager(self.store)
        self.llm = LLMClient(self.settings)
        self.tool_registry = build_tool_registry(self.settings)
        self.compiler = GraphCompiler()
        self.planner = TaskPlanner(llm=self.llm, tools=self.tool_registry)
        self.executor = PlanExecutor(self.tool_registry)
        self.validator = RuntimeValidator(self.settings)

    def run_spec(self, spec_path: str, input_payload: dict[str, Any]) -> dict[str, Any]:
        session = self.create_session(spec_path, input_payload, trigger="manual")
        return self.resume_session(session["id"], trigger="manual")

    def create_session(self, spec_path: str, input_payload: dict[str, Any], trigger: str = "manual") -> dict[str, Any]:
        spec = load_runtime_spec(spec_path)
        compiled = self.compiler.compile(spec)
        session_id = self._new_session_id()
        session = self.session_manager.create_session(
            session_id=session_id,
            spec_path=spec_path,
            compiled=compiled,
            input_payload=input_payload,
            trigger=trigger,
        )
        self.store.append_event(
            session_id=session.id,
            node_name="session",
            event_type="session.created",
            payload={"input": input_payload, "spec": compiled.name, "trigger": trigger},
        )
        self.session_manager.persist_session(session, node_name="session")
        return session.model_dump()

    def resume_session(self, session_id: str, trigger: str = "manual", max_cycles: int | None = None) -> dict[str, Any]:
        session = self._get_required_session(session_id)
        if self._is_done(session.state):
            return ensure_jsonable(session.state)

        session.current_trigger = trigger
        session.state["trigger"] = trigger
        self._touch_state(session.state)
        self.store.append_event(
            session_id=session.id,
            node_name="session",
            event_type="session.triggered",
            payload={"trigger": trigger, "status": session.status},
        )
        self.session_manager.persist_session(session, node_name="session")

        cycles = 0
        while not self._is_done(session.state):
            if max_cycles is not None and cycles >= max_cycles:
                break
            action = self._decide_next_action(session.state)
            if action == "planner":
                self._run_planner(session)
            elif action == "executor":
                self._run_executor_step(session)
            elif action == "validator":
                self._run_validator(session)
            else:
                session.state["status"] = "failed"
                session.state["error"] = f"Unknown next action: {action}"
                session.state["done"] = True
                break
            cycles += 1

        if self._is_done(session.state):
            self._finalize_session(session)
        else:
            self.session_manager.persist_session(session, node_name="loop")
        return ensure_jsonable(session.state)

    def trigger_session(self, session_id: str, trigger: str = "manual", max_cycles: int | None = None) -> dict[str, Any]:
        return self.resume_session(session_id, trigger=trigger, max_cycles=max_cycles)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        return {
            "session": session.model_dump(),
            "events": self.store.get_events(session_id),
            "latest_snapshot": self.store.get_latest_snapshot(session_id),
        }

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return [session.model_dump() for session in self.session_manager.list_sessions(limit=limit)]

    def validate_spec(self, spec_path: str) -> CompiledRuntimeSpec:
        spec = load_runtime_spec(spec_path)
        compiled = self.compiler.compile(spec)
        for tool_name in compiled.tools:
            self.tool_registry.get(tool_name)
        return compiled

    # Backward-compatible run-centric accessors.
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.get_session(run_id)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.list_sessions(limit=limit)

    def _run_planner(self, session: AgentSession) -> None:
        state = session.state
        compiled = CompiledRuntimeSpec.model_validate(session.compiled_spec)
        metrics = dict(state.get("metrics", {}))
        metrics["llm_calls"] = int(metrics.get("llm_calls", 0)) + 1
        state["metrics"] = metrics
        self.store.append_event(
            session_id=session.id,
            node_name="planner",
            event_type="planner.started",
            payload={"retry": state.get("retries", 0), "attempt": state.get("attempt", 0) + 1},
        )
        try:
            plan = self.planner.build_plan(
                goal=str(state.get("goal", "")),
                planner=compiled.planner,
                allowed_tools=compiled.tools,
                input_payload=dict(state.get("input", {})),
                failure_context=state.get("failure_context"),
            )
            state.update(
                {
                    "current_node": "planner",
                    "next_action": "executor",
                    "status": "executing",
                    "plan": plan.model_dump(),
                    "attempt_history": [],
                    "current_step_index": 0,
                    "attempt": int(state.get("attempt", 0)) + 1,
                    "validation": None,
                    "validation_checks": [],
                    "failure_context": None,
                    "error": None,
                }
            )
            self.store.append_event(
                session_id=session.id,
                node_name="planner",
                event_type="planner.completed",
                payload=plan.model_dump(),
            )
        except Exception as exc:  # noqa: BLE001
            state.update(
                {
                    "current_node": "planner",
                    "next_action": "",
                    "status": "failed",
                    "done": True,
                    "error": str(exc),
                    "final_output": {"content": str(exc), "artifacts": [], "metadata": {"goal": state.get("goal", "")}},
                }
            )
            self.store.append_event(
                session_id=session.id,
                node_name="planner",
                event_type="planner.failed",
                payload={"error": str(exc)},
            )
        self._persist(session, node_name="planner")

    def _run_executor_step(self, session: AgentSession) -> None:
        state = session.state
        plan = ExecutionPlan.model_validate(state.get("plan", {}))
        step_index = int(state.get("current_step_index", 0))
        if step_index >= len(plan.steps):
            state.update({"current_node": "executor", "next_action": "validator", "status": "validating"})
            self._persist(session, node_name="executor")
            return

        step = plan.steps[step_index]
        self.store.append_event(
            session_id=session.id,
            node_name="executor",
            event_type="executor.step.started",
            payload={"step": step.model_dump(), "step_index": step_index},
        )
        log = self.executor.execute_step(step)
        log_payload = log.model_dump()

        history = list(state.get("history", []))
        attempt_history = list(state.get("attempt_history", []))
        history.append(log_payload)
        attempt_history.append(log_payload)
        state["history"] = history
        state["attempt_history"] = attempt_history

        metrics = dict(state.get("metrics", {}))
        metrics["steps_executed"] = int(metrics.get("steps_executed", 0)) + 1
        state["metrics"] = metrics

        self.store.append_event(
            session_id=session.id,
            node_name="executor",
            event_type="executor.step.completed",
            payload=log_payload,
        )

        if not log.success:
            self._handle_retry_or_failure(
                session,
                node_name="executor",
                reason="tool_execution_failed",
                detail=log.error or "step failed",
                extra_context={
                    "step": step.model_dump(),
                    "history": attempt_history,
                },
            )
            return

        next_index = step_index + 1
        if next_index >= len(plan.steps):
            attempt_models = [self._step_log_from_dict(item) for item in attempt_history]
            final_output = summarize_final_output(str(state.get("goal", "")), attempt_models)
            state.update(
                {
                    "current_node": "executor",
                    "next_action": "validator",
                    "status": "validating",
                    "current_step_index": next_index,
                    "final_output": final_output,
                    "error": None,
                }
            )
        else:
            state.update(
                {
                    "current_node": "executor",
                    "next_action": "executor",
                    "status": "executing",
                    "current_step_index": next_index,
                    "error": None,
                }
            )
        self._persist(session, node_name="executor")

    def _run_validator(self, session: AgentSession) -> None:
        state = session.state
        attempt_history = [self._step_log_from_dict(item) for item in state.get("attempt_history", [])]
        validation, checks = self.validator.validate(attempt_history, goal=str(state.get("goal", "")))
        payload = validation.model_dump()
        self.store.append_event(
            session_id=session.id,
            node_name="validator",
            event_type="validator.completed",
            payload={"result": payload, "checks": checks},
        )

        if validation.success:
            final_output = state.get("final_output") or FinalOutput(metadata={"goal": state.get("goal", "")}).model_dump()
            state.update(
                {
                    "current_node": "validator",
                    "next_action": "",
                    "status": "completed",
                    "done": True,
                    "validation": payload,
                    "validation_checks": checks,
                    "error": None,
                    "final_output": final_output,
                }
            )
            self._persist(session, node_name="validator")
            return

        self._handle_retry_or_failure(
            session,
            node_name="validator",
            reason="validation_failed",
            detail=validation.reason or "validation failed",
            extra_context={
                "validation": payload,
                "validation_checks": checks,
                "plan": state.get("plan", {}),
                "history": state.get("attempt_history", []),
            },
        )

    def _handle_retry_or_failure(
        self,
        session: AgentSession,
        *,
        node_name: str,
        reason: str,
        detail: str,
        extra_context: dict[str, Any],
    ) -> None:
        state = session.state
        compiled = CompiledRuntimeSpec.model_validate(session.compiled_spec)
        retries = int(state.get("retries", 0))
        should_retry = retries < min(compiled.runtime.max_retries, self.settings.max_plan_retries)
        failure_context = {"reason": reason, "error": detail, **extra_context}

        self.store.append_event(
            session_id=session.id,
            node_name=node_name,
            event_type=f"{node_name}.failed",
            payload=failure_context,
        )

        if should_retry:
            state.update(
                {
                    "current_node": node_name,
                    "next_action": "planner",
                    "status": "retrying",
                    "retries": retries + 1,
                    "failure_context": failure_context,
                    "error": detail,
                    "plan": {},
                    "attempt_history": [],
                    "current_step_index": 0,
                }
            )
        else:
            state.update(
                {
                    "current_node": node_name,
                    "next_action": "",
                    "status": "failed",
                    "done": True,
                    "failure_context": failure_context,
                    "error": detail,
                    "final_output": {
                        "content": detail,
                        "artifacts": list((state.get("final_output") or {}).get("artifacts", [])),
                        "metadata": {"goal": state.get("goal", ""), "failure_reason": reason},
                    },
                }
            )
        self._persist(session, node_name=node_name)

    def _finalize_session(self, session: AgentSession) -> None:
        state = session.state
        metrics = RunMetrics.model_validate(state.get("metrics", {})).model_dump()
        metrics["retries"] = int(state.get("retries", 0))
        state["metrics"] = metrics
        status = str(state.get("status", "failed"))
        if status not in TERMINAL_STATUSES:
            status = "failed"
            state["status"] = status
            state["done"] = True

        final_output = state.get("final_output") or {"content": "", "artifacts": [], "metadata": {}}
        if status == "failed" and not str(final_output.get("content", "")).strip():
            final_output = {
                "content": str(state.get("error") or "Task failed."),
                "artifacts": list(final_output.get("artifacts", [])),
                "metadata": dict(final_output.get("metadata", {})),
            }
        state.update(
            {
                "current_node": "finalize",
                "next_action": "",
                "status": status,
                "done": True,
                "final_output": final_output,
            }
        )
        self.store.append_event(
            session_id=session.id,
            node_name="finalize",
            event_type="finalize.completed",
            payload={"status": status, "final_output": final_output, "metrics": metrics},
        )
        self._persist(session, node_name="finalize")

    def _persist(self, session: AgentSession, *, node_name: str) -> None:
        self._touch_state(session.state)
        session.status = str(session.state.get("status", session.status))
        session.current_trigger = str(session.state.get("trigger", session.current_trigger))
        session.history = list(session.state.get("history", session.history))
        self.session_manager.persist_session(session, node_name=node_name)

    @staticmethod
    def _step_log_from_dict(payload: dict[str, Any]):
        from aor_runtime.core.contracts import StepLog

        return StepLog.model_validate(payload)

    @staticmethod
    def _new_session_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid.uuid4().hex[:10]}"

    @staticmethod
    def _is_done(state: dict[str, Any]) -> bool:
        return bool(state.get("done")) or str(state.get("status", "")) in TERMINAL_STATUSES

    @staticmethod
    def _decide_next_action(state: RuntimeState) -> str:
        next_action = str(state.get("next_action", "")).strip()
        if next_action:
            return next_action
        status = str(state.get("status", "planning"))
        if status in {"planning", "retrying"}:
            return "planner"
        if status == "executing":
            return "executor"
        if status == "validating":
            return "validator"
        return "planner"

    @staticmethod
    def _touch_state(state: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        state["updated_at"] = now.isoformat()
        started_at = str(state.get("started_at", state["updated_at"]))
        try:
            started = datetime.fromisoformat(started_at)
            elapsed_ms = round((now - started).total_seconds() * 1000, 2)
        except Exception:  # noqa: BLE001
            elapsed_ms = float((state.get("metrics") or {}).get("latency_ms", 0.0))
        metrics = dict(state.get("metrics", {}))
        metrics["latency_ms"] = elapsed_ms
        metrics["retries"] = int(state.get("retries", 0))
        state["metrics"] = metrics

    def _get_required_session(self, session_id: str) -> AgentSession:
        session = self.session_manager.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")
        return session
