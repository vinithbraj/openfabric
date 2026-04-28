from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from aor_runtime import __version__
from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import AgentSession, ExecutionPlan, ExecutionStep, FinalOutput, RunMetrics
from aor_runtime.core.utils import ensure_jsonable
from aor_runtime.dsl.loader import load_runtime_spec
from aor_runtime.dsl.models import CompiledRuntimeSpec
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.compiler import GraphCompiler
from aor_runtime.runtime.dataflow import resolve_execution_step
from aor_runtime.runtime.decomposer import is_complex_goal
from aor_runtime.runtime.error_normalization import normalize_planner_error, normalize_runtime_failure
from aor_runtime.runtime.executor import PlanExecutor, summarize_final_output
from aor_runtime.runtime.failure_classifier import classify_failure, generate_prompt_suggestions
from aor_runtime.runtime.planner import TaskPlanner, summarize_plan, summarize_planner_raw_output
from aor_runtime.runtime.policies import PlanContractViolation
from aor_runtime.runtime.prompt_suggestions import append_prompt_suggestions
from aor_runtime.runtime.sessions import SessionManager
from aor_runtime.runtime.state import RuntimeState
from aor_runtime.runtime.store import SQLiteRunStore
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.factory import build_tool_registry
from aor_runtime.tools.filesystem import resolve_path
from aor_runtime.tools.gateway import resolve_execution_node


TERMINAL_STATUSES = {"completed", "failed"}
DANGEROUS_SHELL_PATTERN = re.compile(r"(?:^|&&|\|\||;|\|)\s*(?:sudo\s+)?(?:rm|rmdir|unlink)\b")
FAILURE_SUMMARY_MAX_STEPS = 3
FAILURE_SUMMARY_MAX_BYTES = 4096
FAILURE_SUMMARY_MAX_STRING = 240
NON_RETRYABLE_FAILURE_MESSAGES = {
    "Unsafe query",
    "Only SELECT and WITH queries are allowed.",
    "Multiple SQL statements are not allowed.",
    "Empty SQL query.",
}
STARTUP_BANNER = r"""
   ___   ____  ____
  / _ | / __ \/ __/
 / __ |/ /_/ / /_
/_/ |_|\____/\__/
""".strip("\n")


def render_startup_banner() -> str:
    return f"{STARTUP_BANNER}\naor-runtime v{__version__}"


def _safe_execution_plan(plan_data: Any) -> ExecutionPlan | None:
    if not isinstance(plan_data, dict) or not plan_data:
        return None
    try:
        return ExecutionPlan.model_validate(plan_data)
    except Exception:  # noqa: BLE001
        return None


def summarize_failure_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in list(history or [])[-FAILURE_SUMMARY_MAX_STEPS:]:
        payload = dict(item or {})
        step_payload = dict(payload.get("step") or {})
        result = payload.get("result")
        summary.append(
            {
                "step_id": step_payload.get("id"),
                "action": step_payload.get("action"),
                "success": bool(payload.get("success")),
                "result_type": type(result).__name__,
                "result_keys": sorted(str(key) for key in result.keys())[:8] if isinstance(result, dict) else None,
                "size_hint": _value_size_hint(result),
                "error_excerpt": _truncate_string(payload.get("error")),
            }
        )
    return summary


def _value_size_hint(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _truncate_string(value: Any, limit: int = FAILURE_SUMMARY_MAX_STRING) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 13)]}...[truncated]"


def _summarize_step_payload(step_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(step_payload, dict) or not step_payload:
        return None
    summary: dict[str, Any] = {}
    if "id" in step_payload:
        summary["id"] = step_payload.get("id")
    if "action" in step_payload:
        summary["action"] = step_payload.get("action")
    raw_args = step_payload.get("args")
    if isinstance(raw_args, dict) and raw_args:
        summarized_args: dict[str, Any] = {}
        for key in ("database", "node", "path", "src", "dst", "command", "query"):
            if key not in raw_args:
                continue
            value = raw_args.get(key)
            if isinstance(value, str):
                summarized_args[key] = _truncate_string(value)
            elif isinstance(value, (int, float, bool)) or value is None:
                summarized_args[key] = value
            else:
                summarized_args[key] = type(value).__name__
        if not summarized_args:
            summarized_args["arg_keys"] = sorted(str(key) for key in raw_args.keys() if not str(key).startswith("__"))[:8]
        summary["args"] = summarized_args
    return summary


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False, sort_keys=True))


def _cap_failure_context_size(failure_context: dict[str, Any], *, limit: int = FAILURE_SUMMARY_MAX_BYTES) -> dict[str, Any]:
    capped = dict(failure_context)
    if _serialized_size(capped) <= limit:
        return capped

    summary = list(capped.get("summary") or [])
    while summary and _serialized_size(capped) > limit:
        summary = summary[1:]
        capped["summary"] = summary
    if _serialized_size(capped) <= limit:
        return capped

    capped = _truncate_failure_context_strings(capped)
    if _serialized_size(capped) <= limit:
        return capped

    if summary:
        capped["summary"] = [{"truncated": True}]
    if isinstance(capped.get("step"), dict):
        step_summary = dict(capped["step"])
        step_summary.pop("args", None)
        step_summary["truncated"] = True
        capped["step"] = step_summary
    if _serialized_size(capped) <= limit:
        return capped

    return {
        "reason": capped.get("reason"),
        "error": _truncate_string(capped.get("error"), limit=512),
        "failed_step": capped.get("failed_step"),
        "summary": [{"truncated": True}],
        "truncated": True,
        **{key: capped[key] for key in ("error_source", "error_kind", "error_target", "error_detail") if key in capped},
    }


def _truncate_failure_context_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _truncate_failure_context_strings(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_truncate_failure_context_strings(item) for item in value]
    if isinstance(value, str):
        return _truncate_string(value)
    return value


def _is_non_retryable_failure(reason: str, detail: str, extra_context: dict[str, Any]) -> bool:
    if str(extra_context.get("violation_tier") or "").strip().lower() == "hard":
        return True
    normalized = str(detail or "").strip()
    if normalized in NON_RETRYABLE_FAILURE_MESSAGES:
        return True
    if reason == "planner_contract_failed":
        return True
    return False


class ExecutionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._emit_startup_banner()
        self.base_settings = settings or get_settings()
        self.settings = self.base_settings
        self.store = SQLiteRunStore(self.settings.run_store_path)
        self.session_manager = SessionManager(self.store)
        self.llm = LLMClient(self.settings)
        self.tool_registry = build_tool_registry(self.settings)
        self.compiler = GraphCompiler()
        self.planner = TaskPlanner(llm=self.llm, tools=self.tool_registry, settings=self.settings)
        self.executor = PlanExecutor(self.tool_registry)
        self.validator = RuntimeValidator(self.settings)

    def _emit_startup_banner(self) -> None:
        sys.stderr.write(f"{render_startup_banner()}\n")
        sys.stderr.flush()

    def run_spec(self, spec_path: str, input_payload: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        session = self.create_session(spec_path, input_payload, trigger="manual", dry_run=dry_run)
        return self.resume_session(session["id"], trigger="manual")

    def create_session(
        self,
        spec_path: str,
        input_payload: dict[str, Any],
        trigger: str = "manual",
        dry_run: bool = False,
        stream_shell_output: bool = False,
    ) -> dict[str, Any]:
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
        session.state["dry_run"] = bool(dry_run)
        session.state["awaiting_confirmation"] = False
        session.state["confirmation_kind"] = None
        session.state["confirmation_step"] = None
        session.state["confirmation_message"] = None
        session.state["policies_used"] = []
        session.state["stream_shell_output"] = bool(stream_shell_output)
        self.store.append_event(
            session_id=session.id,
            node_name="session",
            event_type="session.created",
            payload={"input": input_payload, "spec": compiled.name, "trigger": trigger},
        )
        self.session_manager.persist_session(session, node_name="session")
        return session.model_dump()

    def resume_session(
        self,
        session_id: str,
        trigger: str = "manual",
        max_cycles: int | None = None,
        approve_dangerous: bool = False,
        stream_shell_output: bool = False,
    ) -> dict[str, Any]:
        session = self._get_required_session(session_id)
        if self._is_done(session.state):
            return ensure_jsonable(session.state)

        approved_dangerous_step_id: int | None = None
        if bool(session.state.get("awaiting_confirmation")):
            confirmation_kind = str(session.state.get("confirmation_kind") or "")
            if confirmation_kind == "dangerous_step":
                if not approve_dangerous:
                    return ensure_jsonable(session.state)
                confirmation_step = session.state.get("confirmation_step") or {}
                if isinstance(confirmation_step.get("id"), int):
                    approved_dangerous_step_id = int(confirmation_step["id"])
            self._clear_confirmation_state(session.state)
            session.state["dry_run"] = False
        session.current_trigger = trigger
        session.state["trigger"] = trigger
        session.state["stream_shell_output"] = bool(session.state.get("stream_shell_output", False) or stream_shell_output)
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
                self._run_executor_step(session, approved_dangerous_step_id=approved_dangerous_step_id)
                approved_dangerous_step_id = None
            elif action == "validator":
                self._run_validator(session)
            else:
                session.state["status"] = "failed"
                session.state["error"] = f"Unknown next action: {action}"
                session.state["done"] = True
                break
            if bool(session.state.get("awaiting_confirmation")):
                break
            cycles += 1

        if self._is_done(session.state):
            self._finalize_session(session)
        else:
            self.session_manager.persist_session(session, node_name="loop")
        if bool(session.state.get("awaiting_confirmation")) and str(session.state.get("confirmation_kind") or "") == "dry_run":
            return self._dry_run_preview(session.state)
        return ensure_jsonable(session.state)

    def trigger_session(
        self,
        session_id: str,
        trigger: str = "manual",
        max_cycles: int | None = None,
        approve_dangerous: bool = False,
        stream_shell_output: bool = False,
    ) -> dict[str, Any]:
        return self.resume_session(
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=approve_dangerous,
            stream_shell_output=stream_shell_output,
        )

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
        self._configure_runtime_for_compiled(compiled)
        goal = str(state.get("goal", ""))
        planning_mode = "hierarchical" if is_complex_goal(goal) else "direct"
        self.store.append_event(
            session_id=session.id,
            node_name="planner",
            event_type="planner.started",
            payload={
                "retry": state.get("retries", 0),
                "attempt": state.get("attempt", 0) + 1,
                "planning_mode": planning_mode,
            },
        )
        try:
            plan = self.planner.build_plan(
                goal=goal,
                planner=compiled.planner,
                allowed_tools=compiled.tools,
                input_payload=dict(state.get("input", {})),
                failure_context=state.get("failure_context"),
            )
            metrics = dict(state.get("metrics", {}))
            metrics["llm_calls"] = int(metrics.get("llm_calls", 0)) + int(self.planner.last_llm_calls)
            metrics["llm_intent_calls"] = int(metrics.get("llm_intent_calls", 0)) + int(self.planner.last_llm_intent_calls)
            metrics["raw_planner_llm_calls"] = int(metrics.get("raw_planner_llm_calls", 0)) + int(self.planner.last_raw_planner_llm_calls)
            state["metrics"] = metrics
            plan_summary = summarize_plan(plan)
            policies_used = list(self.planner.last_policies_used)
            high_level_plan = list(self.planner.last_high_level_plan) if self.planner.last_high_level_plan is not None else None
            repair_trace = list(self.planner.last_plan_repairs)
            canonicalized = bool(self.planner.last_plan_canonicalized)
            original_execution_plan = self.planner.last_original_execution_plan if canonicalized else None
            resolved_planning_mode = str(self.planner.last_planning_mode or planning_mode)
            planning_metadata = {
                "planning_mode": resolved_planning_mode,
                "capability": self.planner.last_capability_name,
                "llm_intent_type": self.planner.last_llm_intent_type,
                "llm_intent_confidence": self.planner.last_llm_intent_confidence,
                "llm_intent_reason": self.planner.last_llm_intent_reason,
                "llm_intent_calls": int(self.planner.last_llm_intent_calls),
                "raw_planner_llm_calls": int(self.planner.last_raw_planner_llm_calls),
            }
            planning_metadata.update(
                {
                    key: value
                    for key, value in dict(getattr(self.planner, "last_capability_metadata", {}) or {}).items()
                    if str(key).startswith("sql_")
                }
            )
            awaiting_confirmation = bool(state.get("dry_run"))
            state.update(
                {
                    "current_node": "planner",
                    "next_action": "executor",
                    "status": "executing",
                    "awaiting_confirmation": awaiting_confirmation,
                    "confirmation_kind": "dry_run" if awaiting_confirmation else None,
                    "confirmation_step": None,
                    "confirmation_message": None,
                    "policies_used": policies_used,
                    "high_level_plan": high_level_plan,
                    "step_outputs": {},
                    "plan": plan.model_dump(),
                    "plan_summary": plan_summary,
                    "plan_canonicalized": canonicalized,
                    "plan_repairs": repair_trace,
                    "attempt_history": [],
                    "current_step_index": 0,
                    "attempt": int(state.get("attempt", 0)) + 1,
                    "validation": None,
                    "validation_checks": [],
                    "planning_metadata": planning_metadata,
                    "failure_context": None,
                    "error": None,
                }
            )
            self.store.append_event(
                session_id=session.id,
                node_name="planner",
                event_type="planner.completed",
                payload={
                    **plan.model_dump(),
                    "goal": goal,
                    **planning_metadata,
                    "high_level_plan": high_level_plan,
                    "execution_plan": plan.model_dump(),
                    "original_execution_plan": original_execution_plan,
                    "canonicalization_changed": canonicalized,
                    "repair_trace": repair_trace,
                    "policies": policies_used,
                },
            )
        except Exception as exc:  # noqa: BLE001
            metrics = dict(state.get("metrics", {}))
            metrics["llm_calls"] = int(metrics.get("llm_calls", 0)) + int(self.planner.last_llm_calls)
            metrics["llm_intent_calls"] = int(metrics.get("llm_intent_calls", 0)) + int(self.planner.last_llm_intent_calls)
            metrics["raw_planner_llm_calls"] = int(metrics.get("raw_planner_llm_calls", 0)) + int(self.planner.last_raw_planner_llm_calls)
            state["metrics"] = metrics
            failed_policies = list(self.planner.last_policies_used)
            planner_error_type = str(self.planner.last_error_type or type(exc).__name__)
            planner_error_stage = str(self.planner.last_error_stage or planning_mode)
            raw_output_preview = summarize_planner_raw_output(self.planner.last_raw_output)
            normalized_error = normalize_planner_error(
                error_type=planner_error_type,
                detail=str(exc),
                llm_base_url=self.settings.llm_base_url,
            )
            final_error = normalized_error.message if normalized_error is not None else str(exc)
            final_output_metadata = {"goal": state.get("goal", ""), "planner_error_type": planner_error_type}
            final_output_metadata.update(
                {
                    "planning_mode": str(self.planner.last_planning_mode or planning_mode),
                    "capability": self.planner.last_capability_name,
                    "llm_intent_type": self.planner.last_llm_intent_type,
                    "llm_intent_confidence": self.planner.last_llm_intent_confidence,
                    "llm_intent_reason": self.planner.last_llm_intent_reason,
                    "llm_intent_calls": int(self.planner.last_llm_intent_calls),
                    "raw_planner_llm_calls": int(self.planner.last_raw_planner_llm_calls),
                }
            )
            if raw_output_preview is not None:
                final_output_metadata["planner_raw_output_preview"] = raw_output_preview
            if normalized_error is not None:
                final_output_metadata.update(normalized_error.as_metadata())
            if isinstance(exc, PlanContractViolation):
                final_output_metadata.update(exc.as_metadata())
            suggestion_content, suggestion_metadata = self._failure_output_with_suggestions(
                goal=goal,
                message=final_error,
                metadata=final_output_metadata,
                error=exc,
                plan=_safe_execution_plan(self.planner.last_canonicalized_execution_plan),
            )
            state.update(
                {
                    "current_node": "planner",
                    "next_action": "",
                    "status": "failed",
                    "done": True,
                    "awaiting_confirmation": False,
                    "confirmation_kind": None,
                    "confirmation_step": None,
                    "confirmation_message": None,
                    "policies_used": [],
                    "high_level_plan": None,
                    "step_outputs": {},
                    "plan_summary": None,
                    "plan_canonicalized": False,
                    "plan_repairs": [],
                    "error": final_error,
                    "final_output": {"content": suggestion_content, "artifacts": [], "metadata": suggestion_metadata},
                }
            )
            failure_payload = {
                "error": final_error,
                "error_type": planner_error_type,
                "stage": planner_error_stage,
                "planning_mode": str(self.planner.last_planning_mode or planning_mode),
                "capability": self.planner.last_capability_name,
                "llm_intent_type": self.planner.last_llm_intent_type,
                "llm_intent_confidence": self.planner.last_llm_intent_confidence,
                "llm_intent_reason": self.planner.last_llm_intent_reason,
                "llm_intent_calls": int(self.planner.last_llm_intent_calls),
                "raw_planner_llm_calls": int(self.planner.last_raw_planner_llm_calls),
                "policies": failed_policies,
            }
            if raw_output_preview is not None:
                failure_payload["raw_output_preview"] = raw_output_preview
            if self.planner.last_original_execution_plan is not None:
                failure_payload["original_execution_plan"] = self.planner.last_original_execution_plan
            if self.planner.last_canonicalized_execution_plan is not None:
                failure_payload["execution_plan"] = self.planner.last_canonicalized_execution_plan
            if self.planner.last_plan_repairs:
                failure_payload["repair_trace"] = list(self.planner.last_plan_repairs)
                failure_payload["canonicalization_changed"] = bool(self.planner.last_plan_canonicalized)
            if normalized_error is not None:
                failure_payload.update(normalized_error.as_metadata())
            if isinstance(exc, PlanContractViolation):
                failure_payload.update(exc.as_metadata())
            self.store.append_event(
                session_id=session.id,
                node_name="planner",
                event_type="planner.failed",
                payload=failure_payload,
            )
        self._persist(session, node_name="planner")

    def _run_executor_step(self, session: AgentSession, approved_dangerous_step_id: int | None = None) -> None:
        state = session.state
        compiled = CompiledRuntimeSpec.model_validate(session.compiled_spec)
        self._configure_runtime_for_compiled(compiled)
        plan = ExecutionPlan.model_validate(state.get("plan", {}))
        step_index = int(state.get("current_step_index", 0))
        if step_index >= len(plan.steps):
            state.update({"current_node": "executor", "next_action": "validator", "status": "validating"})
            self._persist(session, node_name="executor")
            return

        step = plan.steps[step_index]
        try:
            resolved_step = resolve_execution_step(step, dict(state.get("step_outputs", {})))
        except Exception as exc:  # noqa: BLE001
            self._handle_retry_or_failure(
                session,
                node_name="executor",
                reason="tool_execution_failed",
                detail=str(exc),
                extra_context={
                    "step": step.model_dump(),
                    "history": list(state.get("attempt_history", [])),
                },
            )
            return

        dangerous_message = self._dangerous_step_message(resolved_step)
        if dangerous_message is not None and approved_dangerous_step_id != step.id:
            state.update(
                {
                    "current_node": "executor",
                    "next_action": "executor",
                    "status": "executing",
                    "awaiting_confirmation": True,
                    "confirmation_kind": "dangerous_step",
                    "confirmation_step": resolved_step.model_dump(),
                    "confirmation_message": dangerous_message,
                    "error": None,
                }
            )
            self.store.append_event(
                session_id=session.id,
                node_name="executor",
                event_type="executor.step.awaiting_confirmation",
                payload={"step": resolved_step.model_dump(), "step_index": step_index, "message": dangerous_message},
            )
            self._persist(session, node_name="executor")
            return

        self._clear_confirmation_state(state)
        step_preview = self.executor.describe_step(resolved_step)
        self.store.append_event(
            session_id=session.id,
            node_name="executor",
            event_type="executor.step.started",
            payload={"step": resolved_step.model_dump(), "step_index": step_index, **step_preview},
        )
        stream_shell_output = bool(state.get("stream_shell_output", False))

        def emit_step_output(payload: dict[str, Any]) -> None:
            self.store.append_event(
                session_id=session.id,
                node_name="executor",
                event_type="executor.step.output",
                payload={
                    "step_id": int(payload.get("step_id") or resolved_step.id),
                    "step_index": step_index,
                    "action": str(payload.get("action") or resolved_step.action),
                    "channel": str(payload.get("channel") or ""),
                    "text": str(payload.get("text") or ""),
                    "node": str(payload.get("node") or resolved_step.args.get("node") or resolved_step.args.get("gateway_node") or ""),
                    "command": str(payload.get("command") or resolved_step.args.get("command", "")),
                },
            )

        log = self.executor.execute_step(resolved_step, event_sink=emit_step_output if stream_shell_output else None)
        log_payload = log.model_dump()

        history = list(state.get("history", []))
        attempt_history = list(state.get("attempt_history", []))
        step_outputs = dict(state.get("step_outputs", {}))
        history.append(log_payload)
        attempt_history.append(log_payload)
        state["history"] = history
        state["attempt_history"] = attempt_history
        if log.success and log.step.output:
            step_outputs[log.step.output] = log.result
        state["step_outputs"] = step_outputs

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
            final_output = summarize_final_output(
                str(state.get("goal", "")),
                attempt_models,
                settings=self.settings,
                metadata=dict(state.get("planning_metadata") or {}),
            )
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
        compiled = CompiledRuntimeSpec.model_validate(session.compiled_spec)
        self._configure_runtime_for_compiled(compiled)
        attempt_history = [self._step_log_from_dict(item) for item in state.get("attempt_history", [])]
        self.store.append_event(
            session_id=session.id,
            node_name="validator",
            event_type="validator.started",
            payload={"attempt_history_length": len(attempt_history), "goal": str(state.get("goal", ""))},
        )
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
        self._configure_runtime_for_compiled(compiled)
        retries = int(state.get("retries", 0))
        retryable = not _is_non_retryable_failure(reason, detail, extra_context)
        should_retry = retryable and retries < min(compiled.runtime.max_retries, self.settings.max_plan_retries)
        step_payload = extra_context.get("step")
        normalized_error = normalize_runtime_failure(
            reason=reason,
            detail=detail,
            step=step_payload if isinstance(step_payload, dict) else None,
            settings=self.settings,
        )
        final_detail = normalized_error.message if normalized_error is not None else detail
        history_payload = extra_context.get("history")
        validation_payload = extra_context.get("validation")
        validation_checks = extra_context.get("validation_checks")
        failure_context: dict[str, Any] = {
            "reason": reason,
            "error": final_detail,
            "failed_step": str(step_payload.get("action") or "") if isinstance(step_payload, dict) else None,
            "step": _summarize_step_payload(step_payload if isinstance(step_payload, dict) else None),
            "summary": summarize_failure_history(history_payload if isinstance(history_payload, list) else []),
        }
        if isinstance(validation_payload, dict):
            failure_context["validation"] = {
                "success": validation_payload.get("success"),
                "reason": _truncate_string(validation_payload.get("reason")),
            }
        if isinstance(validation_checks, list):
            failure_context["validation_checks"] = [
                _truncate_string(item, limit=120) or ""
                for item in validation_checks[:5]
                if _truncate_string(item, limit=120)
            ]
        if normalized_error is not None:
            failure_context.update(normalized_error.as_metadata())
        failure_context["retryable"] = retryable
        if "contract_violation" in extra_context:
            failure_context["contract_violation"] = bool(extra_context.get("contract_violation"))
        if extra_context.get("violation_tier"):
            failure_context["violation_tier"] = extra_context.get("violation_tier")
        if extra_context.get("violation_code"):
            failure_context["violation_code"] = extra_context.get("violation_code")
        failure_context = _cap_failure_context_size(failure_context)

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
                    "error": final_detail,
                    "awaiting_confirmation": False,
                    "confirmation_kind": None,
                    "confirmation_step": None,
                    "confirmation_message": None,
                    "policies_used": [],
                    "high_level_plan": None,
                    "step_outputs": {},
                    "plan": {},
                    "plan_summary": None,
                    "plan_canonicalized": False,
                    "plan_repairs": [],
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
                    "error": final_detail,
                    "awaiting_confirmation": False,
                    "confirmation_kind": None,
                    "confirmation_step": None,
                    "confirmation_message": None,
                    "policies_used": [],
                    "high_level_plan": None,
                    "step_outputs": {},
                    "plan_summary": None,
                    "plan_canonicalized": False,
                    "plan_repairs": [],
                    "final_output": {
                        "content": "",
                        "artifacts": list((state.get("final_output") or {}).get("artifacts", [])),
                        "metadata": {
                            "goal": state.get("goal", ""),
                            "failure_reason": reason,
                            "retryable": retryable,
                            **(normalized_error.as_metadata() if normalized_error is not None else {}),
                        },
                    },
                }
            )
            final_output = dict(state.get("final_output") or {})
            plan = _safe_execution_plan(state.get("plan", {}))
            suggestion_content, suggestion_metadata = self._failure_output_with_suggestions(
                goal=str(state.get("goal", "")),
                message=final_detail,
                metadata={
                    **dict(final_output.get("metadata") or {}),
                    **({"reason": reason} if reason else {}),
                },
                error=RuntimeError(final_detail),
                plan=plan,
            )
            final_output["content"] = suggestion_content
            final_output["metadata"] = suggestion_metadata
            state["final_output"] = final_output
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
        final_output = self._decorate_final_output(state, final_output, status=status, metrics=metrics)
        self.store.append_event(
            session_id=session.id,
            node_name="finalize",
            event_type="finalize.started",
            payload={"status": status},
        )
        state.update(
            {
                "current_node": "finalize",
                "next_action": "",
                "status": status,
                "done": True,
                "awaiting_confirmation": False,
                "confirmation_kind": None,
                "confirmation_step": None,
                "confirmation_message": None,
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

    def _decorate_final_output(
        self,
        state: RuntimeState,
        final_output: dict[str, Any],
        *,
        status: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(final_output.get("metadata") or {})
        content = str(final_output.get("content") or "")
        goal = str(state.get("goal", ""))
        planning_metadata = {
            key: value
            for key, value in dict(state.get("planning_metadata") or {}).items()
            if value is not None
        }

        if status == "failed" and "failure_type" not in metadata:
            failure_context = dict(state.get("failure_context") or {})
            failure_message = str(state.get("error") or content or "Task failed.")
            plan = _safe_execution_plan(state.get("plan", {}))
            suggestion_content, suggestion_metadata = self._failure_output_with_suggestions(
                goal=goal,
                message=failure_message,
                metadata={**metadata, **failure_context},
                error=RuntimeError(failure_message),
                plan=plan,
            )
            return {
                "content": suggestion_content,
                "artifacts": list(final_output.get("artifacts", [])),
                "metadata": suggestion_metadata,
            }

        if status == "completed" and int(metrics.get("llm_calls", 0)) > 0 and "prompt_suggestions" not in metadata:
            metadata.update(planning_metadata)
            suggestion_result = generate_prompt_suggestions(
                goal,
                classify_failure(
                    goal,
                    metadata={**metadata, "status": status, "llm_calls": int(metrics.get("llm_calls", 0))},
                ),
                context=self._prompt_suggestion_context(state, metadata),
            )
            metadata.update(suggestion_result.metadata_payload())
            return {
                "content": content,
                "artifacts": list(final_output.get("artifacts", [])),
                "metadata": metadata,
            }

        return {
            "content": content,
            "artifacts": list(final_output.get("artifacts", [])),
            "metadata": metadata,
        }

    def _failure_output_with_suggestions(
        self,
        *,
        goal: str,
        message: str,
        metadata: dict[str, Any],
        error: Exception | None = None,
        plan: ExecutionPlan | None = None,
    ) -> tuple[str, dict[str, Any]]:
        suggestion_result = generate_prompt_suggestions(
            goal,
            classify_failure(goal, error=error, plan=plan, metadata=metadata),
            context=self._prompt_suggestion_context(None, metadata),
        )
        enriched_metadata = dict(metadata)
        enriched_metadata.update(suggestion_result.metadata_payload())
        return append_prompt_suggestions(message, suggestion_result), enriched_metadata

    def _prompt_suggestion_context(self, state: RuntimeState | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        context: dict[str, Any] = {
            "workspace_root": str(self.settings.workspace_root),
            "outputs_dir": str(self.settings.workspace_root / "outputs"),
        }
        if state is not None and state.get("goal"):
            context["goal"] = str(state.get("goal"))
        if metadata:
            context.update(dict(metadata))
        return context

    def _persist(self, session: AgentSession, *, node_name: str) -> None:
        self._touch_state(session.state)
        session.status = str(session.state.get("status", session.status))
        session.current_trigger = str(session.state.get("trigger", session.current_trigger))
        session.history = list(session.state.get("history", session.history))
        self.session_manager.persist_session(session, node_name=node_name)

    def _settings_for_compiled(self, compiled: CompiledRuntimeSpec) -> Settings:
        payload = self.base_settings.model_dump()
        endpoints = {endpoint.name: endpoint.url for endpoint in compiled.nodes.endpoints}
        if endpoints:
            payload["available_nodes_raw"] = ",".join(endpoints)
            payload["gateway_endpoints"] = endpoints
            payload["gateway_url"] = None
            payload["default_node"] = compiled.nodes.default
        return Settings.model_validate(payload)

    def _configure_runtime_for_compiled(self, compiled: CompiledRuntimeSpec) -> None:
        effective_settings = self._settings_for_compiled(compiled)
        self.settings = effective_settings
        self.llm.settings = effective_settings
        self.tool_registry = build_tool_registry(effective_settings)
        self.planner.settings = effective_settings
        self.planner.tools = self.tool_registry
        self.executor.tools = self.tool_registry
        self.validator.settings = effective_settings

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

    @staticmethod
    def _clear_confirmation_state(state: RuntimeState) -> None:
        state["awaiting_confirmation"] = False
        state["confirmation_kind"] = None
        state["confirmation_step"] = None
        state["confirmation_message"] = None

    def is_dangerous(self, step: ExecutionStep) -> bool:
        return self._dangerous_step_message(step) is not None

    def _dangerous_step_message(self, step: ExecutionStep) -> str | None:
        args = dict(step.args)

        if step.action == "fs.write":
            path = str(args.get("path", "")).strip()
            if path:
                target = resolve_path(self.settings, path)
                if target.exists():
                    return f"Overwrite existing path via fs.write: {target}"
            return None

        if step.action == "fs.copy":
            dst = str(args.get("dst", "")).strip()
            if dst:
                target = resolve_path(self.settings, dst)
                if target.exists():
                    return f"Overwrite existing path via fs.copy: {target}"
            return None

        if step.action == "shell.exec":
            command = str(args.get("command", "")).strip()
            if command and DANGEROUS_SHELL_PATTERN.search(command.lower()):
                try:
                    target = resolve_execution_node(self.settings, str(args.get("node", "")))
                except Exception:  # noqa: BLE001
                    return None
                return f"Run potentially destructive shell command on {target}: {command}"
            return None

        return None

    @staticmethod
    def _dry_run_preview(state: RuntimeState) -> dict[str, Any]:
        return {
            "session_id": str(state.get("session_id", "")),
            "plan": state.get("plan", {}),
            "summary": state.get("plan_summary"),
        }
