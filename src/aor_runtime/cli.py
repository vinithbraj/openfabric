from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import typer
import uvicorn

from aor_runtime.app_config import APP_CONFIG_PATH_ENV
from aor_runtime.config import Settings, get_settings
from aor_runtime.core.utils import dumps_json
from aor_runtime.dsl.loader import load_runtime_spec
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.capabilities.registry import build_default_capability_registry

app = typer.Typer(help="OpenFABRIC CLI")
runs_app = typer.Typer(help="Inspect persisted runs")
sessions_app = typer.Typer(help="Manage long-running agent sessions")
app.add_typer(runs_app, name="runs")
app.add_typer(sessions_app, name="sessions")

CHAT_COMMANDS = ("/exit", "/quit", "/history", "/last", "/new", "/clear", "/capabilities")


def _is_dry_run_preview(payload: dict[str, Any]) -> bool:
    return all(key in payload for key in ("session_id", "plan", "summary")) and "status" not in payload


def _is_dangerous_confirmation_pause(payload: dict[str, Any]) -> bool:
    return bool(payload.get("awaiting_confirmation")) and str(payload.get("confirmation_kind") or "") == "dangerous_step"


def _dangerous_confirmation_preview(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": str(payload.get("session_id", "")),
        "confirmation_message": payload.get("confirmation_message"),
        "confirmation_step": payload.get("confirmation_step"),
    }


def _resolve_dangerous_confirmation(
    engine: ExecutionEngine,
    state: dict[str, Any],
    *,
    trigger: str,
    max_cycles: int | None = None,
    approve_dangerous: bool = False,
    stream_shell_output: bool = False,
) -> dict[str, Any]:
    auto_approve = approve_dangerous
    while _is_dangerous_confirmation_pause(state):
        typer.echo(dumps_json(_dangerous_confirmation_preview(state), indent=2))
        if not auto_approve and not typer.confirm("Continue with dangerous operation?", default=False):
            return state
        state = engine.resume_session(
            str(state["session_id"]),
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=True,
            stream_shell_output=stream_shell_output,
        )
        auto_approve = False
    return state


def _final_answer_from_state(state: dict[str, Any]) -> str:
    if bool(state.get("awaiting_confirmation")):
        confirmation_message = state.get("confirmation_message")
        if isinstance(confirmation_message, str) and confirmation_message.strip():
            return confirmation_message.strip()

    final_output = state.get("final_output")
    if isinstance(final_output, dict):
        content = final_output.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    error = state.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    validation = state.get("validation")
    if isinstance(validation, dict):
        detail = validation.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    return dumps_json(state, indent=2)


def _reset_chat_history(session_history: list[dict[str, str]]) -> None:
    session_history.clear()


def _session_history_window(session_history: list[dict[str, str]], max_history: int) -> list[dict[str, str]]:
    if max_history <= 0 or not session_history:
        return []
    return session_history[-max_history:]


def _chat_commands_banner() -> str:
    return f"Commands: {', '.join(CHAT_COMMANDS)}"


def _start_background(target):
    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["result"] = target()
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, outcome


def _render_progress_event(event: dict[str, Any]) -> None:
    event_type = str(event.get("event_type") or "")
    payload = dict(event.get("payload") or {})
    if event_type == "planner.started":
        typer.echo("Thinking...")
        return
    if event_type == "planner.completed":
        execution_plan = dict(payload.get("execution_plan") or {})
        steps = list(execution_plan.get("steps") or [])
        typer.echo(f"Plan ready: {len(steps)} step{'s' if len(steps) != 1 else ''}")
        return
    if event_type == "executor.step.started":
        step = dict(payload.get("step") or {})
        node = str(payload.get("node") or "").strip()
        command = str(payload.get("command") or "").strip()
        suffix = f" on {node}" if node else ""
        typer.echo(f"Executing: {step.get('action', 'step')}{suffix}")
        if command:
            typer.echo(f"Command: {command}")
        return
    if event_type == "executor.step.output":
        text = str(payload.get("text") or "")
        if not text:
            return
        if str(payload.get("channel") or "") == "stderr":
            typer.echo(text, nl=False, err=True)
        else:
            typer.echo(text, nl=False)
        return
    if event_type == "executor.step.completed":
        step = dict(payload.get("step") or {})
        typer.echo(f"Completed: {step.get('action', 'step')}")
        return
    if event_type == "validator.started":
        typer.echo("Validating...")
        return
    if event_type == "validator.completed":
        result = dict(payload.get("result") or {})
        typer.echo("Validation passed" if bool(result.get("success")) else "Validation failed")
        return
    if event_type.endswith(".failed"):
        phase = event_type.split(".", 1)[0]
        typer.echo(f"Failed during {phase}: {payload.get('error', 'Task failed.')}")
        return
    if event_type == "finalize.completed":
        typer.echo(f"Finished: {payload.get('status', 'completed')}")


def _drain_progress_events(engine: ExecutionEngine, session_id: str, after_id: int | None) -> int | None:
    cursor = after_id
    for event in engine.store.get_events_after(session_id, after_id=after_id):
        _render_progress_event(event)
        cursor = int(event["id"])
    return cursor


def _run_session_with_progress(
    engine: ExecutionEngine,
    session_id: str,
    *,
    trigger: str,
    max_cycles: int | None = None,
    approve_dangerous: bool = False,
) -> dict[str, Any]:
    worker, outcome = _start_background(
        lambda: engine.resume_session(
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=approve_dangerous,
            stream_shell_output=True,
        )
    )
    cursor: int | None = None
    while worker.is_alive():
        cursor = _drain_progress_events(engine, session_id, cursor)
        time.sleep(0.05)
    worker.join()
    cursor = _drain_progress_events(engine, session_id, cursor)
    if "error" in outcome:
        raise outcome["error"]
    state = dict(outcome.get("result") or {})
    if _is_dangerous_confirmation_pause(state):
        typer.echo(dumps_json(_dangerous_confirmation_preview(state), indent=2))
        if not approve_dangerous and not typer.confirm("Continue with dangerous operation?", default=False):
            return state
        return _run_session_with_progress(
            engine,
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=True,
        )
    return state


def _chat_capabilities_payload(spec_path: Path) -> dict[str, Any]:
    spec = load_runtime_spec(spec_path)
    registry = build_default_capability_registry()
    return {
        "runtime_spec": {
            "name": spec.name,
            "description": spec.description,
            "path": str(spec_path),
            "version": spec.version,
            "tools": list(spec.tools),
            "default_node": spec.nodes.default,
            "nodes": [endpoint.name for endpoint in spec.nodes.endpoints],
        },
        "capability_packs": [
            {
                "name": pack.name,
                "intent_types": [intent_type.__name__ for intent_type in pack.intent_types],
            }
            for pack in registry.packs
        ],
        "chat_commands": list(CHAT_COMMANDS),
    }


def _build_engine(config_path: Path | None = None) -> ExecutionEngine:
    settings = get_settings(config_path=config_path)
    return ExecutionEngine(settings)


def _resolve_server_binding(settings: Settings, host: str | None, port: int | None) -> tuple[str, int]:
    return (host or settings.server_host, port or settings.server_port)


@app.command()
def validate(
    spec_path: Path,
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    engine = _build_engine(config)
    compiled = engine.validate_spec(str(spec_path))
    typer.echo(dumps_json(compiled.model_dump(), indent=2))


@app.command()
def run(
    spec_path: Path,
    input: str = typer.Option("{}", help="JSON input payload"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview the plan and confirm before execution."),
    progress: bool = typer.Option(False, "--progress", help="Print live planning and execution events."),
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    payload = json.loads(input)
    engine = _build_engine(config)
    if progress:
        session = engine.create_session(str(spec_path), payload, trigger="manual", dry_run=dry_run, stream_shell_output=True)
        state = _run_session_with_progress(engine, str(session["id"]), trigger="manual")
    else:
        state = engine.run_spec(str(spec_path), payload, dry_run=dry_run)
    if dry_run and _is_dry_run_preview(state):
        typer.echo(dumps_json(state, indent=2))
        if typer.confirm("Run this plan?", default=False):
            if progress:
                resumed = _run_session_with_progress(engine, str(state["session_id"]), trigger="manual")
            else:
                resumed = engine.resume_session(str(state["session_id"]), trigger="manual")
                resumed = _resolve_dangerous_confirmation(engine, resumed, trigger="manual")
            typer.echo(dumps_json(resumed, indent=2))
        return
    if not progress:
        state = _resolve_dangerous_confirmation(engine, state, trigger="manual")
    typer.echo(dumps_json(state, indent=2))


@app.command()
def resume(
    session_id: str,
    trigger: str = "manual",
    max_cycles: int | None = None,
    approve_dangerous: bool = typer.Option(
        False,
        "--approve-dangerous",
        help="Approve a paused dangerous step before resuming execution.",
    ),
    progress: bool = typer.Option(False, "--progress", help="Print live planning and execution events."),
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    engine = _build_engine(config)
    if progress:
        state = _run_session_with_progress(
            engine,
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=approve_dangerous,
        )
    else:
        state = engine.resume_session(
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=approve_dangerous,
        )
        state = _resolve_dangerous_confirmation(engine, state, trigger=trigger, max_cycles=max_cycles)
    typer.echo(dumps_json(state, indent=2))


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help="Host for the API server."),
    port: int | None = typer.Option(None, "--port", help="Port for the API server."),
    reload: bool = False,
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    settings = get_settings(config_path=config)
    resolved_host, resolved_port = _resolve_server_binding(settings, host, port)
    if settings.app_config_path is not None:
        os.environ[APP_CONFIG_PATH_ENV] = str(settings.app_config_path)
    uvicorn.run("aor_runtime.api.app:create_app", host=resolved_host, port=resolved_port, reload=reload, factory=True)


@app.command()
def chat(
    spec_path: Path,
    system_note: str = typer.Option("", help="Optional operator note injected into each turn."),
    max_history: int = typer.Option(0, help="Number of prior turns to include in session context."),
    progress: bool = typer.Option(False, "--progress", help="Print live planning and execution events."),
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    engine = _build_engine(config)
    session_history: list[dict[str, str]] = []

    typer.echo(f"Interactive session started for {spec_path}. Type /exit to quit.")
    typer.echo(_chat_commands_banner())

    while True:
        prompt = typer.prompt("\nYou").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            typer.echo("Session ended.")
            break
        if prompt == "/history":
            typer.echo(dumps_json(session_history, indent=2))
            continue
        if prompt == "/last":
            if not session_history:
                typer.echo("No turns yet.")
            else:
                typer.echo(dumps_json(session_history[-2:], indent=2))
            continue
        if prompt == "/new":
            _reset_chat_history(session_history)
            typer.echo("Started a new conversation.")
            continue
        if prompt == "/clear":
            _reset_chat_history(session_history)
            typer.echo("\033[2J\033[H", nl=False)
            typer.echo(f"Interactive session started for {spec_path}. Type /exit to quit.")
            typer.echo(_chat_commands_banner())
            continue
        if prompt == "/capabilities":
            typer.echo(dumps_json(_chat_capabilities_payload(spec_path), indent=2))
            continue

        payload: dict[str, Any] = {"task": prompt}
        if system_note.strip():
            payload["system_note"] = system_note.strip()
        history_window = _session_history_window(session_history, max_history)
        if history_window:
            payload["session_history"] = history_window

        if progress:
            session = engine.create_session(str(spec_path), payload, trigger="manual", stream_shell_output=True)
            state = _run_session_with_progress(engine, str(session["id"]), trigger="manual")
        else:
            state = engine.run_spec(str(spec_path), payload)
            state = _resolve_dangerous_confirmation(engine, state, trigger="manual")
        answer = _final_answer_from_state(state)
        run_id = state.get("run_id", "")

        typer.echo(f"\nAssistant [{run_id}]")
        typer.echo(answer)

        session_history.append({"role": "user", "content": prompt})
        session_history.append({"role": "assistant", "content": answer})


@runs_app.command("list")
def list_runs(limit: int = 20, config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config.")) -> None:
    engine = _build_engine(config)
    typer.echo(dumps_json(engine.list_runs(limit=limit), indent=2))


@runs_app.command("show")
def show_run(run_id: str, config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config.")) -> None:
    engine = _build_engine(config)
    payload = engine.get_run(run_id)
    if payload is None:
        raise typer.Exit(code=1)
    typer.echo(dumps_json(payload, indent=2))


@sessions_app.command("create")
def create_session(
    spec_path: Path,
    input: str = typer.Option("{}", help="JSON input payload"),
    run_immediately: bool = typer.Option(True, help="Run the loop immediately after creating the session."),
    progress: bool = typer.Option(False, "--progress", help="Print live planning and execution events."),
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    payload = json.loads(input)
    engine = _build_engine(config)
    session = engine.create_session(str(spec_path), payload, trigger="manual", stream_shell_output=progress)
    if run_immediately:
        if progress:
            state = _run_session_with_progress(engine, str(session["id"]), trigger="manual")
        else:
            state = engine.resume_session(session["id"], trigger="manual")
            state = _resolve_dangerous_confirmation(engine, state, trigger="manual")
        typer.echo(dumps_json(state, indent=2))
        return
    typer.echo(dumps_json(session, indent=2))


@sessions_app.command("resume")
def resume_session(
    session_id: str,
    trigger: str = "manual",
    max_cycles: int | None = None,
    approve_dangerous: bool = typer.Option(
        False,
        "--approve-dangerous",
        help="Approve a paused dangerous step before resuming execution.",
    ),
    progress: bool = typer.Option(False, "--progress", help="Print live planning and execution events."),
    config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config."),
) -> None:
    engine = _build_engine(config)
    if progress:
        state = _run_session_with_progress(
            engine,
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=approve_dangerous,
        )
    else:
        state = engine.resume_session(
            session_id,
            trigger=trigger,
            max_cycles=max_cycles,
            approve_dangerous=approve_dangerous,
        )
        state = _resolve_dangerous_confirmation(engine, state, trigger=trigger, max_cycles=max_cycles)
    typer.echo(dumps_json(state, indent=2))


@sessions_app.command("list")
def list_sessions(limit: int = 20, config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config.")) -> None:
    engine = _build_engine(config)
    typer.echo(dumps_json(engine.list_sessions(limit=limit), indent=2))


@sessions_app.command("show")
def show_session(session_id: str, config: Path | None = typer.Option(None, "--config", help="Path to the YAML app config.")) -> None:
    engine = _build_engine(config)
    payload = engine.get_session(session_id)
    if payload is None:
        raise typer.Exit(code=1)
    typer.echo(dumps_json(payload, indent=2))


if __name__ == "__main__":
    app()
