from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import uvicorn

from aor_runtime.core.utils import dumps_json
from aor_runtime.runtime.engine import ExecutionEngine

app = typer.Typer(help="Agent Orchestration Runtime CLI")
runs_app = typer.Typer(help="Inspect persisted runs")
sessions_app = typer.Typer(help="Manage long-running agent sessions")
app.add_typer(runs_app, name="runs")
app.add_typer(sessions_app, name="sessions")


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


@app.command()
def validate(spec_path: Path) -> None:
    engine = ExecutionEngine()
    compiled = engine.validate_spec(str(spec_path))
    typer.echo(dumps_json(compiled.model_dump(), indent=2))


@app.command()
def run(
    spec_path: Path,
    input: str = typer.Option("{}", help="JSON input payload"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview the plan and confirm before execution."),
) -> None:
    payload = json.loads(input)
    engine = ExecutionEngine()
    state = engine.run_spec(str(spec_path), payload, dry_run=dry_run)
    if dry_run and _is_dry_run_preview(state):
        typer.echo(dumps_json(state, indent=2))
        if typer.confirm("Run this plan?", default=False):
            resumed = engine.resume_session(str(state["session_id"]), trigger="manual")
            resumed = _resolve_dangerous_confirmation(engine, resumed, trigger="manual")
            typer.echo(dumps_json(resumed, indent=2))
        return
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
) -> None:
    engine = ExecutionEngine()
    state = engine.resume_session(
        session_id,
        trigger=trigger,
        max_cycles=max_cycles,
        approve_dangerous=approve_dangerous,
    )
    state = _resolve_dangerous_confirmation(engine, state, trigger=trigger, max_cycles=max_cycles)
    typer.echo(dumps_json(state, indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8011, reload: bool = False) -> None:
    uvicorn.run("aor_runtime.api.app:create_app", host=host, port=port, reload=reload, factory=True)


@app.command()
def chat(
    spec_path: Path,
    system_note: str = typer.Option("", help="Optional operator note injected into each turn."),
    max_history: int = typer.Option(8, help="Number of prior turns to include in session context."),
) -> None:
    engine = ExecutionEngine()
    session_history: list[dict[str, str]] = []

    typer.echo(f"Interactive session started for {spec_path}. Type /exit to quit.")
    typer.echo("Commands: /exit, /quit, /history, /last")

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

        payload: dict[str, Any] = {"task": prompt}
        if system_note.strip():
            payload["system_note"] = system_note.strip()
        if session_history:
            payload["session_history"] = session_history[-max_history:]

        state = engine.run_spec(str(spec_path), payload)
        state = _resolve_dangerous_confirmation(engine, state, trigger="manual")
        answer = _final_answer_from_state(state)
        run_id = state.get("run_id", "")

        typer.echo(f"\nAssistant [{run_id}]")
        typer.echo(answer)

        session_history.append({"role": "user", "content": prompt})
        session_history.append({"role": "assistant", "content": answer})


@runs_app.command("list")
def list_runs(limit: int = 20) -> None:
    engine = ExecutionEngine()
    typer.echo(dumps_json(engine.list_runs(limit=limit), indent=2))


@runs_app.command("show")
def show_run(run_id: str) -> None:
    engine = ExecutionEngine()
    payload = engine.get_run(run_id)
    if payload is None:
        raise typer.Exit(code=1)
    typer.echo(dumps_json(payload, indent=2))


@sessions_app.command("create")
def create_session(
    spec_path: Path,
    input: str = typer.Option("{}", help="JSON input payload"),
    run_immediately: bool = typer.Option(True, help="Run the loop immediately after creating the session."),
) -> None:
    payload = json.loads(input)
    engine = ExecutionEngine()
    session = engine.create_session(str(spec_path), payload, trigger="manual")
    if run_immediately:
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
) -> None:
    engine = ExecutionEngine()
    state = engine.resume_session(
        session_id,
        trigger=trigger,
        max_cycles=max_cycles,
        approve_dangerous=approve_dangerous,
    )
    state = _resolve_dangerous_confirmation(engine, state, trigger=trigger, max_cycles=max_cycles)
    typer.echo(dumps_json(state, indent=2))


@sessions_app.command("list")
def list_sessions(limit: int = 20) -> None:
    engine = ExecutionEngine()
    typer.echo(dumps_json(engine.list_sessions(limit=limit), indent=2))


@sessions_app.command("show")
def show_session(session_id: str) -> None:
    engine = ExecutionEngine()
    payload = engine.get_session(session_id)
    if payload is None:
        raise typer.Exit(code=1)
    typer.echo(dumps_json(payload, indent=2))


if __name__ == "__main__":
    app()
