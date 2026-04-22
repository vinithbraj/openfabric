from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from aor_runtime.core.utils import dumps_json
from aor_runtime.runtime.engine import ExecutionEngine

app = typer.Typer(help="Agent Orchestration Runtime CLI")
runs_app = typer.Typer(help="Inspect persisted runs")
app.add_typer(runs_app, name="runs")


@app.command()
def validate(spec_path: Path) -> None:
    engine = ExecutionEngine()
    compiled = engine.validate_spec(str(spec_path))
    typer.echo(dumps_json(compiled.model_dump(), indent=2))


@app.command()
def run(spec_path: Path, input: str = typer.Option("{}", help="JSON input payload")) -> None:
    payload = json.loads(input)
    engine = ExecutionEngine()
    state = engine.run_spec(str(spec_path), payload)
    typer.echo(dumps_json(state, indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8011, reload: bool = False) -> None:
    uvicorn.run("aor_runtime.api.app:create_app", host=host, port=port, reload=reload, factory=True)


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


if __name__ == "__main__":
    app()
