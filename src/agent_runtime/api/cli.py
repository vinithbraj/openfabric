"""Typer CLI for the OpenFABRIC agent runtime.

Purpose:
    Keep the ``aor`` command as a lightweight entrypoint into the current
    runtime and server surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console

from agent_runtime.api.app import create_app
from agent_runtime.api.app_config import APP_CONFIG_PATH_ENV
from agent_runtime.api.config import Settings, get_settings
from agent_runtime.api.constants import DEFAULT_COMPAT_SPEC_PATH
from agent_runtime.api.runtime.engine import ExecutionEngine, extract_prompt


app = typer.Typer(help="OpenFABRIC agent runtime.")
console = Console()


def _settings(config: Path | None) -> Settings:
    """Load runtime settings for CLI commands.

    Used by:
        ``serve``, ``run``, and ``chat`` commands.
    """

    return get_settings(config_path=str(config) if config else None)


def _payload_from_input(input_json: str | None, prompt: str | None) -> dict[str, object]:
    """Build the engine input payload from CLI options.

    Used by:
        ``run`` command.
    """

    if input_json:
        payload = json.loads(input_json)
        if not isinstance(payload, dict):
            raise typer.BadParameter("--input must decode to a JSON object.")
        return payload
    return {"task": prompt or ""}


@app.command()
def serve(
    config: Annotated[Path | None, typer.Option("--config", help="Path to config.yaml.")] = None,
    host: Annotated[str | None, typer.Option("--host", help="Override configured host.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Override configured port.")] = None,
) -> None:
    """Start the OpenWebUI/OpenAI-compatible API."""

    settings = _settings(config)
    if config:
        import os

        os.environ[APP_CONFIG_PATH_ENV] = str(config)
    uvicorn.run(
        create_app(settings),
        host=host or settings.server_host,
        port=port or settings.server_port,
    )


@app.command()
def run(
    spec_path: Annotated[
        str,
        typer.Argument(help="Optional compatibility spec path; retained for older surfaces but ignored by the current runtime."),
    ] = DEFAULT_COMPAT_SPEC_PATH,
    input_json: Annotated[str | None, typer.Option("--input", help="JSON input containing task/prompt.")] = None,
    prompt: Annotated[str | None, typer.Option("--prompt", help="Prompt text to run.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Path to config.yaml.")] = None,
) -> None:
    """Run one prompt once and print the final rendered result."""

    engine = ExecutionEngine(_settings(config))
    payload = _payload_from_input(input_json, prompt)
    result = engine.run_spec(spec_path, payload)
    content = str((result.get("final_output") or {}).get("content") or "")
    console.print(content)


@app.command()
def chat(
    spec_path: Annotated[
        str,
        typer.Argument(help="Optional compatibility spec path; retained for older surfaces but ignored by the current runtime."),
    ] = DEFAULT_COMPAT_SPEC_PATH,
    config: Annotated[Path | None, typer.Option("--config", help="Path to config.yaml.")] = None,
) -> None:
    """Start a tiny interactive chat loop through the compatibility runtime."""

    engine = ExecutionEngine(_settings(config))
    console.print("[bold]OpenFABRIC agent runtime[/bold]. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = typer.prompt("you")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if prompt.strip().lower() in {"exit", "quit"}:
            return
        result = engine.run_spec(spec_path, {"task": prompt})
        console.print(str((result.get("final_output") or {}).get("content") or extract_prompt({"task": prompt})))


if __name__ == "__main__":
    app()
