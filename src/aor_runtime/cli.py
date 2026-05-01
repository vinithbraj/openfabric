"""Typer CLI for the V10 echo reset runtime.

Purpose:
    Keep the ``aor`` command usable while the internal agent runtime is rebuilt.

Responsibilities:
    Load configuration, start the FastAPI server, and provide local run/chat
    commands that echo prompt text.

Data flow / Interfaces:
    CLI arguments are converted into prompt payloads for ``ExecutionEngine`` or
    server settings for ``uvicorn``.

Boundaries:
    The CLI does not invoke tools, LLMs, SQL, SLURM, shell commands, or gateway
    calls. It only echoes prompt text through the reset runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console

from aor_runtime.api.app import create_app
from aor_runtime.app_config import APP_CONFIG_PATH_ENV
from aor_runtime.config import Settings, get_settings
from aor_runtime.runtime.engine import ExecutionEngine, extract_prompt


app = typer.Typer(help="OpenFABRIC V10 echo reset runtime.")
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
    """Start the OpenWebUI/OpenAI-compatible echo API."""

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
    spec_path: Annotated[str, typer.Argument(help="Compatibility spec path; retained but not executed.")],
    input_json: Annotated[str | None, typer.Option("--input", help="JSON input containing task/prompt.")] = None,
    prompt: Annotated[str | None, typer.Option("--prompt", help="Prompt text to echo.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Path to config.yaml.")] = None,
) -> None:
    """Echo a prompt once and print the result."""

    engine = ExecutionEngine(_settings(config))
    payload = _payload_from_input(input_json, prompt)
    result = engine.run_spec(spec_path, payload)
    content = str((result.get("final_output") or {}).get("content") or "")
    console.print(content)


@app.command()
def chat(
    spec_path: Annotated[str, typer.Argument(help="Compatibility spec path; retained but not executed.")],
    config: Annotated[Path | None, typer.Option("--config", help="Path to config.yaml.")] = None,
) -> None:
    """Start a tiny interactive echo loop."""

    engine = ExecutionEngine(_settings(config))
    console.print("[bold]OpenFABRIC echo runtime[/bold]. Type 'exit' or 'quit' to stop.")
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
