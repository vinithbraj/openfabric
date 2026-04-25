from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aor_runtime import __version__
from aor_runtime.app_config import APP_CONFIG_PATH_ENV
from aor_runtime.config import Settings, get_settings
from aor_runtime.runtime.engine import ExecutionEngine


class RunRequest(BaseModel):
    spec_path: str
    input: dict[str, Any] = Field(default_factory=dict)


class SessionTriggerRequest(BaseModel):
    trigger: str = "manual"
    max_cycles: int | None = None
    approve_dangerous: bool = False


class ValidateRequest(BaseModel):
    spec_path: str


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Agent Orchestration Runtime", version=__version__)
    configured_settings = settings or get_settings(config_path=os.getenv(APP_CONFIG_PATH_ENV) or None)
    engine = ExecutionEngine(configured_settings)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/compile")
    def compile_spec(request: ValidateRequest) -> dict[str, Any]:
        compiled = engine.validate_spec(request.spec_path)
        return compiled.model_dump()

    @app.post("/sessions")
    def create_session(request: RunRequest, run_immediately: bool = True) -> dict[str, Any]:
        session = engine.create_session(request.spec_path, request.input, trigger="manual")
        if not run_immediately:
            return session
        return engine.resume_session(session["id"], trigger="manual")

    @app.post("/sessions/{session_id}/trigger")
    def trigger_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        try:
            return engine.trigger_session(
                session_id,
                trigger=request.trigger,
                max_cycles=request.max_cycles,
                approve_dangerous=request.approve_dangerous,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume")
    def resume_session(session_id: str, request: SessionTriggerRequest) -> dict[str, Any]:
        try:
            return engine.resume_session(
                session_id,
                trigger=request.trigger,
                max_cycles=request.max_cycles,
                approve_dangerous=request.approve_dangerous,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions")
    def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
        return engine.list_sessions(limit=limit)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        payload = engine.get_session(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return payload

    @app.post("/runs")
    def create_run(request: RunRequest) -> dict[str, Any]:
        final_state = engine.run_spec(request.spec_path, request.input)
        return final_state

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        return engine.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        payload = engine.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    return app
