from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aor_runtime.runtime.engine import ExecutionEngine


class RunRequest(BaseModel):
    spec_path: str
    input: dict[str, Any] = Field(default_factory=dict)


class ValidateRequest(BaseModel):
    spec_path: str


def create_app() -> FastAPI:
    app = FastAPI(title="Agent Orchestration Runtime", version="0.1.0")
    engine = ExecutionEngine()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/compile")
    def compile_spec(request: ValidateRequest) -> dict[str, Any]:
        compiled = engine.validate_spec(request.spec_path)
        return compiled.model_dump()

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
