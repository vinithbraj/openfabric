from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from gateway_agent import __version__
from gateway_agent.config import Settings, get_settings
from gateway_agent.executor import execute_command
from gateway_agent.models import CapabilitiesResponse, CapabilityInfo, ExecRequest, ExecResponse, HealthResponse


LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    app = FastAPI(title="OpenFabric Gateway Agent", version=__version__)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", node=runtime_settings.node_name)

    @app.get("/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            node=runtime_settings.node_name,
            version=__version__,
            capabilities=[
                CapabilityInfo(name="healthz", description="Report agent health and the configured logical node."),
                CapabilityInfo(
                    name="exec",
                    description="Execute a local shell command when the request node matches the configured node.",
                ),
            ],
        )

    @app.post("/exec", response_model=ExecResponse)
    def exec_command(payload: ExecRequest) -> ExecResponse:
        node = str(payload.node or "").strip()
        command = str(payload.command or "").strip()

        if not node:
            raise HTTPException(status_code=400, detail="Node is required.")
        if not command:
            raise HTTPException(status_code=400, detail="Command is required.")
        if node != runtime_settings.node_name:
            raise HTTPException(
                status_code=400,
                detail=f"Node mismatch. This agent serves node '{runtime_settings.node_name}'.",
            )

        try:
            return execute_command(runtime_settings, command)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive server guard
            LOGGER.exception("Gateway agent execution failed.")
            raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return app


app = create_app()
