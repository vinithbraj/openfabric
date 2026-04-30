"""OpenFABRIC Runtime Module: aor_runtime.api.__init__

Purpose:
    Expose the FastAPI control plane, session APIs, streaming endpoints, and OpenAI-compatible chat surface.

Responsibilities:
    Translate HTTP requests into engine calls, stream safe progress, and preserve API compatibility for OpenWebUI and integration clients.

Data flow / Interfaces:
    Receives HTTP payloads and returns JSON, SSE events, or OpenAI-compatible chat responses backed by ExecutionEngine sessions.

Boundaries:
    Protects API/user-mode boundaries by hiding raw runtime internals unless an explicit debug or raw surface is selected.
"""
