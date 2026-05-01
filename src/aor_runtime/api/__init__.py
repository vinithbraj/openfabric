"""OpenFABRIC V10 echo API package.

Purpose:
    Expose the FastAPI control plane and OpenAI-compatible chat surface.

Responsibilities:
    Translate HTTP requests into echo engine calls and preserve API
    compatibility for OpenWebUI and integration clients.

Data flow / Interfaces:
    Receives HTTP payloads and returns JSON, SSE events, or
    OpenAI-compatible chat responses backed by echo sessions.

Boundaries:
    Performs no tool execution and no LLM calls; the assistant response is the
    selected user prompt.
"""
