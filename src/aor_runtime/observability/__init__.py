"""OpenFABRIC Runtime Module: aor_runtime.observability.__init__

Purpose:
    Provide observability-facing helpers for runtime diagnostics and event inspection. Existing module summary: Observability helpers.

Responsibilities:
    Keep tracing and monitoring concerns separate from planning, execution, and presentation code.

Data flow / Interfaces:
    Consumes session/runtime metadata and exposes safe observability helpers.

Boundaries:
    Must avoid turning raw tool payloads or sensitive values into broad logs or user-visible traces.
"""
