"""OpenFABRIC Runtime Module: aor_runtime.dsl.__init__

Purpose:
    Load and model runtime YAML specifications. Existing module summary: YAML DSL models and loaders.

Responsibilities:
    Parse assistant specs into typed configuration objects before compilation and execution.

Data flow / Interfaces:
    Receives YAML/runtime spec data and returns validated DSL models for the compiler.

Boundaries:
    Keeps configuration parsing separate from request-time LLM planning and tool execution.
"""
