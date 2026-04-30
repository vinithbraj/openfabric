"""OpenFABRIC Runtime Module: aor_runtime.llm.__init__

Purpose:
    Provide the LLM client abstraction used by planner and optional presentation layers. Existing module summary: LLM integration layer.

Responsibilities:
    Normalize configured provider access, model selection, token accounting, and structured response handling.

Data flow / Interfaces:
    Receives compact planner/presentation prompts and returns model responses plus usage metadata.

Boundaries:
    Must not be handed raw result rows or sensitive payloads except through explicitly sanitized prompt construction.
"""
