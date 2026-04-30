"""OpenFABRIC Runtime Module: aor_runtime.tools.__init__

Purpose:
    Implement registered tools that perform local or gateway-routed work after validation. Existing module summary: Tool registry and built-in tools.

Responsibilities:
    Expose typed tool arguments/results for filesystem, SQL, shell, SLURM, text formatting, Python, and runtime return operations.

Data flow / Interfaces:
    Receives validated tool arguments from the executor and returns structured result models for downstream contracts and presenters.

Boundaries:
    Does not decide user intent; every tool must preserve safety, allowed-root, read-only, timeout, and result-shape boundaries.
"""
