"""OpenFABRIC V10 echo runtime package.

Purpose:
    Host the minimal echo engine used during the architecture reset.

Responsibilities:
    Expose a tiny in-memory engine that echoes prompt text and records
    compatibility events.

Data flow / Interfaces:
    Consumes prompt dictionaries from API/CLI callers and produces echo
    session dictionaries.

Boundaries:
    Performs no planning or tool execution, keeping the reset branch free of
    old internal behavior.
"""
