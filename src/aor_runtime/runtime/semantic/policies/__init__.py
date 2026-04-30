"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.policies

Purpose:
    Group domain-owned semantic policy modules.

Responsibilities:
    Provide stable import paths for domain meaning rules such as SLURM accounting state policy.

Data flow / Interfaces:
    Imported by semantic canonicalizers, compilers, and tests that need domain policy behavior.

Boundaries:
    Policies interpret user meaning and must not execute tools or produce executable payloads.
"""

from __future__ import annotations

