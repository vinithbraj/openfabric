"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.compilers

Purpose:
    Group domain-owned semantic compiler modules.

Responsibilities:
    Provide stable import paths for compiling domain semantic frames into execution plans.

Data flow / Interfaces:
    Imported by tests and future dispatcher code that need direct access to domain compiler seams.

Boundaries:
    Domain compilers choose safe execution strategies but do not bypass validation, coverage, or tool contracts.
"""

from __future__ import annotations

