"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic

Purpose:
    Group the semantic-frame implementation behind the public compatibility facade.

Responsibilities:
    Expose focused semantic submodules for shared models, capability metadata, extraction, canonicalization, compilation, coverage, and projection.

Data flow / Interfaces:
    Imported by runtime semantic-frame wrappers and domain semantic modules.

Boundaries:
    Keeps domain policy and compiler seams separate from the legacy public `runtime.semantic_frame` import path.
"""

from __future__ import annotations

