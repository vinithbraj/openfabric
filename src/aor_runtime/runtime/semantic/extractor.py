"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.extractor

Purpose:
    Provide semantic-frame extraction imports for deterministic and LLM-assisted meaning extraction.

Responsibilities:
    Expose safe semantic prompt construction, LLM extraction, deterministic extraction, and metric normalization helpers.

Data flow / Interfaces:
    Consumes user goals, runtime settings, and allowed tool metadata; returns semantic-frame extraction results.

Boundaries:
    Extraction describes intent only and must not emit executable tools, shell commands, SQL text, or raw payloads.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import (
    LLMSemanticFrameExtractor,
    build_semantic_frame_prompt,
    deterministic_semantic_frame,
    normalize_metric_name,
)

__all__ = [
    "LLMSemanticFrameExtractor",
    "build_semantic_frame_prompt",
    "deterministic_semantic_frame",
    "normalize_metric_name",
]
