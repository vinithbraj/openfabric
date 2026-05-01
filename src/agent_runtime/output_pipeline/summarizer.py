"""Summarization interface for composed output."""

from __future__ import annotations

from agent_runtime.core.types import RenderedOutput


class Summarizer:
    """Placeholder summarizer that leaves output unchanged."""

    def summarize(self, output: RenderedOutput) -> RenderedOutput:
        """Return rendered output without modification."""

        return output
