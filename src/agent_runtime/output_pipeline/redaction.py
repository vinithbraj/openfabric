"""Redaction boundary for output composition."""

from __future__ import annotations

from typing import Any


class Redactor:
    """Placeholder redactor for rendered values."""

    def redact(self, value: Any) -> Any:
        """Return the value unchanged until policies are defined."""

        return value
