"""Validation helpers for input semantic contracts."""

from __future__ import annotations

from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import TaskFrame


class TaskFrameValidator:
    """Validate required task-frame invariants."""

    def validate(self, frame: TaskFrame) -> None:
        """Raise when a frame is too incomplete to lower."""

        if not frame.description.strip():
            raise ValidationError("task frame description is required")
        if not frame.semantic_verb:
            raise ValidationError("task frame semantic_verb is required")
        if not frame.object_type.strip():
            raise ValidationError("task frame object_type is required")
