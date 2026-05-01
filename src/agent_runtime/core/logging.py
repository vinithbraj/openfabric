"""Logging helpers for runtime components."""

from __future__ import annotations

import json
import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for an agent-runtime module."""

    return logging.getLogger(f"agent_runtime.{name}")


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit one structured JSON log event."""

    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, sort_keys=True))
