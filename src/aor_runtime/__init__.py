"""Agent Orchestration Runtime."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["__package_version__", "__version__", "get_runtime_version"]

__package_version__ = "0.1.0"


@lru_cache(maxsize=1)
def get_runtime_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return __package_version__
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return __package_version__
    revision = result.stdout.strip()
    if result.returncode == 0 and revision:
        return revision
    return __package_version__


__version__ = get_runtime_version()
