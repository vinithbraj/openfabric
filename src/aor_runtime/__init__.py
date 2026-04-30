"""OpenFABRIC runtime."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["__package_version__", "__version__", "get_runtime_version"]

__package_version__ = "0.4.2"


@lru_cache(maxsize=1)
def get_runtime_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return __package_version__
    revision = _run_git_command(repo_root, ["rev-parse", "--short", "HEAD"])
    if not revision:
        return __package_version__
    version = f"{__package_version__}+{revision}"
    if _git_worktree_is_dirty(repo_root):
        version = f"{version}.dirty"
    return version


def _git_worktree_is_dirty(repo_root: Path) -> bool:
    status = _run_git_command(repo_root, ["status", "--short"])
    return bool(status)


def _run_git_command(repo_root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


__version__ = get_runtime_version()
