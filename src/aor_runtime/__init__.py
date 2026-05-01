"""OpenFABRIC V10 echo reset package metadata.

Purpose:
    Expose package version metadata and runtime version helpers for the
    reset runtime.

Responsibilities:
    Resolve package version, Git revision, and dirty-worktree suffix for API/model identity surfaces.

Data flow / Interfaces:
    Exports __version__, __package_version__, and get_runtime_version for CLI, API, and OpenWebUI model naming.

Boundaries:
    Performs read-only Git inspection only and must not mutate repository state.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["__package_version__", "__version__", "get_runtime_version"]

__package_version__ = "0.10.0"


@lru_cache(maxsize=1)
def get_runtime_version() -> str:
    """Get runtime version for the surrounding runtime workflow.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.__init__.get_runtime_version.
    """
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
    """Handle the internal git worktree is dirty helper path for this module.

    Inputs:
        Receives repo_root for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.__init__._git_worktree_is_dirty.
    """
    status = _run_git_command(repo_root, ["status", "--short"])
    return bool(status)


def _run_git_command(repo_root: Path, args: list[str]) -> str:
    """Handle the internal run git command helper path for this module.

    Inputs:
        Receives repo_root, args for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by OpenFABRIC runtime support code paths that import or call aor_runtime.__init__._run_git_command.
    """
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
