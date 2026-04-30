"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.policies.slurm

Purpose:
    Provide SLURM-specific semantic policy imports.

Responsibilities:
    Expose SLURM accounting state policy, all-state phrase recognition, and canonical state-filter handling.

Data flow / Interfaces:
    Used by SLURM semantic compilers and tests to map user wording such as "all jobs" or "completed jobs" to safe tool args.

Boundaries:
    This module owns SLURM meaning policy only; SLURM command construction remains in the tool layer.
"""

from __future__ import annotations

from aor_runtime.runtime.semantic.core import (
    SLURM_ALL_STATES_RE,
    SlurmAccountingStatePolicy,
    _canonical_slurm_accounting_filters as canonical_slurm_accounting_filters,
    _extract_partition_targets as extract_partition_targets,
    _extract_state_filter as extract_state_filter,
    _slurm_accounting_state_policy as slurm_accounting_state_policy,
    _slurm_all_states_phrase as slurm_all_states_phrase,
)

__all__ = [
    "SLURM_ALL_STATES_RE",
    "SlurmAccountingStatePolicy",
    "canonical_slurm_accounting_filters",
    "extract_partition_targets",
    "extract_state_filter",
    "slurm_accounting_state_policy",
    "slurm_all_states_phrase",
]
