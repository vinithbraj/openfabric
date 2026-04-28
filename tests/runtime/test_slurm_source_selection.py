from __future__ import annotations

from aor_runtime.runtime.slurm_semantics import extract_slurm_semantic_frame


def test_live_queue_prompts_select_squeue() -> None:
    assert extract_slurm_semantic_frame("show current running jobs").source == "squeue"
    assert extract_slurm_semantic_frame("count pending jobs right now").source == "squeue"


def test_runtime_and_history_prompts_select_sacct() -> None:
    assert extract_slurm_semantic_frame("how long did jobs take on totalseg partition").source == "sacct"
    assert extract_slurm_semantic_frame("average runtime by partition").source == "sacct"
    assert extract_slurm_semantic_frame("show failed jobs yesterday").source == "sacct"


def test_inventory_and_health_prompts_select_inventory_sources() -> None:
    assert extract_slurm_semantic_frame("show node status").source == "sinfo"
    assert extract_slurm_semantic_frame("show partition summary").source == "sinfo"
    assert extract_slurm_semantic_frame("is slurmdbd healthy").source == "sacctmgr"


def test_cluster_health_selects_derived_source() -> None:
    frame = extract_slurm_semantic_frame("what is the status of my slurm cluster?")

    assert frame.source == "derived"
    assert frame.compound_requests
