from __future__ import annotations

from agent_runtime.core.semantic_compatibility import (
    canonical_domain,
    canonical_object_family,
    domains_compatible,
    has_hard_cross_domain_conflict,
    object_types_compatible,
)


def test_canonical_domain_aliases() -> None:
    assert canonical_domain("system_administration") == "system"
    assert canonical_domain("operating_system") == "system"
    assert canonical_domain("system administration") == "system"
    assert canonical_domain("resource_usage") == "system"
    assert canonical_domain("repo") == "git"
    assert canonical_domain("repository") == "git"
    assert canonical_domain("csv") == "table"
    assert canonical_domain("database") == "sql"
    assert canonical_domain("hpc") == "slurm"
    assert canonical_domain("workflow") == "airflow"
    assert canonical_domain("capabilities") == "runtime"


def test_canonical_object_family_aliases() -> None:
    assert canonical_object_family("memory") == "system.memory"
    assert canonical_object_family("free_memory") == "system.memory"
    assert canonical_object_family("ram") == "system.memory"
    assert canonical_object_family("swap") == "system.memory"
    assert canonical_object_family("cpu_load") == "system.cpu"
    assert canonical_object_family("disk_usage") == "system.disk"
    assert canonical_object_family("uptime") == "system.uptime"
    assert canonical_object_family("folder") == "filesystem.directory"
    assert canonical_object_family("file") == "filesystem.file"
    assert canonical_object_family("repo") == "git.repository"
    assert canonical_object_family("csv") == "table.csv"


def test_domains_compatible_aliases() -> None:
    assert domains_compatible("system_administration", "system")
    assert domains_compatible("operating_system", "system")
    assert domains_compatible(
        None,
        "system",
        ["system_administration", "operating_system", "system"],
        "memory",
        ["system.memory", "memory"],
    )
    assert domains_compatible("repo", "git")
    assert domains_compatible("csv", "table")
    assert domains_compatible("database", "sql")
    assert domains_compatible("hpc", "slurm")
    assert domains_compatible("workflow", "airflow")


def test_object_types_compatible_aliases() -> None:
    assert object_types_compatible("memory", ["system.memory"])
    assert object_types_compatible("memory", ["memory", "ram", "swap"])
    assert object_types_compatible("system.memory", ["system.resources"])
    assert object_types_compatible("folder", ["filesystem.directory"])
    assert object_types_compatible("repo", ["git.repository"])
    assert object_types_compatible("csv", ["table.csv", "table"])
    assert object_types_compatible("database", ["sql.database"])


def test_hard_cross_domain_conflicts() -> None:
    assert has_hard_cross_domain_conflict(
        "system",
        "filesystem",
        "system.memory",
        ["filesystem.directory"],
        ["system"],
    )
    assert has_hard_cross_domain_conflict(
        "system",
        "filesystem",
        "system.cpu",
        ["filesystem.file"],
        ["system"],
    )
    assert has_hard_cross_domain_conflict(
        "system",
        "shell",
        "system.disk",
        ["system.process"],
        ["system"],
    )
    assert has_hard_cross_domain_conflict(
        "sql",
        "shell",
        "sql.patient",
        ["system.process"],
        ["sql"],
    )
    assert has_hard_cross_domain_conflict(
        "table",
        "shell",
        "table.csv",
        ["system.process"],
        ["table"],
    )
    assert has_hard_cross_domain_conflict(
        "runtime",
        "filesystem",
        "runtime.capabilities",
        ["filesystem.file"],
        ["runtime"],
    )
