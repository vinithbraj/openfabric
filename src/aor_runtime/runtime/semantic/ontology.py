"""OpenFABRIC Runtime Module: aor_runtime.runtime.semantic.ontology

Purpose:
    Centralize semantic aliases for domains, entities, dimensions, and filters.

Responsibilities:
    Normalize LLM-proposed meaning into canonical labels and expose safe ontology
    metadata for prompts, matrix matching, compiler lowering, and tests.

Data flow / Interfaces:
    Consumes plain text labels or user prompts; returns canonical semantic tokens
    without producing executable tools, SQL, shell commands, or data payloads.

Boundaries:
    Owns language-to-token normalization only. It does not compile tools, run
    queries, inspect rows, or render user-visible results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OntologyMatch:
    """Represent one explicit semantic alias found in user text.

    Inputs:
        Built from domain, canonical token, and matched alias text.

    Returns:
        Immutable prompt-grounding evidence for canonicalization and tests.

    Used by:
        detect_prompt_entities and semantic-frame canonicalization.
    """

    domain: str
    canonical: str
    alias: str


SQL_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "patients": ("patient", "patients", "patieint", "patieints"),
    "studies": ("study", "studies"),
    "series": ("series",),
    "instances": ("instance", "instances", "image", "images"),
    "rtplan": ("rtplan", "rtplans", "rt plan", "rt plans"),
    "rtdose": ("rtdose", "rtdoses", "rt dose", "rt doses"),
    "rtstruct": ("rtstruct", "rtstructs", "rt structure", "rt structures"),
    "ct": ("ct",),
    "mr": ("mr", "mri"),
    "pt": ("pt", "pet"),
    "modalities": ("modality", "modalities"),
    "tables": ("table", "tables"),
    "schema": ("schema", "catalog"),
    "columns": ("column", "columns"),
    "relationships": ("relationship", "relationships", "join path", "join paths"),
}

SLURM_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "nodes": ("node", "nodes", "unique node", "unique nodes", "compute node", "compute nodes"),
    "partitions": ("partition", "partitions", "queue partition", "queue partitions"),
    "jobs": ("job", "jobs", "queue", "queued jobs", "slurm jobs"),
    "accounting_jobs": ("accounting job", "accounting jobs", "sacct jobs"),
    "cluster": ("cluster", "slurm cluster"),
    "accounting": ("accounting", "slurmdbd", "slurmdbd health"),
}

FILESYSTEM_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "files": ("file", "files"),
    "directory": ("folder", "folders", "directory", "directories"),
    "content": ("content", "matches", "references"),
    "configs": ("config", "configs", "configuration", "configurations"),
    "docs": ("doc", "docs", "documentation", "markdown"),
}

SHELL_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "processes": ("process", "processes"),
    "ports": ("port", "ports", "listening port", "listening ports"),
    "disk": ("disk", "disk usage"),
    "memory": ("memory", "memory usage"),
    "cpu": ("cpu", "cpu info"),
    "hostname": ("hostname", "host name"),
    "uptime": ("uptime",),
    "mounts": ("mount", "mounts", "mounted filesystems"),
    "docker": ("docker", "container", "containers"),
    "user": ("user", "current user"),
    "pwd": ("pwd", "working directory", "current directory"),
}

ENTITY_ALIASES_BY_DOMAIN: dict[str, dict[str, tuple[str, ...]]] = {
    "sql": SQL_ENTITY_ALIASES,
    "slurm": SLURM_ENTITY_ALIASES,
    "filesystem": FILESYSTEM_ENTITY_ALIASES,
    "shell": SHELL_ENTITY_ALIASES,
}

DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "modality": ("modality", "modalities"),
    "partition": ("partition", "partitions"),
    "state": ("state", "states", "status"),
    "user": ("user", "users"),
    "node": ("node", "nodes"),
    "job_name": ("job name", "job_name", "jobname"),
    "extension": ("extension", "extensions", "file type", "file types"),
    "path": ("path", "paths"),
    "type": ("type", "types"),
    "year": ("year", "years"),
    "month": ("month", "months"),
    "decade": ("decade", "decades", "age decade", "age decades"),
}

FILTER_ALIASES: dict[str, tuple[str, ...]] = {
    "database": ("database", "db"),
    "age": ("age", "patient age", "patient_age", "patient_age_years"),
    "concept_term": ("concept", "concept term", "term", "description"),
    "modality": ("modality", "modalities"),
    "missing_value": ("missing", "missing value", "absent"),
    "partition": ("partition", "partitions"),
    "state": ("state", "status"),
    "user": ("user", "users"),
    "extension": ("extension", "file type", "type"),
    "path": ("path", "folder", "directory"),
    "pattern": ("pattern", "glob"),
    "needle": ("needle", "search"),
    "process": ("process", "processes"),
}


def normalize_semantic_entity(domain: str, value: Any) -> str:
    """Normalize an entity label within a semantic domain.

    Inputs:
        Receives a domain name and raw entity label from LLM output or tests.

    Returns:
        Canonical entity token when known, otherwise a normalized token.

    Used by:
        Semantic frame canonicalization, matrix matching, and SQL/SLURM compilers.
    """
    token = normalize_semantic_token(value)
    if not token:
        return ""
    aliases = ENTITY_ALIASES_BY_DOMAIN.get(normalize_semantic_token(domain), {})
    for canonical, raw_aliases in aliases.items():
        if token == canonical or token in {normalize_semantic_token(alias) for alias in raw_aliases}:
            return canonical
    return token


def normalize_semantic_dimension(value: Any) -> str:
    """Normalize a grouping or target dimension label.

    Inputs:
        Receives an arbitrary dimension label.

    Returns:
        Canonical dimension token when known.

    Used by:
        Semantic target, output group_by, and matrix matching helpers.
    """
    token = normalize_semantic_token(value)
    for canonical, aliases in DIMENSION_ALIASES.items():
        if token == canonical or token in {normalize_semantic_token(alias) for alias in aliases}:
            return canonical
    return token


def normalize_semantic_filter_field(value: Any) -> str:
    """Normalize a filter field label.

    Inputs:
        Receives an arbitrary semantic filter field.

    Returns:
        Canonical filter token when known.

    Used by:
        Semantic frame canonicalization and matrix matching.
    """
    token = normalize_semantic_token(value)
    for canonical, aliases in FILTER_ALIASES.items():
        if token == canonical or token in {normalize_semantic_token(alias) for alias in aliases}:
            return canonical
    return token


def normalize_semantic_token(value: Any) -> str:
    """Normalize arbitrary semantic text into a stable token.

    Inputs:
        Receives a raw string-like value.

    Returns:
        Lowercase underscore-separated token.

    Used by:
        All ontology normalization helpers.
    """
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def detect_prompt_entities(domain: str, prompt: str) -> tuple[OntologyMatch, ...]:
    """Detect explicit entity aliases in the user prompt for one domain.

    Inputs:
        Receives a semantic domain and original user prompt text.

    Returns:
        Ordered unique ontology matches for explicitly mentioned entities.

    Used by:
        SemanticFrameCanonicalizer to ground LLM frames in the user's words.
    """
    normalized_domain = normalize_semantic_token(domain)
    aliases = ENTITY_ALIASES_BY_DOMAIN.get(normalized_domain, {})
    text = str(prompt or "")
    matches: list[OntologyMatch] = []
    seen: set[str] = set()
    for canonical, raw_aliases in aliases.items():
        for alias in sorted(raw_aliases, key=len, reverse=True):
            if _alias_in_text(alias, text):
                if canonical not in seen:
                    seen.add(canonical)
                    matches.append(OntologyMatch(normalized_domain, canonical, alias))
                break
    return tuple(matches)


def ontology_prompt_metadata() -> dict[str, Any]:
    """Return safe ontology metadata for LLM semantic-frame prompts.

    Inputs:
        Uses static ontology declarations only.

    Returns:
        JSON-compatible alias metadata without tools, rows, commands, or payloads.

    Used by:
        build_semantic_frame_prompt.
    """
    return {
        "entities": {domain: {canonical: list(aliases) for canonical, aliases in mapping.items()} for domain, mapping in ENTITY_ALIASES_BY_DOMAIN.items()},
        "dimensions": {canonical: list(aliases) for canonical, aliases in DIMENSION_ALIASES.items()},
        "filters": {canonical: list(aliases) for canonical, aliases in FILTER_ALIASES.items()},
    }


def _alias_in_text(alias: str, text: str) -> bool:
    """Return whether an alias appears as a word-like phrase in text.

    Inputs:
        Receives one alias and a user prompt.

    Returns:
        True when the alias appears with non-word boundaries.

    Used by:
        detect_prompt_entities.
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(str(alias or "").strip()) + r"(?![A-Za-z0-9_])"
    return bool(re.search(pattern, text, re.IGNORECASE))


__all__ = [
    "OntologyMatch",
    "detect_prompt_entities",
    "normalize_semantic_dimension",
    "normalize_semantic_entity",
    "normalize_semantic_filter_field",
    "normalize_semantic_token",
    "ontology_prompt_metadata",
]
