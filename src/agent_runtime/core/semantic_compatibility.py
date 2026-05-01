"""Canonical semantic compatibility helpers used by planning validation."""

from __future__ import annotations

import re
from typing import Optional

_SEPARATOR_RE = re.compile(r"[\s\-\./:]+")
_UNDERSCORE_RE = re.compile(r"_+")


def normalize_token(value: Optional[str]) -> Optional[str]:
    """Normalize one semantic token for comparison."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = _SEPARATOR_RE.sub("_", text)
    text = _UNDERSCORE_RE.sub("_", text).strip("_")
    return text or None


_DOMAIN_GROUPS = {
    "runtime": {
        "runtime",
        "agent",
        "registry",
        "capability",
        "capabilities",
        "tool",
        "tools",
        "pipeline",
        "plan",
        "dag",
    },
    "filesystem": {
        "filesystem",
        "file_system",
        "fs",
        "file",
        "files",
        "directory",
        "directories",
        "folder",
        "folders",
        "path",
        "paths",
    },
    "system": {
        "system",
        "system_administration",
        "sysadmin",
        "operating_system",
        "os",
        "host",
        "machine",
        "computer",
        "node",
        "local_system",
        "remote_system",
        "system_resources",
        "resources",
        "resource_usage",
        "system_metrics",
        "metrics",
        "performance",
        "utilization",
    },
    "shell": {
        "shell",
        "command",
        "commands",
        "process",
        "processes",
        "terminal",
        "cli",
    },
    "git": {
        "git",
        "repository",
        "repo",
        "version_control",
        "source_control",
        "commit",
        "branch",
        "diff",
    },
    "code": {
        "code",
        "source",
        "source_code",
        "program",
        "module",
        "function",
        "class",
        "symbol",
        "import",
        "test_file",
    },
    "data": {
        "data",
        "records",
        "structured_data",
        "list",
        "dictionary",
        "dict",
        "json",
        "payload",
    },
    "table": {
        "table",
        "csv",
        "tsv",
        "dataframe",
        "spreadsheet",
        "rows",
        "columns",
        "tabular",
    },
    "sql": {
        "sql",
        "database",
        "db",
        "postgres",
        "postgresql",
        "relational",
        "table_schema",
        "query",
    },
    "dicom": {
        "dicom",
        "pacs",
        "study",
        "series",
        "instance",
        "rtplan",
        "rtdose",
        "rtstruct",
        "medical_image",
        "imaging",
    },
    "slurm": {
        "slurm",
        "cluster",
        "hpc",
        "job",
        "queue",
        "partition",
        "gpu_job",
    },
    "airflow": {
        "airflow",
        "dag_run",
        "workflow",
        "orchestration",
        "task_instance",
    },
    "web": {
        "web",
        "http",
        "url",
        "webpage",
        "website",
        "browser",
    },
}

_DOMAIN_LOOKUP = {
    normalize_token(alias): canonical
    for canonical, aliases in _DOMAIN_GROUPS.items()
    for alias in aliases
    if normalize_token(alias) is not None
}


def canonical_domain(value: Optional[str]) -> Optional[str]:
    """Return the canonical domain label for one semantic value."""

    token = normalize_token(value)
    if token is None:
        return None
    if token in _DOMAIN_LOOKUP:
        return _DOMAIN_LOOKUP[token]
    if "." in str(value):
        prefix = normalize_token(str(value).split(".", 1)[0])
        if prefix in _DOMAIN_LOOKUP:
            return _DOMAIN_LOOKUP[prefix]
    prefix = token.split("_", 1)[0]
    if prefix in _DOMAIN_LOOKUP:
        return _DOMAIN_LOOKUP[prefix]
    return token


_OBJECT_FAMILY_GROUPS = {
    "runtime.capabilities": {
        "runtime.capabilities",
        "capability",
        "capabilities",
        "tool",
        "tools",
        "registry",
    },
    "runtime.pipeline": {
        "runtime.pipeline",
        "pipeline",
        "architecture",
        "plan",
        "dag",
    },
    "system.memory": {
        "memory",
        "free_memory",
        "memory_usage",
        "ram",
        "swap",
        "mem",
        "system_memory",
        "system.memory",
    },
    "system.cpu": {
        "cpu",
        "processor",
        "load",
        "cpu_load",
        "load_average",
        "system_cpu",
        "system.cpu",
    },
    "system.disk": {
        "disk",
        "disk_usage",
        "free_disk",
        "storage",
        "storage_usage",
        "filesystem_storage",
        "system_disk",
        "system.disk",
    },
    "system.uptime": {"uptime", "system_uptime", "system.uptime"},
    "system.environment": {
        "environment",
        "system_environment",
        "env_summary",
        "system.environment",
    },
    "system.process": {
        "process",
        "processes",
        "process_list",
        "running_processes",
        "system.process",
    },
    "filesystem.file": {"file", "filesystem.file"},
    "filesystem.directory": {"directory", "folder", "filesystem.directory"},
    "git.repository": {"repo", "repository", "git.repository"},
    "code.symbol": {"function", "class", "symbol", "code.symbol"},
    "table.csv": {"csv", "table.csv"},
    "table": {"dataframe", "rows", "spreadsheet", "table"},
    "sql.database": {"database", "db", "sql.database"},
    "sql.patient": {"patient", "sql.patient"},
    "dicom.study": {"dicom.study"},
}

_OBJECT_FAMILY_ALIASES = {
    normalize_token(alias): canonical
    for canonical, aliases in _OBJECT_FAMILY_GROUPS.items()
    for alias in aliases
    if normalize_token(alias) is not None
}


def canonical_object_family(value: Optional[str]) -> Optional[str]:
    """Return a canonical object family for one semantic object label."""

    token = normalize_token(value)
    if token is None:
        return None
    if token in _OBJECT_FAMILY_ALIASES:
        return _OBJECT_FAMILY_ALIASES[token]
    dotted = str(value)
    if "." in dotted:
        dotted_token = normalize_token(dotted.replace(".", "_"))
        if dotted_token in _OBJECT_FAMILY_ALIASES:
            return _OBJECT_FAMILY_ALIASES[dotted_token]
        first = dotted.split(".", 1)[0]
        domain = canonical_domain(first)
        if domain:
            return f"{domain}.{normalize_token(dotted.split('.', 1)[1])}"
    domain = canonical_domain(token)
    if domain and domain != token:
        if domain in {"filesystem", "git", "sql", "system"}:
            return domain
    return token


def known_object_families() -> list[str]:
    """Return the runtime's known canonical object-family vocabulary."""

    return sorted(_OBJECT_FAMILY_GROUPS.keys())


def match_object_family_from_text(
    text: Optional[str],
    allowed_object_families: list[str] | None = None,
) -> Optional[str]:
    """Return the best matching canonical object family from free text, if any."""

    normalized_text = normalize_token(text)
    if normalized_text is None:
        return None
    haystack = f"_{normalized_text}_"
    allowed = set(allowed_object_families or known_object_families())
    best_match: tuple[int, str] | None = None

    for family, aliases in _OBJECT_FAMILY_GROUPS.items():
        if family not in allowed:
            continue
        for alias in aliases:
            normalized_alias = normalize_token(alias)
            if normalized_alias is None:
                continue
            if f"_{normalized_alias}_" in haystack or haystack.endswith(f"_{normalized_alias}") or haystack.startswith(
                f"{normalized_alias}_"
            ):
                score = len(normalized_alias)
                if best_match is None or score > best_match[0]:
                    best_match = (score, family)
    return best_match[1] if best_match is not None else None


_VERB_ALIASES = {
    "read": "read",
    "search": "search",
    "create": "create",
    "update": "update",
    "delete": "delete",
    "transform": "transform",
    "analyze": "analyze",
    "summarize": "summarize",
    "compare": "compare",
    "execute": "execute",
    "render": "render",
    "unknown": "unknown",
    "list": "search",
    "show": "read",
    "inspect": "read",
    "compute": "analyze",
    "calculate": "analyze",
    "count": "analyze",
    "sum": "analyze",
    "merge": "transform",
    "filter": "transform",
    "sort": "transform",
    "run": "execute",
    "display": "render",
    "format": "render",
}


def canonical_semantic_verb(value: Optional[str]) -> str:
    """Return a canonical semantic verb."""

    token = normalize_token(value)
    if token is None:
        return "unknown"
    return _VERB_ALIASES.get(token, "unknown")


def object_domain(value: Optional[str]) -> Optional[str]:
    """Return the canonical domain implied by one object family."""

    family = canonical_object_family(value)
    if family is None:
        return None
    return canonical_domain(family)


def object_types_compatible(
    task_object_type: Optional[str],
    manifest_object_types: list[str],
    context_domains: Optional[list[str]] = None,
) -> bool:
    """Return whether one task object family is compatible with manifest object types."""

    task_family = canonical_object_family(task_object_type)
    manifest_families = {
        family
        for family in (canonical_object_family(value) for value in manifest_object_types)
        if family is not None
    }
    if task_family is None:
        return True
    if task_family in manifest_families:
        return True

    broad_matches = {
        "runtime.capabilities": {"runtime"},
        "runtime.pipeline": {"runtime"},
        "system.memory": {"system.resources", "system"},
        "system.cpu": {"system.resources", "system"},
        "system.disk": {"system.resources", "system"},
        "system.uptime": {"system.resources", "system"},
        "system.environment": {"system.resources", "system"},
        "filesystem.file": {"filesystem"},
        "filesystem.directory": {"filesystem"},
        "table.csv": {"table"},
        "git.repository": {"git"},
        "sql.database": {"sql"},
        "sql.patient": {"sql", "sql.database"},
        "dicom.study": {"dicom"},
    }
    if task_family in broad_matches and manifest_families & broad_matches[task_family]:
        return True

    task_domain = canonical_domain(task_family)
    manifest_domains = {
        domain
        for domain in (canonical_domain(value) for value in manifest_object_types)
        if domain is not None
    }
    if task_domain and task_domain in manifest_domains:
        if task_domain == "system" and manifest_families & {"system.resources", "system"}:
            return True
        if task_domain in {"filesystem", "git", "table", "sql", "dicom"}:
            return True

    context = {canonical_domain(domain) for domain in (context_domains or []) if canonical_domain(domain)}
    if task_domain and task_domain in context and task_domain in manifest_domains:
        return True
    return False


def semantic_verbs_compatible(task_verb: str, manifest_verbs: list[str]) -> bool:
    """Return whether one task verb is semantically compatible with manifest verbs."""

    task = canonical_semantic_verb(task_verb)
    verbs = {canonical_semantic_verb(verb) for verb in manifest_verbs}
    if task in verbs:
        return True
    if task == "analyze" and verbs & {"read", "analyze", "summarize"}:
        return True
    if task == "summarize" and verbs & {"summarize", "analyze", "read"}:
        return True
    if task == "render" and verbs & {"render"}:
        return True
    if task in {"delete", "update", "create"}:
        return False
    return False


def has_hard_cross_domain_conflict(
    task_domain,
    manifest_domain,
    task_object_type,
    manifest_object_types,
    context_domains,
) -> bool:
    """Return whether task and capability live in a fundamentally unsafe mismatch."""

    task_dom = canonical_domain(task_domain) or object_domain(task_object_type)
    manifest_dom = canonical_domain(manifest_domain)
    task_family = canonical_object_family(task_object_type)
    manifest_families = {
        family
        for family in (canonical_object_family(value) for value in (manifest_object_types or []))
        if family is not None
    }
    context = {
        domain
        for domain in (canonical_domain(value) for value in (context_domains or []))
        if domain is not None
    }

    conflict_pairs = {
        ("system.memory", "filesystem"),
        ("system.cpu", "filesystem"),
        ("system.disk", "shell"),
        ("sql.patient", "shell"),
        ("table.csv", "shell"),
        ("runtime.capabilities", "filesystem"),
    }
    if task_family and manifest_dom and (task_family, manifest_dom) in conflict_pairs:
        return True

    if task_family == "dicom.study" and manifest_dom == "filesystem":
        # Only allow filesystem when the prompt is clearly about local DICOM files by path.
        if "filesystem" not in context and "dicom" in context:
            return True

    if task_dom == "runtime" and manifest_dom != "runtime":
        return True
    if task_dom == "system" and manifest_dom == "filesystem":
        return True
    if task_dom == "sql" and manifest_dom == "shell":
        return True
    return False


def domains_compatible(
    task_domain: Optional[str],
    manifest_domain: Optional[str],
    likely_domains: Optional[list[str]] = None,
    task_object_type: Optional[str] = None,
    manifest_object_types: Optional[list[str]] = None,
) -> bool:
    """Return whether task and manifest domains are canonically compatible."""

    task_dom = canonical_domain(task_domain)
    manifest_dom = canonical_domain(manifest_domain)
    likely = {
        canonical_domain(value)
        for value in (likely_domains or [])
        if canonical_domain(value) is not None
    }
    task_object_domain = object_domain(task_object_type)
    manifest_object_domains = {
        object_domain(value)
        for value in (manifest_object_types or [])
        if object_domain(value) is not None
    }

    if task_dom and manifest_dom and task_dom == manifest_dom:
        return True
    if manifest_dom and manifest_dom in likely:
        return True
    if task_object_domain and manifest_dom and task_object_domain == manifest_dom:
        return True
    if task_dom and task_dom in manifest_object_domains:
        return True
    if object_types_compatible(task_object_type, manifest_object_types or [], list(likely)):
        if not has_hard_cross_domain_conflict(
            task_dom,
            manifest_dom,
            task_object_type,
            manifest_object_types or [],
            list(likely),
        ):
            return True
    return False


__all__ = [
    "canonical_domain",
    "canonical_object_family",
    "canonical_semantic_verb",
    "domains_compatible",
    "has_hard_cross_domain_conflict",
    "known_object_families",
    "match_object_family_from_text",
    "normalize_token",
    "object_domain",
    "object_types_compatible",
    "semantic_verbs_compatible",
]
