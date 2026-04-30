"""OpenFABRIC Runtime Module: aor_runtime.runtime.auto_artifact

Purpose:
    Materialize large list/table outputs to safe local artifacts instead of streaming rows into chat.

Responsibilities:
    Select collection candidates, write CSV/TXT artifacts, and replace final content with count plus file link.

Data flow / Interfaces:
    Consumes StepLog history, goal output contracts, normalized output envelopes, and filesystem write policy.

Boundaries:
    Uses presentation row count for thresholds and keeps generated paths inside configured workspace roots.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import StepLog
from aor_runtime.runtime.output_envelope import normalize_tool_output
from aor_runtime.runtime.output_shape import EXPORT_GOAL_RE, infer_goal_output_contract
from aor_runtime.tools.filesystem import fs_write
from aor_runtime.tools.text_format import format_data


EXPLICIT_TXT_RE = re.compile(r"\b[\w.-]+\.txt\b", re.IGNORECASE)
LIST_TABLE_GOAL_RE = re.compile(
    r"\b(?:list|show|display|return|give\s+me|all|patients|jobs|nodes|files|rows|records|tables|matches|entries)\b",
    re.IGNORECASE,
)
SUMMARY_STATUS_GOAL_RE = re.compile(r"\b(?:status|summary|overview|health|availability|utilization|usage)\b", re.IGNORECASE)


@dataclass(frozen=True)
class AutoArtifact:
    """Represent auto artifact within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by AutoArtifact.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.auto_artifact.AutoArtifact and related tests.
    """
    path: str
    filename: str
    rows_written: int
    presentation_count: int
    source_count: int | None
    format: str
    source_tool: str
    source_field: str
    truncated: bool = False
    bytes_written: int = 0
    query_used: str | None = None

    def metadata(self) -> dict[str, Any]:
        """Metadata for AutoArtifact instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifact.metadata calls and related tests.
        """
        return {
            "auto_artifact": True,
            "path": self.path,
            "filename": self.filename,
            "rows_written": self.rows_written,
            "presentation_count": self.presentation_count,
            "format": self.format,
            "source_tool": self.source_tool,
            "source_field": self.source_field,
            "truncated": self.truncated,
            "bytes_written": self.bytes_written,
            **({"query_used": self.query_used} if self.query_used else {}),
            **({"source_count": self.source_count} if self.source_count is not None and self.source_count != self.presentation_count else {}),
        }


@dataclass(frozen=True)
class AutoArtifactResult:
    """Represent auto artifact result within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by AutoArtifactResult.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.auto_artifact.AutoArtifactResult and related tests.
    """
    applied: bool
    final_output: dict[str, Any]
    artifact: AutoArtifact | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _CollectionCandidate:
    """Represent collection candidate within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by _CollectionCandidate.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.auto_artifact._CollectionCandidate and related tests.
    """
    log: StepLog
    field: str
    values: list[Any]
    presentation_count: int
    source_count: int | None
    truncated: bool


class AutoArtifactMaterializer:
    """Represent auto artifact materializer within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by AutoArtifactMaterializer.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.auto_artifact.AutoArtifactMaterializer and related tests.
    """
    def __init__(self, settings: Settings) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives settings for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer.__init__ calls and related tests.
        """
        self.settings = settings

    def maybe_materialize(
        self,
        *,
        goal: str,
        history: list[StepLog],
        final_output: dict[str, Any] | None,
    ) -> AutoArtifactResult:
        """Maybe materialize for AutoArtifactMaterializer instances.

        Inputs:
            Receives goal, history, final_output for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer.maybe_materialize calls and related tests.
        """
        output = dict(final_output or {})
        metadata = dict(output.get("metadata") or {})
        if not self.settings.auto_artifacts_enabled:
            return AutoArtifactResult(False, output, reason="disabled")
        threshold = int(self.settings.auto_artifact_row_threshold)
        if threshold <= 0:
            return AutoArtifactResult(False, output, reason="threshold_disabled")
        if self._has_explicit_file_artifact(goal, history):
            return AutoArtifactResult(False, output, reason="explicit_file_artifact")
        contract = infer_goal_output_contract(goal)
        if contract.kind in {"scalar", "file", "status"}:
            return AutoArtifactResult(False, output, reason=f"contract_{contract.kind}")
        if not self._looks_like_list_or_table_goal(goal):
            return AutoArtifactResult(False, output, reason="not_list_or_table_goal")

        candidate = self._find_candidate(history)
        if candidate is None:
            return AutoArtifactResult(False, output, reason="no_collection")
        if candidate.presentation_count <= threshold:
            return AutoArtifactResult(False, output, reason="below_threshold", metadata={"row_count": candidate.presentation_count})

        artifact = self._write_candidate(goal=goal, candidate=candidate)
        output["content"] = self._render_artifact_markdown(artifact)
        output["artifacts"] = list(dict.fromkeys([*list(output.get("artifacts") or []), artifact.path]))
        output["metadata"] = {
            **metadata,
            **artifact.metadata(),
            "goal": metadata.get("goal") or goal,
        }
        return AutoArtifactResult(True, output, artifact=artifact, metadata=artifact.metadata())

    def _has_explicit_file_artifact(self, goal: str, history: list[StepLog]) -> bool:
        """Handle the internal has explicit file artifact helper path for this module.

        Inputs:
            Receives goal, history for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer._has_explicit_file_artifact calls and related tests.
        """
        if EXPORT_GOAL_RE.search(str(goal or "")):
            return True
        return any(item.success and item.step.action == "fs.write" for item in history)

    def _looks_like_list_or_table_goal(self, goal: str) -> bool:
        """Handle the internal looks like list or table goal helper path for this module.

        Inputs:
            Receives goal for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer._looks_like_list_or_table_goal calls and related tests.
        """
        text = str(goal or "")
        if not LIST_TABLE_GOAL_RE.search(text):
            return False
        if SUMMARY_STATUS_GOAL_RE.search(text) and not re.search(
            r"\b(?:list|all|table|rows|records|jobs|nodes|patients|files|matches|entries)\b",
            text,
            re.IGNORECASE,
        ):
            return False
        return True

    def _find_candidate(self, history: list[StepLog]) -> _CollectionCandidate | None:
        """Handle the internal find candidate helper path for this module.

        Inputs:
            Receives history for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer._find_candidate calls and related tests.
        """
        for item in reversed(history):
            if not item.success or item.step.action in {"runtime.return", "text.format", "fs.write"}:
                continue
            envelope = normalize_tool_output(item.step.action, item.result)
            if envelope.kind != "table" or not envelope.rows:
                continue
            return _CollectionCandidate(
                log=item,
                field=envelope.source_field or "rows",
                values=list(envelope.rows),
                presentation_count=envelope.presentation_count,
                source_count=envelope.source_count,
                truncated=envelope.truncated,
            )
        return None

    def _write_candidate(self, *, goal: str, candidate: _CollectionCandidate) -> AutoArtifact:
        """Handle the internal write candidate helper path for this module.

        Inputs:
            Receives goal, candidate for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer._write_candidate calls and related tests.
        """
        output_format = "csv"
        extension = "txt" if EXPLICIT_TXT_RE.search(str(goal or "")) else "csv"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"aor_export_{timestamp}_{secrets.token_hex(4)}.{extension}"
        path = self._safe_artifact_path(filename)
        query_used = (
            str(candidate.log.result.get("sql_final") or candidate.log.result.get("sql_normalized") or candidate.log.step.args.get("query") or "")
            if candidate.log.step.action == "sql.query"
            else ""
        )
        formatted = format_data(candidate.values, output_format, query_used=query_used, output_path=str(path))
        write_result = fs_write(self.settings, str(path), str(formatted.get("content") or ""))
        return AutoArtifact(
            path=str(write_result.get("path") or path),
            filename=filename,
            rows_written=len(candidate.values),
            presentation_count=candidate.presentation_count,
            source_count=candidate.source_count,
            format=output_format,
            source_tool=candidate.log.step.action,
            source_field=candidate.field,
            truncated=candidate.truncated,
            bytes_written=int(write_result.get("bytes_written") or 0),
            query_used=query_used or None,
        )

    def _safe_artifact_path(self, filename: str) -> Path:
        """Handle the internal safe artifact path helper path for this module.

        Inputs:
            Receives filename for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer._safe_artifact_path calls and related tests.
        """
        raw_dir = str(self.settings.auto_artifact_dir or "outputs").strip() or "outputs"
        root = self.settings.workspace_root.resolve()
        candidate_dir = Path(raw_dir).expanduser()
        if candidate_dir.is_absolute():
            resolved_dir = candidate_dir.resolve()
        else:
            resolved_dir = (root / candidate_dir).resolve()
        try:
            resolved_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Auto-artifact directory must stay inside the workspace root: {raw_dir}") from exc
        if ".." in Path(raw_dir).parts:
            raise ValueError(f"Auto-artifact directory must not contain path traversal: {raw_dir}")
        return resolved_dir / filename

    def _render_artifact_markdown(self, artifact: AutoArtifact) -> str:
        """Handle the internal render artifact markdown helper path for this module.

        Inputs:
            Receives artifact for this AutoArtifactMaterializer method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through AutoArtifactMaterializer._render_artifact_markdown calls and related tests.
        """
        file_link = f"[{artifact.filename}]({artifact.path})"
        lines = [
            "## Result",
            "",
            f"Rows written: **{artifact.rows_written:,}**",
            "",
            f"Output file: {file_link}",
            "",
            f"Source: **{_artifact_source_label(artifact.source_tool, artifact.source_field)}**",
            "",
            f"The result was saved automatically because it contains more than {self.settings.auto_artifact_row_threshold:,} display rows.",
        ]
        if artifact.source_count is not None and artifact.source_count != artifact.presentation_count:
            lines.append(f"The source contained **{artifact.source_count:,}** records used to compute these display rows.")
        if artifact.truncated:
            lines.append(
                f"The source reported **{artifact.presentation_count:,}** display rows, but **{artifact.rows_written:,}** returned rows were available to write."
            )
        if artifact.query_used:
            lines.extend(["", "## Query Used", "", "```sql", artifact.query_used, "```"])
        return "\n".join(lines).strip()


def _artifact_source_label(source_tool: str, source_field: str) -> str:
    """Handle the internal artifact source label helper path for this module.

    Inputs:
        Receives source_tool, source_field for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.auto_artifact._artifact_source_label.
    """
    tool = str(source_tool or "").strip()
    field = str(source_field or "").strip()
    if tool == "shell.exec" and field == "stdout":
        return "system shell output"
    if tool == "slurm.queue" and field == "grouped":
        return "SLURM queue grouped counts"
    if tool == "slurm.queue" and field == "jobs":
        return "SLURM queue jobs"
    if tool == "slurm.accounting" and field == "grouped":
        return "SLURM accounting grouped counts"
    if tool == "slurm.accounting" and field == "jobs":
        return "SLURM accounting jobs"
    if tool == "slurm.nodes" and field == "nodes":
        return "SLURM nodes"
    if tool == "slurm.partitions" and field == "partitions":
        return "SLURM partitions"
    if tool == "sql.query" and field == "rows":
        return "SQL query rows"
    if tool.startswith("fs.") and field in {"matches", "entries"}:
        return "filesystem results"
    return f"{tool} {field}".strip()
