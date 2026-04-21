import json
import os
from pathlib import Path
from typing import Any

from .run_inspector import (
    build_persisted_workflow_graph,
    build_run_inspection,
    build_run_summary,
    render_workflow_graph_mermaid,
)


RUN_STORE_SCHEMA_VERSION = "phase1"


class RunStore:
    DEFAULT_BASE_DIR = "artifacts/runtime_runs"

    def __init__(self, base_dir: str | None = None):
        raw_base_dir = (
            base_dir
            or os.getenv("OPENFABRIC_RUN_STORE_DIR", "").strip()
            or self.DEFAULT_BASE_DIR
        )
        self.base_dir = Path(raw_base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state.json"

    def timeline_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "timeline.jsonl"

    def summary_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "summary.json"

    def graph_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "graph.json"

    def graph_mermaid_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "graph.mmd"

    def exists(self, run_id: str) -> bool:
        return self.state_path(run_id).exists()

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self.state_path(run_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None

    def save(self, state: dict[str, Any], *, stage: str) -> str:
        run_id = str(state.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("Run state is missing run_id.")

        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.state_path(run_id)
        tmp_path = state_path.with_suffix(".tmp")

        payload = dict(state)
        payload.setdefault("schema_version", RUN_STORE_SCHEMA_VERSION)

        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        tmp_path.replace(state_path)

        timeline_entry = {
            "schema_version": RUN_STORE_SCHEMA_VERSION,
            "run_id": run_id,
            "stage": stage,
            "status": payload.get("status"),
            "updated_at": payload.get("updated_at"),
            "current_attempt_index": payload.get("current_attempt_index"),
            "selected_attempt_index": payload.get("selected_attempt_index"),
            "terminal_event": payload.get("terminal_event"),
        }
        with self.timeline_path(run_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(timeline_entry, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

        self._persist_derived_artifacts(payload)
        return str(state_path)

    def load_timeline(self, run_id: str) -> list[dict[str, Any]]:
        path = self.timeline_path(run_id)
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
        return entries

    def load_summary(self, run_id: str) -> dict[str, Any] | None:
        return self._load_json_dict(self.summary_path(run_id))

    def load_graph(self, run_id: str) -> dict[str, Any] | None:
        return self._load_json_dict(self.graph_path(run_id))

    def load_graph_mermaid(self, run_id: str) -> str | None:
        path = self.graph_mermaid_path(run_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_runs(self, *, limit: int | None = 20, status: str | None = None) -> list[dict[str, Any]]:
        target_status = str(status or "").strip()
        summaries: list[dict[str, Any]] = []
        for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            summary = self.load_summary(run_dir.name)
            if not isinstance(summary, dict):
                session = self.load(run_dir.name)
                if not isinstance(session, dict):
                    continue
                timeline = self.load_timeline(run_dir.name)
                graph = self.load_graph(run_dir.name)
                if not isinstance(graph, dict):
                    graph = build_persisted_workflow_graph(session)
                summary = build_run_summary(session, timeline=timeline, graph=graph)
            if target_status and str(summary.get("status") or "").strip() != target_status:
                continue
            summaries.append(summary)

        summaries.sort(
            key=lambda item: (
                str(item.get("updated_at") or item.get("created_at") or ""),
                str(item.get("run_id") or ""),
            ),
            reverse=True,
        )
        if limit is None:
            return summaries
        return summaries[: max(0, int(limit))]

    def inspect(self, run_id: str, *, include_timeline: bool = True) -> dict[str, Any] | None:
        session = self.load(run_id)
        if not isinstance(session, dict):
            return None
        timeline = self.load_timeline(run_id) if include_timeline else []
        return build_run_inspection(session, timeline=timeline)

    def _load_json_dict(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        tmp_path.replace(path)

    def _write_text(self, path: Path, payload: str) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            if payload and not payload.endswith("\n"):
                handle.write("\n")
        tmp_path.replace(path)

    def _persist_derived_artifacts(self, state: dict[str, Any]) -> None:
        try:
            run_id = str(state.get("run_id") or "").strip()
            if not run_id:
                return
            timeline = self.load_timeline(run_id)
            graph = build_persisted_workflow_graph(state)
            summary = build_run_summary(state, timeline=timeline, graph=graph)
            self._write_json(self.summary_path(run_id), summary)
            self._write_json(self.graph_path(run_id), graph)
            self._write_text(self.graph_mermaid_path(run_id), render_workflow_graph_mermaid(graph))
        except Exception:
            # Derived inspection artifacts must not block persistence of the run state itself.
            return
