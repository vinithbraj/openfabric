import json
import os
from pathlib import Path
from typing import Any


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

        return str(state_path)
