from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from aor_runtime.core.contracts import RunEvent, RunSummary
from aor_runtime.core.utils import ensure_jsonable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRunStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    spec_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    final_state_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_run(self, *, run_id: str, spec_name: str, input_payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, spec_name, status, input_json, final_state_json, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    spec_name,
                    "running",
                    json.dumps(ensure_jsonable(input_payload), default=str),
                    None,
                    json.dumps(ensure_jsonable(metadata or {}), default=str),
                    now,
                    now,
                ),
            )

    def append_event(self, *, run_id: str, node_name: str, event_type: str, payload: dict[str, Any]) -> None:
        event = RunEvent(run_id=run_id, node=node_name, event_type=event_type, payload=ensure_jsonable(payload))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events(run_id, node_name, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (event.run_id, event.node, event.event_type, json.dumps(event.payload, default=str), event.created_at),
            )
            conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (event.created_at, run_id))

    def save_snapshot(self, *, run_id: str, node_name: str, state: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots(run_id, node_name, state_json, created_at) VALUES (?, ?, ?, ?)",
                (run_id, node_name, json.dumps(ensure_jsonable(state), default=str), utc_now_iso()),
            )

    def finalize_run(self, *, run_id: str, status: str, final_state: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, final_state_json = ?, updated_at = ? WHERE run_id = ?",
                (status, json.dumps(ensure_jsonable(final_state), default=str), utc_now_iso(), run_id),
            )

    def get_run(self, run_id: str) -> RunSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunSummary(
            run_id=row["run_id"],
            spec_name=row["spec_name"],
            status=row["status"],
            input=json.loads(row["input_json"]),
            final_state=json.loads(row["final_state_json"]) if row["final_state_json"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for row in rows:
            results.append(
                RunSummary(
                    run_id=row["run_id"],
                    spec_name=row["spec_name"],
                    status=row["status"],
                    input=json.loads(row["input_json"]),
                    final_state=json.loads(row["final_state_json"]) if row["final_state_json"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return results

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM events WHERE run_id = ? ORDER BY id ASC", (run_id,)).fetchall()
        return [
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "node_name": row["node_name"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
