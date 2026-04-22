from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from aor_runtime.core.contracts import AgentSession, RunEvent
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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    spec_name TEXT NOT NULL,
                    spec_path TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    compiled_spec_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    history_json TEXT NOT NULL,
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

    def create_session(self, session: AgentSession) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(
                    session_id, spec_name, spec_path, goal, status, trigger_type,
                    input_json, compiled_spec_json, state_json, history_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.spec_name,
                    session.spec_path,
                    session.goal,
                    session.status,
                    session.current_trigger,
                    json.dumps(ensure_jsonable(session.input), default=str),
                    json.dumps(ensure_jsonable(session.compiled_spec), default=str),
                    json.dumps(ensure_jsonable(session.state), default=str),
                    json.dumps(ensure_jsonable(session.history), default=str),
                    session.created_at,
                    session.updated_at,
                ),
            )

    def update_session(self, session: AgentSession) -> None:
        session.updated_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = ?, trigger_type = ?, state_json = ?, history_json = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    session.status,
                    session.current_trigger,
                    json.dumps(ensure_jsonable(session.state), default=str),
                    json.dumps(ensure_jsonable(session.history), default=str),
                    session.updated_at,
                    session.id,
                ),
            )

    def get_session(self, session_id: str) -> AgentSession | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return AgentSession(
            id=row["session_id"],
            spec_name=row["spec_name"],
            spec_path=row["spec_path"],
            goal=row["goal"],
            input=json.loads(row["input_json"]),
            compiled_spec=json.loads(row["compiled_spec_json"]),
            state=json.loads(row["state_json"]),
            history=json.loads(row["history_json"]),
            status=row["status"],
            current_trigger=row["trigger_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_sessions(self, limit: int = 50) -> list[AgentSession]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [
            AgentSession(
                id=row["session_id"],
                spec_name=row["spec_name"],
                spec_path=row["spec_path"],
                goal=row["goal"],
                input=json.loads(row["input_json"]),
                compiled_spec=json.loads(row["compiled_spec_json"]),
                state=json.loads(row["state_json"]),
                history=json.loads(row["history_json"]),
                status=row["status"],
                current_trigger=row["trigger_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def append_event(self, *, session_id: str, node_name: str, event_type: str, payload: dict[str, Any]) -> None:
        event = RunEvent(run_id=session_id, node=node_name, event_type=event_type, payload=ensure_jsonable(payload))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events(run_id, node_name, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (event.run_id, event.node, event.event_type, json.dumps(event.payload, default=str), event.created_at),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (event.created_at, session_id))

    def save_snapshot(self, *, session_id: str, node_name: str, state: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots(run_id, node_name, state_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, node_name, json.dumps(ensure_jsonable(state), default=str), utc_now_iso()),
            )

    def get_latest_snapshot(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM snapshots WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    def get_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM events WHERE run_id = ? ORDER BY id ASC", (session_id,)).fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["run_id"],
                "node_name": row["node_name"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # Backward-compatible helpers for older run-centric callers.
    def get_run(self, run_id: str) -> AgentSession | None:
        return self.get_session(run_id)

    def list_runs(self, limit: int = 50) -> list[AgentSession]:
        return self.list_sessions(limit=limit)
