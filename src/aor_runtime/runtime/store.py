"""OpenFABRIC Runtime Module: aor_runtime.runtime.store

Purpose:
    Persist sessions, events, and snapshots in SQLite.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

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
    """Utc now iso for the surrounding runtime workflow.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.store.utc_now_iso.
    """
    return datetime.now(timezone.utc).isoformat()


class SQLiteRunStore:
    """Represent s q lite run store within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SQLiteRunStore.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.store.SQLiteRunStore and related tests.
    """
    def __init__(self, db_path: str | Path) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives db_path for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.__init__ calls and related tests.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Handle the internal connect helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore._connect calls and related tests.
        """
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Handle the internal initialize helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore._initialize calls and related tests.
        """
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
        """Create session for SQLiteRunStore instances.

        Inputs:
            Receives session for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.create_session calls and related tests.
        """
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
        """Update session for SQLiteRunStore instances.

        Inputs:
            Receives session for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.update_session calls and related tests.
        """
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
        """Get session for SQLiteRunStore instances.

        Inputs:
            Receives session_id for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.get_session calls and related tests.
        """
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
        """List sessions for SQLiteRunStore instances.

        Inputs:
            Receives limit for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.list_sessions calls and related tests.
        """
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
        """Append event for SQLiteRunStore instances.

        Inputs:
            Receives session_id, node_name, event_type, payload for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.append_event calls and related tests.
        """
        event = RunEvent(run_id=session_id, node=node_name, event_type=event_type, payload=ensure_jsonable(payload))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events(run_id, node_name, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (event.run_id, event.node, event.event_type, json.dumps(event.payload, default=str), event.created_at),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (event.created_at, session_id))

    def save_snapshot(self, *, session_id: str, node_name: str, state: dict[str, Any]) -> None:
        """Save snapshot for SQLiteRunStore instances.

        Inputs:
            Receives session_id, node_name, state for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.save_snapshot calls and related tests.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots(run_id, node_name, state_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, node_name, json.dumps(ensure_jsonable(state), default=str), utc_now_iso()),
            )

    def get_latest_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """Get latest snapshot for SQLiteRunStore instances.

        Inputs:
            Receives session_id for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.get_latest_snapshot calls and related tests.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM snapshots WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    def get_events(self, session_id: str) -> list[dict[str, Any]]:
        """Get events for SQLiteRunStore instances.

        Inputs:
            Receives session_id for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.get_events calls and related tests.
        """
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

    def get_events_after(self, session_id: str, after_id: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Get events after for SQLiteRunStore instances.

        Inputs:
            Receives session_id, after_id, limit for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.get_events_after calls and related tests.
        """
        query = "SELECT * FROM events WHERE run_id = ?"
        params: list[Any] = [session_id]
        if after_id is not None:
            query += " AND id > ?"
            params.append(int(after_id))
        query += " ORDER BY id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
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
        """Get run for SQLiteRunStore instances.

        Inputs:
            Receives run_id for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.get_run calls and related tests.
        """
        return self.get_session(run_id)

    def list_runs(self, limit: int = 50) -> list[AgentSession]:
        """List runs for SQLiteRunStore instances.

        Inputs:
            Receives limit for this SQLiteRunStore method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SQLiteRunStore.list_runs calls and related tests.
        """
        return self.list_sessions(limit=limit)
