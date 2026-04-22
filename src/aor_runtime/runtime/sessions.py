from __future__ import annotations

from aor_runtime.core.contracts import AgentSession
from aor_runtime.dsl.models import CompiledRuntimeSpec
from typing import cast

from aor_runtime.runtime.state import RuntimeState, initial_runtime_state
from aor_runtime.runtime.store import SQLiteRunStore


class SessionManager:
    def __init__(self, store: SQLiteRunStore) -> None:
        self.store = store

    def create_session(
        self,
        *,
        session_id: str,
        spec_path: str,
        compiled: CompiledRuntimeSpec,
        input_payload: dict,
        trigger: str = "manual",
    ) -> AgentSession:
        state = initial_runtime_state(
            session_id=session_id,
            spec_name=compiled.name,
            spec_path=spec_path,
            input_payload=input_payload,
            compiled_spec=compiled.model_dump(),
            trigger=trigger,
        )
        session = AgentSession(
            id=session_id,
            spec_name=compiled.name,
            spec_path=spec_path,
            goal=state["goal"],
            input=input_payload,
            compiled_spec=compiled.model_dump(),
            state=state,
            history=[],
            status=state["status"],
            current_trigger=trigger,
        )
        self.store.create_session(session)
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self.store.get_session(session_id)

    def list_sessions(self, limit: int = 50) -> list[AgentSession]:
        return self.store.list_sessions(limit=limit)

    def persist_session(self, session: AgentSession, *, node_name: str) -> None:
        session.status = str(session.state.get("status", session.status))
        session.history = list(session.state.get("history", session.history))
        self.store.update_session(session)
        self.store.save_snapshot(session_id=session.id, node_name=node_name, state=session.state)

    @staticmethod
    def state(session: AgentSession) -> RuntimeState:
        return cast(RuntimeState, session.state)
