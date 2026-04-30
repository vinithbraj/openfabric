"""OpenFABRIC Runtime Module: aor_runtime.runtime.sessions

Purpose:
    Create and manipulate persisted runtime session records.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

from aor_runtime.core.contracts import AgentSession
from aor_runtime.dsl.models import CompiledRuntimeSpec
from typing import cast

from aor_runtime.runtime.state import RuntimeState, initial_runtime_state
from aor_runtime.runtime.store import SQLiteRunStore


class SessionManager:
    """Represent session manager within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by SessionManager.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.sessions.SessionManager and related tests.
    """
    def __init__(self, store: SQLiteRunStore) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives store for this SessionManager method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through SessionManager.__init__ calls and related tests.
        """
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
        """Create session for SessionManager instances.

        Inputs:
            Receives session_id, spec_path, compiled, input_payload, trigger for this SessionManager method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SessionManager.create_session calls and related tests.
        """
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
        """Get session for SessionManager instances.

        Inputs:
            Receives session_id for this SessionManager method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SessionManager.get_session calls and related tests.
        """
        return self.store.get_session(session_id)

    def list_sessions(self, limit: int = 50) -> list[AgentSession]:
        """List sessions for SessionManager instances.

        Inputs:
            Receives limit for this SessionManager method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SessionManager.list_sessions calls and related tests.
        """
        return self.store.list_sessions(limit=limit)

    def persist_session(self, session: AgentSession, *, node_name: str) -> None:
        """Persist session for SessionManager instances.

        Inputs:
            Receives session, node_name for this SessionManager method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through SessionManager.persist_session calls and related tests.
        """
        session.status = str(session.state.get("status", session.status))
        session.history = list(session.state.get("history", session.history))
        self.store.update_session(session)
        self.store.save_snapshot(session_id=session.id, node_name=node_name, state=session.state)

    @staticmethod
    def state(session: AgentSession) -> RuntimeState:
        """State for SessionManager instances.

        Inputs:
            Receives session for this SessionManager method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through SessionManager.state calls and related tests.
        """
        return cast(RuntimeState, session.state)
