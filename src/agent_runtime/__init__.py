"""Schema-driven intelligent agent runtime foundation.

The package is intentionally skeletal. It defines typed contracts and clean
pipeline interfaces before real planning, execution, or rendering behavior is
introduced.
"""

from agent_runtime.core.types import (
    ActionDAG,
    ActionNode,
    CapabilityRef,
    DisplayPlan,
    ExecutionResult,
    RenderedOutput,
    ResultBundle,
    TaskFrame,
    UserRequest,
)
from agent_runtime.core.orchestrator import AgentRuntime

__all__ = [
    "ActionDAG",
    "AgentRuntime",
    "ActionNode",
    "CapabilityRef",
    "DisplayPlan",
    "ExecutionResult",
    "RenderedOutput",
    "ResultBundle",
    "TaskFrame",
    "UserRequest",
]
