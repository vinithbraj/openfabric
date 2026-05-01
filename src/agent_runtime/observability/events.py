"""Structured user-visible pipeline events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.core.ids import new_id

EventLevel = Literal["debug", "info", "warning", "error"]

STAGE_REQUEST_RECEIVED = "request_received"
STAGE_PROMPT_CLASSIFICATION = "prompt_classification"
STAGE_DECOMPOSITION = "decomposition"
STAGE_VERB_ASSIGNMENT = "verb_assignment"
STAGE_CAPABILITY_SELECTION = "capability_selection"
STAGE_CAPABILITY_FIT = "capability_fit"
STAGE_DATAFLOW_PLANNING = "dataflow_planning"
STAGE_ARGUMENT_EXTRACTION = "argument_extraction"
STAGE_DAG_CONSTRUCTION = "dag_construction"
STAGE_DAG_REVIEW = "dag_review"
STAGE_SAFETY_EVALUATION = "safety_evaluation"
STAGE_EXECUTION = "execution"
STAGE_FAILURE_REPAIR = "failure_repair"
STAGE_OUTPUT_PLANNING = "output_planning"
STAGE_RENDERING = "rendering"
STAGE_COMPLETED = "completed"

EVENT_STAGE_STARTED = "stage.started"
EVENT_STAGE_COMPLETED = "stage.completed"
EVENT_STAGE_FAILED = "stage.failed"
EVENT_LLM_PROPOSAL_RECEIVED = "llm.proposal.received"
EVENT_VALIDATION_ACCEPTED = "validation.accepted"
EVENT_VALIDATION_REJECTED = "validation.rejected"
EVENT_CAPABILITY_CANDIDATE = "capability.candidate"
EVENT_CAPABILITY_SELECTED = "capability.selected"
EVENT_CAPABILITY_REJECTED = "capability.rejected"
EVENT_CAPABILITY_GAP = "capability.gap"
EVENT_DATAFLOW_REF_PROPOSED = "dataflow.ref.proposed"
EVENT_DATAFLOW_REF_ACCEPTED = "dataflow.ref.accepted"
EVENT_DATAFLOW_REF_REJECTED = "dataflow.ref.rejected"
EVENT_DATAFLOW_DERIVED_TASK_CREATED = "dataflow.derived_task.created"
EVENT_ARGUMENT_EXTRACTED = "argument.extracted"
EVENT_ARGUMENT_REJECTED = "argument.rejected"
EVENT_DAG_NODE_CREATED = "dag.node.created"
EVENT_DAG_EDGE_CREATED = "dag.edge.created"
EVENT_SAFETY_ALLOWED = "safety.allowed"
EVENT_SAFETY_BLOCKED = "safety.blocked"
EVENT_EXECUTION_NODE_STARTED = "execution.node.started"
EVENT_EXECUTION_NODE_COMPLETED = "execution.node.completed"
EVENT_EXECUTION_NODE_FAILED = "execution.node.failed"
EVENT_REPAIR_PROPOSED = "repair.proposed"
EVENT_REPAIR_ACCEPTED = "repair.accepted"
EVENT_REPAIR_REJECTED = "repair.rejected"
EVENT_OUTPUT_SHAPE_SELECTED = "output.shape.selected"
EVENT_RENDERING_COMPLETED = "rendering.completed"


class PipelineEvent(BaseModel):
    """One safe, structured event emitted during runtime processing."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    request_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    level: EventLevel
    stage: str
    event_type: str
    title: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    safe_for_user: bool = True
    debug_only: bool = False

