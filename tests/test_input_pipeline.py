from __future__ import annotations

from agent_runtime.core.types import TaskFrame, UserRequest
from agent_runtime.input_pipeline.orchestrator import InputPipelineOrchestrator


def test_parse_fake_task_frame() -> None:
    orchestrator = InputPipelineOrchestrator()

    frame = orchestrator.parse_frame(
        {
            "description": "list files",
            "semantic_verb": "read",
            "object_type": "filesystem",
            "intent_confidence": 0.9,
            "constraints": {
                "capability_id": "filesystem",
                "operation_id": "read",
                "arguments": {"path": "."},
            },
        }
    )

    assert isinstance(frame, TaskFrame)
    assert frame.semantic_verb == "read"
    assert frame.object_type == "filesystem"


def test_infer_frame_from_request() -> None:
    orchestrator = InputPipelineOrchestrator()

    frame = orchestrator.infer_frame(UserRequest(raw_prompt="list repository files"))

    assert frame.semantic_verb == "read"
    assert frame.object_type == "filesystem"
    assert frame.raw_evidence == "list repository files"
