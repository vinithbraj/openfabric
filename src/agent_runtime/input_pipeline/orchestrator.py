"""Orchestrator for the input semantic pipeline."""

from __future__ import annotations

from typing import Any

from agent_runtime.core.types import ActionDAG, TaskFrame, UserRequest
from agent_runtime.input_pipeline.argument_extraction import ArgumentExtractor
from agent_runtime.input_pipeline.dag_builder import DAGBuilder
from agent_runtime.input_pipeline.decomposition import Decomposer
from agent_runtime.input_pipeline.domain_selection import DomainSelector
from agent_runtime.input_pipeline.validators import TaskFrameValidator
from agent_runtime.input_pipeline.verb_classification import VerbClassifier


class InputPipelineOrchestrator:
    """Coordinate placeholder semantic parsing and DAG construction."""

    def __init__(self) -> None:
        self.decomposer = Decomposer()
        self.verb_classifier = VerbClassifier()
        self.domain_selector = DomainSelector()
        self.argument_extractor = ArgumentExtractor()
        self.validator = TaskFrameValidator()
        self.dag_builder = DAGBuilder()

    def parse_frame(self, payload: dict[str, Any]) -> TaskFrame:
        """Validate a caller-supplied task frame payload."""

        frame = TaskFrame.model_validate(payload)
        self.validator.validate(frame)
        return frame

    def infer_frame(self, request: UserRequest) -> TaskFrame:
        """Create a minimal frame from a user request using placeholders."""

        decomposition = self.decomposer.decompose(request)
        prompt = decomposition.tasks[0].description if decomposition.tasks else request.raw_prompt
        verb = self.verb_classifier.classify(prompt)
        domain = self.domain_selector.select(prompt)
        return TaskFrame(
            description=prompt,
            semantic_verb=verb,
            object_type=domain,
            intent_confidence=0.5,
            constraints={
                "capability_id": domain,
                "operation_id": verb,
                "arguments": {
                    argument.name: argument.value for argument in self.argument_extractor.extract(prompt)
                },
            },
            raw_evidence=prompt,
        )

    def build_dag(self, frame: TaskFrame) -> ActionDAG:
        """Validate a frame and build an action DAG."""

        self.validator.validate(frame)
        return self.dag_builder.build(frame)
