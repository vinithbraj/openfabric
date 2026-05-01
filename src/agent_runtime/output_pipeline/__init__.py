"""Output composition pipeline package."""

from agent_runtime.output_pipeline.display_selection import (
    DisplaySelectionInput,
    select_display_plan,
)
from agent_runtime.output_pipeline.orchestrator import (
    OutputPipelineOrchestrator,
    compose_output,
)
from agent_runtime.output_pipeline.result_shapes import (
    AggregateResult,
    CapabilityListResult,
    DirectoryListingResult,
    ErrorResult,
    FileContentResult,
    MultiSectionResult,
    ProcessListResult,
    RecordListResult,
    ResultShape,
    ScalarResult,
    TableResult,
    TextResult,
    normalize_execution_result,
)

__all__ = [
    "AggregateResult",
    "CapabilityListResult",
    "DisplaySelectionInput",
    "DirectoryListingResult",
    "ErrorResult",
    "FileContentResult",
    "MultiSectionResult",
    "OutputPipelineOrchestrator",
    "ProcessListResult",
    "RecordListResult",
    "ResultShape",
    "ScalarResult",
    "TableResult",
    "TextResult",
    "compose_output",
    "normalize_execution_result",
    "select_display_plan",
]
