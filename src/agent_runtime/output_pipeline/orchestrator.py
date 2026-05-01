"""Orchestrator for the output composition pipeline."""

from __future__ import annotations

from typing import Any

from agent_runtime.core.types import ActionDAG, RenderedOutput, ResultBundle, UserRequest
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)
from agent_runtime.observability import (
    EVENT_OUTPUT_SHAPE_SELECTED,
    EVENT_RENDERING_COMPLETED,
    ObservabilityContext,
    STAGE_OUTPUT_PLANNING,
    STAGE_RENDERING,
)
from agent_runtime.output_pipeline.display_selection import (
    DisplaySelectionInput,
    DisplaySelector,
    select_display_plan,
)
from agent_runtime.output_pipeline.redaction import Redactor
from agent_runtime.output_pipeline.renderers import render_result_shape, render_display_plan
from agent_runtime.output_pipeline.result_shapes import (
    AggregateResult,
    DirectoryListingResult,
    ErrorResult,
    MultiSectionResult,
    ResultShape,
    normalize_execution_result,
)
from agent_runtime.output_pipeline.summarizer import Summarizer


def _result_store_from_request(user_request: UserRequest):
    """Resolve a result store from request context when available."""

    for context in (
        user_request.safety_context,
        user_request.session_context,
        user_request.user_context,
    ):
        store = context.get("result_store")
        if store is not None:
            return store
    return None


def _observability_from_request(user_request: UserRequest) -> ObservabilityContext | None:
    """Resolve a request-scoped observability context when present."""

    for context in (
        user_request.safety_context,
        user_request.session_context,
        user_request.user_context,
    ):
        observability = context.get("observability")
        if isinstance(observability, ObservabilityContext):
            return observability
    return None


def _allow_full_output_access(user_request: UserRequest) -> bool:
    """Return whether full result dereferencing is permitted."""

    for context in (
        user_request.safety_context,
        user_request.session_context,
        user_request.user_context,
    ):
        if bool(context.get("allow_full_output_access", False)):
            return True
    return False


def _summarize_dag(dag: ActionDAG) -> dict[str, Any]:
    """Build a compact DAG summary for display planning."""

    return {
        "dag_id": dag.dag_id,
        "node_count": len(dag.nodes),
        "requires_confirmation": dag.requires_confirmation,
        "nodes": [
            {
                "node_id": node.id,
                "task_id": node.task_id,
                "semantic_verb": node.semantic_verb,
                "capability_id": node.capability_id,
                "operation_id": node.operation_id,
            }
            for node in dag.nodes
        ],
    }


def _summarize_results(result_bundle: ResultBundle) -> dict[str, Any]:
    """Build a compact result summary for display planning."""

    status_counts: dict[str, int] = {}
    for result in result_bundle.results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    return {
        "dag_id": result_bundle.dag_id,
        "status": result_bundle.status,
        "safe_summary": result_bundle.safe_summary,
        "status_counts": status_counts,
        "result_count": len(result_bundle.results),
    }


def _build_safe_previews(result_bundle: ResultBundle, redactor: Redactor) -> list[dict[str, Any]]:
    """Build safe previews for LLM display selection."""

    safe_previews: list[dict[str, Any]] = []
    for result in result_bundle.results:
        safe_previews.append(
            {
                "node_id": result.node_id,
                "status": result.status,
                "data_ref": result.data_ref.ref_id if result.data_ref is not None else None,
                "preview": redactor.redact(result.data_preview) if result.data_preview is not None else None,
                "shape_type": normalize_execution_result(result).shape_type,
                "error": result.error,
            }
        )
    return safe_previews


def _source_lookup(
    result_bundle: ResultBundle,
    safe_previews: list[dict[str, Any]],
    user_request: UserRequest,
) -> dict[str, dict[str, Any]]:
    """Build source lookup entries from safe previews and optional full data."""

    lookup: dict[str, dict[str, Any]] = {}
    result_store = _result_store_from_request(user_request)
    allow_full = _allow_full_output_access(user_request)

    for preview in safe_previews:
        payload = preview.get("preview")
        data_ref = preview.get("data_ref")
        if allow_full and data_ref and result_store is not None:
            try:
                payload = result_store.get(str(data_ref))
            except Exception:
                payload = preview.get("preview")
        record = {
            "node_id": preview.get("node_id"),
            "data_ref": data_ref,
            "payload": payload,
            "shape": normalize_execution_result(
                next(
                    result
                    for result in result_bundle.results
                    if result.node_id == preview.get("node_id")
                ),
                result_store if allow_full else None,
            ),
            "status": preview.get("status"),
            "error": preview.get("error"),
        }
        node_id = preview.get("node_id")
        if node_id is not None:
            lookup[f"node:{node_id}"] = record
        if data_ref is not None:
            lookup[f"data:{data_ref}"] = record
    return lookup


def _safe_error_text(message: str | None) -> str:
    """Strip traceback-like content from user-visible error text."""

    text = str(message or "").strip()
    if not text:
        return "Execution failed."
    if "Traceback (most recent call last)" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if ":" in line and not line.startswith("Traceback"):
                return line
        return "Execution failed."
    return text.splitlines()[0].strip()


def _safe_summary_text(message: str | None) -> str:
    """Remove traceback text from summaries while preserving normal prose."""

    text = str(message or "").strip()
    if not text:
        return ""
    if "Traceback (most recent call last)" in text:
        return _safe_error_text(text)
    return text


def _render_error_bundle(result_bundle: ResultBundle) -> str:
    """Render an error bundle deterministically."""

    lines = ["Execution failed."]
    blocked_reasons = result_bundle.metadata.get("blocked_reasons", [])
    if blocked_reasons:
        lines.append("")
        lines.append("Blocked reasons:")
        lines.extend(f"- {reason}" for reason in blocked_reasons)
    safe_summary = _safe_summary_text(result_bundle.safe_summary)
    if safe_summary:
        lines.append("")
        lines.append(safe_summary)
    return "\n".join(lines)


def _render_partial_bundle(result_bundle: ResultBundle) -> str:
    """Render a partial bundle deterministically."""

    lines = ["Partial results"]
    safe_summary = _safe_summary_text(result_bundle.safe_summary)
    if safe_summary:
        lines.extend(["", safe_summary])

    successes = [result for result in result_bundle.results if result.status == "success"]
    skipped = [result for result in result_bundle.results if result.status == "skipped"]
    errors = [result for result in result_bundle.results if result.status == "error"]

    if successes:
        lines.extend(["", "Completed:"])
        for result in successes:
            lines.append(f"- {result.node_id}: {result.data_preview}")
    if skipped:
        lines.extend(["", "Skipped:"])
        for result in skipped:
            lines.append(f"- {result.node_id}: {result.error or 'Skipped.'}")
    if errors:
        lines.extend(["", "Errors:"])
        for result in errors:
            lines.append(f"- {result.node_id}: {_safe_error_text(result.error)}")
    return "\n".join(lines)


def _default_shape_title(shape: ResultShape) -> str | None:
    """Return a deterministic title for one normalized shape."""

    if getattr(shape, "title", None):
        return getattr(shape, "title")
    if isinstance(shape, AggregateResult):
        return shape.label or "Aggregate Result"
    if isinstance(shape, DirectoryListingResult):
        return "Directory Listing"
    return None


def _shape_with_default_title(shape: ResultShape) -> ResultShape:
    """Ensure fallback-rendered sections have stable, human-readable titles."""

    if getattr(shape, "title", None):
        return shape
    default_title = _default_shape_title(shape)
    if not default_title:
        return shape
    return shape.model_copy(update={"title": default_title})


def _fallback_section_order(prompt: str, shapes: list[ResultShape]) -> list[ResultShape]:
    """Order normalized shapes deterministically for fallback rendering."""

    lowered = str(prompt or "").lower()
    asked_to_list_files = "list" in lowered and "file" in lowered
    ordered = list(shapes)
    if asked_to_list_files:
        ordered.sort(
            key=lambda shape: (
                0
                if isinstance(shape, DirectoryListingResult)
                else 1
                if isinstance(shape, AggregateResult)
                else 2
            )
        )
    return ordered


def _fallback_render_success(
    user_request: UserRequest,
    result_bundle: ResultBundle,
) -> str:
    """Render successful bundles deterministically from normalized result shapes."""

    result_store = _result_store_from_request(user_request)
    shapes = [
        normalize_execution_result(result, result_store)
        for result in result_bundle.results
        if result.status == "success"
    ]
    if not shapes:
        return "No results available."

    ordered_shapes = [
        _shape_with_default_title(shape)
        for shape in _fallback_section_order(user_request.raw_prompt, shapes)
    ]
    if len(ordered_shapes) == 1:
        shape = ordered_shapes[0]
        return render_result_shape(shape, title=_default_shape_title(shape))

    multi = MultiSectionResult(
        node_id="multi-section",
        title=None,
        sections=ordered_shapes,
    )
    return render_result_shape(multi)


def _fallback_rendered_output(
    user_request: UserRequest,
    result_bundle: ResultBundle,
) -> RenderedOutput:
    """Build a deterministic rendered output when display planning fails."""

    content = _fallback_render_success(user_request, result_bundle)
    return RenderedOutput(
        content=content,
        display_plan=DisplaySelector().select(
            DisplaySelectionInput(
                original_prompt=user_request.raw_prompt,
                dag_summary={},
                result_summary=_summarize_results(result_bundle),
                safe_previews=[],
                available_display_types=[
                    "plain_text",
                    "markdown",
                    "table",
                    "json",
                    "code_block",
                    "multi_section",
                ],
            )
        ),
        metadata={"fallback": True},
    )


def compose_output(
    user_request: UserRequest,
    dag: ActionDAG,
    result_bundle: ResultBundle,
    llm_client,
) -> str:
    """Compose final user-facing output from a DAG and result bundle."""

    observability = _observability_from_request(user_request)
    if result_bundle.status == "error":
        if observability is not None:
            observability.stage_started(
                STAGE_RENDERING,
                "Rendering started",
                "The runtime is rendering an error result deterministically.",
            )
        content = _render_error_bundle(result_bundle)
        if observability is not None:
            observability.info(
                STAGE_RENDERING,
                EVENT_RENDERING_COMPLETED,
                "Rendering completed",
                "The runtime rendered an error result deterministically.",
                details={"content_length": len(content)},
            )
            observability.stage_completed(
                STAGE_RENDERING,
                "Rendering completed",
                "Rendering finished with an error bundle view.",
                details={"content_length": len(content)},
            )
        return content
    if result_bundle.status == "partial":
        if observability is not None:
            observability.stage_started(
                STAGE_RENDERING,
                "Rendering started",
                "The runtime is rendering partial results deterministically.",
            )
        content = _render_partial_bundle(result_bundle)
        if observability is not None:
            observability.info(
                STAGE_RENDERING,
                EVENT_RENDERING_COMPLETED,
                "Rendering completed",
                "The runtime rendered partial results deterministically.",
                details={"content_length": len(content)},
            )
            observability.stage_completed(
                STAGE_RENDERING,
                "Rendering completed",
                "Rendering finished with a partial-results view.",
                details={"content_length": len(content)},
            )
        return content

    redactor = Redactor()
    safe_previews = _build_safe_previews(result_bundle, redactor)
    if observability is not None:
        observability.stage_started(
            STAGE_OUTPUT_PLANNING,
            "Output planning started",
            "The runtime is selecting result shapes and a display plan.",
            details={"safe_preview_count": len(safe_previews)},
        )
    selection_input = DisplaySelectionInput(
        original_prompt=user_request.raw_prompt,
        dag_summary=_summarize_dag(dag),
        result_summary=_summarize_results(result_bundle),
        safe_previews=safe_previews,
        available_display_types=[
            "plain_text",
            "markdown",
            "table",
            "json",
            "code_block",
            "multi_section",
        ],
    )
    display_plan = None
    trace = user_request.safety_context.get("planning_trace")
    try:
        if observability is not None:
            result_store = _result_store_from_request(user_request)
            shape_types = [
                normalize_execution_result(result, result_store).shape_type
                for result in result_bundle.results
            ]
            observability.info(
                STAGE_OUTPUT_PLANNING,
                EVENT_OUTPUT_SHAPE_SELECTED,
                "Result shapes selected",
                "Execution results were normalized into semantic result shapes.",
                details={"shape_types": shape_types},
            )
        display_plan = select_display_plan(selection_input, llm_client)
        if isinstance(trace, PlanningTrace):
            model_name, temperature = llm_client_metadata(llm_client)
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="display_plan_selection",
                    request_id=user_request.request_id,
                    model_name=model_name,
                    llm_temperature=temperature,
                    prompt_template_id="display_plan_selection",
                    parsed_proposal=display_plan.model_dump(mode="json"),
                    selected_candidate=display_plan.model_dump(mode="json"),
                    ),
                )
        if observability is not None:
            observability.stage_completed(
                STAGE_OUTPUT_PLANNING,
                "Output planning completed",
                "A display plan was selected for rendering.",
                details={
                    "display_type": display_plan.display_type,
                    "section_count": len(display_plan.sections),
                },
            )
            observability.stage_started(
                STAGE_RENDERING,
                "Rendering started",
                "The runtime is rendering the validated display plan.",
                details={"display_type": display_plan.display_type},
            )
        lookup = _source_lookup(result_bundle, safe_previews, user_request)
        rendered = render_display_plan(display_plan, lookup)
        if observability is not None:
            observability.info(
                STAGE_RENDERING,
                EVENT_RENDERING_COMPLETED,
                "Rendering completed",
                "The final response was rendered successfully.",
                details={
                    "display_type": display_plan.display_type,
                    "content_length": len(rendered.content),
                },
            )
            observability.stage_completed(
                STAGE_RENDERING,
                "Rendering completed",
                "Rendering finished successfully.",
                details={"content_length": len(rendered.content)},
            )
        return rendered.content
    except Exception:
        if observability is not None:
            observability.warning(
                STAGE_OUTPUT_PLANNING,
                "validation.rejected",
                "Display plan rejected",
                "The runtime fell back to deterministic result-shape rendering.",
            )
        content = _fallback_render_success(user_request, result_bundle)
        if observability is not None:
            observability.stage_started(
                STAGE_RENDERING,
                "Rendering started",
                "The runtime is using deterministic result-shape rendering.",
            )
            observability.info(
                STAGE_RENDERING,
                EVENT_RENDERING_COMPLETED,
                "Rendering completed",
                "The runtime rendered the response using the deterministic fallback.",
                details={"content_length": len(content), "fallback": True},
            )
            observability.stage_completed(
                STAGE_RENDERING,
                "Rendering completed",
                "Deterministic fallback rendering finished successfully.",
                details={"content_length": len(content)},
            )
        return content


class OutputPipelineOrchestrator:
    """Select, redact, render, and summarize a result bundle."""

    def __init__(self) -> None:
        self.display_selector = DisplaySelector()
        self.redactor = Redactor()
        self.summarizer = Summarizer()

    def render(
        self,
        bundle: ResultBundle,
        user_request: UserRequest | None = None,
        dag: ActionDAG | None = None,
        llm_client=None,
    ) -> RenderedOutput:
        """Render a result bundle through the deterministic output pipeline."""

        if user_request is not None and dag is not None and llm_client is not None:
            content = compose_output(user_request, dag, bundle, llm_client)
            rendered = RenderedOutput(
                content=content,
                display_plan=self.display_selector.select(
                    DisplaySelectionInput(
                        original_prompt=user_request.raw_prompt,
                        dag_summary=_summarize_dag(dag),
                        result_summary=_summarize_results(bundle),
                        safe_previews=_build_safe_previews(bundle, self.redactor),
                        available_display_types=[
                            "plain_text",
                            "markdown",
                            "table",
                            "json",
                            "code_block",
                            "multi_section",
                        ],
                    )
                ),
                metadata={},
            )
            return self.summarizer.summarize(rendered)

        safe_previews = _build_safe_previews(bundle, self.redactor)
        selection_input = DisplaySelectionInput(
            original_prompt="",
            dag_summary={},
            result_summary=_summarize_results(bundle),
            safe_previews=safe_previews,
            available_display_types=[
                "plain_text",
                "markdown",
                "table",
                "json",
                "code_block",
                "multi_section",
            ],
        )
        display_plan = self.display_selector.select(selection_input)
        lookup = {
            f"node:{preview['node_id']}": {
                "node_id": preview["node_id"],
                "data_ref": preview.get("data_ref"),
                "payload": preview.get("preview"),
                "shape": normalize_execution_result(result),
            }
            for result, preview in zip(bundle.results, safe_previews)
        }
        try:
            rendered = render_display_plan(display_plan, lookup)
        except Exception:
            rendered = _fallback_rendered_output(
                user_request or UserRequest(raw_prompt=""),
                bundle,
            )
        return self.summarizer.summarize(rendered)
