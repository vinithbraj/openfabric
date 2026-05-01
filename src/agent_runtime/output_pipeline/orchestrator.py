"""Orchestrator for the output composition pipeline."""

from __future__ import annotations

from typing import Any

from agent_runtime.core.errors import ValidationError
from agent_runtime.core.types import ActionDAG, RenderedOutput, ResultBundle, UserRequest
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)
from agent_runtime.output_pipeline.display_selection import (
    DisplaySelectionInput,
    DisplaySelector,
    select_display_plan,
)
from agent_runtime.output_pipeline.redaction import Redactor
from agent_runtime.output_pipeline.renderers import render_display_plan
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
            "status": preview.get("status"),
            "error": preview.get("error"),
        }
        node_id = preview.get("node_id")
        if node_id is not None:
            lookup[f"node:{node_id}"] = record
        if data_ref is not None:
            lookup[f"data:{data_ref}"] = record
    return lookup


def _resolve_section_payload(section: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> Any:
    """Resolve one section payload from safe or full-data lookup entries."""

    source_node_id = section.get("source_node_id")
    if source_node_id is not None:
        key = f"node:{source_node_id}"
        if key not in lookup:
            raise ValidationError(f"Display plan references missing node source: {source_node_id}")
        return lookup[key]["payload"]

    source_data_ref = section.get("source_data_ref")
    if source_data_ref is not None:
        key = f"data:{source_data_ref}"
        if key not in lookup:
            raise ValidationError(f"Display plan references missing data source: {source_data_ref}")
        return lookup[key]["payload"]

    raise ValidationError("Display plan section must reference source_node_id or source_data_ref.")


def _render_error_bundle(result_bundle: ResultBundle) -> str:
    """Render an error bundle deterministically."""

    lines = ["Execution failed."]
    blocked_reasons = result_bundle.metadata.get("blocked_reasons", [])
    if blocked_reasons:
        lines.append("")
        lines.append("Blocked reasons:")
        lines.extend(f"- {reason}" for reason in blocked_reasons)
    if result_bundle.safe_summary:
        lines.append("")
        lines.append(result_bundle.safe_summary)
    return "\n".join(lines)


def _render_partial_bundle(result_bundle: ResultBundle) -> str:
    """Render a partial bundle deterministically."""

    lines = ["Partial results"]
    if result_bundle.safe_summary:
        lines.extend(["", result_bundle.safe_summary])

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
            lines.append(f"- {result.node_id}: {result.error or 'Execution failed.'}")
    return "\n".join(lines)


def compose_output(
    user_request: UserRequest,
    dag: ActionDAG,
    result_bundle: ResultBundle,
    llm_client,
) -> str:
    """Compose final user-facing output from a DAG and result bundle."""

    if result_bundle.status == "error":
        return _render_error_bundle(result_bundle)
    if result_bundle.status == "partial":
        return _render_partial_bundle(result_bundle)

    redactor = Redactor()
    safe_previews = _build_safe_previews(result_bundle, redactor)
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
    display_plan = select_display_plan(selection_input, llm_client)
    trace = user_request.safety_context.get("planning_trace")
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
    lookup = _source_lookup(result_bundle, safe_previews, user_request)
    rendered = render_display_plan(display_plan, lookup, _resolve_section_payload)
    return rendered.content


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
            }
            for preview in safe_previews
        }
        rendered = render_display_plan(display_plan, lookup, _resolve_section_payload)
        return self.summarizer.summarize(rendered)
