"""Top-level orchestration for the schema-driven agent runtime."""

from __future__ import annotations

from typing import Any

from agent_runtime.capabilities import CapabilityRegistry
from agent_runtime.core.errors import AgentRuntimeError, ValidationError
from agent_runtime.core.logging import get_logger, log_event
from agent_runtime.core.types import ActionDAG, ResultBundle, UserRequest
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.failure_repair import attempt_failure_repair
from agent_runtime.execution.safety import evaluate_dag_safety
from agent_runtime.input_pipeline.argument_extraction import extract_arguments
from agent_runtime.input_pipeline.capability_fit import assess_capability_fit
from agent_runtime.input_pipeline.dataflow_planning import plan_dataflow
from agent_runtime.input_pipeline.dag_builder import build_action_dag
from agent_runtime.input_pipeline.decomposition import classify_prompt, decompose_prompt
from agent_runtime.input_pipeline.domain_selection import select_capabilities
from agent_runtime.input_pipeline.planning_review import review_action_dag
from agent_runtime.input_pipeline.verb_classification import assign_semantic_verbs
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    hash_action_dag,
    hash_capability_manifest,
    llm_client_metadata,
    replay_from_validated_dag,
)
from agent_runtime.observability import (
    EVENT_ARGUMENT_EXTRACTED,
    EVENT_ARGUMENT_REJECTED,
    EVENT_CAPABILITY_CANDIDATE,
    EVENT_CAPABILITY_GAP,
    EVENT_CAPABILITY_REJECTED,
    EVENT_CAPABILITY_SELECTED,
    EVENT_DAG_EDGE_CREATED,
    EVENT_DAG_NODE_CREATED,
    EVENT_DATAFLOW_DERIVED_TASK_CREATED,
    EVENT_DATAFLOW_REF_ACCEPTED,
    EVENT_DATAFLOW_REF_REJECTED,
    EVENT_LLM_PROPOSAL_RECEIVED,
    EVENT_OUTPUT_SHAPE_SELECTED,
    EVENT_REPAIR_ACCEPTED,
    EVENT_REPAIR_PROPOSED,
    EVENT_REPAIR_REJECTED,
    EVENT_SAFETY_ALLOWED,
    EVENT_SAFETY_BLOCKED,
    EVENT_VALIDATION_ACCEPTED,
    EVENT_VALIDATION_REJECTED,
    STAGE_ARGUMENT_EXTRACTION,
    STAGE_CAPABILITY_FIT,
    STAGE_CAPABILITY_SELECTION,
    STAGE_COMPLETED,
    STAGE_DAG_CONSTRUCTION,
    STAGE_DAG_REVIEW,
    STAGE_DATAFLOW_PLANNING,
    STAGE_DECOMPOSITION,
    STAGE_EXECUTION,
    STAGE_FAILURE_REPAIR,
    STAGE_OUTPUT_PLANNING,
    STAGE_PROMPT_CLASSIFICATION,
    STAGE_REQUEST_RECEIVED,
    STAGE_SAFETY_EVALUATION,
    STAGE_VERB_ASSIGNMENT,
    ObservabilityContext,
    build_observability_context,
)
from agent_runtime.output_pipeline.orchestrator import OutputPipelineOrchestrator


class AgentRuntime:
    """Coordinate the full typed agent pipeline from prompt to rendered output."""

    def __init__(
        self,
        llm_client,
        registry: CapabilityRegistry,
        execution_engine: ExecutionEngine,
        output_orchestrator: OutputPipelineOrchestrator,
    ) -> None:
        self.llm_client = llm_client
        self.registry = registry
        self.execution_engine = execution_engine
        self.output_orchestrator = output_orchestrator
        self.logger = get_logger("orchestrator")
        self.last_planning_trace: PlanningTrace | None = None
        self.last_plan_summary: dict[str, Any] | None = None
        self.last_failure_summary: dict[str, Any] | None = None

    @staticmethod
    def _observability(user_request: UserRequest) -> ObservabilityContext | None:
        """Return the request-scoped observability context when available."""

        observability = user_request.safety_context.get("observability")
        if isinstance(observability, ObservabilityContext):
            return observability
        return None

    def _runtime_state_snapshot(self) -> dict[str, Any]:
        """Return the current runtime-owned introspection state."""

        return {
            "last_plan": self.last_plan_summary,
            "last_failure": self.last_failure_summary,
        }

    def _record_last_failure(
        self,
        *,
        request_id: str,
        prompt: str,
        category: str,
        stage: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a safe summary of the most recent failure."""

        self.last_failure_summary = {
            "request_id": request_id,
            "prompt": prompt,
            "category": category,
            "stage": stage,
            "reason": reason,
            "metadata": dict(metadata or {}),
        }

    def _record_last_plan(
        self,
        *,
        user_request: UserRequest,
        tasks,
        selections,
        dag: ActionDAG,
        safety_decision,
    ) -> None:
        """Persist a safe summary of the most recent trusted plan."""

        rows = [
            {
                "node_id": node.id,
                "task_id": node.task_id,
                "capability_id": node.capability_id,
                "operation_id": node.operation_id,
                "depends_on": ", ".join(node.depends_on) if node.depends_on else "-",
            }
            for node in dag.nodes
        ]
        self.last_plan_summary = {
            "request_id": user_request.request_id,
            "prompt": user_request.raw_prompt,
            "task_count": len(tasks),
            "tasks": [
                {
                    "task_id": task.id,
                    "description": task.description,
                    "semantic_verb": task.semantic_verb,
                    "object_type": task.object_type,
                    "dependencies": list(task.dependencies),
                }
                for task in tasks
            ],
            "selected_capabilities": [
                {
                    "task_id": selection.task_id,
                    "capability_id": (
                        selection.selected.capability_id if selection.selected is not None else None
                    ),
                    "operation_id": (
                        selection.selected.operation_id if selection.selected is not None else None
                    ),
                    "unresolved_reason": selection.unresolved_reason,
                }
                for selection in selections
            ],
            "dag_id": dag.dag_id,
            "rows": rows,
            "safety_decision": {
                "allowed": safety_decision.allowed,
                "requires_confirmation": safety_decision.requires_confirmation,
                "blocked_reasons": list(safety_decision.blocked_reasons),
                "warnings": list(safety_decision.warnings),
            },
        }

    def _build_user_request(self, raw_prompt: str, context: dict[str, Any]) -> UserRequest:
        """Build a typed user request with runtime-owned safety context."""

        payload = dict(context or {})
        user_context = dict(payload.pop("user_context", {}))
        session_context = dict(payload.pop("session_context", {}))
        safety_context = dict(payload.pop("safety_context", {}))
        session_context.update(payload)
        session_context.setdefault("runtime_state", self._runtime_state_snapshot())
        safety_context.setdefault("capability_registry", self.registry)
        safety_context.setdefault("result_store", self.execution_engine.result_store)
        safety_context.setdefault("allow_full_output_access", False)
        safety_context.setdefault(
            "planning_trace",
            PlanningTrace(
                request_id="pending",
                raw_prompt=raw_prompt,
                capability_manifest_hash=hash_capability_manifest(
                    self.registry.export_llm_manifest()
                ),
            ),
        )
        return UserRequest(
            raw_prompt=raw_prompt,
            user_context=user_context,
            session_context=session_context,
            safety_context=safety_context,
        )

    def _bind_trace_to_request(self, user_request: UserRequest) -> PlanningTrace:
        """Attach a request-scoped planning trace to the user request."""

        trace = user_request.safety_context.get("planning_trace")
        if not isinstance(trace, PlanningTrace):
            trace = PlanningTrace(
                request_id=user_request.request_id,
                raw_prompt=user_request.raw_prompt,
                capability_manifest_hash=hash_capability_manifest(
                    self.registry.export_llm_manifest()
                ),
            )
            user_request.safety_context["planning_trace"] = trace
        trace.request_id = user_request.request_id
        trace.raw_prompt = user_request.raw_prompt
        trace.capability_manifest_hash = hash_capability_manifest(
            self.registry.export_llm_manifest()
        )
        trace.capability_manifest_version = trace.capability_manifest_hash
        self.last_planning_trace = trace
        return trace

    def _mark_dag_execution_ready(self, dag: ActionDAG, safety_decision) -> ActionDAG:
        """Return a trusted DAG annotated for execution."""

        prepared = dag.model_copy(
            update={
                "execution_ready": bool(safety_decision.allowed),
                "safety_decision": safety_decision.model_dump(mode="json"),
            }
        )
        final_hash = hash_action_dag(prepared)
        return prepared.model_copy(update={"final_dag_hash": final_hash})

    def _record_dag_review(
        self,
        trace: PlanningTrace,
        request_id: str,
        review,
        dag: ActionDAG,
    ) -> None:
        """Append one DAG review entry to the planning trace."""

        model_name, temperature = llm_client_metadata(self.llm_client)
        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="dag_review",
                request_id=request_id,
                model_name=model_name,
                llm_temperature=temperature,
                prompt_template_id="dag_review",
                raw_llm_response=review.model_dump(mode="json"),
                parsed_proposal=review.model_dump(mode="json"),
                selected_candidate={
                    "dag_id": dag.dag_id,
                    "node_count": len(dag.nodes),
                },
                rejection_reasons=(
                    ["recommended_repair_advisory_only"] if review.recommended_repair else []
                ),
            ),
        )

    def _record_safety(self, trace: PlanningTrace, request_id: str, safety_decision, dag_hash: str | None) -> None:
        """Append deterministic safety information to the planning trace."""

        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="safety_evaluation",
                request_id=request_id,
                prompt_template_id="safety_evaluation",
                raw_llm_response=None,
                parsed_proposal=None,
                safety_decision=safety_decision.model_dump(mode="json"),
                final_dag_hash=dag_hash,
            ),
        )

    def _placeholder_direct_answer(self, user_request: UserRequest) -> str:
        """Return the current direct-answer placeholder for no-tool questions."""

        _ = user_request
        return "Direct answering without tools is not implemented yet in this runtime."

    def _render_capability_gaps(self, gaps) -> str:
        """Render one or more capability-gap descriptions safely for the user."""

        if not gaps:
            return "I understood the request, but this runtime does not currently have a compatible capability for it."
        if len(gaps) == 1:
            return gaps[0].user_facing_message
        lines = ["Some parts of that request are not currently supported by this runtime:"]
        lines.extend(f"- {gap.user_facing_message}" for gap in gaps)
        return "\n".join(lines)

    def _safe_failure(self, user_request: UserRequest, stage: str, message: str) -> str:
        """Log and return a user-safe failure message."""

        observability = self._observability(user_request)
        if observability is not None:
            observability.stage_failed(
                stage,
                "Request failed",
                message,
                details={"request_id": user_request.request_id},
            )
            observability.stage_completed(
                STAGE_COMPLETED,
                "Request completed with error",
                "The request ended with a user-safe runtime error.",
                details={"final_status": "error", "failed_stage": stage},
            )
        log_event(
            self.logger,
            "agent_runtime.failure",
            request_id=user_request.request_id,
            stage=stage,
            message=message,
        )
        self._record_last_failure(
            request_id=user_request.request_id,
            prompt=user_request.raw_prompt,
            category="runtime_error",
            stage=stage,
            reason=message,
        )
        trace = user_request.safety_context.get("planning_trace")
        if isinstance(trace, PlanningTrace) and message not in trace.user_facing_errors:
            trace.user_facing_errors.append(message)
        return f"I couldn't complete that request safely at the {stage} stage. {message}"

    def handle_request(self, raw_prompt: str, context: dict[str, Any] = {}) -> str:
        """Run the full agent pipeline and return final user-facing text."""

        user_request = self._build_user_request(raw_prompt, dict(context or {}))
        trace = self._bind_trace_to_request(user_request)
        observability = build_observability_context(user_request.request_id, dict(context or {}))
        user_request.safety_context["observability"] = observability
        user_request.session_context["observability"] = observability
        log_event(
            self.logger,
            "agent_runtime.request_received",
            request_id=user_request.request_id,
            prompt=user_request.raw_prompt,
        )
        observability.info(
            STAGE_REQUEST_RECEIVED,
            "request.received",
            "Request received",
            "The runtime accepted a new request and is starting the planning pipeline.",
            details={
                "request_id": user_request.request_id,
                "prompt_preview": user_request.raw_prompt[:240],
            },
        )

        try:
            observability.stage_started(
                STAGE_PROMPT_CLASSIFICATION,
                "Prompt classification started",
                "The runtime is classifying the prompt and deciding whether tools are needed.",
            )
            classification = classify_prompt(user_request, self.llm_client)
            observability.info(
                STAGE_PROMPT_CLASSIFICATION,
                EVENT_LLM_PROPOSAL_RECEIVED,
                "Classification proposal received",
                "The runtime received a structured classification proposal.",
                details={
                    "prompt_type": classification.prompt_type,
                    "likely_domains": classification.likely_domains,
                    "risk_level": classification.risk_level,
                    "needs_clarification": classification.needs_clarification,
                },
            )
            observability.info(
                STAGE_PROMPT_CLASSIFICATION,
                EVENT_VALIDATION_ACCEPTED,
                "Classification accepted",
                "The classification passed deterministic validation.",
                details={
                    "prompt_type": classification.prompt_type,
                    "requires_tools": classification.requires_tools,
                },
            )
            log_event(
                self.logger,
                "agent_runtime.classified",
                request_id=user_request.request_id,
                prompt_type=classification.prompt_type,
                requires_tools=classification.requires_tools,
                likely_domains=classification.likely_domains,
            )
            observability.stage_completed(
                STAGE_PROMPT_CLASSIFICATION,
                "Prompt classification completed",
                "Prompt classification finished successfully.",
                details={
                    "prompt_type": classification.prompt_type,
                    "requires_tools": classification.requires_tools,
                },
            )

            if classification.prompt_type == "simple_question" and not classification.requires_tools:
                content = self._placeholder_direct_answer(user_request)
                observability.stage_completed(
                    STAGE_COMPLETED,
                    "Request completed",
                    "The request finished without invoking any capabilities.",
                    details={"final_status": "success", "output_type": "direct_placeholder"},
                )
                return content

            observability.stage_started(
                STAGE_DECOMPOSITION,
                "Task decomposition started",
                "The runtime is breaking the prompt into atomic tasks.",
            )
            decomposition = decompose_prompt(
                user_request,
                classification,
                self.llm_client,
                available_domains=sorted({manifest.domain for manifest in self.registry.list_manifests()}),
            )
            observability.info(
                STAGE_DECOMPOSITION,
                EVENT_LLM_PROPOSAL_RECEIVED,
                "Task decomposition received",
                "The runtime received a structured task decomposition.",
                details={
                    "task_count": len(decomposition.tasks),
                    "tasks": [
                        {
                            "task_id": task.id,
                            "description": task.description,
                            "depends_on": list(task.dependencies),
                        }
                        for task in decomposition.tasks
                    ],
                    "unresolved_references": decomposition.unresolved_references,
                },
            )
            log_event(
                self.logger,
                "agent_runtime.decomposed",
                request_id=user_request.request_id,
                task_count=len(decomposition.tasks),
                global_constraints=decomposition.global_constraints,
            )
            observability.stage_completed(
                STAGE_DECOMPOSITION,
                "Task decomposition completed",
                "Task decomposition completed with validated task frames.",
                details={
                    "task_count": len(decomposition.tasks),
                    "global_constraints": decomposition.global_constraints,
                    "unresolved_references": decomposition.unresolved_references,
                },
            )

            enriched_tasks = []
            for task in decomposition.tasks:
                task.constraints = {
                    **dict(task.constraints),
                    "global_constraints": dict(decomposition.global_constraints),
                }
                enriched_tasks.append(task)

            observability.stage_started(
                STAGE_VERB_ASSIGNMENT,
                "Semantic verb assignment started",
                "The runtime is assigning semantic verbs and object types to each task.",
            )
            typed_tasks = assign_semantic_verbs(enriched_tasks, self.llm_client, trace=trace)
            observability.info(
                STAGE_VERB_ASSIGNMENT,
                EVENT_VALIDATION_ACCEPTED,
                "Verb assignments accepted",
                "Semantic task annotations passed deterministic validation.",
                details={
                    "tasks": [
                        {
                            "task_id": task.id,
                            "semantic_verb": task.semantic_verb,
                            "object_type": task.object_type,
                            "risk_level": task.risk_level,
                        }
                        for task in typed_tasks
                    ]
                },
            )
            log_event(
                self.logger,
                "agent_runtime.verbs_assigned",
                request_id=user_request.request_id,
                tasks=[{"task_id": task.id, "verb": task.semantic_verb} for task in typed_tasks],
            )
            observability.stage_completed(
                STAGE_VERB_ASSIGNMENT,
                "Semantic verb assignment completed",
                "Each task now has a semantic verb and object type.",
                details={"task_count": len(typed_tasks)},
            )

            classification_context = {
                "original_prompt": user_request.raw_prompt,
                "prompt_type": classification.prompt_type,
                "likely_domains": classification.likely_domains,
                "risk_level": classification.risk_level,
            }

            observability.stage_started(
                STAGE_CAPABILITY_SELECTION,
                "Capability selection started",
                "The runtime is ranking capability candidates for each task.",
            )
            selections = select_capabilities(
                typed_tasks,
                self.registry,
                self.llm_client,
                classification_context=classification_context,
                trace=trace,
            )
            for selection in selections:
                observability.info(
                    STAGE_CAPABILITY_SELECTION,
                    EVENT_CAPABILITY_CANDIDATE,
                    "Capability candidates considered",
                    "The runtime ranked capability candidates for one task.",
                    details={
                        "task_id": selection.task_id,
                        "candidates": [
                            {
                                "capability_id": candidate.capability_id,
                                "operation_id": candidate.operation_id,
                                "confidence": candidate.confidence,
                            }
                            for candidate in selection.candidates
                        ],
                    },
                )
                if selection.selected is not None:
                    observability.info(
                        STAGE_CAPABILITY_SELECTION,
                        EVENT_CAPABILITY_SELECTED,
                        "Capability selected",
                        "A candidate capability was selected for this task.",
                        details={
                            "task_id": selection.task_id,
                            "capability_id": selection.selected.capability_id,
                            "operation_id": selection.selected.operation_id,
                            "confidence": selection.selected.confidence,
                        },
                    )
                elif selection.unresolved_reason:
                    observability.warning(
                        STAGE_CAPABILITY_SELECTION,
                        EVENT_VALIDATION_REJECTED,
                        "Capability unresolved",
                        "No capability candidate was accepted for this task.",
                        details={
                            "task_id": selection.task_id,
                            "reason": selection.unresolved_reason,
                        },
                    )
            log_event(
                self.logger,
                "agent_runtime.capabilities_selected",
                request_id=user_request.request_id,
                selections=[
                    {
                        "task_id": result.task_id,
                        "selected": (
                            result.selected.capability_id if result.selected is not None else None
                        ),
                        "unresolved_reason": result.unresolved_reason,
                    }
                    for result in selections
                ],
            )
            observability.stage_completed(
                STAGE_CAPABILITY_SELECTION,
                "Capability selection completed",
                "Capability ranking and preliminary selection completed.",
                details={"task_count": len(selections)},
            )

            observability.stage_started(
                STAGE_CAPABILITY_FIT,
                "Capability fit started",
                "The runtime is checking whether selected capabilities truly fit each task.",
            )
            fit_decisions, capability_gaps = assess_capability_fit(
                typed_tasks,
                selections,
                self.registry,
                classification_context,
                self.llm_client,
                trace=trace,
            )
            fit_by_task = {decision.task_id: decision for decision in fit_decisions}
            for decision in fit_decisions:
                if decision.is_fit:
                    llm_raw_preview = (
                        decision.llm_diagnostics.raw_response_preview
                        if observability.debug and decision.llm_diagnostics is not None
                        else None
                    )
                    observability.info(
                        STAGE_CAPABILITY_FIT,
                        EVENT_VALIDATION_ACCEPTED,
                        "Capability fit accepted",
                        "The selected capability passed semantic and deterministic fit checks.",
                        details={
                            "task_id": decision.task_id,
                            "capability_id": decision.candidate_capability_id,
                            "llm_fits": (
                                decision.llm_proposal.fits
                                if decision.llm_proposal is not None
                                else None
                            ),
                            "llm_primary_failure_mode": (
                                decision.llm_proposal.primary_failure_mode
                                if decision.llm_proposal is not None
                                else None
                            ),
                            "llm_confidence": (
                                decision.llm_proposal.confidence
                                if decision.llm_proposal is not None
                                else None
                            ),
                            "llm_error_kind": (
                                decision.llm_diagnostics.error_kind
                                if decision.llm_diagnostics is not None
                                else None
                            ),
                            "llm_error_message": (
                                decision.llm_diagnostics.error_message
                                if decision.llm_diagnostics is not None
                                else None
                            ),
                            "llm_raw_response_preview": llm_raw_preview,
                            "status": decision.status,
                            "reasons": decision.reasons,
                            "normalized_task_domain": decision.normalized_task_domain,
                            "normalized_likely_domains": decision.normalized_likely_domains,
                            "normalized_task_object_type": decision.normalized_task_object_type,
                            "normalized_manifest_domain": decision.normalized_manifest_domain,
                            "normalized_manifest_object_types": decision.normalized_manifest_object_types,
                        },
                    )
                else:
                    llm_raw_preview = (
                        decision.llm_diagnostics.raw_response_preview
                        if observability.debug and decision.llm_diagnostics is not None
                        else None
                    )
                    observability.warning(
                        STAGE_CAPABILITY_FIT,
                        EVENT_CAPABILITY_REJECTED,
                        "Capability rejected",
                        "A selected candidate was rejected by capability fit validation.",
                        details={
                            "task_id": decision.task_id,
                            "capability_id": decision.candidate_capability_id,
                            "llm_fits": (
                                decision.llm_proposal.fits
                                if decision.llm_proposal is not None
                                else None
                            ),
                            "llm_primary_failure_mode": (
                                decision.llm_proposal.primary_failure_mode
                                if decision.llm_proposal is not None
                                else None
                            ),
                            "llm_confidence": (
                                decision.llm_proposal.confidence
                                if decision.llm_proposal is not None
                                else None
                            ),
                            "llm_error_kind": (
                                decision.llm_diagnostics.error_kind
                                if decision.llm_diagnostics is not None
                                else None
                            ),
                            "llm_error_message": (
                                decision.llm_diagnostics.error_message
                                if decision.llm_diagnostics is not None
                                else None
                            ),
                            "llm_raw_response_preview": llm_raw_preview,
                            "status": decision.status,
                            "reasons": decision.reasons,
                            "deterministic_rejections": decision.deterministic_rejections,
                            "normalized_task_domain": decision.normalized_task_domain,
                            "normalized_likely_domains": decision.normalized_likely_domains,
                            "normalized_task_object_type": decision.normalized_task_object_type,
                            "normalized_manifest_domain": decision.normalized_manifest_domain,
                            "normalized_manifest_object_types": decision.normalized_manifest_object_types,
                        },
                    )
            for gap in capability_gaps:
                observability.warning(
                    STAGE_CAPABILITY_FIT,
                    EVENT_CAPABILITY_GAP,
                    "Capability gap detected",
                    "The runtime understood the task but does not have a compatible capability.",
                    details={
                        "task_id": gap.task_id,
                        "suggested_domain": gap.suggested_domain,
                        "suggested_object_type": gap.suggested_object_type,
                        "message": gap.user_facing_message,
                    },
                )
            log_event(
                self.logger,
                "agent_runtime.capability_fit_assessed",
                request_id=user_request.request_id,
                decisions=[
                    {
                        "task_id": decision.task_id,
                        "capability_id": decision.candidate_capability_id,
                        "status": decision.status,
                    }
                    for decision in fit_decisions
                ],
            )
            observability.stage_completed(
                STAGE_CAPABILITY_FIT,
                "Capability fit completed",
                "Capability fit validation completed.",
                details={
                    "fit_count": sum(1 for decision in fit_decisions if decision.is_fit),
                    "gap_count": len(capability_gaps),
                },
            )

            fit_tasks = [task for task in typed_tasks if fit_by_task.get(task.id, None) is None or fit_by_task[task.id].is_fit]
            fit_selections = [
                selection
                for selection in selections
                if fit_by_task.get(selection.task_id, None) is None or fit_by_task[selection.task_id].is_fit
            ]
            if not fit_tasks:
                message = self._render_capability_gaps(capability_gaps)
                self._record_last_failure(
                    request_id=user_request.request_id,
                    prompt=user_request.raw_prompt,
                    category="capability_gap",
                    stage="capability_fit",
                    reason=message,
                    metadata={
                        "gaps": [gap.model_dump(mode="json") for gap in capability_gaps],
                    },
                )
                if message not in trace.user_facing_errors:
                    trace.user_facing_errors.append(message)
                observability.stage_completed(
                    STAGE_COMPLETED,
                    "Request completed with capability gap",
                    "The runtime could not find a compatible capability for the request.",
                    details={"final_status": "unsupported", "gap_count": len(capability_gaps)},
                )
                return message

            observability.stage_started(
                STAGE_DATAFLOW_PLANNING,
                "Dataflow planning started",
                "The runtime is looking for producer-consumer relationships between tasks.",
            )
            dataflow_plan = plan_dataflow(
                original_prompt=user_request.raw_prompt,
                tasks=fit_tasks,
                capability_selections=fit_selections,
                registry=self.registry,
                llm_client=self.llm_client,
                trace=trace,
            )
            for ref in dataflow_plan.refs:
                observability.info(
                    STAGE_DATAFLOW_PLANNING,
                    EVENT_DATAFLOW_REF_ACCEPTED,
                    "Dataflow reference accepted",
                    "A producer output was wired into a downstream task argument.",
                    details={
                        "consumer_task_id": ref.consumer_task_id,
                        "consumer_argument_name": ref.consumer_argument_name,
                        "producer_task_id": ref.producer_task_id,
                        "producer_output_key": ref.producer_output_key,
                    },
                )
            for rejected in dataflow_plan.rejected_refs:
                observability.warning(
                    STAGE_DATAFLOW_PLANNING,
                    EVENT_DATAFLOW_REF_REJECTED,
                    "Dataflow reference rejected",
                    "A proposed producer-consumer reference did not pass validation.",
                    details=rejected,
                )
            for derived in dataflow_plan.derived_tasks:
                observability.info(
                    STAGE_DATAFLOW_PLANNING,
                    EVENT_DATAFLOW_DERIVED_TASK_CREATED,
                    "Derived task created",
                    "The runtime created a derived internal data task.",
                    details={
                        "task_id": derived.task.id,
                        "description": derived.task.description,
                        "capability_id": derived.capability_id,
                        "depends_on": derived.depends_on,
                    },
                )
            log_event(
                self.logger,
                "agent_runtime.dataflow_planned",
                request_id=user_request.request_id,
                ref_count=len(dataflow_plan.refs),
                derived_task_count=len(dataflow_plan.derived_tasks),
                unresolved_dataflows=dataflow_plan.unresolved_dataflows,
            )
            observability.stage_completed(
                STAGE_DATAFLOW_PLANNING,
                "Dataflow planning completed",
                "Dataflow planning completed.",
                details={
                    "accepted_refs": len(dataflow_plan.refs),
                    "derived_tasks": len(dataflow_plan.derived_tasks),
                    "unresolved_dataflows": dataflow_plan.unresolved_dataflows,
                },
            )

            observability.stage_started(
                STAGE_ARGUMENT_EXTRACTION,
                "Argument extraction started",
                "The runtime is filling validated capability arguments.",
            )
            extraction_results = extract_arguments(
                typed_tasks,
                fit_selections,
                self.registry,
                self.llm_client,
                trace=trace,
                dataflow_plan=dataflow_plan,
                capability_fit_decisions=fit_decisions,
            )
            for result in extraction_results:
                details = {
                    "task_id": result.task_id,
                    "capability_id": result.capability_id,
                    "arguments": {
                        key: ("input_ref" if hasattr(value, "source_node_id") else value)
                        for key, value in result.arguments.items()
                    },
                    "missing_required_arguments": result.missing_required_arguments,
                }
                if result.missing_required_arguments:
                    observability.warning(
                        STAGE_ARGUMENT_EXTRACTION,
                        EVENT_ARGUMENT_REJECTED,
                        "Missing required arguments",
                        "Argument extraction could not fully satisfy this task.",
                        details=details,
                    )
                else:
                    observability.info(
                        STAGE_ARGUMENT_EXTRACTION,
                        EVENT_ARGUMENT_EXTRACTED,
                        "Arguments extracted",
                        "Validated arguments are ready for DAG construction.",
                        details=details,
                    )
            log_event(
                self.logger,
                "agent_runtime.arguments_extracted",
                request_id=user_request.request_id,
                extraction_results=[
                    {
                        "task_id": result.task_id,
                        "capability_id": result.capability_id,
                        "missing_required_arguments": result.missing_required_arguments,
                    }
                    for result in extraction_results
                ],
            )
            observability.stage_completed(
                STAGE_ARGUMENT_EXTRACTION,
                "Argument extraction completed",
                "Argument extraction completed for fit-approved tasks.",
                details={"task_count": len(extraction_results)},
            )

            decomposition.tasks = typed_tasks
            observability.stage_started(
                STAGE_DAG_CONSTRUCTION,
                "DAG construction started",
                "The runtime is building a validated action DAG.",
            )
            dag = build_action_dag(
                user_request,
                decomposition,
                fit_selections,
                extraction_results,
                dataflow_plan=dataflow_plan,
                capability_fit_decisions=fit_decisions,
            )
            trace.dag_raw = dag.model_dump(mode="json")
            append_trace_entry(
                trace,
                PlanningTraceEntry(
                    stage="dag_construction",
                    request_id=user_request.request_id,
                    prompt_template_id="dag_construction",
                    raw_llm_response=trace.dag_raw,
                    selected_candidate=trace.dag_raw,
                ),
            )
            for node in dag.nodes:
                observability.info(
                    STAGE_DAG_CONSTRUCTION,
                    EVENT_DAG_NODE_CREATED,
                    "DAG node created",
                    "An executable node was added to the action DAG.",
                    details={
                        "node_id": node.id,
                        "task_id": node.task_id,
                        "capability_id": node.capability_id,
                        "operation_id": node.operation_id,
                        "depends_on": list(node.depends_on),
                    },
                )
            for source, target in dag.edges:
                observability.info(
                    STAGE_DAG_CONSTRUCTION,
                    EVENT_DAG_EDGE_CREATED,
                    "DAG edge created",
                    "A dependency edge was added to the action DAG.",
                    details={"source_node_id": source, "target_node_id": target},
                )
            observability.stage_completed(
                STAGE_DAG_CONSTRUCTION,
                "DAG construction completed",
                "The action DAG is validated and ready for review.",
                details={"node_count": len(dag.nodes), "edge_count": len(dag.edges)},
            )

            observability.stage_started(
                STAGE_DAG_REVIEW,
                "DAG review started",
                "The runtime is running an advisory DAG review over sanitized metadata.",
            )
            review = review_action_dag(user_request, dag, self.llm_client, registry=self.registry)
            self._record_dag_review(trace, user_request.request_id, review, dag)
            observability.info(
                STAGE_DAG_REVIEW,
                EVENT_LLM_PROPOSAL_RECEIVED,
                "DAG review received",
                "The advisory DAG review completed.",
                details={
                    "missing_user_intents": review.missing_user_intents,
                    "suspicious_nodes": review.suspicious_nodes,
                    "dependency_warnings": review.dependency_warnings,
                    "dataflow_warnings": review.dataflow_warnings,
                    "output_expectation_warnings": review.output_expectation_warnings,
                },
            )
            log_event(
                self.logger,
                "agent_runtime.dag_built",
                request_id=user_request.request_id,
                dag_id=dag.dag_id,
                node_count=len(dag.nodes),
            )
            observability.stage_completed(
                STAGE_DAG_REVIEW,
                "DAG review completed",
                "Advisory DAG review finished without changing the trusted DAG.",
                details={"confidence": review.confidence},
            )

            observability.stage_started(
                STAGE_SAFETY_EVALUATION,
                "Safety evaluation started",
                "The runtime is checking the DAG against deterministic safety policy.",
            )
            safety_decision = evaluate_dag_safety(
                dag,
                self.registry,
                self.execution_engine.safety_policy.config,
            )
            ready_dag = self._mark_dag_execution_ready(dag, safety_decision)
            trace.final_dag_hash = ready_dag.final_dag_hash
            trace.dag_validated = ready_dag.model_dump(mode="json")
            trace.validated_dag = trace.dag_validated
            trace.safety_decision = safety_decision.model_dump(mode="json")
            trace.execution_ready = bool(safety_decision.allowed)
            self._record_safety(trace, user_request.request_id, safety_decision, ready_dag.final_dag_hash)
            self._record_last_plan(
                user_request=user_request,
                tasks=typed_tasks,
                selections=fit_selections,
                dag=ready_dag,
                safety_decision=safety_decision,
            )
            if safety_decision.allowed:
                observability.info(
                    STAGE_SAFETY_EVALUATION,
                    EVENT_SAFETY_ALLOWED,
                    "Safety evaluation allowed execution",
                    "The DAG passed deterministic safety checks.",
                    details={
                        "requires_confirmation": safety_decision.requires_confirmation,
                        "warnings": safety_decision.warnings,
                    },
                )
            else:
                observability.warning(
                    STAGE_SAFETY_EVALUATION,
                    EVENT_SAFETY_BLOCKED,
                    "Safety evaluation blocked execution",
                    "The DAG was blocked by deterministic safety policy.",
                    details={
                        "blocked_reasons": safety_decision.blocked_reasons,
                        "warnings": safety_decision.warnings,
                    },
                )
            log_event(
                self.logger,
                "agent_runtime.safety_evaluated",
                request_id=user_request.request_id,
                allowed=safety_decision.allowed,
                requires_confirmation=safety_decision.requires_confirmation,
                blocked_reasons=safety_decision.blocked_reasons,
            )
            observability.stage_completed(
                STAGE_SAFETY_EVALUATION,
                "Safety evaluation completed",
                "Safety evaluation finished.",
                details={
                    "allowed": safety_decision.allowed,
                    "requires_confirmation": safety_decision.requires_confirmation,
                },
            )

            execution_context = dict(user_request.session_context)
            execution_context["observability"] = observability
            result_bundle: ResultBundle
            observability.stage_started(
                STAGE_EXECUTION,
                "Execution started",
                "The runtime is executing the validated DAG.",
                details={"dag_id": ready_dag.dag_id, "node_count": len(ready_dag.nodes)},
            )
            if not safety_decision.allowed:
                result_bundle = ResultBundle(
                    dag_id=dag.dag_id,
                    results=[],
                    status="error",
                    safe_summary="Execution blocked by safety policy.",
                    metadata={
                        "blocked_reasons": list(safety_decision.blocked_reasons),
                        "warnings": list(safety_decision.warnings),
                        "confirmation_required": False,
                    },
                )
                self._record_last_failure(
                    request_id=user_request.request_id,
                    prompt=user_request.raw_prompt,
                    category="safety_block",
                    stage="safety",
                    reason="Execution blocked by safety policy.",
                    metadata={"blocked_reasons": list(safety_decision.blocked_reasons)},
                )
                observability.warning(
                    STAGE_EXECUTION,
                    EVENT_SAFETY_BLOCKED,
                    "Execution blocked",
                    "Execution did not start because safety policy blocked the DAG.",
                    details={"blocked_reasons": safety_decision.blocked_reasons},
                )
            elif safety_decision.requires_confirmation and not bool(execution_context.get("confirmation", False)):
                result_bundle = ResultBundle(
                    dag_id=dag.dag_id,
                    results=[],
                    status="error",
                    safe_summary="Execution requires confirmation before proceeding.",
                    metadata={
                        "blocked_reasons": [],
                        "warnings": list(safety_decision.warnings),
                        "confirmation_required": True,
                    },
                )
                self._record_last_failure(
                    request_id=user_request.request_id,
                    prompt=user_request.raw_prompt,
                    category="confirmation_required",
                    stage="safety",
                    reason="Execution requires confirmation before proceeding.",
                )
                observability.warning(
                    STAGE_EXECUTION,
                    EVENT_SAFETY_BLOCKED,
                    "Confirmation required",
                    "Execution is waiting for explicit confirmation.",
                    details={"warnings": safety_decision.warnings},
                )
            else:
                result_bundle = self.execution_engine.execute(ready_dag, execution_context)

            repair_metadata: dict[str, Any] | None = None
            if (
                result_bundle.status in {"error", "partial"}
                and not bool(execution_context.get("repair_attempted", False))
            ):
                observability.stage_started(
                    STAGE_FAILURE_REPAIR,
                    "Failure repair started",
                    "The runtime is evaluating one safe repair attempt.",
                )
                repaired_dag, repair_metadata = attempt_failure_repair(
                    user_request=user_request,
                    dag=ready_dag,
                    result_bundle=result_bundle,
                    registry=self.registry,
                    llm_client=self.llm_client,
                    safety_config=self.execution_engine.safety_policy.config,
                    repair_attempt_count=int(execution_context.get("repair_attempt_count", 0)),
                )
                if repair_metadata is not None:
                    observability.info(
                        STAGE_FAILURE_REPAIR,
                        EVENT_REPAIR_PROPOSED,
                        "Repair proposal received",
                        "The runtime evaluated one advisory repair proposal.",
                        details={
                            "repair_attempt_count": repair_metadata.get("repair_attempt_count"),
                            "proposed_action": (
                                dict(repair_metadata.get("proposal") or {}).get("proposed_action")
                            ),
                        },
                    )
                if repaired_dag is not None:
                    repaired_decision = evaluate_dag_safety(
                        repaired_dag,
                        self.registry,
                        self.execution_engine.safety_policy.config,
                    )
                    if repaired_decision.allowed:
                        ready_repaired_dag = self._mark_dag_execution_ready(repaired_dag, repaired_decision)
                        trace.final_dag_hash = ready_repaired_dag.final_dag_hash
                        trace.validated_dag = ready_repaired_dag.model_dump(mode="json")
                        trace.safety_decision = repaired_decision.model_dump(mode="json")
                        execution_context["repair_attempted"] = True
                        execution_context["repair_attempt_count"] = int(
                            repair_metadata.get("repair_attempt_count", 1)
                        )
                        observability.info(
                            STAGE_FAILURE_REPAIR,
                            EVENT_REPAIR_ACCEPTED,
                            "Repair accepted",
                            "A repaired DAG passed validation and safety checks, so execution will retry once.",
                            details={
                                "repair_attempt_count": execution_context["repair_attempt_count"],
                            },
                        )
                        result_bundle = self.execution_engine.execute(ready_repaired_dag, execution_context)
                if repair_metadata is not None:
                    model_name, temperature = llm_client_metadata(self.llm_client)
                    append_trace_entry(
                        trace,
                        PlanningTraceEntry(
                            stage="failure_repair",
                            request_id=user_request.request_id,
                            model_name=model_name,
                            llm_temperature=temperature,
                            prompt_template_id="failure_repair",
                            raw_llm_response=repair_metadata.get("proposal"),
                            parsed_proposal=repair_metadata.get("proposal"),
                            selected_candidate=repair_metadata,
                            rejection_reasons=(
                                [str(repair_metadata.get("rejected"))]
                                if repair_metadata.get("rejected")
                                else []
                            ),
                        ),
                    )
                    if repair_metadata.get("rejected"):
                        observability.warning(
                            STAGE_FAILURE_REPAIR,
                            EVENT_REPAIR_REJECTED,
                            "Repair rejected",
                            "The repair proposal did not pass deterministic validation.",
                            details={"reason": repair_metadata.get("rejected")},
                        )
                    observability.stage_completed(
                        STAGE_FAILURE_REPAIR,
                        "Failure repair completed",
                        "Failure repair handling completed.",
                        details={"attempted": bool(repair_metadata.get("attempted", False))},
                    )

            log_event(
                self.logger,
                "agent_runtime.executed",
                request_id=user_request.request_id,
                dag_id=dag.dag_id,
                bundle_status=result_bundle.status,
                result_count=len(result_bundle.results),
            )
            observability.stage_completed(
                STAGE_EXECUTION,
                "Execution completed",
                "DAG execution finished.",
                details={
                    "bundle_status": result_bundle.status,
                    "result_count": len(result_bundle.results),
                },
            )
            if result_bundle.status in {"error", "partial"}:
                first_error = next((result.error for result in result_bundle.results if result.error), None)
                self._record_last_failure(
                    request_id=user_request.request_id,
                    prompt=user_request.raw_prompt,
                    category="execution_error",
                    stage="execution",
                    reason=first_error or str(result_bundle.safe_summary or "Execution failed."),
                    metadata={
                        "bundle_status": result_bundle.status,
                        "safe_summary": result_bundle.safe_summary,
                    },
                )
            if repair_metadata is not None:
                result_bundle.metadata["repair_attempt"] = repair_metadata

            rendered = self.output_orchestrator.render(
                result_bundle,
                user_request=user_request,
                dag=dag,
                llm_client=self.llm_client,
            )
            log_event(
                self.logger,
                "agent_runtime.rendered",
                request_id=user_request.request_id,
                dag_id=dag.dag_id,
                content_length=len(rendered.content),
            )
            observability.stage_completed(
                STAGE_COMPLETED,
                "Request completed",
                "The request completed and a final response was rendered.",
                details={
                    "final_status": result_bundle.status,
                    "content_length": len(rendered.content),
                    "display_type": rendered.display_plan.display_type,
                },
            )
            return rendered.content

        except (AgentRuntimeError, ValidationError, ValueError) as exc:
            return self._safe_failure(user_request, "runtime", str(exc))
        except Exception as exc:  # pragma: no cover - kept as a final safety boundary
            observability = self._observability(user_request)
            if observability is not None:
                observability.stage_failed(
                    "runtime",
                    "Unexpected runtime error",
                    "The runtime hit an unexpected internal error.",
                    details={"error_type": type(exc).__name__},
                )
                observability.stage_completed(
                    STAGE_COMPLETED,
                    "Request completed with error",
                    "The request ended with an internal runtime error.",
                    details={"final_status": "error", "failed_stage": "runtime"},
                )
            log_event(
                self.logger,
                "agent_runtime.unexpected_error",
                request_id=user_request.request_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            self._record_last_failure(
                request_id=user_request.request_id,
                prompt=user_request.raw_prompt,
                category="unexpected_error",
                stage="runtime",
                reason=str(exc),
                metadata={"error_type": type(exc).__name__},
            )
            return "I hit an internal error while handling that request."

    def replay_from_trace(self, trace: PlanningTrace, context: dict[str, Any] | None = None) -> str:
        """Replay a validated DAG from trace without calling the LLM again."""

        result_bundle = replay_from_validated_dag(trace, self.execution_engine, dict(context or {}))
        rendered = self.output_orchestrator.render(result_bundle)
        return rendered.content
