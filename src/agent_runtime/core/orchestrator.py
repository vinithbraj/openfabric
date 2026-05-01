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

    def _build_user_request(self, raw_prompt: str, context: dict[str, Any]) -> UserRequest:
        """Build a typed user request with runtime-owned safety context."""

        payload = dict(context or {})
        user_context = dict(payload.pop("user_context", {}))
        session_context = dict(payload.pop("session_context", {}))
        safety_context = dict(payload.pop("safety_context", {}))
        session_context.update(payload)
        safety_context.setdefault("capability_registry", self.registry)
        safety_context.setdefault("result_store", self.execution_engine.result_store)
        safety_context.setdefault("allow_full_output_access", False)
        safety_context.setdefault(
            "planning_trace",
            PlanningTrace(
                request_id="pending",
                capability_manifest_version=hash_capability_manifest(
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
                capability_manifest_version=hash_capability_manifest(
                    self.registry.export_llm_manifest()
                ),
            )
            user_request.safety_context["planning_trace"] = trace
        trace.request_id = user_request.request_id
        trace.capability_manifest_version = hash_capability_manifest(
            self.registry.export_llm_manifest()
        )
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

    def _safe_failure(self, user_request: UserRequest, stage: str, message: str) -> str:
        """Log and return a user-safe failure message."""

        log_event(
            self.logger,
            "agent_runtime.failure",
            request_id=user_request.request_id,
            stage=stage,
            message=message,
        )
        return f"I couldn't complete that request safely at the {stage} stage. {message}"

    def handle_request(self, raw_prompt: str, context: dict[str, Any] = {}) -> str:
        """Run the full agent pipeline and return final user-facing text."""

        user_request = self._build_user_request(raw_prompt, dict(context or {}))
        trace = self._bind_trace_to_request(user_request)
        log_event(
            self.logger,
            "agent_runtime.request_received",
            request_id=user_request.request_id,
            prompt=user_request.raw_prompt,
        )

        try:
            classification = classify_prompt(user_request, self.llm_client)
            log_event(
                self.logger,
                "agent_runtime.classified",
                request_id=user_request.request_id,
                prompt_type=classification.prompt_type,
                requires_tools=classification.requires_tools,
                likely_domains=classification.likely_domains,
            )

            if classification.prompt_type == "simple_question" and not classification.requires_tools:
                return self._placeholder_direct_answer(user_request)

            decomposition = decompose_prompt(
                user_request,
                classification,
                self.llm_client,
                available_domains=sorted({manifest.domain for manifest in self.registry.list_manifests()}),
            )
            log_event(
                self.logger,
                "agent_runtime.decomposed",
                request_id=user_request.request_id,
                task_count=len(decomposition.tasks),
                global_constraints=decomposition.global_constraints,
            )

            enriched_tasks = []
            for task in decomposition.tasks:
                task.constraints = {
                    **dict(task.constraints),
                    "global_constraints": dict(decomposition.global_constraints),
                }
                enriched_tasks.append(task)

            typed_tasks = assign_semantic_verbs(enriched_tasks, self.llm_client, trace=trace)
            log_event(
                self.logger,
                "agent_runtime.verbs_assigned",
                request_id=user_request.request_id,
                tasks=[{"task_id": task.id, "verb": task.semantic_verb} for task in typed_tasks],
            )

            selections = select_capabilities(typed_tasks, self.registry, self.llm_client, trace=trace)
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

            extraction_results = extract_arguments(
                typed_tasks,
                selections,
                self.registry,
                self.llm_client,
                trace=trace,
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

            decomposition.tasks = typed_tasks
            dag = build_action_dag(
                user_request,
                decomposition,
                selections,
                extraction_results,
            )
            review = review_action_dag(user_request, dag, self.llm_client)
            self._record_dag_review(trace, user_request.request_id, review, dag)
            log_event(
                self.logger,
                "agent_runtime.dag_built",
                request_id=user_request.request_id,
                dag_id=dag.dag_id,
                node_count=len(dag.nodes),
            )

            safety_decision = evaluate_dag_safety(
                dag,
                self.registry,
                self.execution_engine.safety_policy.config,
            )
            ready_dag = self._mark_dag_execution_ready(dag, safety_decision)
            trace.final_dag_hash = ready_dag.final_dag_hash
            trace.validated_dag = ready_dag.model_dump(mode="json")
            trace.safety_decision = safety_decision.model_dump(mode="json")
            self._record_safety(trace, user_request.request_id, safety_decision, ready_dag.final_dag_hash)
            log_event(
                self.logger,
                "agent_runtime.safety_evaluated",
                request_id=user_request.request_id,
                allowed=safety_decision.allowed,
                requires_confirmation=safety_decision.requires_confirmation,
                blocked_reasons=safety_decision.blocked_reasons,
            )

            execution_context = dict(user_request.session_context)
            result_bundle: ResultBundle
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
            else:
                result_bundle = self.execution_engine.execute(ready_dag, execution_context)

            repair_metadata: dict[str, Any] | None = None
            if (
                result_bundle.status in {"error", "partial"}
                and not bool(execution_context.get("repair_attempted", False))
            ):
                repaired_dag, repair_metadata = attempt_failure_repair(
                    user_request=user_request,
                    dag=ready_dag,
                    result_bundle=result_bundle,
                    registry=self.registry,
                    llm_client=self.llm_client,
                    safety_config=self.execution_engine.safety_policy.config,
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

            log_event(
                self.logger,
                "agent_runtime.executed",
                request_id=user_request.request_id,
                dag_id=dag.dag_id,
                bundle_status=result_bundle.status,
                result_count=len(result_bundle.results),
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
            return rendered.content

        except (AgentRuntimeError, ValidationError, ValueError) as exc:
            return self._safe_failure(user_request, "runtime", str(exc))
        except Exception as exc:  # pragma: no cover - kept as a final safety boundary
            log_event(
                self.logger,
                "agent_runtime.unexpected_error",
                request_id=user_request.request_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return "I hit an internal error while handling that request."

    def replay_from_trace(self, trace: PlanningTrace, context: dict[str, Any] | None = None) -> str:
        """Replay a validated DAG from trace without calling the LLM again."""

        if trace.validated_dag is None:
            raise ValidationError("PlanningTrace does not contain a validated DAG for replay.")
        dag = ActionDAG.model_validate(trace.validated_dag)
        result_bundle = self.execution_engine.execute(dag, dict(context or {}))
        rendered = self.output_orchestrator.render(result_bundle)
        return rendered.content
