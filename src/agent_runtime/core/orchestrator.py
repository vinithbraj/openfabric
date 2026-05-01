"""Top-level orchestration for the schema-driven agent runtime."""

from __future__ import annotations

from typing import Any

from agent_runtime.capabilities import CapabilityRegistry
from agent_runtime.core.errors import AgentRuntimeError, ValidationError
from agent_runtime.core.logging import get_logger, log_event
from agent_runtime.core.types import ResultBundle, UserRequest
from agent_runtime.execution.engine import ExecutionEngine
from agent_runtime.execution.safety import evaluate_dag_safety
from agent_runtime.input_pipeline.argument_extraction import extract_arguments
from agent_runtime.input_pipeline.dag_builder import build_action_dag
from agent_runtime.input_pipeline.decomposition import classify_prompt, decompose_prompt
from agent_runtime.input_pipeline.domain_selection import select_capabilities
from agent_runtime.input_pipeline.verb_classification import assign_semantic_verbs
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
        return UserRequest(
            raw_prompt=raw_prompt,
            user_context=user_context,
            session_context=session_context,
            safety_context=safety_context,
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

            decomposition = decompose_prompt(user_request, classification, self.llm_client)
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

            typed_tasks = assign_semantic_verbs(enriched_tasks, self.llm_client)
            log_event(
                self.logger,
                "agent_runtime.verbs_assigned",
                request_id=user_request.request_id,
                tasks=[{"task_id": task.id, "verb": task.semantic_verb} for task in typed_tasks],
            )

            selections = select_capabilities(typed_tasks, self.registry, self.llm_client)
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
                result_bundle = self.execution_engine.execute(dag, execution_context)

            log_event(
                self.logger,
                "agent_runtime.executed",
                request_id=user_request.request_id,
                dag_id=dag.dag_id,
                bundle_status=result_bundle.status,
                result_count=len(result_bundle.results),
            )

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
