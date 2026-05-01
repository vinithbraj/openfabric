"""Prompt classification and typed task decomposition interfaces."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import model_json_schema

from agent_runtime.core.types import TaskFrame, UserRequest
from agent_runtime.input_pipeline.plan_selection import CandidateEvaluation, select_best_candidate
from agent_runtime.llm.critique import critique_decomposition, critique_requires_repair
from agent_runtime.llm.proposals import (
    PromptClassificationProposal,
    TaskDecompositionProposal,
    collect_n_best_structured_attempts,
)
from agent_runtime.llm.reproducibility import (
    PlanningTrace,
    PlanningTraceEntry,
    append_trace_entry,
    llm_client_metadata,
)
from agent_runtime.llm.structured_call import structured_call


class PromptClassification(BaseModel):
    """Typed prompt classification produced by the first input-pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    prompt_type: Literal[
        "simple_question",
        "simple_tool_task",
        "compound_tool_task",
        "complex_workflow",
        "ambiguous",
        "unsupported",
    ]
    requires_tools: bool
    likely_domains: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"]
    needs_clarification: bool
    clarification_question: str | None = None
    reason: str


class DecompositionResult(BaseModel):
    """Typed decomposition output for a user prompt."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskFrame] = Field(default_factory=list)
    global_constraints: dict[str, Any] = Field(default_factory=dict)
    unresolved_references: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> "DecompositionResult":
        """Ensure task dependencies point at task ids within the same result."""

        task_ids = {task.id for task in self.tasks}
        for task in self.tasks:
            for dependency in task.dependencies:
                if dependency not in task_ids:
                    raise ValueError(
                        f"Task {task.id} depends on unknown task id {dependency}."
                    )
        return self


class Decomposer:
    """Minimal decomposer that keeps a prompt as one task."""

    def decompose(self, request: UserRequest) -> DecompositionResult:
        """Return the request prompt as a single undecomposed task."""

        return DecompositionResult(
            tasks=[
                TaskFrame(
                    description=request.raw_prompt,
                    semantic_verb="unknown",
                    object_type="unknown",
                    intent_confidence=0.0,
                    constraints={},
                    raw_evidence=request.raw_prompt,
                )
            ]
        )


def _classification_schema_description() -> str:
    """Return a short human-readable summary of the classification schema."""

    return (
        "PromptClassification fields: "
        "prompt_type, requires_tools, likely_domains, risk_level, "
        "needs_clarification, clarification_question, reason."
    )


def _decomposition_schema_description() -> str:
    """Return a short human-readable summary of the decomposition schema."""

    return (
        "DecompositionResult fields: tasks, global_constraints, "
        "unresolved_references, assumptions. "
        "Each task is a TaskFrame with id, description, semantic_verb, "
        "object_type, intent_confidence, constraints, dependencies, "
        "raw_evidence, requires_confirmation, risk_level."
    )


def _build_classification_prompt(user_request: UserRequest) -> str:
    """Build the strict JSON-only prompt sent to the LLM classifier."""

    schema = model_json_schema(PromptClassification)
    return "\n".join(
        [
            "You are classifying a user prompt for an intelligent agent runtime.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not produce commands, shell syntax, SQL, code, or executable plans.",
            "Only classify the prompt. Do not solve it.",
            _classification_schema_description(),
            "JSON schema:",
            str(schema),
            "User prompt:",
            user_request.raw_prompt,
        ]
    )


def _validate_classification_proposal(
    proposal: PromptClassificationProposal,
) -> tuple[PromptClassification, list[str]]:
    """Convert one untrusted classification proposal into the trusted model."""

    normalized = PromptClassification.model_validate(
        {
            "prompt_type": proposal.prompt_type,
            "requires_tools": proposal.requires_tools,
            "likely_domains": list(proposal.likely_domains),
            "risk_level": proposal.risk_level,
            "needs_clarification": proposal.needs_clarification,
            "clarification_question": proposal.clarification_question,
            "reason": proposal.reason,
        }
    )
    return normalized, [f"proposal_confidence={proposal.confidence}"] if proposal.confidence is not None else []


_SHELL_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcurrent working directory\b"),
    re.compile(r"\bpwd\b"),
    re.compile(r"\bgit status\b"),
    re.compile(r"\bwhich\b"),
    re.compile(r"\bprocess(?:es)?\b"),
    re.compile(r"\bport\b"),
    re.compile(r"\brun\b.*\breadonly tests\b"),
    re.compile(r"\brun\b.*\btests\b"),
)

_SYSTEM_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bfree memory\b"),
    re.compile(r"\bmemory usage\b"),
    re.compile(r"\bram\b"),
    re.compile(r"\bswap\b"),
    re.compile(r"\bdisk usage\b"),
    re.compile(r"\bfree disk\b"),
    re.compile(r"\bdisk space\b"),
    re.compile(r"\bcpu load\b"),
    re.compile(r"\bload average\b"),
    re.compile(r"\buptime\b"),
    re.compile(r"\benvironment summary\b"),
)

_RUNTIME_INTROSPECTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "runtime.describe_capabilities": (
        re.compile(r"\bwhat are my capabilities\b"),
        re.compile(r"\bwhat tools do you have\b"),
        re.compile(r"\bwhat can this runtime do\b"),
        re.compile(r"\bshow available capabilities\b"),
        re.compile(r"\blist registered capabilities\b"),
        re.compile(r"\bwhat operations are available\b"),
        re.compile(r"\bwhat tools are enabled\b"),
    ),
    "runtime.describe_pipeline": (
        re.compile(r"\bhow does this runtime work\b"),
        re.compile(r"\bshow pipeline\b"),
        re.compile(r"\bexplain your architecture\b"),
        re.compile(r"\bwhat stages do you run\b"),
    ),
    "runtime.show_last_plan": (
        re.compile(r"\bshow last plan\b"),
        re.compile(r"\bwhat plan did you make\b"),
        re.compile(r"\bshow the dag\b"),
        re.compile(r"\bwhy did you choose that tool\b"),
    ),
    "runtime.explain_last_failure": (
        re.compile(r"\bwhy did that fail\b"),
        re.compile(r"\bexplain the last error\b"),
        re.compile(r"\bwhat went wrong\b"),
    ),
}


def _looks_like_shell_tool_prompt(raw_prompt: str) -> bool:
    """Return whether a prompt strongly implies a shell/system inspection tool."""

    lowered = str(raw_prompt or "").strip().lower()
    if not lowered:
        return False
    return any(pattern.search(lowered) for pattern in _SHELL_TOOL_PATTERNS)


def _runtime_introspection_capability_id(raw_prompt: str) -> str | None:
    """Return the runtime introspection capability implied by one prompt, if any."""

    lowered = str(raw_prompt or "").strip().lower()
    if not lowered:
        return None
    for capability_id, patterns in _RUNTIME_INTROSPECTION_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            return capability_id
    return None


def _looks_like_system_tool_prompt(raw_prompt: str) -> bool:
    """Return whether a prompt strongly implies a safe system-inspection capability."""

    lowered = str(raw_prompt or "").strip().lower()
    if not lowered:
        return False
    return any(pattern.search(lowered) for pattern in _SYSTEM_TOOL_PATTERNS)


def _normalize_classification(
    user_request: UserRequest,
    classification: PromptClassification,
) -> PromptClassification:
    """Apply conservative deterministic normalization after LLM classification."""

    if classification.needs_clarification:
        return classification

    if _looks_like_shell_tool_prompt(user_request.raw_prompt) and (
        classification.prompt_type == "simple_question" or not classification.requires_tools
    ):
        likely_domains = list(classification.likely_domains)
        if "shell" not in {domain.strip().lower() for domain in likely_domains}:
            likely_domains.append("shell")
        return classification.model_copy(
            update={
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": likely_domains,
                "reason": f"{classification.reason} Deterministically normalized to a shell tool task.",
            }
        )

    if _looks_like_system_tool_prompt(user_request.raw_prompt) and (
        classification.prompt_type == "simple_question" or not classification.requires_tools
    ):
        likely_domains = list(classification.likely_domains)
        if "system" not in {domain.strip().lower() for domain in likely_domains}:
            likely_domains.append("system")
        return classification.model_copy(
            update={
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": likely_domains,
                "reason": f"{classification.reason} Deterministically normalized to a system inspection tool task.",
            }
        )

    runtime_capability = _runtime_introspection_capability_id(user_request.raw_prompt)
    if runtime_capability is not None:
        likely_domains = list(classification.likely_domains)
        if "runtime" not in {domain.strip().lower() for domain in likely_domains}:
            likely_domains.append("runtime")
        return classification.model_copy(
            update={
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": likely_domains,
                "reason": (
                    f"{classification.reason} Deterministically normalized to runtime introspection via "
                    f"{runtime_capability}."
                ),
            }
        )

    return classification


def classify_prompt(user_request: UserRequest, llm_client) -> PromptClassification:
    """Classify a user prompt through a strict structured LLM call."""

    prompt = _build_classification_prompt(user_request)
    raw_response = llm_client.complete_json(prompt, PromptClassificationProposal.model_json_schema())
    proposal = PromptClassificationProposal.model_validate(raw_response)
    classification, normalizations = _validate_classification_proposal(proposal)
    normalized = _normalize_classification(user_request, classification)
    if normalized != classification:
        normalizations.append("normalized_shell_tool_prompt")
    trace = user_request.safety_context.get("planning_trace")
    if isinstance(trace, PlanningTrace):
        model_name, temperature = llm_client_metadata(llm_client)
        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="prompt_classification",
                request_id=user_request.request_id,
                model_name=model_name,
                llm_temperature=temperature,
                prompt_template_id="prompt_classification",
                raw_llm_response=raw_response,
                parsed_proposal=proposal.model_dump(mode="json"),
                selected_candidate=normalized.model_dump(mode="json"),
                deterministic_normalizations=normalizations,
            ),
        )
    return normalized


def _build_decomposition_prompt(
    user_request: UserRequest,
    classification: PromptClassification,
    critique_feedback: dict[str, Any] | None = None,
) -> str:
    """Build the strict JSON-only prompt sent to the LLM decomposer."""

    schema = model_json_schema(DecompositionResult)
    return "\n".join(
        [
            "You are decomposing a user prompt into atomic user-meaningful tasks.",
            "Return JSON only.",
            "Do not return markdown fences.",
            "Do not select tools.",
            "Do not generate commands.",
            "Do not generate SQL.",
            "Do not generate shell syntax.",
            "Break compound prompts into atomic user-meaningful tasks.",
            "Preserve ordering and dependencies.",
            "Extract global constraints such as limits, date ranges, output preferences, paths, table names, row limits, and safety constraints.",
            "Use task dependencies to express ordering.",
            "Only describe meaning. Do not solve the task.",
            _decomposition_schema_description(),
            "Prompt classification context:",
            str(classification.model_dump()),
            *(
                [
                    "Critique feedback from a previous proposal attempt:",
                    str(critique_feedback),
                ]
                if critique_feedback
                else []
            ),
            "JSON schema:",
            str(schema),
            "User prompt:",
            user_request.raw_prompt,
        ]
    )


def _normalize_decomposition_proposal(
    proposal: TaskDecompositionProposal,
) -> DecompositionResult:
    """Convert one untrusted decomposition proposal into the trusted model."""

    tasks = [TaskFrame.model_validate(task.model_dump(mode="json")) for task in proposal.tasks]
    return DecompositionResult(
        tasks=tasks,
        global_constraints=dict(proposal.global_constraints),
        unresolved_references=list(proposal.unresolved_references),
        assumptions=list(proposal.assumptions),
    )


_RISK_SCORES: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def _evaluate_decomposition_attempts(
    attempts,
) -> list[CandidateEvaluation[DecompositionResult]]:
    """Evaluate N-best decomposition attempts deterministically."""

    evaluations: list[CandidateEvaluation[DecompositionResult]] = []
    for attempt in attempts:
        evaluation = CandidateEvaluation[DecompositionResult](
            raw_response=attempt.raw_llm_response,
            validation_errors=list(attempt.validation_errors),
        )
        if attempt.parsed_proposal is None or attempt.validation_errors:
            evaluations.append(evaluation)
            continue
        proposal = attempt.parsed_proposal
        try:
            normalized = _normalize_decomposition_proposal(proposal)
            evaluation.proposal = normalized
            evaluation.confidence = float(proposal.confidence or 0.0)
            evaluation.assumption_count = len(proposal.assumptions)
            evaluation.unresolved_count = len(proposal.unresolved_references)
            evaluation.risk_score = sum(
                _RISK_SCORES.get(task.risk_level.strip().lower(), 0) for task in proposal.tasks
            )
        except Exception as exc:
            evaluation.validation_errors.append(str(exc))
        evaluations.append(evaluation)
    return evaluations


def decompose_prompt(
    user_request: UserRequest,
    classification: PromptClassification,
    llm_client,
    available_domains: list[str] | None = None,
    n_best: int = 3,
) -> DecompositionResult:
    """Decompose a prompt into ordered atomic tasks through a structured LLM call."""

    prompt = _build_decomposition_prompt(user_request, classification)
    attempts = collect_n_best_structured_attempts(
        llm_client=llm_client,
        system_prompt=prompt,
        user_payload={
            "prompt": user_request.raw_prompt,
            "classification": classification.model_dump(mode="json"),
        },
        output_model=TaskDecompositionProposal,
        n=n_best,
    )
    evaluations = _evaluate_decomposition_attempts(attempts)
    selected = select_best_candidate(evaluations)
    if selected is None or selected.proposal is None or not selected.is_valid:
        errors = [
            error
            for evaluation in evaluations
            for error in (evaluation.validation_errors + evaluation.rejection_reasons)
        ]
        raise ValueError(
            "No valid decomposition proposal was accepted."
            + (f" Errors: {' | '.join(errors)}" if errors else "")
        )

    available_domains = available_domains or []
    selected_proposal = TaskDecompositionProposal.model_validate(
        attempts[evaluations.index(selected)].parsed_proposal.model_dump(mode="json")
        if attempts[evaluations.index(selected)].parsed_proposal is not None
        else {}
    )
    critique = critique_decomposition(user_request, selected_proposal, available_domains, llm_client)
    repaired = False
    if critique_requires_repair(critique):
        repair_prompt = _build_decomposition_prompt(
            user_request,
            classification,
            critique_feedback=critique.model_dump(mode="json"),
        )
        repair_attempts = collect_n_best_structured_attempts(
            llm_client=llm_client,
            system_prompt=repair_prompt,
            user_payload={
                "prompt": user_request.raw_prompt,
                "classification": classification.model_dump(mode="json"),
                "critique": critique.model_dump(mode="json"),
            },
            output_model=TaskDecompositionProposal,
            n=1,
        )
        repair_evaluations = _evaluate_decomposition_attempts(repair_attempts)
        repaired_candidate = select_best_candidate(repair_evaluations)
        if repaired_candidate is not None and repaired_candidate.is_valid and repaired_candidate.proposal is not None:
            selected = repaired_candidate
            repaired = True

    trace = user_request.safety_context.get("planning_trace")
    if isinstance(trace, PlanningTrace):
        model_name, temperature = llm_client_metadata(llm_client)
        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="task_decomposition",
                request_id=user_request.request_id,
                model_name=model_name,
                llm_temperature=temperature,
                prompt_template_id="task_decomposition",
                raw_llm_response=[attempt.raw_llm_response for attempt in attempts],
                parsed_proposal=[
                    attempt.parsed_proposal.model_dump(mode="json")
                    if attempt.parsed_proposal is not None
                    else None
                    for attempt in attempts
                ],
                validation_errors=[
                    error for attempt in attempts for error in attempt.validation_errors
                ],
                selected_candidate=selected.proposal.model_dump(mode="json"),
                rejection_reasons=[
                    reason
                    for evaluation in evaluations
                    if evaluation is not selected
                    for reason in (evaluation.rejection_reasons + evaluation.validation_errors)
                ],
                deterministic_normalizations=(
                    ["repaired_after_critique"] if repaired else []
                ),
            ),
        )
        append_trace_entry(
            trace,
            PlanningTraceEntry(
                stage="task_decomposition_critique",
                request_id=user_request.request_id,
                model_name=model_name,
                llm_temperature=temperature,
                prompt_template_id="task_decomposition_critique",
                raw_llm_response=critique.model_dump(mode="json"),
                parsed_proposal=critique.model_dump(mode="json"),
                selected_candidate=selected.proposal.model_dump(mode="json"),
            ),
        )
    return selected.proposal
