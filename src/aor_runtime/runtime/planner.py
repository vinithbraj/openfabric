"""OpenFABRIC Runtime Module: aor_runtime.runtime.planner

Purpose:
    Wrap the LLM action planner as the single natural-language planning path.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import re
from typing import Any

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.llm.client import LLMClient
from aor_runtime.runtime.action_planner import LLMActionPlanner
from aor_runtime.runtime.dataflow import normalize_execution_plan_dataflow
from aor_runtime.runtime.llm_intent_extractor import LLMIntentExtractor
from aor_runtime.runtime.plan_canonicalizer import canonicalize_plan
from aor_runtime.runtime.policies import (
    PlanContractViolation,
    classify_plan_violations,
    validate_plan_contract,
)
from aor_runtime.runtime.semantic_frame import SemanticCompilationResult, SemanticFramePlanner, semantic_frame_mode
from aor_runtime.tools.base import ToolRegistry
from aor_runtime.tools.sql import resolve_sql_databases


INTERNAL_ALLOWED_TOOLS = {"runtime.return"}
ACTIVE_PLANNING_MODE = "validator_enforced_action_planner"


DEFAULT_PLANNER_PROMPT = """Legacy raw ExecutionPlan prompt retired.

The runtime no longer loads this prompt. Natural-language planning is handled
by LLMActionPlanner, which emits structured action plans that are normalized,
validated, compiled, executed, and rendered by deterministic runtime code.
"""

DATABASE_NAME_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*_db\b")
STORAGE_TOKEN_RE = re.compile(r"[a-z0-9_]+")
DU_COMMAND_RE = re.compile(r"\bdu\b")
DF_COMMAND_RE = re.compile(r"\bdf\b")
PLANNER_RAW_OUTPUT_PREVIEW_CHARS = 600
TOOL_INTENT_PATTERNS = {
    "shell.exec": [r"\b(?:using|use|with)\s+shell(?:\.exec)?\b"],
    "python.exec": [r"\b(?:using|use|with)\s+python(?:\.exec)?\b"],
    "sql.query": [r"\b(?:using|use|with)\s+sql(?:\.query)?\b"],
    "fs.*": [r"\b(?:using|use|with)\s+(?:filesystem|fs)\b"],
}
SHELL_NATIVE_FIRST_INTENT_RE = re.compile(
    r"\b(?:using|use|with)\s+shell(?:\.exec)?\b.*\b(?:total\s+file\s+size|total\s+size|sum(?:\s+the)?\s+size|how\s+much\s+space|disk\s+space\s+used\s+by|size\s+of\s+all|total\s+bytes|count\s+and\s+total\s+size|matching\s+lines?|lines?\s+containing|files?\s+containing|contents?\s+(?:include|contain)|mention(?:ing)?)\b",
    re.IGNORECASE,
)
SHELL_NATIVE_FIRST_FETCH_RE = re.compile(
    r"(?:\b(?:fetch|curl)\b.*\b(?:title|head)\b.*\b(?:using|use|with)\s+shell(?:\.exec)?\b|\b(?:using|use|with)\s+shell(?:\.exec)?\b.*\b(?:fetch|curl)\b.*\b(?:title|head)\b)",
    re.IGNORECASE,
)
FILESYSTEM_TOOL_INTENT_PATTERNS = {
    "fs.copy": [r"\b(?:using|use|with)\s+fs\.copy\b"],
    "fs.exists": [r"\b(?:using|use|with)\s+fs\.exists\b"],
    "fs.find": [r"\b(?:using|use|with)\s+fs\.find\b"],
    "fs.list": [r"\b(?:using|use|with)\s+fs\.list\b"],
    "fs.mkdir": [r"\b(?:using|use|with)\s+fs\.mkdir\b"],
    "fs.not_exists": [r"\b(?:using|use|with)\s+fs\.not_exists\b"],
    "fs.read": [r"\b(?:using|use|with)\s+fs\.read\b"],
    "fs.size": [r"\b(?:using|use|with)\s+fs\.size\b"],
    "fs.write": [r"\b(?:using|use|with)\s+fs\.write\b"],
}


def summarize_plan(plan: ExecutionPlan) -> str:
    """Summarize plan for the surrounding runtime workflow.

    Inputs:
        Receives plan for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.planner.summarize_plan.
    """
    actions = [step.action for step in plan.steps]
    return f"Plan with {len(actions)} steps: " + ", ".join(actions)


def summarize_planner_raw_output(raw_output: str | None, limit: int = PLANNER_RAW_OUTPUT_PREVIEW_CHARS) -> str | None:
    """Summarize planner raw output for the surrounding runtime workflow.

    Inputs:
        Receives raw_output, limit for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.planner.summarize_planner_raw_output.
    """
    text = str(raw_output or "").strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_explicit_tool_intent(goal: str, allowed_tools: list[str]) -> list[str]:
    """Extract explicit tool intent for the surrounding runtime workflow.

    Inputs:
        Receives goal, allowed_tools for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.planner.extract_explicit_tool_intent.
    """
    goal_text = str(goal or "").lower()
    requested: list[str] = []
    for tool_name, patterns in TOOL_INTENT_PATTERNS.items():
        if tool_name != "fs.*" and tool_name not in allowed_tools:
            continue
        if tool_name == "fs.*" and not any(name.startswith("fs.") for name in allowed_tools):
            continue
        if tool_name == "shell.exec" and (SHELL_NATIVE_FIRST_INTENT_RE.search(goal_text) or SHELL_NATIVE_FIRST_FETCH_RE.search(goal_text)):
            continue
        if any(re.search(pattern, goal_text) for pattern in patterns):
            requested.append(tool_name)
    for tool_name, patterns in FILESYSTEM_TOOL_INTENT_PATTERNS.items():
        if tool_name not in allowed_tools:
            continue
        if any(re.search(pattern, goal_text) for pattern in patterns):
            requested.append(tool_name)
    return list(dict.fromkeys(requested))


class TaskPlanner:
    """Represent task planner within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by TaskPlanner.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.planner.TaskPlanner and related tests.
    """
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        settings: Settings | None = None,
        capability_registry: Any | None = None,
    ) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Receives llm, tools, settings, capability_registry for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner.__init__ calls and related tests.
        """
        self.llm = llm
        self.tools = tools
        self.settings = settings or get_settings()
        # Deprecated compatibility parameter. Natural-language planning is always
        # handled by LLMActionPlanner; capability registries are no longer a
        # runtime router.
        self.capability_registry = capability_registry
        # Kept only for capability-pack unit tests and transitional helper APIs.
        # TaskPlanner never uses it as a planning route.
        self.llm_intent_extractor = LLMIntentExtractor(llm=llm, settings=self.settings)
        self.last_policies_used: list[str] = []
        self.last_high_level_plan: list[str] | None = None
        self.last_planning_mode: str = ACTIVE_PLANNING_MODE
        self.last_llm_calls: int = 0
        self.last_llm_intent_calls: int = 0
        self.last_raw_planner_llm_calls: int = 0
        self.last_llm_intent_type: str | None = None
        self.last_llm_intent_confidence: float | None = None
        self.last_llm_intent_reason: str | None = None
        self.last_capability_name: str | None = None
        self.last_error_stage: str | None = None
        self.last_raw_output: str | None = None
        self.last_error_type: str | None = None
        self.last_original_execution_plan: dict[str, Any] | None = None
        self.last_canonicalized_execution_plan: dict[str, Any] | None = None
        self.last_plan_repairs: list[str] = []
        self.last_plan_canonicalized: bool = False
        self.last_capability_metadata: dict[str, Any] = {}
        self.last_semantic_compilation: SemanticCompilationResult | None = None

    def build_plan(
        self,
        *,
        goal: str,
        planner: PlannerConfig,
        allowed_tools: list[str],
        input_payload: dict[str, Any],
        failure_context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Build plan for TaskPlanner instances.

        Inputs:
            Receives goal, planner, allowed_tools, input_payload, failure_context for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner.build_plan calls and related tests.
        """
        self._reset_tracking()
        self.last_planning_mode = ACTIVE_PLANNING_MODE
        self.last_capability_name = "action_planner"
        self.last_policies_used = [ACTIVE_PLANNING_MODE]
        action_planner: LLMActionPlanner | None = None
        try:
            semantic_result = self._try_semantic_frame_plan(
                goal=goal,
                planner=planner,
                allowed_tools=allowed_tools,
            )
            if semantic_result is not None:
                finalized_semantic_plan = self._finalize_plan(
                    goal,
                    semantic_result.plan,
                    allowed_tools,
                    extract_explicit_tool_intent(goal, allowed_tools),
                    allow_internal_tools=INTERNAL_ALLOWED_TOOLS | {"text.format", "sql.schema", "sql.validate"},
                )
                self.last_capability_name = "semantic_frame"
                self.last_policies_used = [ACTIVE_PLANNING_MODE, "semantic_frame"]
                self.last_capability_metadata = self._semantic_frame_metadata(semantic_result)
                self.last_llm_calls = int(semantic_result.metadata.get("semantic_llm_calls") or 0)
                self.last_raw_planner_llm_calls = 0
                self.last_error_stage = None
                return finalized_semantic_plan
            action_planner = LLMActionPlanner(llm=self.llm, tools=self.tools, settings=self.settings)
            plan = action_planner.build_plan(
                goal=goal,
                planner=planner,
                allowed_tools=allowed_tools,
                input_payload=input_payload,
                failure_context=failure_context,
            )
            self.last_llm_calls = 1 + int(action_planner.last_temporal_llm_calls)
            self.last_raw_planner_llm_calls = 0
            self.last_raw_output = action_planner.last_raw_output
            self.last_capability_metadata = self._action_planner_metadata(action_planner)
            finalized_plan = self._finalize_plan(
                goal,
                plan,
                allowed_tools,
                extract_explicit_tool_intent(goal, allowed_tools),
                allow_internal_tools=INTERNAL_ALLOWED_TOOLS | {"text.format", "sql.schema", "sql.validate"},
            )
            self.last_error_stage = None
            return finalized_plan
        except Exception as exc:
            self.last_error_type = type(exc).__name__
            self.last_error_stage = "action_planner"
            if action_planner is not None:
                self.last_raw_output = action_planner.last_raw_output
                self.last_llm_calls = (1 if action_planner.last_raw_output is not None else 0) + int(
                    action_planner.last_temporal_llm_calls
                )
                self.last_raw_planner_llm_calls = 0
                self.last_capability_metadata = self._action_planner_metadata(action_planner)
            raise

    def _reset_tracking(self) -> None:
        """Handle the internal reset tracking helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._reset_tracking calls and related tests.
        """
        self.last_policies_used = []
        self.last_high_level_plan = None
        self.last_planning_mode = ACTIVE_PLANNING_MODE
        self.last_llm_calls = 0
        self.last_llm_intent_calls = 0
        self.last_raw_planner_llm_calls = 0
        self.last_llm_intent_type = None
        self.last_llm_intent_confidence = None
        self.last_llm_intent_reason = None
        self.last_capability_name = None
        self.last_error_stage = None
        self.last_raw_output = None
        self.last_error_type = None
        self.last_original_execution_plan = None
        self.last_canonicalized_execution_plan = None
        self.last_plan_repairs = []
        self.last_plan_canonicalized = False
        self.last_capability_metadata = {}
        self.last_semantic_compilation = None

    def _action_planner_metadata(self, action_planner: LLMActionPlanner) -> dict[str, Any]:
        """Handle the internal action planner metadata helper path for this module.

        Inputs:
            Receives action_planner for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._action_planner_metadata calls and related tests.
        """
        return {
            "capability_pack": "action_planner",
            "planning_mode": ACTIVE_PLANNING_MODE,
            "action_prompt": action_planner.last_prompt,
            "raw_action_plan": action_planner.last_raw_action_plan,
            "normalized_action_plan": action_planner.last_normalized_action_plan,
            "canonicalized_action_plan": action_planner.last_canonicalized_action_plan,
            "canonicalization_repairs": action_planner.last_canonicalization_repairs,
            "database_propagation_repairs": action_planner.last_database_propagation_repairs,
            "dataflow_canonicalization_repairs": action_planner.last_dataflow_canonicalization_repairs,
            "temporal_normalization_repairs": action_planner.last_temporal_normalization_repairs,
            "temporal_normalization_metadata": action_planner.last_temporal_normalization_metadata,
            "tool_argument_canonicalization": action_planner.last_tool_argument_canonicalization_metadata,
            "action_validation_errors": action_planner.last_validation_errors,
            "contract_validation_errors": action_planner.last_contract_validation_errors,
            "domain_validation_errors": action_planner.last_domain_validation_errors,
        }

    def _try_semantic_frame_plan(
        self,
        *,
        goal: str,
        planner: PlannerConfig,
        allowed_tools: list[str],
    ) -> SemanticCompilationResult | None:
        """Try to build an execution plan from a typed semantic frame.

        Inputs:
            Receives the user goal, planner config, and tool allow-list.

        Returns:
            A semantic compilation result, or None when fallback action planning should continue.

        Used by:
            TaskPlanner.build_plan before invoking the general LLM action planner.
        """
        if semantic_frame_mode(self.settings) == "off":
            return None
        semantic_planner = SemanticFramePlanner(settings=self.settings, llm=self.llm, allowed_tools=allowed_tools)
        result = semantic_planner.try_build_plan(
            goal=goal,
            model=planner.model,
            temperature=planner.temperature,
        )
        self.last_semantic_compilation = result
        return result

    def _semantic_frame_metadata(self, result: SemanticCompilationResult) -> dict[str, Any]:
        """Build planner metadata for semantic-frame compiled plans.

        Inputs:
            Receives a semantic compilation result.

        Returns:
            Metadata safe to persist in planner events and stats surfaces.

        Used by:
            TaskPlanner.build_plan when semantic-frame compilation succeeds.
        """
        return {
            "capability_pack": "semantic_frame",
            "planning_mode": ACTIVE_PLANNING_MODE,
            "semantic_frame_mode": result.metadata.get("semantic_frame_mode"),
            "semantic_frame_source": result.metadata.get("semantic_frame_source"),
            "semantic_strategy": result.metadata.get("semantic_strategy"),
            "semantic_targets": result.metadata.get("semantic_targets"),
            "semantic_frame": result.metadata.get("semantic_frame"),
            "semantic_coverage": result.metadata.get("semantic_coverage"),
        }

    def _finalize_plan(
        self,
        goal: str,
        plan: ExecutionPlan,
        allowed_tools: list[str],
        explicit_tool_intent: list[str],
        *,
        allow_internal_tools: set[str] | None = None,
        canonicalize_soft_violations: bool = True,
    ) -> ExecutionPlan:
        """Handle the internal finalize plan helper path for this module.

        Inputs:
            Receives goal, plan, allowed_tools, explicit_tool_intent, allow_internal_tools, canonicalize_soft_violations for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._finalize_plan calls and related tests.
        """
        self._apply_storage_shell_semantics(goal, plan)
        normalize_execution_plan_dataflow(plan)
        self.last_original_execution_plan = plan.model_dump()
        initial_violations = classify_plan_violations(plan, goal=goal)
        if initial_violations.hard:
            raise PlanContractViolation.from_violations(initial_violations)
        if initial_violations.soft and not canonicalize_soft_violations:
            raise PlanContractViolation.from_violations(initial_violations)

        if initial_violations.soft:
            canonicalized = canonicalize_plan(plan, goal, allowed_tools)
            plan = canonicalized.plan
            self.last_plan_repairs = list(canonicalized.repairs)
            self.last_plan_canonicalized = canonicalized.changed
            self.last_canonicalized_execution_plan = plan.model_dump() if canonicalized.changed else None
            normalize_execution_plan_dataflow(plan)
        else:
            self.last_plan_repairs = []
            self.last_plan_canonicalized = False
            self.last_canonicalized_execution_plan = None

        permitted_actions = set(allowed_tools)
        permitted_actions.update(allow_internal_tools or set())
        for step in plan.steps:
            if step.action not in permitted_actions:
                raise ValueError(f"Planner selected disallowed tool {step.action!r}.")
            self.tools.validate_step(step.action, step.args)
        self._validate_explicit_database_targets(goal, plan)
        self._validate_shell_targets(plan)
        self._validate_explicit_tool_intent(plan, explicit_tool_intent)
        validate_plan_contract(plan, goal=goal)
        return plan

    def _validate_explicit_tool_intent(self, plan: ExecutionPlan, explicit_tool_intent: list[str]) -> None:
        """Handle the internal validate explicit tool intent helper path for this module.

        Inputs:
            Receives plan, explicit_tool_intent for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._validate_explicit_tool_intent calls and related tests.
        """
        if not explicit_tool_intent:
            return
        actions = [step.action for step in plan.steps]
        for requested_tool in explicit_tool_intent:
            if requested_tool == "fs.*":
                if not any(action.startswith("fs.") for action in actions):
                    raise ValueError("Planner ignored the explicit filesystem tool request.")
                continue
            if requested_tool not in actions:
                raise ValueError(f"Planner ignored the explicit tool request for {requested_tool}.")

    def _validate_explicit_database_targets(self, goal: str, plan: ExecutionPlan) -> None:
        """Handle the internal validate explicit database targets helper path for this module.

        Inputs:
            Receives goal, plan for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._validate_explicit_database_targets calls and related tests.
        """
        configured_databases = resolve_sql_databases(self.settings)
        if not configured_databases:
            return

        goal_text = str(goal or "").lower()
        requested_names = {match.group(0).lower() for match in DATABASE_NAME_RE.finditer(goal_text)}
        for database_name in configured_databases:
            if database_name.lower() in goal_text or database_name.replace("_", " ").lower() in goal_text:
                requested_names.add(database_name.lower())

        if not requested_names:
            return

        for step in plan.steps:
            if step.action != "sql.query":
                continue
            database_name = step.args.get("database")
            if not isinstance(database_name, str) or not database_name.strip():
                raise ValueError("Planner must include an explicit database name for SQL steps when the goal names a database.")
            normalized = re.sub(r"[^a-z0-9_]+", "_", database_name.strip().lower()).strip("_")
            if normalized not in requested_names:
                requested = ", ".join(sorted(requested_names))
                raise ValueError(f"Planner changed the requested database target. Expected one of: {requested}.")

    def _validate_shell_targets(self, plan: ExecutionPlan) -> None:
        """Handle the internal validate shell targets helper path for this module.

        Inputs:
            Receives plan for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._validate_shell_targets calls and related tests.
        """
        allowed_nodes = self.settings.available_nodes
        default_node = str(self.settings.resolved_default_node() or "").strip()
        for step in plan.steps:
            if step.action != "shell.exec":
                continue
            node = str(step.args.get("node", "")).strip()
            if node and node not in allowed_nodes:
                allowed = ", ".join(allowed_nodes) or "<none configured>"
                raise ValueError(f"Planner selected a disallowed node {node!r}. Available nodes: {allowed}.")
            if not node and not default_node:
                raise ValueError(
                    "Planner must include an explicit node name for shell.exec steps when no default node is configured."
                )

    def _apply_storage_shell_semantics(self, goal: str, plan: ExecutionPlan) -> None:
        """Handle the internal apply storage shell semantics helper path for this module.

        Inputs:
            Receives goal, plan for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns None; side effects are limited to the local runtime operation described above.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._apply_storage_shell_semantics calls and related tests.
        """
        intent = self._classify_storage_intent(goal)
        if intent is None:
            return

        preferred_command = self._preferred_storage_command(intent)
        if preferred_command is None:
            return

        for step in plan.steps:
            if step.action != "shell.exec":
                continue
            command = str(step.args.get("command", ""))
            if intent.startswith("folder_usage"):
                if not DU_COMMAND_RE.search(command):
                    step.args["command"] = preferred_command
                return
            if intent == "filesystem_capacity":
                if not DF_COMMAND_RE.search(command):
                    step.args["command"] = preferred_command
                return

    def _classify_storage_intent(self, goal: str) -> str | None:
        """Handle the internal classify storage intent helper path for this module.

        Inputs:
            Receives goal for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._classify_storage_intent calls and related tests.
        """
        tokens = set(STORAGE_TOKEN_RE.findall(str(goal or "").lower()))
        if not tokens:
            return None

        folder_terms = {"folder", "folders", "directory", "directories"}
        filesystem_terms = {"disk", "disks", "filesystem", "filesystems", "partition", "partitions", "mount", "mounted", "drive", "drives"}
        usage_terms = {"space", "size", "usage", "used", "consuming", "largest", "biggest", "heaviest", "most"}
        system_scope_terms = {"computer", "system", "root", "whole", "entire"}

        mentions_usage = bool(tokens & usage_terms)
        mentions_folder = bool(tokens & folder_terms)
        mentions_filesystem = bool(tokens & filesystem_terms)
        mentions_system_scope = bool(tokens & system_scope_terms)

        if mentions_folder and mentions_usage:
            return "folder_usage_system" if mentions_system_scope else "folder_usage_workspace"
        if mentions_filesystem and mentions_usage:
            return "filesystem_capacity"
        return None

    def _preferred_storage_command(self, intent: str) -> str | None:
        """Handle the internal preferred storage command helper path for this module.

        Inputs:
            Receives intent for this TaskPlanner method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through TaskPlanner._preferred_storage_command calls and related tests.
        """
        if intent == "folder_usage_workspace":
            return "du -sh * | sort -hr"
        if intent == "folder_usage_system":
            return "du -xhd 1 / 2>/dev/null | sort -hr"
        if intent == "filesystem_capacity":
            return "df -h /"
        return None
