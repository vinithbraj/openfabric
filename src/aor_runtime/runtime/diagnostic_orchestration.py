"""OpenFABRIC Runtime Module: aor_runtime.runtime.diagnostic_orchestration

Purpose:
    Bound broad multi-domain diagnostic prompts into compact staged sections.

Responsibilities:
    Recognize diagnostic goals and define budgets, sections, and partial-completion behavior.

Data flow / Interfaces:
    Provides planner-time guidance for filesystem, config, SQL, shell, and capability summaries.

Boundaries:
    Prevents broad diagnostics from monopolizing request time or dumping raw lists into final output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestBudget:
    """Represent request budget within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RequestBudget.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.diagnostic_orchestration.RequestBudget and related tests.
    """
    max_wall_time_seconds: int = 60
    max_llm_calls: int = 2
    max_actions: int = 8
    max_output_facts_per_section: int = 8

    def model_dump(self) -> dict[str, int]:
        """Model dump for RequestBudget instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through RequestBudget.model_dump calls and related tests.
        """
        return {
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_llm_calls": self.max_llm_calls,
            "max_actions": self.max_actions,
            "max_output_facts_per_section": self.max_output_facts_per_section,
        }


@dataclass(frozen=True)
class DiagnosticSectionPlan:
    """Represent diagnostic section plan within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by DiagnosticSectionPlan.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.diagnostic_orchestration.DiagnosticSectionPlan and related tests.
    """
    name: str
    allowed_tools: tuple[str, ...]
    instruction: str

    def model_dump(self) -> dict[str, Any]:
        """Model dump for DiagnosticSectionPlan instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through DiagnosticSectionPlan.model_dump calls and related tests.
        """
        return {"name": self.name, "allowed_tools": list(self.allowed_tools), "instruction": self.instruction}


@dataclass(frozen=True)
class DiagnosticPlan:
    """Represent diagnostic plan within the OpenFABRIC runtime.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by DiagnosticPlan.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.diagnostic_orchestration.DiagnosticPlan and related tests.
    """
    budget: RequestBudget = field(default_factory=RequestBudget)
    sections: tuple[DiagnosticSectionPlan, ...] = ()

    def model_dump(self) -> dict[str, Any]:
        """Model dump for DiagnosticPlan instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through DiagnosticPlan.model_dump calls and related tests.
        """
        return {"budget": self.budget.model_dump(), "sections": [section.model_dump() for section in self.sections]}


DIAGNOSTIC_SECTIONS: tuple[DiagnosticSectionPlan, ...] = (
    DiagnosticSectionPlan(
        name="workspace summary",
        allowed_tools=("fs.list", "fs.find", "fs.size"),
        instruction="Summarize top-level workspace facts; do not return raw file contents.",
    ),
    DiagnosticSectionPlan(
        name="OpenFABRIC config flags",
        allowed_tools=("fs.search_content",),
        instruction="Search for AOR_/OpenFABRIC config flags and summarize compactly.",
    ),
    DiagnosticSectionPlan(
        name="SQL capabilities",
        allowed_tools=("sql.schema",),
        instruction="Inspect configured SQL metadata only; do not execute expensive data queries.",
    ),
    DiagnosticSectionPlan(
        name="filesystem capabilities",
        allowed_tools=("fs.search_content",),
        instruction="Identify filesystem tool definitions and safety checks from source references.",
    ),
    DiagnosticSectionPlan(
        name="shell inspection capabilities",
        allowed_tools=("fs.search_content",),
        instruction="Identify shell inspection and safety policy components from source references.",
    ),
)


def is_broad_diagnostic_goal(goal: str) -> bool:
    """Is broad diagnostic goal for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.diagnostic_orchestration.is_broad_diagnostic_goal.
    """
    text = str(goal or "").lower()
    if not re.search(r"\b(?:diagnostic|summarize|summary|available|capabilities|config\s+flags)\b", text):
        return False
    domain_hits = sum(
        1
        for pattern in (
            r"\bworkspace\b|\brepository\b|\brepo\b|\bfiles\b",
            r"\bconfig\b|\bflags?\b|\baor_\b|\bopenfabric\b",
            r"\bsql\b|\bdatabase\b",
            r"\bfilesystem\b|\bfs\.",
            r"\bshell\b|\bcommand\b|\binspection\b",
        )
        if re.search(pattern, text)
    )
    return domain_hits >= 3


def diagnostic_plan_for_goal(goal: str) -> DiagnosticPlan | None:
    """Diagnostic plan for goal for the surrounding runtime workflow.

    Inputs:
        Receives goal for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.diagnostic_orchestration.diagnostic_plan_for_goal.
    """
    if not is_broad_diagnostic_goal(goal):
        return None
    return DiagnosticPlan(sections=DIAGNOSTIC_SECTIONS)
