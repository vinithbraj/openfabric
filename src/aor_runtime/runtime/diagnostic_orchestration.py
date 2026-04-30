from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestBudget:
    max_wall_time_seconds: int = 60
    max_llm_calls: int = 2
    max_actions: int = 8
    max_output_facts_per_section: int = 8

    def model_dump(self) -> dict[str, int]:
        return {
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_llm_calls": self.max_llm_calls,
            "max_actions": self.max_actions,
            "max_output_facts_per_section": self.max_output_facts_per_section,
        }


@dataclass(frozen=True)
class DiagnosticSectionPlan:
    name: str
    allowed_tools: tuple[str, ...]
    instruction: str

    def model_dump(self) -> dict[str, Any]:
        return {"name": self.name, "allowed_tools": list(self.allowed_tools), "instruction": self.instruction}


@dataclass(frozen=True)
class DiagnosticPlan:
    budget: RequestBudget = field(default_factory=RequestBudget)
    sections: tuple[DiagnosticSectionPlan, ...] = ()

    def model_dump(self) -> dict[str, Any]:
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
    if not is_broad_diagnostic_goal(goal):
        return None
    return DiagnosticPlan(sections=DIAGNOSTIC_SECTIONS)
