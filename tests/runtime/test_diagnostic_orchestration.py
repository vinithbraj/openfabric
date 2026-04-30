from __future__ import annotations

from aor_runtime.runtime.diagnostic_orchestration import diagnostic_plan_for_goal, is_broad_diagnostic_goal


def test_broad_diagnostic_goal_is_detected() -> None:
    goal = (
        "Run a read-only end-to-end diagnostic: summarize current workspace files, "
        "active AOR config flags found in the repo, available SQL-related capabilities, "
        "available filesystem capabilities, and available shell inspection capabilities."
    )

    assert is_broad_diagnostic_goal(goal)
    plan = diagnostic_plan_for_goal(goal)
    assert plan is not None
    assert plan.budget.max_actions == 8
    assert [section.name for section in plan.sections] == [
        "workspace summary",
        "OpenFABRIC config flags",
        "SQL capabilities",
        "filesystem capabilities",
        "shell inspection capabilities",
    ]


def test_normal_single_domain_prompt_is_not_diagnostic() -> None:
    assert not is_broad_diagnostic_goal("count the number of patients in dicom")
