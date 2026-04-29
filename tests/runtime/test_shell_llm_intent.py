from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import PlannerConfig
from aor_runtime.runtime.capabilities.base import ClassificationContext
from aor_runtime.runtime.capabilities.shell import ShellCapabilityPack
from aor_runtime.runtime.planner import ACTIVE_PLANNING_MODE, TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


SHELL_ALLOWED_TOOLS = ["shell.exec", "fs.search_content", "fs.aggregate", "python.exec"]


class FakeLLM:
    def __init__(self, responses: list[str | dict] | None = None) -> None:
        self.responses = [json.dumps(response) if isinstance(response, dict) else response for response in list(responses or [])]
        self.call_count = 0

    def load_prompt(self, path: str | None, fallback: str) -> str:
        raise AssertionError("Legacy planner prompt should not be loaded")

    def complete(self, **_: object) -> str:
        self.call_count += 1
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return self.responses.pop(0)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    payload = {
        "workspace_root": tmp_path,
        "run_store_path": tmp_path / "runtime.db",
        "available_nodes_raw": "edge-1,edge-2",
        "default_node": "edge-1",
        "enable_llm_intent_extraction": True,
    }
    payload.update(overrides)
    return Settings(**payload)


def _classification_context(tmp_path: Path, **settings_overrides: object) -> ClassificationContext:
    return ClassificationContext(
        schema_payload=None,
        allowed_tools=SHELL_ALLOWED_TOOLS,
        settings=_settings(tmp_path, **settings_overrides),
    )


def _shell_action_plan(command: str) -> dict:
    return {
        "goal": "Run shell inspection.",
        "actions": [
            {
                "id": "inspect",
                "tool": "shell.exec",
                "purpose": "Run safe shell inspection.",
                "inputs": {"command": command},
                "output_binding": "inspection",
                "expected_result_shape": {"kind": "text"},
            },
            {
                "id": "return_result",
                "tool": "runtime.return",
                "purpose": "Return stdout.",
                "inputs": {"value": {"$ref": "inspection", "path": "stdout"}, "mode": "text"},
                "depends_on": ["inspect"],
                "output_binding": "runtime_return_result",
                "expected_result_shape": {"kind": "text"},
            },
        ],
        "expected_final_shape": {"kind": "text"},
        "notes": [],
    }


def test_task_planner_uses_action_planner_for_shell_prompt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM([_shell_action_plan("uptime")])
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="check uptime",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "check uptime"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "text.format", "runtime.return"]
    assert plan.steps[0].args["command"] == "uptime"
    assert planner.last_planning_mode == ACTIVE_PLANNING_MODE
    assert planner.last_llm_calls == 1
    assert planner.last_raw_planner_llm_calls == 0


def test_shell_pack_llm_extractor_remains_available_as_helper(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.93, '
            '"arguments": {"kind": "uptime"}, "reason": "Read-only uptime summary."}'
        ]
    )
    settings = _settings(tmp_path)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    result = ShellCapabilityPack().try_llm_extract("check uptime", _classification_context(tmp_path), planner.llm_intent_extractor)

    assert result.matched is True
    assert result.intent.__class__.__name__ == "ShellInspectionIntent"


def test_disabled_flag_skips_shell_llm_intent_extraction(tmp_path: Path) -> None:
    llm = FakeLLM()
    settings = _settings(tmp_path, enable_llm_intent_extraction=False)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    context = _classification_context(tmp_path, enable_llm_intent_extraction=False)

    result = ShellCapabilityPack().try_llm_extract("check uptime", context, planner.llm_intent_extractor)

    assert result.matched is False
    assert llm.call_count == 0
