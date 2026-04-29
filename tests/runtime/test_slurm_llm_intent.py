from __future__ import annotations

import json
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import PlannerConfig
from aor_runtime.runtime.capabilities.base import ClassificationContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.planner import ACTIVE_PLANNING_MODE, TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


SLURM_ALLOWED_TOOLS = [
    "slurm.queue",
    "slurm.job_detail",
    "slurm.nodes",
    "slurm.node_detail",
    "slurm.partitions",
    "slurm.accounting",
    "slurm.metrics",
    "slurm.slurmdbd_health",
]


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
    return ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path, **settings_overrides))


def _slurm_metrics_action_plan() -> dict:
    return {
        "goal": "Inspect SLURM cluster status.",
        "actions": [
            {
                "id": "metrics",
                "tool": "slurm.metrics",
                "purpose": "Get cluster metrics.",
                "inputs": {"metric_group": "cluster_summary"},
                "output_binding": "cluster_metrics",
                "expected_result_shape": {"kind": "json"},
            },
            {
                "id": "return_result",
                "tool": "runtime.return",
                "purpose": "Return metrics.",
                "inputs": {"value": {"$ref": "cluster_metrics", "path": "payload"}, "mode": "markdown"},
                "depends_on": ["metrics"],
                "output_binding": "runtime_return_result",
                "expected_result_shape": {"kind": "text"},
            },
        ],
        "expected_final_shape": {"kind": "text"},
        "notes": [],
    }


def test_task_planner_uses_action_planner_for_slurm_prompt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM([_slurm_metrics_action_plan()])
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    plan = planner.build_plan(
        goal="How saturated is the HPC scheduler right now?",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "How saturated is the HPC scheduler right now?"},
    )

    assert [step.action for step in plan.steps] == ["slurm.metrics", "text.format", "runtime.return"]
    assert planner.last_planning_mode == ACTIVE_PLANNING_MODE
    assert planner.last_llm_calls == 1
    assert planner.last_raw_planner_llm_calls == 0


def test_slurm_pack_llm_extractor_remains_available_as_helper(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.91, '
            '"arguments": {"metric_group": "cluster_summary", "output_mode": "json"}, '
            '"reason": "Broad cluster summary."}'
        ]
    )
    settings = _settings(tmp_path)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)

    result = SlurmCapabilityPack().try_llm_extract(
        "Is the cluster busy right now?",
        _classification_context(tmp_path),
        planner.llm_intent_extractor,
    )

    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmMetricsIntent"


def test_fuzzy_prompt_does_not_use_llm_intent_extractor_when_disabled(tmp_path: Path) -> None:
    llm = FakeLLM()
    settings = _settings(tmp_path, enable_llm_intent_extraction=False)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    context = _classification_context(tmp_path, enable_llm_intent_extraction=False)

    result = SlurmCapabilityPack().try_llm_extract("Is the cluster busy right now?", context, planner.llm_intent_extractor)

    assert result.matched is False
    assert llm.call_count == 0
