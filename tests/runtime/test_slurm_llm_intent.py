from __future__ import annotations

import getpass
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import PlannerConfig
from aor_runtime.runtime.capabilities.base import ClassificationContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.planner import TaskPlanner
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
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.call_count = 0

    def load_prompt(self, path: str | None, fallback: str) -> str:
        return fallback

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


def _planner(tmp_path: Path, llm: FakeLLM, **settings_overrides: object) -> TaskPlanner:
    settings = _settings(tmp_path, **settings_overrides)
    return TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)


def _classification_context(tmp_path: Path, **settings_overrides: object) -> ClassificationContext:
    return ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path, **settings_overrides))


def test_cluster_busy_maps_to_metrics_intent(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.91, "arguments": {"metric_group": "cluster_summary", "output_mode": "json"}, "reason": "Broad cluster summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Is the cluster busy right now?",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "Is the cluster busy right now?"},
    )

    assert [step.action for step in plan.steps] == ["slurm.metrics", "runtime.return"]
    assert planner.last_planning_mode == "llm_intent_extractor"
    assert planner.last_llm_calls == 1
    assert planner.last_llm_intent_calls == 1
    assert planner.last_raw_planner_llm_calls == 0


def test_gpu_availability_maps_to_gpu_summary(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.9, "arguments": {"metric_group": "gpu_summary", "output_mode": "json"}, "reason": "GPU availability."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Are GPUs available?",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "Are GPUs available?"},
    )

    assert plan.steps[0].action == "slurm.metrics"
    assert plan.steps[0].args["metric_group"] == "gpu_summary"


def test_failed_recently_maps_to_accounting_failed(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmAccountingIntent", "confidence": 0.87, "arguments": {"state": "FAILED", "output_mode": "json"}, "reason": "Recent failures."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="What failed recently?",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "What failed recently?"},
    )

    assert [step.action for step in plan.steps] == ["slurm.accounting", "runtime.return"]
    assert plan.steps[0].args["state"] == "FAILED"


def test_stuck_jobs_maps_to_pending_queue_for_current_user(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmQueueIntent", "confidence": 0.84, "arguments": {"output_mode": "json"}, "reason": "Pending queue is safest."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Are my jobs stuck?",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "Are my jobs stuck?"},
    )

    assert [step.action for step in plan.steps] == ["slurm.queue", "runtime.return"]
    assert plan.steps[0].args["state"] == "PENDING"
    assert plan.steps[0].args["user"] == getpass.getuser()


def test_slurmdbd_health_maps_to_dedicated_intent(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmDBDHealthIntent", "confidence": 0.96, "arguments": {"output_mode": "json"}, "reason": "Explicit SLURMDBD health check."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Is slurmdbd healthy?",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "Is slurmdbd healthy?"},
    )

    assert [step.action for step in plan.steps] == ["slurm.slurmdbd_health", "runtime.return"]


def test_mutating_cancel_request_is_rejected_without_slurm_tool(tmp_path: Path) -> None:
    llm = FakeLLM()
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Cancel my job 123",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "Cancel my job 123"},
    )

    assert [step.action for step in plan.steps] == ["runtime.return"]
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0


def test_mutating_drain_request_is_rejected_without_slurm_tool(tmp_path: Path) -> None:
    llm = FakeLLM()
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Drain node slurm-worker-agatha",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "Drain node slurm-worker-agatha"},
    )

    assert [step.action for step in plan.steps] == ["runtime.return"]
    assert planner.last_llm_calls == 0


def test_malicious_llm_argument_payload_is_rejected(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmQueueIntent", "confidence": 0.88, "arguments": {"state": "PENDING; rm -rf /", "output_mode": "json"}, "reason": "unsafe"}'
        ]
    )
    pack = SlurmCapabilityPack()
    planner = _planner(tmp_path, llm)
    result = pack.try_llm_extract("Are my jobs stuck?", _classification_context(tmp_path), planner.llm_intent_extractor)

    assert result.matched is False


def test_deterministic_prompt_still_uses_zero_llm_calls(tmp_path: Path) -> None:
    llm = FakeLLM()
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="show slurm queue",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SLURM_ALLOWED_TOOLS,
        input_payload={"task": "show slurm queue"},
    )

    assert [step.action for step in plan.steps] == ["slurm.queue", "runtime.return"]
    assert planner.last_llm_calls == 0
    assert planner.last_llm_intent_calls == 0
    assert planner.last_raw_planner_llm_calls == 0
    assert llm.call_count == 0


def test_fuzzy_prompt_does_not_use_llm_intent_extractor_when_disabled(tmp_path: Path) -> None:
    llm = FakeLLM()
    pack = SlurmCapabilityPack()
    context = _classification_context(tmp_path, enable_llm_intent_extraction=False)
    planner = _planner(tmp_path, llm, enable_llm_intent_extraction=False)

    result = pack.try_llm_extract("Is the cluster busy right now?", context, planner.llm_intent_extractor)

    assert result.matched is False
    assert llm.call_count == 0
