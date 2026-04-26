from __future__ import annotations

import os
from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.slurm import SlurmCapabilityPack
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.eval_fixtures import rebuild_eval_workspace
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry
from aor_runtime.tools.slurm import SLURM_FIXTURE_DIR_ENV


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"
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
        self.user_prompts: list[str] = []

    def load_prompt(self, path: str | None, fallback: str) -> str:
        return fallback

    def complete(self, **_: object) -> str:
        self.user_prompts.append("called")
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.user_prompts)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        available_nodes_raw="edge-1,edge-2",
        default_node="edge-1",
    )


def _context(tmp_path: Path) -> ClassificationContext:
    return ClassificationContext(schema_payload=None, allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path))


def test_classify_show_queue(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show slurm queue", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmQueueIntent"


def test_classify_show_running_jobs(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show running slurm jobs", _context(tmp_path))
    assert result.matched is True
    assert result.intent.state == "RUNNING"


def test_classify_count_pending_jobs(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("count pending jobs in slurm", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmJobCountIntent"
    assert result.intent.state == "PENDING"


def test_classify_job_detail(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show details for slurm job 12345", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmJobDetailIntent"
    assert result.intent.job_id == "12345"


def test_classify_sacct_failed_jobs(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show recent failed jobs from sacct", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmAccountingIntent"
    assert result.intent.state == "FAILED"


def test_classify_node_status(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show idle slurm nodes", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmNodeStatusIntent"
    assert result.intent.state == "idle"


def test_classify_node_detail(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show slurm node details for slurm-worker-agatha", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmNodeDetailIntent"
    assert result.intent.node == "slurm-worker-agatha"


def test_classify_node_detail_on_cluster_gateway_node(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify(
        "show slurm node details for slurm-worker-agatha on cluster edge-2",
        _context(tmp_path),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmNodeDetailIntent"
    assert result.intent.node == "slurm-worker-agatha"
    assert result.intent.gateway_node == "edge-2"


def test_classify_node_detail_with_trailing_gateway_node(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify(
        "show slurm node details for slurm-worker-agatha on edge-2",
        _context(tmp_path),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmNodeDetailIntent"
    assert result.intent.node == "slurm-worker-agatha"
    assert result.intent.gateway_node == "edge-2"


def test_classify_partition_summary(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show slurm partitions", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmPartitionSummaryIntent"


def test_classify_gpu_availability(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show slurm gpu availability", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmMetricsIntent"
    assert result.intent.metric_group == "gpu_summary"


def test_classify_slurmdbd_health(tmp_path: Path) -> None:
    result = SlurmCapabilityPack().classify("show slurmdbd health", _context(tmp_path))
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SlurmMetricsIntent"
    assert result.intent.metric_group == "slurmdbd_health"


def test_compile_plans_use_slurm_tools_and_runtime_return(tmp_path: Path) -> None:
    pack = SlurmCapabilityPack()
    context = _context(tmp_path)
    result = pack.classify("show slurm queue as json on cluster edge-2", context)
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))
    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["slurm.queue", "runtime.return"]
    assert plan.plan.steps[0].args["gateway_node"] == "edge-2"


def test_supported_prompts_use_zero_llm_calls(monkeypatch, tmp_path: Path) -> None:
    fixtures = rebuild_eval_workspace(tmp_path / "fixtures")
    monkeypatch.setenv(SLURM_FIXTURE_DIR_ENV, fixtures.variables["slurm_fixture_dir"])
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases=fixtures.sql_databases,
        sql_default_database=fixtures.sql_default_database,
    )
    llm = FakeLLM()
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    plan = planner.build_plan(
        goal="show slurm queue as json",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=[*SLURM_ALLOWED_TOOLS],
        input_payload={"task": "show slurm queue as json"},
    )
    assert isinstance(plan, ExecutionPlan)
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0
    assert [step.action for step in plan.steps] == ["slurm.queue", "runtime.return"]


def test_mutating_prompts_do_not_compile_to_a_mutating_tool(tmp_path: Path) -> None:
    pack = SlurmCapabilityPack()
    result = pack.classify("cancel job 12345 in slurm", _context(tmp_path))
    plan = pack.compile(result.intent, CompileContext(allowed_tools=SLURM_ALLOWED_TOOLS, settings=_settings(tmp_path)))
    assert plan is not None
    assert [step.action for step in plan.plan.steps] == ["runtime.return"]


def test_engine_runs_supported_slurm_prompt_without_live_cluster(monkeypatch, tmp_path: Path) -> None:
    fixtures = rebuild_eval_workspace(tmp_path / "fixtures")
    monkeypatch.setenv(SLURM_FIXTURE_DIR_ENV, fixtures.variables["slurm_fixture_dir"])
    settings = Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases=fixtures.sql_databases,
        sql_default_database=fixtures.sql_default_database,
    )
    engine = ExecutionEngine(settings)
    state = engine.run_spec(str(SPEC_PATH), {"task": "show slurm queue as json"})
    assert state["metrics"]["llm_calls"] == 0
    assert '"jobs"' in state["final_output"]["content"]
