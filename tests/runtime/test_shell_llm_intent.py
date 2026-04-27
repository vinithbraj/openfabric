from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import PlannerConfig
from aor_runtime.runtime.capabilities.base import ClassificationContext
from aor_runtime.runtime.capabilities.shell import ShellCapabilityPack
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


SHELL_ALLOWED_TOOLS = [
    "shell.exec",
    "fs.search_content",
    "fs.aggregate",
    "python.exec",
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
    return ClassificationContext(
        schema_payload=None,
        allowed_tools=SHELL_ALLOWED_TOOLS,
        settings=_settings(tmp_path, **settings_overrides),
    )


def test_check_uptime_maps_to_shell_inspection(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.93, "arguments": {"kind": "uptime"}, "reason": "Read-only uptime summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="check uptime",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "check uptime"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "uptime"
    assert planner.last_planning_mode == "llm_intent_extractor"
    assert planner.last_llm_calls == 1
    assert planner.last_llm_intent_calls == 1
    assert planner.last_raw_planner_llm_calls == 0


def test_show_disk_usage_maps_to_fixed_template(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.9, "arguments": {"kind": "disk_usage"}, "reason": "Disk usage summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="show disk usage",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "show disk usage"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "df -h"


def test_show_memory_usage_maps_to_fixed_template(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.9, "arguments": {"kind": "memory_summary"}, "reason": "Memory summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="show memory usage",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "show memory usage"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "free -h"


def test_show_top_processes_maps_to_fixed_template_with_limit(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.9, "arguments": {"kind": "process_summary", "limit": 10}, "reason": "Top processes summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="show top processes",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "show top processes"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "ps -eo pid,user,comm,pcpu,pmem --sort=-pcpu | head -n 11"


def test_show_listening_ports_maps_to_fixed_template(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.9, "arguments": {"kind": "listening_ports"}, "reason": "Listening ports summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="show listening ports",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "show listening ports"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "ss -tuln"


def test_show_network_summary_maps_to_fixed_template(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.9, "arguments": {"kind": "network_summary"}, "reason": "Network summary."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="show network summary",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "show network summary"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert plan.steps[0].args["command"] == "ip addr"


def test_shell_framed_file_aggregate_routes_to_filesystem_plan(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "FileAggregateIntent", "confidence": 0.9, "arguments": {"path": "/tmp/work", "pattern": "*.mp4", "recursive": true, "file_only": true, "aggregate": "total_size", "output_mode": "text", "size_unit": "auto"}, "reason": "Filesystem aggregate."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="using shell, how much space do mp4s take under /tmp/work",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "using shell, how much space do mp4s take under /tmp/work"},
    )

    assert [step.action for step in plan.steps] == ["fs.aggregate", "runtime.return"]
    assert planner.last_planning_mode == "llm_intent_extractor"
    assert planner.last_capability_name == "filesystem"


def test_shell_framed_content_files_routes_to_search_then_return(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "FileContentDiagnosticIntent", "confidence": 0.88, "arguments": {"path": "/tmp/work", "needle": "cinnamon", "result_kind": "files", "output_mode": "json"}, "reason": "Content diagnostic files."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="with shell, show files containing cinnamon under /tmp/work as json",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "with shell, show files containing cinnamon under /tmp/work as json"},
    )

    assert [step.action for step in plan.steps] == ["fs.search_content", "runtime.return"]
    assert planner.last_capability_name == "shell"


def test_shell_framed_content_lines_routes_to_search_then_transform(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "FileContentDiagnosticIntent", "confidence": 0.89, "arguments": {"path": "/tmp/work", "needle": "TODO", "result_kind": "lines", "output_mode": "json"}, "reason": "Content diagnostic lines."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="with shell, show matching lines containing TODO under /tmp/work",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "with shell, show matching lines containing TODO under /tmp/work"},
    )

    assert [step.action for step in plan.steps] == ["fs.search_content", "python.exec", "runtime.return"]
    assert planner.last_capability_name == "shell"


def test_shell_framed_fetch_routes_to_existing_fetch_compile_path(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "FetchExtractIntent", "confidence": 0.9, "arguments": {"url": "https://example.com", "extract": "title", "output_mode": "text"}, "reason": "Fetch title extraction."}'
        ]
    )
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="fetch the title of https://example.com using shell",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["shell.exec"],
        input_payload={"task": "fetch the title of https://example.com using shell"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec"]
    assert "curl -sL" in plan.steps[0].args["command"]
    assert planner.last_capability_name == "fetch"


def test_delete_request_is_rejected_before_llm(tmp_path: Path) -> None:
    llm = FakeLLM()
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="delete all logs",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "delete all logs"},
    )

    assert [step.action for step in plan.steps] == ["runtime.return"]
    assert planner.last_planning_mode == "llm_intent_extractor"
    assert planner.last_llm_calls == 0
    assert planner.last_llm_intent_calls == 0
    assert planner.last_raw_planner_llm_calls == 0
    assert llm.call_count == 0


def test_kill_request_is_rejected_before_llm(tmp_path: Path) -> None:
    llm = FakeLLM()
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="kill process 123",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "kill process 123"},
    )

    assert [step.action for step in plan.steps] == ["runtime.return"]
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0


def test_malicious_llm_command_field_is_rejected(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "ShellInspectionIntent", "confidence": 0.9, "arguments": {"kind": "uptime", "command": "uptime"}, "reason": "unsafe"}'
        ]
    )
    pack = ShellCapabilityPack()
    planner = _planner(tmp_path, llm)

    result = pack.try_llm_extract("check uptime", _classification_context(tmp_path), planner.llm_intent_extractor)

    assert result.matched is False


def test_disabled_flag_skips_shell_llm_intent_extraction(tmp_path: Path) -> None:
    llm = FakeLLM()
    pack = ShellCapabilityPack()
    planner = _planner(tmp_path, llm, enable_llm_intent_extraction=False)
    context = _classification_context(tmp_path, enable_llm_intent_extraction=False)

    result = pack.try_llm_extract("check uptime", context, planner.llm_intent_extractor)

    assert result.matched is False
    assert llm.call_count == 0


def test_deterministic_shell_prompt_still_uses_zero_llm_calls(tmp_path: Path) -> None:
    llm = FakeLLM()
    planner = _planner(tmp_path, llm)

    plan = planner.build_plan(
        goal="Using shell, print alpha then beta on separate lines and return as csv",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=SHELL_ALLOWED_TOOLS,
        input_payload={"task": "Using shell, print alpha then beta on separate lines and return as csv"},
    )

    assert [step.action for step in plan.steps] == ["shell.exec", "runtime.return"]
    assert planner.last_llm_calls == 0
    assert planner.last_llm_intent_calls == 0
    assert planner.last_raw_planner_llm_calls == 0
    assert llm.call_count == 0
