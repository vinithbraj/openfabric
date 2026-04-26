from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.capabilities.base import ClassificationContext, CompileContext
from aor_runtime.runtime.capabilities.registry import CapabilityRegistry, build_default_capability_registry
from aor_runtime.runtime.intents import CountFilesIntent, IntentResult, ReadFileLineIntent
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


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


class StubRegistry:
    def __init__(self, *, matched: bool = True) -> None:
        self.matched = matched
        self.classify_calls: list[tuple[str, ClassificationContext]] = []
        self.compile_calls: list[tuple[object, CompileContext]] = []

    def classify(self, goal: str, context: ClassificationContext) -> IntentResult:
        self.classify_calls.append((goal, context))
        if not self.matched:
            return IntentResult(matched=False, reason="stub_unmatched")
        return IntentResult(matched=True, intent=ReadFileLineIntent(path="notes.txt", line_number=2))

    def compile(self, intent: object, context: CompileContext) -> ExecutionPlan:
        self.compile_calls.append((intent, context))
        return ExecutionPlan.model_validate({"steps": [{"id": 1, "action": "fs.read", "args": {"path": "notes.txt"}}]})


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **overrides)


def _schema_payload() -> dict:
    return {
        "databases": [
            {
                "name": "book_club_db",
                "dialect": "sqlite",
                "tables": [{"name": "members", "columns": [{"name": "name", "type": "TEXT"}]}],
            }
        ]
    }


def test_registry_classifies_read_line_prompt(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "read line 2 from notes.txt",
        ClassificationContext(schema_payload=None, allowed_tools=["fs.read", "python.exec"], settings=_settings(tmp_path)),
    )
    assert result.matched is True
    assert isinstance(result.intent, ReadFileLineIntent)


def test_registry_classifies_count_files_prompt(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "how many txt files are in /tmp/example?",
        ClassificationContext(schema_payload=None, allowed_tools=["fs.find", "fs.glob"], settings=_settings(tmp_path)),
    )
    assert result.matched is True
    assert isinstance(result.intent, CountFilesIntent)


def test_registry_classifies_list_files_prompt(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "list top-level *.txt files in /tmp/example as csv",
        ClassificationContext(schema_payload=None, allowed_tools=["fs.glob"], settings=_settings(tmp_path)),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "ListFilesIntent"


def test_registry_classifies_write_return_prompt(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "write hello to /tmp/welcome.txt and return it",
        ClassificationContext(schema_payload=None, allowed_tools=["fs.write", "fs.read"], settings=_settings(tmp_path)),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "WriteTextIntent"


def test_registry_classifies_sql_select_prompt_with_schema(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "list name from members in book_club_db as csv",
        ClassificationContext(
            schema_payload=_schema_payload(),
            allowed_tools=["sql.query"],
            settings=_settings(tmp_path),
        ),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "SqlSelectIntent"


def test_registry_classifies_shell_csv_prompt(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "using shell, print alpha then beta on separate lines and return as csv",
        ClassificationContext(schema_payload=None, allowed_tools=["shell.exec"], settings=_settings(tmp_path)),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "ShellCommandIntent"


def test_registry_classifies_compound_prompt(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    result = registry.classify(
        "Read line 2 from /tmp/x/notes.txt, convert it to uppercase, save it to /tmp/out/line.txt, and return the saved file.",
        ClassificationContext(
            schema_payload=None,
            allowed_tools=["fs.read", "python.exec", "fs.write", "fs.read"],
            settings=_settings(tmp_path),
        ),
    )
    assert result.matched is True
    assert result.intent.__class__.__name__ == "CompoundIntent"


def test_registry_compile_returns_execution_plan(tmp_path: Path) -> None:
    registry = build_default_capability_registry()
    settings = _settings(tmp_path)
    result = registry.classify(
        "read line 2 from notes.txt",
        ClassificationContext(schema_payload=None, allowed_tools=["fs.read", "python.exec"], settings=settings),
    )
    plan = registry.compile(result.intent, CompileContext(allowed_tools=["fs.read", "python.exec"], settings=settings))
    assert isinstance(plan, ExecutionPlan)
    assert [step.action for step in plan.steps] == ["fs.read", "python.exec"]


def test_deterministic_planner_path_still_reports_zero_llm_calls(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM()
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    plan = planner.build_plan(
        goal="read line 2 from notes.txt",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.read", "python.exec"],
        input_payload={"task": "read line 2 from notes.txt"},
    )
    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == "deterministic_intent"
    assert planner.last_llm_calls == 0
    assert llm.call_count == 0


def test_unmatched_prompt_still_falls_back_to_llm_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM(['{"steps":[{"id":1,"action":"fs.read","args":{"path":"notes.txt"}}]}'])
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    plan = planner.build_plan(
        goal="Summarize the tone of notes.txt in one sentence",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.read", "python.exec"],
        input_payload={"task": "Summarize the tone of notes.txt in one sentence"},
    )
    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == "direct"
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1


def test_planner_uses_injected_capability_registry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeLLM()
    registry = StubRegistry()
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings, capability_registry=registry)
    plan = planner.build_plan(
        goal="read line 2 from notes.txt",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.read", "python.exec"],
        input_payload={"task": "read line 2 from notes.txt"},
    )
    assert isinstance(plan, ExecutionPlan)
    assert len(registry.classify_calls) == 1
    assert len(registry.compile_calls) == 1
    assert planner.last_llm_calls == 0
