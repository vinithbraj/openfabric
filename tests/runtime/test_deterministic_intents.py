from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, PlannerConfig
from aor_runtime.runtime.engine import ExecutionEngine
from aor_runtime.runtime.intent_classifier import classify_intent
from aor_runtime.runtime.intent_compiler import compile_intent_to_plan
from aor_runtime.runtime.intents import (
    CountFilesIntent,
    FetchExtractIntent,
    ListFilesIntent,
    ReadFileLineIntent,
    SearchFileContentsIntent,
    ShellCommandIntent,
    SqlCountIntent,
    SqlSelectIntent,
    WriteTextIntent,
)
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry
from aor_runtime.tools.sql import resolve_database_selection


requires_configured_llm = pytest.mark.skip(reason="LLM-exclusive runtime requires a configured action-planning LLM")


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def load_prompt(self, path: str | None, fallback: str) -> str:
        return fallback

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if not self.responses:
            raise AssertionError("LLM should not have been called")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.user_prompts)


def _planner_settings(tmp_path: Path, **overrides) -> Settings:
    payload = {"response_render_mode": "raw", **overrides}
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db", **payload)


def _schema_payload() -> dict:
    return {
        "databases": [
            {
                "name": "book_club_db",
                "dialect": "sqlite",
                "tables": [
                    {
                        "name": "members",
                        "columns": [{"name": "name", "type": "TEXT"}, {"name": "city", "type": "TEXT"}],
                    },
                    {
                        "name": "books",
                        "columns": [{"name": "title", "type": "TEXT"}, {"name": "author", "type": "TEXT"}],
                    },
                ],
            }
        ]
    }


def _planner(tmp_path: Path, responses: list[str] | None = None, **settings_overrides) -> tuple[TaskPlanner, FakeLLM]:
    settings = _planner_settings(tmp_path, **settings_overrides)
    llm = FakeLLM(responses)
    planner = TaskPlanner(llm=llm, tools=build_tool_registry(settings), settings=settings)
    return planner, llm


def _action_plan(tool: str, inputs: dict, *, binding: str = "result", final_path: str | None = None) -> str:
    value = {"$ref": binding}
    if final_path is not None:
        value["path"] = final_path
    return json.dumps(
        {
            "goal": "plan",
            "actions": [
                {
                    "id": "action_1",
                    "tool": tool,
                    "purpose": "Run the requested action.",
                    "inputs": inputs,
                    "output_binding": binding,
                    "expected_result_shape": {"kind": "text"},
                },
                {
                    "id": "return_result",
                    "tool": "runtime.return",
                    "purpose": "Return the result.",
                    "inputs": {"value": value, "mode": "text"},
                    "depends_on": ["action_1"],
                    "output_binding": "runtime_return_result",
                    "expected_result_shape": {"kind": "text"},
                },
            ],
            "expected_final_shape": {"kind": "text"},
            "notes": [],
        }
    )


def _sql_fixture(path: Path) -> str:
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    try:
        db.executescript(
            """
            CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );
            INSERT INTO members(id, name, city) VALUES
                (1, 'Alice', 'Portland'),
                (2, 'Bob', 'Seattle'),
                (3, 'Carla', 'Portland');
            """
        )
        db.commit()
    finally:
        db.close()
    return f"sqlite:///{path}"


def _run_engine(settings: Settings, prompt: str) -> dict:
    engine = ExecutionEngine(settings)
    return engine.run_spec(str(SPEC_PATH), {"task": prompt})


def test_classifier_matches_read_line_prompt() -> None:
    result = classify_intent("read line 2 from notes.txt")
    assert result.matched is True
    assert isinstance(result.intent, ReadFileLineIntent)
    assert result.intent.path == "notes.txt"
    assert result.intent.line_number == 2


def test_classifier_matches_count_files_prompt() -> None:
    result = classify_intent("how many txt files are in /tmp/example?")
    assert result.matched is True
    assert isinstance(result.intent, CountFilesIntent)
    assert result.intent.path == "/tmp/example"
    assert result.intent.pattern == "*.txt"
    assert result.intent.recursive is True


def test_classifier_matches_top_level_list_files_prompt() -> None:
    result = classify_intent("list top-level *.txt files in /tmp/example as csv")
    assert result.matched is True
    assert isinstance(result.intent, ListFilesIntent)
    assert result.intent.path == "/tmp/example"
    assert result.intent.pattern == "*.txt"
    assert result.intent.recursive is False
    assert result.intent.output_mode == "csv"


def test_classifier_matches_search_contents_prompt() -> None:
    result = classify_intent("find files containing cinnamon under /tmp/pantry")
    assert result.matched is True
    assert isinstance(result.intent, SearchFileContentsIntent)
    assert result.intent.path == "/tmp/pantry"
    assert result.intent.needle == "cinnamon"


def test_classifier_matches_write_and_return_prompt() -> None:
    result = classify_intent("write hello to /tmp/welcome.txt and return it")
    assert result.matched is True
    assert isinstance(result.intent, WriteTextIntent)
    assert result.intent.path == "/tmp/welcome.txt"
    assert result.intent.content == "hello"
    assert result.intent.return_content is True


def test_classifier_matches_fetch_extract_prompt() -> None:
    result = classify_intent("curl example.com and extract title")
    assert result.matched is True
    assert isinstance(result.intent, FetchExtractIntent)
    assert result.intent.url == "https://example.com"
    assert result.intent.extract == "title"


def test_classifier_matches_shell_print_prompt() -> None:
    result = classify_intent("using shell, print alpha then beta on separate lines and return as csv")
    assert result.matched is True
    assert isinstance(result.intent, ShellCommandIntent)
    assert "printf" in result.intent.command
    assert result.intent.output_mode == "csv"


def test_compile_read_line_plan() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(ReadFileLineIntent(path="notes.txt", line_number=2), ["fs.read", "python.exec"], settings)
    assert [step.action for step in plan.steps] == ["fs.read", "python.exec"]
    assert plan.steps[1].args["inputs"]["text"]["path"] == "content"


def test_compile_recursive_count_plan_uses_fs_find_and_runtime_return() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(CountFilesIntent(path="logs", pattern="*.txt", recursive=True), ["fs.find"], settings)
    assert [step.action for step in plan.steps] == ["fs.find", "runtime.return"]
    assert plan.steps[1].args["mode"] == "count"


def test_compile_top_level_count_plan_uses_fs_glob() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(CountFilesIntent(path="logs", pattern="*.txt", recursive=False), ["fs.glob"], settings)
    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[0].args["recursive"] is False


def test_compile_list_files_csv_plan_uses_fs_glob_and_runtime_return() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        ListFilesIntent(path="reports", pattern="*.txt", recursive=False, output_mode="csv"),
        ["fs.glob"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return"]
    assert plan.steps[1].args["mode"] == "csv"


def test_compile_search_contents_plan_uses_search_content_and_runtime_return() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        SearchFileContentsIntent(path="/tmp/docs", needle="cinnamon", output_mode="text"),
        ["fs.search_content"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.search_content", "runtime.return"]
    assert plan.steps[0].args["needle"] == "cinnamon"
    assert plan.steps[1].args["value"]["path"] == "matches"


def test_compile_search_contents_json_output_remains_matches_wrapper() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        SearchFileContentsIntent(path="/tmp/docs", needle="cinnamon", output_mode="json"),
        ["fs.search_content"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.search_content", "runtime.return"]
    assert plan.steps[1].args["output_contract"]["json_shape"] == "matches"


def test_compile_sql_count_intent_uses_explicit_database() -> None:
    settings = _planner_settings(
        Path("/tmp"),
        sql_databases={"book_club_db": "sqlite:///book.db", "other_db": "sqlite:///other.db"},
        sql_default_database="book_club_db",
    )
    plan = compile_intent_to_plan(SqlCountIntent(database="book_club_db", table="members"), ["sql.query"], settings)
    assert [step.action for step in plan.steps] == ["sql.query", "runtime.return"]
    assert plan.steps[0].args["database"] == "book_club_db"
    assert "COUNT(*) AS count_value" in plan.steps[0].args["query"]


def test_compile_sql_select_single_column_csv_plan() -> None:
    settings = _planner_settings(Path("/tmp"), sql_databases={"book_club_db": "sqlite:///book.db"})
    plan = compile_intent_to_plan(
        SqlSelectIntent(database="book_club_db", table="members", columns=["name"], output_mode="csv"),
        ["sql.query"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["sql.query", "runtime.return"]
    assert plan.steps[1].args["mode"] == "csv"


def test_compile_write_text_plan_reads_back_when_requested() -> None:
    settings = _planner_settings(Path("/tmp"))
    plan = compile_intent_to_plan(WriteTextIntent(path="welcome.txt", content="hello", return_content=True), ["fs.write", "fs.read"], settings)
    assert [step.action for step in plan.steps] == ["fs.write", "fs.read", "runtime.return"]


def test_explicit_shell_prompt_still_builds_shell_exec_plan(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [_action_plan("shell.exec", {"command": "printf 'alpha\\nbeta\\n'"}, final_path="stdout")],
    )
    plan = planner.build_plan(
        goal="Using shell, print alpha then beta on separate lines and return as csv",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["shell.exec", "fs.search_content", "python.exec"],
        input_payload={"task": "Using shell, print alpha then beta on separate lines and return as csv"},
    )
    assert isinstance(plan, ExecutionPlan)
    assert [step.action for step in plan.steps] == ["shell.exec", "text.format", "runtime.return"]
    assert planner.last_planning_mode == "validator_enforced_action_planner"
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1


def test_search_prompt_prefers_fs_search_content_even_when_shell_is_available(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [
            _action_plan(
                "fs.search_content",
                {"path": str(tmp_path), "needle": "cinnamon", "pattern": "*.txt", "path_style": "name"},
                binding="matches",
                final_path="matches",
            )
        ],
    )
    plan = planner.build_plan(
        goal="Find txt files containing cinnamon under /tmp/pantry and return the matching filenames as json only.",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.search_content", "shell.exec", "python.exec"],
        input_payload={"task": "Find txt files containing cinnamon under /tmp/pantry and return the matching filenames as json only."},
    )
    assert isinstance(plan, ExecutionPlan)
    assert [step.action for step in plan.steps] == ["fs.search_content", "text.format", "runtime.return"]
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1


def test_planner_uses_action_planner_for_former_deterministic_prompt(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [_action_plan("fs.read", {"path": str(tmp_path / "notes.txt")}, binding="file_content", final_path="content")],
    )
    plan = planner.build_plan(
        goal="Read line 2 from /tmp/example/notes.txt and return just the line.",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.read", "python.exec", "fs.glob", "fs.find", "shell.exec", "fs.write"],
        input_payload={"task": "Read line 2 from /tmp/example/notes.txt and return just the line."},
    )
    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == "validator_enforced_action_planner"
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1


def test_unmatched_prompt_uses_same_action_planner_path(tmp_path: Path) -> None:
    planner, llm = _planner(
        tmp_path,
        [_action_plan("fs.read", {"path": "notes.txt"}, binding="file_content", final_path="content")],
    )
    plan = planner.build_plan(
        goal="Summarize notes.txt in a warm paragraph.",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.read"],
        input_payload={"task": "Summarize notes.txt in a warm paragraph."},
    )
    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == "validator_enforced_action_planner"
    assert planner.last_llm_calls == 1
    assert llm.call_count == 1


def test_sql_default_database_resolution_prefers_explicit_then_default(tmp_path: Path) -> None:
    settings = _planner_settings(
        tmp_path,
        sql_databases={"book_club_db": "sqlite:///book.db", "bakery_db": "sqlite:///bakery.db"},
        sql_default_database="bakery_db",
    )
    assert resolve_database_selection(settings, "book_club_db")[0] == "book_club_db"
    assert resolve_database_selection(settings, None)[0] == "bakery_db"


def test_sql_default_database_resolution_errors_without_default(tmp_path: Path) -> None:
    settings = _planner_settings(
        tmp_path,
        sql_databases={"book_club_db": "sqlite:///book.db", "bakery_db": "sqlite:///bakery.db"},
    )
    with pytest.raises(Exception, match="Database selection is required"):
        resolve_database_selection(settings, None)


def test_sql_classifier_requires_schema_confirmation() -> None:
    result = classify_intent("count members in book_club_db", schema_payload=_schema_payload())
    assert result.matched is True
    assert isinstance(result.intent, SqlCountIntent)
    assert result.intent.database == "book_club_db"
    assert result.intent.table == "members"


def test_sql_select_classifier_requires_explicit_columns() -> None:
    result = classify_intent("list name from members in book_club_db as csv", schema_payload=_schema_payload())
    assert result.matched is True
    assert isinstance(result.intent, SqlSelectIntent)
    assert result.intent.columns == ["name"]
    assert result.intent.output_mode == "csv"


@requires_configured_llm
def test_end_to_end_read_line_prompt(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("alpha\nbeta\ngamma\n")
    settings = _planner_settings(tmp_path)

    state = _run_engine(settings, f"Read line 2 from {notes} and return just the line.")

    assert state["final_output"]["content"] == "beta"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_count_txt_files_prompt(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("a")
    (inputs / "b.txt").write_text("b")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    (inputs / "ignore.md").write_text("x")
    settings = _planner_settings(tmp_path)

    state = _run_engine(settings, f"How many txt files are in {inputs} and tell me the count only.")

    assert state["final_output"]["content"] == "3"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_count_top_level_txt_files_only(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("a")
    (inputs / "b.txt").write_text("b")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    settings = _planner_settings(tmp_path)

    state = _run_engine(settings, f"Count top-level txt files in {inputs} and return the count only.")

    assert state["final_output"]["content"] == "2"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_list_top_level_txt_files_as_csv(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("a")
    (inputs / "b.txt").write_text("b")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    settings = _planner_settings(tmp_path)

    state = _run_engine(settings, f"List top-level txt files in {inputs} as csv.")

    assert state["final_output"]["content"] == "a.txt,b.txt"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_write_text_and_return_it(tmp_path: Path) -> None:
    target = tmp_path / "welcome.txt"
    settings = _planner_settings(tmp_path)

    state = _run_engine(settings, f"Write hello to {target} and return it.")

    assert state["final_output"]["content"] == "hello"
    assert target.read_text() == "hello"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_shell_lines_to_csv(tmp_path: Path) -> None:
    settings = _planner_settings(tmp_path)

    state = _run_engine(settings, "Using shell, print alpha then beta on separate lines and return them as csv.")

    assert state["final_output"]["content"] == "alpha,beta"
    assert state["metrics"]["llm_calls"] == 0
