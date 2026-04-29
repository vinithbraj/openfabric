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
    CompoundIntent,
    CountFilesIntent,
    ListFilesIntent,
    ReadFileLineIntent,
    TransformChainIntent,
    WriteResultIntent,
    WriteTextIntent,
)
from aor_runtime.runtime.planner import TaskPlanner
from aor_runtime.tools.factory import build_tool_registry


requires_configured_llm = pytest.mark.skip(reason="LLM-exclusive runtime requires a configured action-planning LLM")


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"


class FakeLLM:
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = [json.dumps(response) for response in list(responses or [])]

    def load_prompt(self, path: str | None, fallback: str) -> str:
        raise AssertionError("Legacy planner prompt should not be loaded")

    def complete(self, **_: object) -> str:
        if not self.responses:
            raise AssertionError("LLM called more times than expected")
        return self.responses.pop(0)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
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
                    }
                ],
            }
        ]
    }


def _compound_action_plan(output_path: Path) -> dict:
    return {
        "goal": "Count top-level txt files, write result, and return it.",
        "actions": [
            {
                "id": "list_txt",
                "tool": "fs.glob",
                "purpose": "List top-level txt files.",
                "inputs": {"path": str(output_path.parent.parent), "pattern": "*.txt", "recursive": False},
                "output_binding": "txt_matches",
                "expected_result_shape": {"kind": "table"},
            },
            {
                "id": "format_result",
                "tool": "text.format",
                "purpose": "Format the count result.",
                "inputs": {"source": {"$ref": "txt_matches", "path": "matches"}, "format": "txt"},
                "depends_on": ["list_txt"],
                "output_binding": "formatted_count",
                "expected_result_shape": {"kind": "text"},
            },
            {
                "id": "write_result",
                "tool": "fs.write",
                "purpose": "Write the formatted result.",
                "inputs": {"path": str(output_path), "content": {"$ref": "formatted_count", "path": "content"}},
                "depends_on": ["format_result"],
                "output_binding": "written_file",
                "expected_result_shape": {"kind": "file"},
            },
            {
                "id": "return_result",
                "tool": "runtime.return",
                "purpose": "Return the written file path.",
                "inputs": {"value": {"$ref": "written_file"}, "mode": "text"},
                "depends_on": ["write_result"],
                "output_binding": "runtime_return_result",
                "expected_result_shape": {"kind": "text"},
            },
        ],
        "expected_final_shape": {"kind": "text"},
        "notes": [],
    }


def _run_engine(settings: Settings, prompt: str) -> dict:
    engine = ExecutionEngine(settings)
    return engine.run_spec(str(SPEC_PATH), {"task": prompt})


def test_compound_classifier_matches_count_save_return_prompt() -> None:
    result = classify_intent("Count the top-level txt files in /tmp/x, save the result to /tmp/out/count.txt, and return the file contents.")
    assert result.matched is True
    assert isinstance(result.intent, CompoundIntent)
    assert isinstance(result.intent.intents[0], CountFilesIntent)
    assert isinstance(result.intent.intents[1], WriteResultIntent)
    assert result.intent.intents[1].path == "/tmp/out/count.txt"
    assert result.intent.return_policy == "return_written_file"


def test_compound_classifier_matches_list_csv_save_prompt() -> None:
    result = classify_intent("List the top-level txt files in /tmp/x as csv, write them to /tmp/out/files.csv, and return the content.")
    assert result.matched is True
    assert isinstance(result.intent, CompoundIntent)
    assert isinstance(result.intent.intents[0], ListFilesIntent)
    assert isinstance(result.intent.intents[1], TransformChainIntent)
    assert result.intent.intents[1].operations == ["csv"]
    assert isinstance(result.intent.intents[2], WriteResultIntent)
    assert result.intent.return_policy == "return_written_file"


def test_compound_classifier_matches_read_uppercase_prompt() -> None:
    result = classify_intent("Read line 2 from /tmp/x/notes.txt, convert it to uppercase, and return it.")
    assert result.matched is True
    assert isinstance(result.intent, CompoundIntent)
    assert isinstance(result.intent.intents[0], ReadFileLineIntent)
    assert isinstance(result.intent.intents[1], TransformChainIntent)
    assert result.intent.intents[1].operations == ["uppercase"]


def test_compound_classifier_matches_read_uppercase_save_return_prompt() -> None:
    result = classify_intent(
        "Read line 2 from /tmp/x/notes.txt, convert it to uppercase, save it to /tmp/out/line.txt, and return the saved file."
    )
    assert result.matched is True
    assert isinstance(result.intent, CompoundIntent)
    assert isinstance(result.intent.intents[0], ReadFileLineIntent)
    assert isinstance(result.intent.intents[1], TransformChainIntent)
    assert isinstance(result.intent.intents[2], WriteResultIntent)
    assert result.intent.intents[2].read_back is True
    assert result.intent.return_policy == "return_written_file"


def test_create_with_exact_content_and_return_stays_single_write_intent() -> None:
    result = classify_intent("Create /tmp/out/welcome.txt with exact content 'hello world' and then return the content only.")
    assert result.matched is True
    assert isinstance(result.intent, WriteTextIntent)
    assert result.intent.return_content is True


def test_top_level_synonyms_map_to_non_recursive() -> None:
    directly_in = classify_intent("Count txt files directly in /tmp/example and return the count only.")
    not_nested = classify_intent("List txt files not nested in /tmp/example as csv.")
    direct = classify_intent("Count direct txt files in /tmp/example and return the count only.")
    assert isinstance(directly_in.intent, CountFilesIntent)
    assert directly_in.intent.recursive is False
    assert isinstance(not_nested.intent, ListFilesIntent)
    assert not_nested.intent.recursive is False
    assert isinstance(direct.intent, CountFilesIntent)
    assert direct.intent.recursive is False


def test_recursive_synonyms_map_to_recursive() -> None:
    recursively = classify_intent("Count txt files recursively in /tmp/example and return the count only.")
    anywhere_below = classify_intent("List txt files anywhere below /tmp/example as csv.")
    assert isinstance(recursively.intent, CountFilesIntent)
    assert recursively.intent.recursive is True
    assert isinstance(anywhere_below.intent, ListFilesIntent)
    assert anywhere_below.intent.recursive is True


def test_meeting_notes_phrase_is_not_treated_as_path() -> None:
    result = classify_intent("Please read the second line from the meeting notes, then return it as uppercase text only.")
    assert result.matched is False


def test_compile_compound_count_save_return_plan() -> None:
    settings = _settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        CompoundIntent(
            intents=[
                CountFilesIntent(path="logs", pattern="*.txt", recursive=False),
                WriteResultIntent(path="out/count.txt", source_alias="__previous__", output_mode="count", read_back=True),
            ],
            return_policy="return_written_file",
        ),
        ["fs.glob", "fs.write", "fs.read"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return", "fs.write", "fs.read", "runtime.return"]


def test_compile_compound_list_csv_save_return_plan() -> None:
    settings = _settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        CompoundIntent(
            intents=[
                ListFilesIntent(path="logs", pattern="*.txt", recursive=False, output_mode="text"),
                TransformChainIntent(operations=["csv"]),
                WriteResultIntent(path="out/files.csv", source_alias="__previous__", output_mode="csv", read_back=True),
            ],
            return_policy="return_written_file",
        ),
        ["fs.glob", "fs.write", "fs.read"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.glob", "runtime.return", "fs.write", "fs.read", "runtime.return"]


def test_compound_classifier_matches_only_in_this_folder_list_prompt() -> None:
    result = classify_intent("Show the txt filenames only in this folder, /tmp/x, as csv, save them to /tmp/out/files.csv, and return the saved file.")
    assert result.matched is True
    assert isinstance(result.intent, CompoundIntent)
    producer = result.intent.intents[0]
    assert isinstance(producer, ListFilesIntent)
    assert producer.recursive is False
    assert producer.path_style == "name"


def test_compound_classifier_matches_immediate_children_count_prompt() -> None:
    result = classify_intent("Count the txt files immediate children of /tmp/x, save the result to /tmp/out/count.txt, and return the file contents.")
    assert result.matched is True
    assert isinstance(result.intent, CompoundIntent)
    producer = result.intent.intents[0]
    assert isinstance(producer, CountFilesIntent)
    assert producer.recursive is False


def test_compile_read_line_uppercase_plan() -> None:
    settings = _settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        CompoundIntent(
            intents=[
                ReadFileLineIntent(path="notes.txt", line_number=2),
                TransformChainIntent(operations=["uppercase"]),
            ],
            return_policy="return_last",
        ),
        ["fs.read", "python.exec"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.read", "python.exec", "python.exec", "runtime.return"]


def test_compile_read_line_uppercase_save_return_plan() -> None:
    settings = _settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        CompoundIntent(
            intents=[
                ReadFileLineIntent(path="notes.txt", line_number=2),
                TransformChainIntent(operations=["uppercase"]),
                WriteResultIntent(path="out/line.txt", source_alias="__previous__", output_mode="text", read_back=True),
            ],
            return_policy="return_written_file",
        ),
        ["fs.read", "python.exec", "fs.write", "fs.read"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.read", "python.exec", "python.exec", "fs.write", "fs.read", "runtime.return"]


def test_compile_create_write_return_uses_readback_and_runtime_return() -> None:
    settings = _settings(Path("/tmp"))
    plan = compile_intent_to_plan(
        WriteTextIntent(path="welcome.txt", content="hello", return_content=True),
        ["fs.write", "fs.read"],
        settings,
    )
    assert [step.action for step in plan.steps] == ["fs.write", "fs.read", "runtime.return"]


def test_compound_prompt_uses_action_planner(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output_path = tmp_path / "out" / "count.txt"
    planner = TaskPlanner(llm=FakeLLM([_compound_action_plan(output_path)]), tools=build_tool_registry(settings), settings=settings)
    plan = planner.build_plan(
        goal=f"Count the top-level txt files in {tmp_path}, save the result to {output_path}, and return the file contents.",
        planner=PlannerConfig(temperature=0.0),
        allowed_tools=["fs.glob", "fs.find", "fs.read", "fs.write", "python.exec", "shell.exec", "sql.query"],
        input_payload={"task": "compound deterministic"},
    )
    assert isinstance(plan, ExecutionPlan)
    assert planner.last_planning_mode == "validator_enforced_action_planner"
    assert planner.last_llm_calls == 1


@requires_configured_llm
def test_end_to_end_count_top_level_save_return(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("a")
    (inputs / "b.txt").write_text("b")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"Count the top-level txt files in {inputs}, save the result to {tmp_path / 'out' / 'count.txt'}, and return the file contents.",
    )

    assert state["final_output"]["content"] == "2"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_list_csv_save_return(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("a")
    (inputs / "b.txt").write_text("b")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"List the top-level txt files in {inputs} as csv, write them to {tmp_path / 'out' / 'files.csv'}, and return the content.",
    )

    assert state["final_output"]["content"] == "a.txt,b.txt"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_read_uppercase_return(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("alpha\nbeta\ngamma\n")
    settings = _settings(tmp_path)

    state = _run_engine(settings, f"Read line 2 from {notes}, convert it to uppercase, and return it.")

    assert state["final_output"]["content"] == "BETA"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_read_uppercase_save_return(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("alpha\nbeta\ngamma\n")
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"Read line 2 from {notes}, convert it to uppercase, save it to {tmp_path / 'out' / 'line.txt'}, and return the saved file.",
    )

    assert state["final_output"]["content"] == "BETA"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_create_welcome_and_return_content(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"Create {tmp_path / 'welcome.txt'} with exact content 'kind words' and then return the content only.",
    )

    assert state["final_output"]["content"] == "kind words"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_end_to_end_explicit_sql_uppercase_csv_save_return(tmp_path: Path) -> None:
    database_path = tmp_path / "book_club.db"
    db = sqlite3.connect(database_path)
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

    settings = _settings(
        tmp_path,
        sql_databases={"book_club_db": f"sqlite:///{database_path}"},
        sql_default_database="book_club_db",
    )

    state = _run_engine(
        settings,
        f"List name from members in book_club_db, convert them to uppercase csv, save the result to {tmp_path / 'out' / 'names.csv'}, and return the file contents.",
    )

    assert state["final_output"]["content"] == "ALICE,BOB,CARLA"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_top_level_txt_list_as_json_uses_matches_shape(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("a")
    (inputs / "b.txt").write_text("b")
    (inputs / "ignore.md").write_text("ignore")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("c")
    settings = _settings(tmp_path)

    state = _run_engine(settings, f"Give me the immediate-child txt filenames in {inputs} as json only.")

    assert state["final_output"]["content"] == '{"matches": ["a.txt", "b.txt"]}'
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_top_level_txt_list_as_json_from_direct_filenames_prompt_uses_matches_shape(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "apple.txt").write_text("a")
    (inputs / "banana.txt").write_text("b")
    (inputs / "ignore.md").write_text("ignore")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "citrus.txt").write_text("c")
    settings = _settings(tmp_path)

    state = _run_engine(settings, f"Give me the direct txt filenames in {inputs} as json only.")

    assert state["final_output"]["content"] == '{"matches": ["apple.txt", "banana.txt"]}'
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_content_search_txt_filenames_only_returns_basenames_and_excludes_md(tmp_path: Path) -> None:
    root = tmp_path / "search"
    root.mkdir()
    (root / "alpha.txt").write_text("cinnamon tea")
    (root / "beta.txt").write_text("fresh cinnamon")
    (root / "notes.md").write_text("cinnamon markdown")
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"Please search {root} for txt files containing cinnamon and return only the matching filenames as csv.",
    )

    assert state["final_output"]["content"] == "alpha.txt,beta.txt"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_filename_only_csv_save_return_uses_zero_llm_calls(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "north.txt").write_text("n")
    (inputs / "south.txt").write_text("s")
    (inputs / "ignore.md").write_text("ignore")
    nested = inputs / "nested"
    nested.mkdir()
    (nested / "east.txt").write_text("e")
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"Show the txt filenames only in this folder, {inputs}, as csv, save them to {tmp_path / 'out' / 'files.csv'}, and return the saved file.",
    )

    assert state["final_output"]["content"] == "north.txt,south.txt"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_write_exact_quoted_text_returns_only_quoted_content(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    state = _run_engine(
        settings,
        f"Write the exact text 'quiet library' to {tmp_path / 'quoted.txt'} and return it.",
    )

    assert state["final_output"]["content"] == "quiet library"
    assert state["metrics"]["llm_calls"] == 0


@requires_configured_llm
def test_shell_csv_paraphrases_use_zero_llm_calls(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    prompts = [
        "using shell, print alpha and beta on separate lines and return csv",
        "with shell, output alpha then beta, one per line, and give csv",
        "run a shell command that prints alpha then beta and return comma separated",
        "shell: print alpha newline beta; return as csv",
    ]

    for prompt in prompts:
        state = _run_engine(settings, prompt)
        assert state["final_output"]["content"] == "alpha,beta"
        assert state["metrics"]["llm_calls"] == 0
