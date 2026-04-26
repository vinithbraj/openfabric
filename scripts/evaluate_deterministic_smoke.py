from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.runtime.engine import ExecutionEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"
WORKSPACE = REPO_ROOT / "artifacts" / "deterministic_smoke"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "deterministic_smoke_report.json"
PROMPT_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    prompt: str
    expected: Any
    mode: str


def _clean_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _setup_workspace() -> dict[str, Any]:
    _clean_workspace()

    notes = WORKSPACE / "notes"
    _write(notes / "meeting_notes.txt", "agenda\nbudget\nfollow-up\n")
    _write(notes / "shopping.txt", "bread\nmilk\ntea\n")
    _write(notes / "poem.txt", "roses\nviolets\nstarlight\n")
    _write(notes / "todo.txt", "email\ncall\nlunch\n")
    _write(notes / "short.txt", "one\ntwo\n")

    recursive_a = WORKSPACE / "files" / "recursive_a"
    _write(recursive_a / "root_a.txt", "a\n")
    _write(recursive_a / "root_b.txt", "b\n")
    _write(recursive_a / "ignore.md", "ignore\n")
    _write(recursive_a / "nested" / "child.txt", "child\n")

    top_level_a = WORKSPACE / "files" / "top_level_a"
    _write(top_level_a / "a.txt", "a\n")
    _write(top_level_a / "b.txt", "b\n")
    _write(top_level_a / "nested" / "child.txt", "child\n")

    recursive_b = WORKSPACE / "files" / "recursive_b"
    _write(recursive_b / "one.txt", "1\n")
    _write(recursive_b / "two.txt", "2\n")
    _write(recursive_b / "nested" / "three.txt", "3\n")
    _write(recursive_b / "nested" / "four.txt", "4\n")

    top_level_b = WORKSPACE / "files" / "top_level_b"
    _write(top_level_b / "first.txt", "1\n")
    _write(top_level_b / "second.txt", "2\n")
    _write(top_level_b / "nested" / "third.txt", "3\n")

    top_level_c = WORKSPACE / "files" / "top_level_c"
    _write(top_level_c / "only.txt", "only\n")
    _write(top_level_c / "nested" / "extra.txt", "extra\n")

    pantry = WORKSPACE / "search" / "pantry"
    _write(pantry / "cake.txt", "cinnamon sugar\n")
    _write(pantry / "tea.txt", "ginger and cinnamon\n")
    _write(pantry / "salt.txt", "sea salt\n")

    journal = WORKSPACE / "search" / "journal"
    _write(journal / "april.txt", "garden plans\n")
    _write(journal / "may.txt", "garden party\n")
    _write(journal / "june.md", "garden sketch\n")

    stories = WORKSPACE / "search" / "stories"
    _write(stories / "library.txt", "quiet library afternoon\n")
    _write(stories / "park.txt", "sunny park walk\n")

    sql_path = WORKSPACE / "book_club.db"
    db = sqlite3.connect(sql_path)
    try:
        db.executescript(
            """
            CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL
            );
            INSERT INTO members(id, name, city) VALUES
                (1, 'Alice', 'Portland'),
                (2, 'Bob', 'Seattle'),
                (3, 'Carla', 'Portland');
            INSERT INTO books(id, title) VALUES
                (1, 'North Window'),
                (2, 'Evening Train');
            """
        )
        db.commit()
    finally:
        db.close()

    return {
        "sql_databases": {"book_club_db": f"sqlite:///{sql_path}"},
        "sql_default_database": "book_club_db",
    }


def _cases() -> list[Case]:
    return [
        Case("read_1", "read_line", f"Read line 2 from {WORKSPACE / 'notes' / 'meeting_notes.txt'} and return just the line.", "budget", "text"),
        Case("read_2", "read_line", f"Read the second line from {WORKSPACE / 'notes' / 'shopping.txt'} and return just the line.", "milk", "text"),
        Case("read_3", "read_line", f"Read first line from {WORKSPACE / 'notes' / 'poem.txt'} and return just the line.", "roses", "text"),
        Case("read_4", "read_line", f"Read line 3 from {WORKSPACE / 'notes' / 'todo.txt'} and return just the line.", "lunch", "text"),
        Case("read_5", "read_line", f"Read line 9 from {WORKSPACE / 'notes' / 'short.txt'} and return just the line.", "", "text"),
        Case("count_1", "count_files", f"How many txt files are in {WORKSPACE / 'files' / 'recursive_a'} and tell me the count only.", "3", "count"),
        Case("count_2", "count_files", f"Count top-level txt files in {WORKSPACE / 'files' / 'top_level_a'} and return the count only.", "2", "count"),
        Case("count_3", "count_files", f"Count *.txt files under {WORKSPACE / 'files' / 'recursive_b'} and return the count only.", "4", "count"),
        Case("count_4", "count_files", f"How many txt files are in {WORKSPACE / 'files' / 'top_level_b'} and tell me the count only.", "3", "count"),
        Case("count_5", "count_files", f"Count top-level .txt files in {WORKSPACE / 'files' / 'top_level_c'} and return the count only.", "1", "count"),
        Case("list_1", "list_files", f"List top-level txt files in {WORKSPACE / 'files' / 'top_level_a'} as csv.", "a.txt,b.txt", "csv"),
        Case("list_2", "list_files", f"List top-level *.txt files in {WORKSPACE / 'files' / 'top_level_b'} as csv.", "first.txt,second.txt", "csv"),
        Case("list_3", "list_files", f"List top-level txt files in {WORKSPACE / 'files' / 'top_level_c'} as json.", {"matches": ["only.txt"]}, "json"),
        Case("list_4", "list_files", f"List top-level txt files in {WORKSPACE / 'files' / 'top_level_c'} as text.", "only.txt", "text"),
        Case("search_1", "search_files", f"Find files containing cinnamon under {WORKSPACE / 'search' / 'pantry'} and return them as csv.", {"cake.txt", "tea.txt"}, "csv_set"),
        Case("search_2", "search_files", f"Find txt files containing garden under {WORKSPACE / 'search' / 'journal'} and return them as csv.", {"april.txt", "may.txt"}, "csv_set"),
        Case("search_3", "search_files", f"Find files containing library under {WORKSPACE / 'search' / 'stories'} and return them as text.", "library.txt", "text"),
        Case("write_1", "write_return", f"Write hello to {WORKSPACE / 'writes' / 'welcome.txt'} and return it.", "hello", "text"),
        Case("write_2", "write_return", f"Write \"fresh bread\" to {WORKSPACE / 'writes' / 'bread.txt'} and return it.", "fresh bread", "text"),
        Case("write_3", "write_return", f"Write sunshine to {WORKSPACE / 'writes' / 'day.txt'} and return it.", "sunshine", "text"),
        Case("write_4", "write_return", f"Write \"kind words\" to {WORKSPACE / 'writes' / 'quote.txt'} and return it.", "kind words", "text"),
        Case("shell_1", "shell_csv", "Using shell, print alpha then beta on separate lines and return them as csv.", "alpha,beta", "csv"),
        Case("shell_2", "shell_csv", "Using shell, print cedar then maple on separate lines and return them as csv.", "cedar,maple", "csv"),
        Case("sql_1", "sql", "count members in book_club_db and return the count only.", "3", "count"),
        Case("sql_2", "sql", "list name from members in book_club_db as csv.", "Alice,Bob,Carla", "csv"),
    ]


def _run_case(queue: mp.Queue, settings_payload: dict[str, Any], prompt: str) -> None:
    try:
        settings = Settings(
            workspace_root=Path(settings_payload["workspace_root"]),
            run_store_path=Path(settings_payload["run_store_path"]),
            sql_databases=dict(settings_payload.get("sql_databases", {})),
            sql_default_database=settings_payload.get("sql_default_database"),
        )
        engine = ExecutionEngine(settings)
        state = engine.run_spec(str(SPEC_PATH), {"task": prompt})
        queue.put({"ok": True, "state": state})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": str(exc)})


def _strict_pass(case: Case, content: str, llm_calls: int) -> bool:
    if llm_calls != 0:
        return False
    if case.mode == "json":
        try:
            return json.loads(content) == case.expected
        except json.JSONDecodeError:
            return False
    if case.mode == "csv_set":
        return set(_csv_items(content)) == set(case.expected) and content == ",".join(sorted(case.expected))
    return content == case.expected


def _semantic_pass(case: Case, content: str) -> bool:
    if case.mode == "json":
        try:
            return json.loads(content) == case.expected
        except json.JSONDecodeError:
            return False
    if case.mode == "csv_set":
        return set(_csv_items(content)) == set(case.expected)
    return content == case.expected


def _csv_items(content: str) -> list[str]:
    return [item.strip() for item in content.split(",") if item.strip()]


def main() -> None:
    sql_settings = _setup_workspace()
    settings_payload = {
        "workspace_root": str(WORKSPACE),
        "run_store_path": str(WORKSPACE / "runtime.db"),
        "sql_databases": sql_settings["sql_databases"],
        "sql_default_database": sql_settings["sql_default_database"],
    }
    results: list[dict[str, Any]] = []
    strict_pass_count = 0
    semantic_pass_count = 0
    failure_categories: Counter[str] = Counter()

    for case in _cases():
        queue: mp.Queue = mp.Queue()
        started = time.monotonic()
        process = mp.Process(target=_run_case, args=(queue, settings_payload, case.prompt))
        process.start()
        process.join(PROMPT_TIMEOUT_SECONDS)
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)

        if process.is_alive():
            process.terminate()
            process.join()
            failure_categories["timeout"] += 1
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "prompt": case.prompt,
                    "strict_pass": False,
                    "semantic_pass": False,
                    "failure_category": "timeout",
                    "llm_calls": None,
                    "latency_ms": elapsed_ms,
                }
            )
            continue

        payload = queue.get() if not queue.empty() else {"ok": False, "error": "no_result"}
        if not payload.get("ok"):
            failure_categories["runtime_error"] += 1
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "prompt": case.prompt,
                    "strict_pass": False,
                    "semantic_pass": False,
                    "failure_category": "runtime_error",
                    "error": payload.get("error"),
                    "llm_calls": None,
                    "latency_ms": elapsed_ms,
                }
            )
            continue

        state = dict(payload["state"])
        final_output = dict(state.get("final_output", {}))
        content = str(final_output.get("content", ""))
        llm_calls = int(dict(state.get("metrics", {})).get("llm_calls", 0))
        semantic_pass = _semantic_pass(case, content)
        strict_pass = semantic_pass and _strict_pass(case, content, llm_calls)

        if strict_pass:
            strict_pass_count += 1
        if semantic_pass:
            semantic_pass_count += 1

        if not semantic_pass:
            failure_category = "incorrect_output"
        elif llm_calls != 0:
            failure_category = "llm_called"
        elif not strict_pass:
            failure_category = "non_strict_contract"
        else:
            failure_category = None

        if failure_category:
            failure_categories[failure_category] += 1

        results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "prompt": case.prompt,
                "strict_pass": strict_pass,
                "semantic_pass": semantic_pass,
                "failure_category": failure_category,
                "llm_calls": llm_calls,
                "latency_ms": elapsed_ms,
                "content": content,
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(WORKSPACE),
        "prompt_count": len(results),
        "strict_pass_count": strict_pass_count,
        "semantic_pass_count": semantic_pass_count,
        "failure_categories": dict(failure_categories),
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
