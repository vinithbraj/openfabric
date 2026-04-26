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
WORKSPACE = REPO_ROOT / "artifacts" / "compound_deterministic_smoke"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "compound_deterministic_smoke_report.json"
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
    _write(notes / "meeting.txt", "agenda\nbudget\nfollow-up\n")
    _write(notes / "shopping.txt", "bread\nmilk\ntea\n")
    _write(notes / "poem.txt", "roses\nviolets\nstarlight\n")
    _write(notes / "todo.txt", "email\ncall\nlunch\n")
    _write(notes / "hello.txt", "hello\nworld\nkind\n")

    files_a = WORKSPACE / "files" / "a"
    _write(files_a / "root_a.txt", "a\n")
    _write(files_a / "root_b.txt", "b\n")
    _write(files_a / "nested" / "child.txt", "c\n")
    _write(files_a / "ignore.md", "ignore\n")

    files_b = WORKSPACE / "files" / "b"
    _write(files_b / "alpha.txt", "alpha\n")
    _write(files_b / "beta.txt", "beta\n")
    _write(files_b / "nested" / "gamma.txt", "gamma\n")

    files_c = WORKSPACE / "files" / "c"
    _write(files_c / "first.txt", "first\n")
    _write(files_c / "second.txt", "second\n")
    _write(files_c / "nested" / "third.txt", "third\n")

    files_d = WORKSPACE / "files" / "d"
    _write(files_d / "one.txt", "one\n")
    _write(files_d / "two.txt", "two\n")
    _write(files_d / "nested" / "three.txt", "three\n")

    files_e = WORKSPACE / "files" / "e"
    _write(files_e / "apple.txt", "apple\n")
    _write(files_e / "banana.txt", "banana\n")
    _write(files_e / "nested" / "citrus.txt", "citrus\n")

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
            INSERT INTO members(id, name, city) VALUES
                (1, 'Alice', 'Portland'),
                (2, 'Bob', 'Seattle'),
                (3, 'Carla', 'Portland');
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
        Case("count_1", "count_save_return", f"Count the top-level txt files in {WORKSPACE / 'files' / 'a'}, save the result to {WORKSPACE / 'outputs' / 'count_a.txt'}, and return the file contents.", "2", "text"),
        Case("count_2", "count_save_return", f"Count the top-level txt files in {WORKSPACE / 'files' / 'b'}, save the result to {WORKSPACE / 'outputs' / 'count_b.txt'}, and return the file contents.", "2", "text"),
        Case("count_3", "count_save_return", f"Count the top-level txt files in {WORKSPACE / 'files' / 'c'}, save the result to {WORKSPACE / 'outputs' / 'count_c.txt'}, and return the file contents.", "2", "text"),
        Case("count_4", "count_save_return", f"Count the top-level txt files in {WORKSPACE / 'files' / 'd'}, save the result to {WORKSPACE / 'outputs' / 'count_d.txt'}, and return the file contents.", "2", "text"),
        Case("count_5", "count_save_return", f"Count the top-level txt files in {WORKSPACE / 'files' / 'e'}, save the result to {WORKSPACE / 'outputs' / 'count_e.txt'}, and return the file contents.", "2", "text"),
        Case("list_1", "list_csv_save_return", f"List the top-level txt files in {WORKSPACE / 'files' / 'a'} as csv, write them to {WORKSPACE / 'outputs' / 'list_a.csv'}, and return the content.", "root_a.txt,root_b.txt", "text"),
        Case("list_2", "list_csv_save_return", f"List the top-level txt files in {WORKSPACE / 'files' / 'b'} as csv, write them to {WORKSPACE / 'outputs' / 'list_b.csv'}, and return the content.", "alpha.txt,beta.txt", "text"),
        Case("list_3", "list_csv_save_return", f"List the top-level txt files in {WORKSPACE / 'files' / 'c'} as csv, write them to {WORKSPACE / 'outputs' / 'list_c.csv'}, and return the content.", "first.txt,second.txt", "text"),
        Case("list_4", "list_csv_save_return", f"List the top-level txt files in {WORKSPACE / 'files' / 'd'} as csv, write them to {WORKSPACE / 'outputs' / 'list_d.csv'}, and return the content.", "one.txt,two.txt", "text"),
        Case("list_5", "list_csv_save_return", f"List the top-level txt files in {WORKSPACE / 'files' / 'e'} as csv, write them to {WORKSPACE / 'outputs' / 'list_e.csv'}, and return the content.", "apple.txt,banana.txt", "text"),
        Case("read_1", "read_upper_return", f"Read line 2 from {WORKSPACE / 'notes' / 'meeting.txt'}, convert it to uppercase, and return it.", "BUDGET", "text"),
        Case("read_2", "read_upper_return", f"Read line 2 from {WORKSPACE / 'notes' / 'shopping.txt'}, convert it to uppercase, and return it.", "MILK", "text"),
        Case("read_3", "read_upper_return", f"Read line 1 from {WORKSPACE / 'notes' / 'poem.txt'}, convert it to uppercase, and return it.", "ROSES", "text"),
        Case("read_4", "read_upper_return", f"Read line 3 from {WORKSPACE / 'notes' / 'todo.txt'}, convert it to uppercase, and return it.", "LUNCH", "text"),
        Case("read_5", "read_upper_return", f"Read line 2 from {WORKSPACE / 'notes' / 'hello.txt'}, convert it to uppercase, and return it.", "WORLD", "text"),
        Case("save_1", "read_upper_save_return", f"Read line 2 from {WORKSPACE / 'notes' / 'meeting.txt'}, convert it to uppercase, save it to {WORKSPACE / 'outputs' / 'line1.txt'}, and return the saved file.", "BUDGET", "text"),
        Case("save_2", "read_upper_save_return", f"Read line 2 from {WORKSPACE / 'notes' / 'shopping.txt'}, convert it to uppercase, save it to {WORKSPACE / 'outputs' / 'line2.txt'}, and return the saved file.", "MILK", "text"),
        Case("save_3", "read_upper_save_return", f"Read line 1 from {WORKSPACE / 'notes' / 'poem.txt'}, convert it to uppercase, save it to {WORKSPACE / 'outputs' / 'line3.txt'}, and return the saved file.", "ROSES", "text"),
        Case("save_4", "read_upper_save_return", f"Read line 3 from {WORKSPACE / 'notes' / 'todo.txt'}, convert it to uppercase, save it to {WORKSPACE / 'outputs' / 'line4.txt'}, and return the saved file.", "LUNCH", "text"),
        Case("save_5", "read_upper_save_return", f"Read line 2 from {WORKSPACE / 'notes' / 'hello.txt'}, convert it to uppercase, save it to {WORKSPACE / 'outputs' / 'line5.txt'}, and return the saved file.", "WORLD", "text"),
        Case("write_1", "create_write_return", f"Create {WORKSPACE / 'outputs' / 'welcome1.txt'} with exact content 'kind words' and then return the content only.", "kind words", "text"),
        Case("write_2", "create_write_return", f"Create {WORKSPACE / 'outputs' / 'welcome2.txt'} with exact content 'hello world' and then return the content only.", "hello world", "text"),
        Case("write_3", "create_write_return", f"Create {WORKSPACE / 'outputs' / 'welcome3.txt'} with exact content 'fresh bread' and then return the content only.", "fresh bread", "text"),
        Case("write_4", "create_write_return", f"Create {WORKSPACE / 'outputs' / 'welcome4.txt'} with exact content 'gentle rain' and then return the content only.", "gentle rain", "text"),
        Case("write_5", "create_write_return", f"Create {WORKSPACE / 'outputs' / 'welcome5.txt'} with exact content 'quiet library' and then return the content only.", "quiet library", "text"),
        Case("sql_1", "sql_upper_csv_save_return", f"List name from members in book_club_db, convert them to uppercase csv, save the result to {WORKSPACE / 'outputs' / 'names1.csv'}, and return the file contents.", "ALICE,BOB,CARLA", "text"),
        Case("sql_2", "sql_upper_csv_save_return", f"List name from members in book_club_db, convert them to uppercase csv, save the result to {WORKSPACE / 'outputs' / 'names2.csv'}, and return the file contents.", "ALICE,BOB,CARLA", "text"),
        Case("sql_3", "sql_upper_csv_save_return", f"List name from members in book_club_db, convert them to uppercase csv, save the result to {WORKSPACE / 'outputs' / 'names3.csv'}, and return the file contents.", "ALICE,BOB,CARLA", "text"),
        Case("sql_4", "sql_upper_csv_save_return", f"List name from members in book_club_db, convert them to uppercase csv, save the result to {WORKSPACE / 'outputs' / 'names4.csv'}, and return the file contents.", "ALICE,BOB,CARLA", "text"),
        Case("sql_5", "sql_upper_csv_save_return", f"List name from members in book_club_db, convert them to uppercase csv, save the result to {WORKSPACE / 'outputs' / 'names5.csv'}, and return the file contents.", "ALICE,BOB,CARLA", "text"),
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
    return llm_calls == 0 and content == case.expected


def _semantic_pass(case: Case, content: str) -> bool:
    return content == case.expected


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
    fallback_prompts: list[str] = []

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
        content = str(dict(state.get("final_output", {})).get("content", ""))
        llm_calls = int(dict(state.get("metrics", {})).get("llm_calls", 0))
        semantic_pass = _semantic_pass(case, content)
        strict_pass = _strict_pass(case, content, llm_calls)

        if strict_pass:
            strict_pass_count += 1
        if semantic_pass:
            semantic_pass_count += 1

        failure_category: str | None = None
        if llm_calls != 0:
            fallback_prompts.append(case.prompt)
            failure_category = "llm_called"
        if not semantic_pass:
            failure_category = "incorrect_output"
        if not semantic_pass and llm_calls != 0:
            failure_category = "incorrect_output"
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
        "fallback_prompts": fallback_prompts,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
