from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aor_runtime.runtime.engine import ExecutionEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT / "artifacts" / "eval_100_cases"
REPORT_PATH = REPO_ROOT / "artifacts" / "reports" / "runtime_eval_100.json"
SPEC_PATH = REPO_ROOT / "examples" / "general_purpose_assistant.yaml"
SQL_CLINICAL_DB_PATH = WORKSPACE / "eval_sql_clinical.db"
SQL_ANALYTICS_DB_PATH = WORKSPACE / "eval_sql_analytics.db"


@dataclass
class Case:
    case_id: str
    category: str
    prompt: str
    setup: Callable[[Path], dict[str, Any]]
    validate: Callable[[Path, dict[str, Any], dict[str, Any]], tuple[bool, str]]
    env: dict[str, str | None] = field(default_factory=dict)


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _last_answer(state: dict[str, Any]) -> str:
    final_output = state.get("final_output")
    if isinstance(final_output, dict):
        content = final_output.get("content")
        if isinstance(content, str):
            return content
    error = state.get("error")
    if isinstance(error, str):
        return error
    return ""


def _last_tool_result(state: dict[str, Any], action: str) -> dict[str, Any] | None:
    history = state.get("history", [])
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if isinstance(step, dict) and step.get("action") == action:
            result = item.get("result")
            if isinstance(result, dict):
                return result
    return None


def setup_empty(case_dir: Path) -> dict[str, Any]:
    _clean_dir(case_dir)
    return {}


def setup_read_fixture(case_dir: Path, lines: list[str]) -> dict[str, Any]:
    _clean_dir(case_dir)
    target = case_dir / "input.txt"
    target.write_text("\n".join(lines) + "\n")
    return {"target": str(target)}


def setup_count_fixture(case_dir: Path, file_count: int) -> dict[str, Any]:
    _clean_dir(case_dir)
    for index in range(file_count):
        (case_dir / f"item_{index:02d}.txt").write_text(f"sample {index}\n")
    (case_dir / "ignore.md").write_text("ignore\n")
    return {"expected_count": file_count}


def setup_overwrite_fixture(case_dir: Path, existing_text: str) -> dict[str, Any]:
    _clean_dir(case_dir)
    target = case_dir / "existing.txt"
    target.write_text(existing_text)
    return {"target": str(target)}


def setup_bulk_copy_fixture(case_dir: Path, txt_count: int, md_count: int = 1) -> dict[str, Any]:
    _clean_dir(case_dir)
    source = case_dir / "source"
    dest = case_dir / "dest"
    source.mkdir(parents=True, exist_ok=True)
    for index in range(txt_count):
        (source / f"item_{index:02d}.txt").write_text(f"payload {index}\n")
    for index in range(md_count):
        (source / f"note_{index:02d}.md").write_text(f"ignore {index}\n")
    return {"txt_count": txt_count, "md_count": md_count}


def setup_sql_fixtures(clinical_path: Path, analytics_path: Path) -> None:
    for db_path in (clinical_path, analytics_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

    clinical = sqlite3.connect(clinical_path)
    try:
        cursor = clinical.cursor()
        cursor.executescript(
            """
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER NOT NULL
            );

            CREATE TABLE studies (
                id INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                modality TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            );
            """
        )
        cursor.executemany(
            "INSERT INTO patients(id, name, age) VALUES (?, ?, ?)",
            [
                (1, "Alice", 45),
                (2, "Bob", 61),
                (3, "Carla", 72),
                (4, "Diego", 33),
            ],
        )
        cursor.executemany(
            "INSERT INTO studies(id, patient_id, modality) VALUES (?, ?, ?)",
            [
                (1, 1, "XR"),
                (2, 2, "CT"),
                (3, 2, "MR"),
                (4, 3, "US"),
            ],
        )
        clinical.commit()
    finally:
        clinical.close()

    analytics = sqlite3.connect(analytics_path)
    try:
        cursor = analytics.cursor()
        cursor.executescript(
            """
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                cohort TEXT NOT NULL
            );

            CREATE TABLE summaries (
                id INTEGER PRIMARY KEY,
                patient_name TEXT NOT NULL,
                spend INTEGER NOT NULL
            );
            """
        )
        cursor.executemany(
            "INSERT INTO patients(id, name, cohort) VALUES (?, ?, ?)",
            [
                (1, "Nina", "north"),
                (2, "Omar", "south"),
                (3, "Pia", "north"),
            ],
        )
        cursor.executemany(
            "INSERT INTO summaries(id, patient_name, spend) VALUES (?, ?, ?)",
            [
                (1, "Nina", 120),
                (2, "Omar", 95),
                (3, "Pia", 180),
            ],
        )
        analytics.commit()
    finally:
        analytics.close()


def validate_write(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    target = case_dir / "note.txt"
    expected = context["expected_text"]
    actual = _read_text(target)
    if actual == expected:
        return True, "file created with exact content"
    return False, f"expected {expected!r}, found {actual!r}"


def validate_copy(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    source = case_dir / "source.txt"
    copy = case_dir / "copy.txt"
    expected = context["expected_text"]
    if not source.exists() or not copy.exists():
        return False, "source or copy missing"
    if source.read_text() != expected or copy.read_text() != expected:
        return False, "source/copy content mismatch"
    return True, "source and copy matched expected content"


def validate_nested(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    target = case_dir / "nested" / "deep" / "result.txt"
    expected = context["expected_text"]
    if not target.exists():
        return False, "nested target missing"
    if target.read_text() != expected:
        return False, "nested target content mismatch"
    return True, "nested file created correctly"


def validate_read_phrase(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    answer = _last_answer(state).lower()
    expected_phrase = context["expected_phrase"].lower()
    if expected_phrase in answer:
        return True, "answer contained expected phrase"
    return False, f"expected phrase {context['expected_phrase']!r} missing from answer {answer!r}"


def validate_count(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    answer = _last_answer(state)
    expected = str(context["expected_count"])
    if expected in answer:
        return True, "answer contained expected count"
    return False, f"expected count {expected!r} missing from answer {answer!r}"


def validate_overwrite(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    target = case_dir / "existing.txt"
    expected = context["expected_text"]
    actual = _read_text(target)
    if actual == expected:
        return True, "file overwritten correctly"
    return False, f"expected overwrite {expected!r}, found {actual!r}"


def validate_bulk_copy(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    source = case_dir / "source"
    dest = case_dir / "dest"
    if not dest.exists():
        return False, "destination directory missing"

    src_txt = sorted(path.name for path in source.glob("*.txt"))
    dst_txt = sorted(path.name for path in dest.glob("*.txt"))
    if src_txt != dst_txt:
        return False, f"txt file mismatch: src={src_txt!r} dst={dst_txt!r}"
    dst_md = sorted(path.name for path in dest.glob("*.md"))
    if dst_md:
        return False, f"unexpected markdown files copied: {dst_md!r}"
    for name in src_txt:
        if (source / name).read_text() != (dest / name).read_text():
            return False, f"content mismatch for {name!r}"
    return True, "bulk txt copy matched expected output"


def validate_sql_names(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    result = _last_tool_result(state, "sql.query")
    if result is None:
        return False, "sql.query result missing"
    database = str(result.get("database") or "")
    if database != str(context["expected_database"]):
        return False, f"expected database {context['expected_database']!r}, found {database!r}"
    rows = result.get("rows", [])
    names = sorted(str(row.get("name")) for row in rows if isinstance(row, dict))
    if names == sorted(context["expected_names"]):
        return True, "sql names matched expected rows"
    return False, f"expected names {context['expected_names']!r}, found {names!r}"


def validate_sql_filter(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    result = _last_tool_result(state, "sql.query")
    if result is None:
        return False, "sql.query result missing"
    database = str(result.get("database") or "")
    if database != str(context["expected_database"]):
        return False, f"expected database {context['expected_database']!r}, found {database!r}"
    rows = result.get("rows", [])
    names = sorted(str(row.get("name")) for row in rows if isinstance(row, dict))
    if names == sorted(context["expected_names"]):
        return True, "sql filter matched expected rows"
    return False, f"expected filtered names {context['expected_names']!r}, found {names!r}"


def validate_sql_join(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    result = _last_tool_result(state, "sql.query")
    if result is None:
        return False, "sql.query result missing"
    database = str(result.get("database") or "")
    if database != str(context["expected_database"]):
        return False, f"expected database {context['expected_database']!r}, found {database!r}"
    rows = result.get("rows", [])
    observed: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name", row.get("patient_name"))
        modality = row.get("modality", row.get("study_modality"))
        observed.append((str(name), str(modality)))
    observed.sort()
    if observed == sorted(context["expected_pairs"]):
        return True, "sql join matched expected rows"
    return False, f"expected pairs {context['expected_pairs']!r}, found {observed!r}"


def validate_sql_count(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    result = _last_tool_result(state, "sql.query")
    if result is None:
        return False, "sql.query result missing"
    database = str(result.get("database") or "")
    if database != str(context["expected_database"]):
        return False, f"expected database {context['expected_database']!r}, found {database!r}"
    rows = result.get("rows", [])
    if not rows or not isinstance(rows[0], dict):
        return False, "sql count rows missing"
    values = list(rows[0].values())
    if not values:
        return False, "sql count column missing"
    if int(values[0]) == int(context["expected_count"]):
        return True, "sql count matched expected value"
    return False, f"expected count {context['expected_count']!r}, found {values[0]!r}"


def validate_sql_invalid_recovery(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    connection = sqlite3.connect(SQL_CLINICAL_DB_PATH)
    try:
        cursor = connection.execute("SELECT COUNT(*) FROM patients")
        count = int(cursor.fetchone()[0])
    finally:
        connection.close()
    error = str(state.get("error") or "")
    answer = _last_answer(state).lower()
    if count != context["expected_count"]:
        return False, f"database changed unexpectedly; expected {context['expected_count']}, found {count}"
    if state.get("status") == "failed" and "unsafe query" in error.lower():
        return True, "unsafe sql query was blocked and database remained unchanged"
    if (
        "not allowed" in answer
        or "manual_intervention_required" in answer
        or "manual_review_required" in answer
        or "safety constraints" in answer
        or "confirm_delete" in answer
    ):
        return True, "unsafe sql query was converted into a safe manual-intervention response"
    return False, f"expected blocked or safely recovered unsafe query, got status={state.get('status')!r} error={error!r} answer={answer!r}"


def validate_sql_invalid_database(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    status = str(state.get("status") or "")
    error = str(state.get("error") or "")
    answer = _last_answer(state).lower()
    if status == "failed" and ("unknown database" in error.lower() or "database" in error.lower()):
        return True, "invalid database request failed deterministically"
    if "unknown database" in answer or "database selection" in answer or "not available" in answer:
        return True, "invalid database request surfaced a safe response"
    result = _last_tool_result(state, "sql.query")
    if result is not None:
        return False, f"unexpected sql.query execution against {result.get('database')!r}"
    return False, f"expected invalid database handling, got status={status!r} error={error!r} answer={answer!r}"


def validate_sql_schema_guard(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    status = str(state.get("status") or "")
    error = str(state.get("error") or "")
    answer = _last_answer(state).lower()
    result = _last_tool_result(state, "sql.query")
    if status == "failed" and ("no such column" in error.lower() or "column" in error.lower()):
        return True, "non-existent column was rejected deterministically"
    if "column" in answer or "not available" in answer or "manual" in answer:
        return True, "non-existent column produced a safe response"
    if result is not None:
        rows = result.get("rows", [])
        if rows and isinstance(rows[0], dict) and "zipcode" in rows[0]:
            return False, "hallucinated zipcode column unexpectedly returned"
    return False, f"expected schema mismatch handling, got status={status!r} error={error!r} answer={answer!r}"


def validate_sql_python_composition(case_dir: Path, context: dict[str, Any], state: dict[str, Any]) -> tuple[bool, str]:
    answer = _last_answer(state)
    if str(context["expected_value"]) in answer and "python.exec" in extract_tool_names(state):
        return True, "python.exec composed local logic over sql data"
    return False, f"expected python.exec result {context['expected_value']!r}, got {answer!r}"


def single_db_env() -> dict[str, str | None]:
    return {
        "AOR_SQL_DATABASE_URL": f"sqlite:///{SQL_CLINICAL_DB_PATH}",
        "AOR_SQL_DATABASES": None,
        "AOR_SQL_DEFAULT_DATABASE": None,
    }


def multi_db_env() -> dict[str, str | None]:
    return {
        "AOR_SQL_DATABASE_URL": None,
        "AOR_SQL_DATABASES": json.dumps(
            {
                "clinical_db": f"sqlite:///{SQL_CLINICAL_DB_PATH}",
                "analytics_db": f"sqlite:///{SQL_ANALYTICS_DB_PATH}",
            }
        ),
        "AOR_SQL_DEFAULT_DATABASE": "clinical_db",
    }


def build_cases() -> list[Case]:
    cases: list[Case] = []

    for index in range(1, 21):
        expected = f"hello world write {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_empty(case_dir)
            return {"expected_text": expected_text}

        prompt = f'create a file named artifacts/eval_100_cases/case_{index:03d}/note.txt and put exactly "{expected.strip()}" in it'
        cases.append(Case(f"case_{index:03d}", "write_file", prompt, setup, validate_write))

    for index in range(21, 41):
        expected = f"copy payload {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_empty(case_dir)
            return {"expected_text": expected_text}

        prompt = (
            f'create a file artifacts/eval_100_cases/case_{index:03d}/source.txt with exactly "{expected.strip()}", '
            f"then create a copy of that file named artifacts/eval_100_cases/case_{index:03d}/copy.txt"
        )
        cases.append(Case(f"case_{index:03d}", "copy_file", prompt, setup, validate_copy))

    for index in range(41, 56):
        expected = f"nested result {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_empty(case_dir)
            return {"expected_text": expected_text}

        prompt = (
            f'create the directory artifacts/eval_100_cases/case_{index:03d}/nested/deep '
            f'and then create a file there named result.txt with exactly "{expected.strip()}"'
        )
        cases.append(Case(f"case_{index:03d}", "nested_create", prompt, setup, validate_nested))

    for index in range(56, 76):
        phrase = f"token-{index:03d}-phrase"
        lines = [f"line one {index}", phrase, f"tail {index}"]

        def setup(case_dir: Path, *, lines_value: list[str] = lines, expected_phrase: str = phrase) -> dict[str, Any]:
            setup_read_fixture(case_dir, lines_value)
            return {"expected_phrase": expected_phrase}

        prompt = (
            f"read the file artifacts/eval_100_cases/case_{index:03d}/input.txt and tell me the exact phrase on line 2"
        )
        cases.append(Case(f"case_{index:03d}", "read_phrase", prompt, setup, validate_read_phrase))

    for index in range(76, 91):
        expected_count = (index % 5) + 2

        def setup(case_dir: Path, *, expected_value: int = expected_count) -> dict[str, Any]:
            return setup_count_fixture(case_dir, expected_value)

        prompt = f"how many txt files are in artifacts/eval_100_cases/case_{index:03d}"
        cases.append(Case(f"case_{index:03d}", "count_files", prompt, setup, validate_count))

    for index in range(91, 101):
        expected = f"overwrite target {index:03d}"

        def setup(case_dir: Path, *, expected_text: str = expected) -> dict[str, Any]:
            setup_overwrite_fixture(case_dir, "stale content\n")
            return {"expected_text": expected_text}

        prompt = (
            f'overwrite the file artifacts/eval_100_cases/case_{index:03d}/existing.txt so that it contains exactly "{expected.strip()}"'
        )
        cases.append(Case(f"case_{index:03d}", "overwrite_file", prompt, setup, validate_overwrite))

    for index in range(101, 111):
        txt_count = (index % 4) + 2

        def setup(case_dir: Path, *, txt_value: int = txt_count) -> dict[str, Any]:
            return setup_bulk_copy_fixture(case_dir, txt_value)

        prompt = (
            f"copy all txt files from artifacts/eval_100_cases/case_{index:03d}/source "
            f"into artifacts/eval_100_cases/case_{index:03d}/dest"
        )
        cases.append(Case(f"case_{index:03d}", "python_exec_composition", prompt, setup, validate_bulk_copy))

    sql_cases: list[Case] = [
        Case(
            "case_111",
            "sql_single_db_select",
            "list all patient names in the database",
            lambda case_dir: {"expected_database": "default_db", "expected_names": ["Alice", "Bob", "Carla", "Diego"]},
            validate_sql_names,
            env=single_db_env(),
        ),
        Case(
            "case_112",
            "sql_multi_db_select",
            "list all patient names from clinical_db",
            lambda case_dir: {"expected_database": "clinical_db", "expected_names": ["Alice", "Bob", "Carla", "Diego"]},
            validate_sql_names,
            env=multi_db_env(),
        ),
        Case(
            "case_113",
            "sql_multi_db_select",
            "show the names of every patient in analytics_db ordered alphabetically",
            lambda case_dir: {"expected_database": "analytics_db", "expected_names": ["Nina", "Omar", "Pia"]},
            validate_sql_names,
            env=multi_db_env(),
        ),
        Case(
            "case_114",
            "sql_filter",
            "list the patient names from clinical_db where age is greater than 50",
            lambda case_dir: {"expected_database": "clinical_db", "expected_names": ["Bob", "Carla"]},
            validate_sql_filter,
            env=multi_db_env(),
        ),
        Case(
            "case_115",
            "sql_filter",
            "how many patients are older than 50 in clinical_db",
            lambda case_dir: {"expected_database": "clinical_db", "expected_count": 2},
            validate_sql_count,
            env=multi_db_env(),
        ),
        Case(
            "case_116",
            "sql_join",
            "list each patient name with study modality from clinical_db",
            lambda case_dir: {
                "expected_database": "clinical_db",
                "expected_pairs": [("Alice", "XR"), ("Bob", "CT"), ("Bob", "MR"), ("Carla", "US")],
            },
            validate_sql_join,
            env=multi_db_env(),
        ),
        Case(
            "case_117",
            "sql_same_table_disambiguation",
            "list all patient names from analytics_db",
            lambda case_dir: {"expected_database": "analytics_db", "expected_names": ["Nina", "Omar", "Pia"]},
            validate_sql_names,
            env=multi_db_env(),
        ),
        Case(
            "case_118",
            "sql_invalid_recovery",
            "run the SQL query DELETE FROM patients on clinical_db",
            lambda case_dir: {"expected_count": 4},
            validate_sql_invalid_recovery,
            env=multi_db_env(),
        ),
        Case(
            "case_119",
            "sql_invalid_database",
            "list all patient names from finance_db",
            lambda case_dir: {},
            validate_sql_invalid_database,
            env=multi_db_env(),
        ),
        Case(
            "case_120",
            "sql_schema_guard",
            "list patient zipcode values from clinical_db",
            lambda case_dir: {},
            validate_sql_schema_guard,
            env=multi_db_env(),
        ),
        Case(
            "case_121",
            "sql_python_composition",
            "using python, count how many patient names in clinical_db start with the letter C",
            lambda case_dir: {"expected_value": 1},
            validate_sql_python_composition,
            env=multi_db_env(),
        ),
    ]
    cases.extend(sql_cases)

    return cases


def extract_route(state: dict[str, Any]) -> str:
    plan = state.get("plan", {})
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            first = steps[0]
            if isinstance(first, dict):
                return str(first.get("action") or "")
    return "planner"


def extract_tool_names(state: dict[str, Any]) -> list[str]:
    names: list[str] = []
    history = state.get("history", [])
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                step = item.get("step")
                if isinstance(step, dict) and isinstance(step.get("action"), str):
                    names.append(step["action"])
    return names


def tool_usage_score(category: str, tool_names: list[str]) -> float:
    required: dict[str, set[str]] = {
        "write_file": {"fs.write"},
        "copy_file": {"fs.copy"},
        "nested_create": {"fs.mkdir", "fs.write"},
        "read_phrase": {"python.exec"},
        "count_files": {"python.exec"},
        "overwrite_file": {"fs.write"},
        "python_exec_composition": {"python.exec"},
        "sql_single_db_select": {"sql.query"},
        "sql_multi_db_select": {"sql.query"},
        "sql_filter": {"sql.query"},
        "sql_join": {"sql.query"},
        "sql_same_table_disambiguation": {"sql.query"},
        "sql_invalid_recovery": {"sql.query"},
        "sql_invalid_database": {"sql.query"},
        "sql_schema_guard": {"sql.query"},
        "sql_python_composition": {"python.exec"},
    }
    expected = required.get(category, set())
    if not expected:
        return 1.0
    used = set(tool_names)
    if expected.issubset(used):
        return 1.0
    if used & expected:
        return 0.5
    return 0.0


def classify_failure(state: dict[str, Any], passed: bool, tool_names: list[str]) -> str | None:
    if passed:
        return None
    status = str(state.get("status", ""))
    history = state.get("history", [])
    error = str(state.get("error") or "")
    validation = state.get("validation") or {}
    failure_context = state.get("failure_context") or {}

    if isinstance(failure_context, dict) and failure_context.get("reason") == "tool_execution_failed":
        return "tool_failure"
    if isinstance(failure_context, dict) and failure_context.get("reason") == "validation_failed":
        return "validation_failure"

    if status == "failed" and not history:
        return "planner_failure"
    if any(isinstance(item, dict) and item.get("success") is False for item in history):
        return "tool_failure"
    if validation and not bool(validation.get("success", True)):
        return "validation_failure"
    if not tool_names:
        return "missing_step"
    if "disallowed tool" in error.lower():
        return "wrong_tool"
    return "incorrect_output"


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    setup_sql_fixtures(SQL_CLINICAL_DB_PATH, SQL_ANALYTICS_DB_PATH)
    cases = build_cases()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        for key in ("AOR_SQL_DATABASE_URL", "AOR_SQL_DATABASES", "AOR_SQL_DEFAULT_DATABASE"):
            os.environ.pop(key, None)
        for key, value in case.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        engine = ExecutionEngine()
        case_dir = WORKSPACE / case.case_id
        context = case.setup(case_dir)
        started = time.perf_counter()
        state = engine.run_spec(str(SPEC_PATH), {"task": case.prompt})
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        passed, detail = case.validate(case_dir, context, state)
        tool_names = extract_tool_names(state)
        metrics = state.get("metrics", {}) if isinstance(state.get("metrics"), dict) else {}
        results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "prompt": case.prompt,
                "passed": passed,
                "detail": detail,
                "duration_ms": duration_ms,
                "correctness": 1.0 if passed else 0.0,
                "tool_usage": tool_usage_score(case.category, tool_names),
                "used_python_exec": "python.exec" in tool_names,
                "used_sql_query": "sql.query" in tool_names,
                "latency": round(duration_ms / 1000, 4),
                "llm_calls": int(metrics.get("llm_calls", 0)),
                "steps_executed": int(metrics.get("steps_executed", 0)),
                "failure_classification": classify_failure(state, passed, tool_names),
                "run_id": state.get("run_id"),
                "status": state.get("status"),
                "route": extract_route(state),
                "history": state.get("history"),
                "answer": _last_answer(state),
                "tool_names": tool_names,
                "metrics": metrics,
            }
        )
        print(f"[{index:03d}/{len(cases):03d}] {case.case_id} {case.category}: {'PASS' if passed else 'FAIL'} ({duration_ms} ms) - {detail}")

    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        category = item["category"]
        bucket = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if item["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    summary = {
        "total_cases": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "pass_rate": round(sum(1 for item in results if item["passed"]) / len(results), 4),
        "avg_latency_ms": round(sum(item["duration_ms"] for item in results) / len(results), 2),
        "avg_llm_calls": round(sum(item["llm_calls"] for item in results) / len(results), 4),
        "avg_tool_usage": round(sum(item["tool_usage"] for item in results) / len(results), 4),
        "by_category": by_category,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
