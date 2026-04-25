from __future__ import annotations

from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.executor import summarize_final_output


def test_sql_count_goal_returns_number_only() -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="sql.query", args={"database": "dicom", "query": "SELECT COUNT(*) AS patient_count FROM patient"}),
        result={"database": "dicom", "row_count": 1, "rows": [{"patient_count": 10000}]},
        success=True,
    )

    output = summarize_final_output("Count patients in dicom and return the count.", [log])

    assert output["content"] == "10000"


def test_python_csv_goal_returns_csv_string_only() -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="python.exec", args={"code": "result = ','.join(inputs['rows'])"}),
        result={"success": True, "output": "alpha,beta,gamma", "result": "alpha,beta,gamma"},
        success=True,
    )

    output = summarize_final_output("Return the result as csv.", [log])

    assert output["content"] == "alpha,beta,gamma"


def test_python_json_goal_returns_json_object_only() -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="python.exec", args={"code": "result = {'count': 3, 'tables': ['patient']}"}),
        result={"success": True, "output": "{\"count\": 3, \"tables\": [\"patient\"]}", "result": {"count": 3, "tables": ["patient"]}},
        success=True,
    )

    output = summarize_final_output("Return JSON with the counts.", [log])

    assert output["content"] == "{\"count\": 3, \"tables\": [\"patient\"]}"


def test_shell_count_goal_extracts_number_only() -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="shell.exec", args={"command": "wc -l notes.txt"}),
        result={"stdout": "3 /tmp/notes.txt\n", "stderr": "", "returncode": 0},
        success=True,
    )

    output = summarize_final_output("Return just the count.", [log])

    assert output["content"] == "3"
