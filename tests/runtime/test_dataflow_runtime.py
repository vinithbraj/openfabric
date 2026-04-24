from __future__ import annotations

from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep
from aor_runtime.runtime.dataflow import normalize_execution_plan_dataflow, resolve_execution_step
from aor_runtime.runtime.executor import PlanExecutor
from aor_runtime.tools.factory import build_tool_registry


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")


def test_resolve_execution_step_supports_nested_refs(tmp_path: Path) -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 2,
            "action": "fs.write",
            "input": ["csv_result"],
            "args": {
                "path": "patients.csv",
                "content": {"$ref": "csv_result", "path": "csv"},
            },
        }
    )

    resolved = resolve_execution_step(step, {"csv_result": {"csv": "Arun,Lena,Maya"}})

    assert resolved.args["content"] == "Arun,Lena,Maya"


def test_normalize_execution_plan_dataflow_backfills_inputs_from_refs() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "sql.query", "args": {"database": "clinical_db", "query": "SELECT name FROM patients"}, "output": "patient_rows"},
                {
                    "id": 2,
                    "action": "python.exec",
                    "args": {
                        "inputs": {"rows": {"$ref": "patient_rows", "path": "rows"}},
                        "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                    },
                    "output": "patient_csv",
                },
                {
                    "id": 3,
                    "action": "fs.write",
                    "args": {"path": "patients.csv", "content": {"$ref": "patient_csv", "path": "csv"}},
                },
            ]
        }
    )

    normalize_execution_plan_dataflow(plan)

    assert plan.steps[1].input == ["patient_rows"]
    assert plan.steps[2].input == ["patient_csv"]


def test_python_exec_receives_inputs_dict(tmp_path: Path) -> None:
    executor = PlanExecutor(build_tool_registry(_settings(tmp_path)))
    step = ExecutionStep.model_validate(
        {
            "id": 1,
            "action": "python.exec",
            "output": "sum_result",
            "args": {
                "inputs": {"values": [1, 2, 3]},
                "code": "result = {'total': sum(inputs['values'])}",
            },
        }
    )

    log = executor.execute_step(step)

    assert log.success is True
    assert log.step.output == "sum_result"
    assert log.result["output"] == "{\"total\": 6}"
    assert log.result["result"] == {"total": 6}


def test_executor_resolves_ref_before_fs_write(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    executor = PlanExecutor(build_tool_registry(settings))
    step = ExecutionStep.model_validate(
        {
            "id": 2,
            "action": "fs.write",
            "input": ["csv_result"],
            "args": {
                "path": "patients.csv",
                "content": {"$ref": "csv_result", "path": "csv"},
            },
        }
    )

    log = executor.execute_step(step, step_outputs={"csv_result": {"csv": "Arun,Lena,Maya"}})

    assert log.success is True
    assert log.step.args["content"] == "Arun,Lena,Maya"
    assert (tmp_path / "patients.csv").read_text() == "Arun,Lena,Maya"


def test_executor_fails_on_unknown_ref(tmp_path: Path) -> None:
    executor = PlanExecutor(build_tool_registry(_settings(tmp_path)))
    step = ExecutionStep.model_validate(
        {
            "id": 1,
            "action": "fs.write",
            "input": ["missing_alias"],
            "args": {
                "path": "patients.csv",
                "content": {"$ref": "missing_alias", "path": "csv"},
            },
        }
    )

    log = executor.execute_step(step)

    assert log.success is False
    assert "Unknown step output reference" in str(log.error)


def test_resolve_execution_step_supports_python_result_field_shortcuts(tmp_path: Path) -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 2,
            "action": "fs.write",
            "input": ["patient_csv"],
            "args": {
                "path": "patients.csv",
                "content": {"$ref": "patient_csv", "path": "csv"},
            },
        }
    )

    resolved = resolve_execution_step(
        step,
        {
            "patient_csv": {
                "success": True,
                "output": "{\"csv\": \"Arun,Lena,Maya\"}",
                "result": {"csv": "Arun,Lena,Maya"},
            }
        },
    )

    assert resolved.args["content"] == "Arun,Lena,Maya"


def test_resolve_execution_step_supports_textual_python_alias_paths(tmp_path: Path) -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 3,
            "action": "fs.write",
            "input": ["size_summary"],
            "args": {
                "path": "summary.json",
                "content": {"$ref": "size_summary", "path": "json_string"},
            },
        }
    )

    resolved = resolve_execution_step(
        step,
        {
            "size_summary": {
                "success": True,
                "output": "{\"file_count\": 2, \"total_size_bytes\": 11}",
                "result": "{\"file_count\": 2, \"total_size_bytes\": 11}",
            }
        },
    )

    assert resolved.args["content"] == "{\"file_count\": 2, \"total_size_bytes\": 11}"


def test_normalize_execution_plan_dataflow_backfills_python_inputs_from_prior_shell_output() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "shell.exec",
                    "args": {"command": "find src -name '*.py'"},
                    "output": "py_files",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["py_files"],
                    "args": {"code": "result = {'csv': ','.join(inputs['py_files'].splitlines())}"},
                    "output": "csv_result",
                },
            ]
        }
    )

    normalize_execution_plan_dataflow(plan)

    assert plan.steps[1].args["inputs"] == {"py_files": {"$ref": "py_files", "path": "stdout"}}


def test_executor_ignores_internal_canonicalizer_args(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    executor = PlanExecutor(build_tool_registry(settings))
    target = tmp_path / "summary.json"
    target.write_text('{"ok": true}')
    step = ExecutionStep.model_validate(
        {
            "id": 3,
            "action": "fs.read",
            "args": {"path": str(target), "__canonicalizer_added": True},
        }
    )

    log = executor.execute_step(step)

    assert log.success is True
    assert log.result["content"] == '{"ok": true}'
