from __future__ import annotations

from pathlib import Path

import pytest

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionPlan, ExecutionStep
from aor_runtime.runtime.dataflow import normalize_execution_plan_dataflow, resolve_execution_step
from aor_runtime.runtime.executor import PlanExecutor
from aor_runtime.runtime.lifecycle import ActiveRunRegistry, CancellationToken, ToolInvocationContext
from aor_runtime.runtime.plan_canonicalizer import canonicalize_plan
from aor_runtime.tools.factory import build_tool_registry
from aor_runtime.tools.python_exec import PythonExecTool


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


def test_normalize_execution_plan_dataflow_defaults_null_ref_path() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "slurm.metrics", "args": {"metric_group": "cluster_summary"}, "output": "cluster_metrics"},
                {
                    "id": 2,
                    "action": "text.format",
                    "args": {"source": {"$ref": "cluster_metrics", "path": None}, "format": "markdown"},
                    "output": "formatted_output",
                },
            ]
        }
    )

    normalize_execution_plan_dataflow(plan)
    resolved = resolve_execution_step(
        plan.steps[1],
        {"cluster_metrics": {"payload": {"queue_count": 12, "pending_jobs": 10}, "metric_group": "cluster_summary"}},
    )

    assert plan.steps[1].args["source"] == {"$ref": "cluster_metrics", "path": "payload"}
    assert resolved.args["source"] == {"queue_count": 12, "pending_jobs": 10}


def test_resolve_execution_step_treats_null_path_as_whole_result() -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 2,
            "action": "text.format",
            "input": ["summary"],
            "args": {"source": {"$ref": "summary", "path": None}, "format": "markdown"},
        }
    )

    resolved = resolve_execution_step(step, {"summary": {"count": 3}})

    assert resolved.args["source"] == {"count": 3}


def test_invalid_ref_path_lists_available_fields() -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 2,
            "action": "text.format",
            "input": ["summary"],
            "args": {"source": {"$ref": "summary", "path": "missing"}, "format": "markdown"},
        }
    )

    with pytest.raises(ValueError) as exc:
        resolve_execution_step(step, {"summary": {"payload": {"count": 3}, "metric_group": "cluster_summary"}})

    message = str(exc.value)
    assert "Reference path not found: missing" in message
    assert "payload" in message
    assert "metric_group" in message
    assert "None" not in message


def test_shell_exit_code_alias_resolves_to_returncode() -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 2,
            "action": "runtime.return",
            "input": ["shell_result"],
            "args": {"value": {"$ref": "shell_result", "path": "exit_code"}, "mode": "count"},
        }
    )

    resolved = resolve_execution_step(step, {"shell_result": {"stdout": "42\n", "returncode": 0}})

    assert resolved.args["value"] == 0


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


def test_resolve_execution_step_coerces_bare_python_exec_result_for_fs_write(tmp_path: Path) -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 3,
            "action": "fs.write",
            "input": ["summary_json"],
            "args": {
                "path": "summary.json",
                "content": {"$ref": "summary_json"},
            },
        }
    )

    resolved = resolve_execution_step(
        step,
        {
            "summary_json": {
                "success": True,
                "output": "{\"studies\": 12, \"series\": 48}",
                "result": "{\"studies\": 12, \"series\": 48}",
                "error": None,
            }
        },
    )

    assert resolved.args["content"] == "{\"studies\": 12, \"series\": 48}"


def test_resolve_execution_step_coerces_structured_python_exec_result_for_fs_write(tmp_path: Path) -> None:
    step = ExecutionStep.model_validate(
        {
            "id": 3,
            "action": "fs.write",
            "input": ["histogram_result"],
            "args": {
                "path": "histogram.json",
                "content": {"$ref": "histogram_result"},
            },
        }
    )

    resolved = resolve_execution_step(
        step,
        {
            "histogram_result": {
                "success": True,
                "output": "{\"histogram\": {\"2\": 500}}",
                "result": {"histogram": {"2": 500}},
                "error": None,
            }
        },
    )

    assert resolved.args["content"] == "{\"histogram\": {\"2\": 500}}"


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


def test_normalize_execution_plan_dataflow_backfills_python_inputs_from_prior_python_output() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = {'count': 3}"},
                    "output": "summary",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["summary"],
                    "args": {"code": "summary = inputs['summary']; result = summary['count']"},
                },
            ]
        }
    )

    normalize_execution_plan_dataflow(plan)

    assert plan.steps[1].args["inputs"] == {"summary": {"$ref": "summary", "path": "result"}}


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


def test_executor_runs_canonicalized_structured_write_as_python_writer(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    executor = PlanExecutor(build_tool_registry(settings))
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "fs.write",
                    "args": {"path": "reports/histogram.json", "content": {"histogram": {"2": 500}}},
                }
            ]
        }
    )

    canonicalized = canonicalize_plan(plan, "Write reports/histogram.json", ["fs.write", "python.exec"])
    log = executor.execute_step(canonicalized.plan.steps[0])

    assert log.success is True
    assert log.step.action == "python.exec"
    assert (tmp_path / "reports" / "histogram.json").read_text() == '{"histogram": {"2": 500}}'


def test_python_exec_supports_safe_json_serialization_fallback(tmp_path: Path) -> None:
    executor = PlanExecutor(build_tool_registry(_settings(tmp_path)))
    step = ExecutionStep.model_validate(
        {
            "id": 1,
            "action": "python.exec",
            "args": {
                "code": "result = _json_dumps_safe({'value': 1 + 2j})",
            },
        }
    )

    log = executor.execute_step(step)

    assert log.success is True
    assert log.result["result"] == '{"value": "(1+2j)"}'


def test_python_exec_tool_unregisters_managed_worker_after_result(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = ActiveRunRegistry(shutdown_grace_seconds=0.5, process_kill_grace_seconds=0.1)
    context = ToolInvocationContext(
        cancellation=CancellationToken(),
        process_registry=registry,
        worker_join_timeout_seconds=0.5,
        tool_process_kill_grace_seconds=0.1,
    )

    result = PythonExecTool(settings).invoke(
        {"code": "result = {'items': list(range(1000))}"},
        context=context,
    )

    assert result["success"] is True
    assert registry.active_process_count() == 0
