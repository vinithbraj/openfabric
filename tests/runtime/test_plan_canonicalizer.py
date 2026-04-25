from __future__ import annotations

from aor_runtime.core.contracts import ExecutionPlan
from aor_runtime.runtime.plan_canonicalizer import (
    CANONICALIZER_ADDED_ARG,
    CANONICALIZER_WRITE_PATH_ARG,
    canonicalize_plan,
)


def test_canonicalizer_rewrites_generic_aliases_and_downstream_refs() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "SELECT name FROM patient"},
                    "output": "rows",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["rows"],
                    "args": {
                        "inputs": {"rows": {"$ref": "rows", "path": "rows"}},
                        "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                    },
                    "output": "csv",
                },
                {
                    "id": 3,
                    "action": "fs.write",
                    "input": ["csv"],
                    "args": {"path": "patients.csv", "content": {"$ref": "csv", "path": "csv"}},
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Query patient names and save them to patients.csv", ["sql.query", "python.exec", "fs.write"])

    assert result.changed is True
    assert result.plan.steps[0].output == "step_1_rows"
    assert result.plan.steps[1].input == ["step_1_rows"]
    assert result.plan.steps[1].args["inputs"]["rows"]["$ref"] == "step_1_rows"
    assert result.plan.steps[1].output == "step_2_data"
    assert result.plan.steps[2].input == ["step_2_data"]
    assert result.plan.steps[2].args["content"]["$ref"] == "step_2_data"


def test_canonicalizer_repairs_unknown_ref_from_tool_type() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "SELECT name FROM patient"},
                    "output": "patient_rows",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "args": {
                        "inputs": {"rows": {"$ref": "rows", "path": "rows"}},
                        "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                    },
                    "output": "patient_csv",
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Query patient names and format them as csv", ["sql.query", "python.exec"])

    assert result.plan.steps[1].input == ["patient_rows"]
    assert result.plan.steps[1].args["inputs"]["rows"]["$ref"] == "patient_rows"


def test_canonicalizer_repairs_python_inputs_literal_aliases() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "SELECT COUNT(*) AS study_count FROM study"},
                    "output": "study_count",
                },
                {
                    "id": 2,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "SELECT COUNT(*) AS series_count FROM series"},
                    "output": "series_count",
                },
                {
                    "id": 3,
                    "action": "python.exec",
                    "input": ["study_count", "series_count"],
                    "args": {
                        "inputs": {"study_count": "study_count", "series_count": "series_count"},
                        "code": "result = {'json': {'study_count': inputs['study_count'][0]['study_count'], 'series_count': inputs['series_count'][0]['series_count']}}",
                    },
                    "output": "summary_json",
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Count studies and series and return json", ["sql.query", "python.exec"])

    assert result.plan.steps[2].args["inputs"]["study_count"] == {"$ref": "study_count", "path": "rows"}
    assert result.plan.steps[2].args["inputs"]["series_count"] == {"$ref": "series_count", "path": "rows"}


def test_canonicalizer_repairs_unique_relative_path_and_leaves_ambiguous_paths_alone() -> None:
    unique_plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.read", "args": {"path": "nested/egg_report.txt"}, "output": "report_content"},
                {"id": 2, "action": "fs.copy", "args": {"src": "egg_report.txt", "dst": "archive/egg_report.txt"}},
            ]
        }
    )

    unique_result = canonicalize_plan(unique_plan, "Archive egg_report.txt", ["fs.read", "fs.copy"])

    assert unique_result.plan.steps[1].args["src"] == "nested/egg_report.txt"

    ambiguous_plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.read", "args": {"path": "nested/egg_report.txt"}, "output": "first_report"},
                {"id": 2, "action": "fs.read", "args": {"path": "other/egg_report.txt"}, "output": "second_report"},
                {"id": 3, "action": "fs.copy", "args": {"src": "egg_report.txt", "dst": "archive/egg_report.txt"}},
            ]
        }
    )

    ambiguous_result = canonicalize_plan(ambiguous_plan, "Archive egg_report.txt", ["fs.read", "fs.copy"])

    assert ambiguous_result.plan.steps[2].args["src"] == "egg_report.txt"


def test_canonicalizer_preserves_unique_absolute_paths() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.read", "args": {"path": "/tmp/data/nested/egg_report.txt"}, "output": "report_content"},
                {"id": 2, "action": "fs.copy", "args": {"src": "egg_report.txt", "dst": "/tmp/archive/egg_report.txt"}},
            ]
        }
    )

    result = canonicalize_plan(plan, "Archive /tmp/data/nested/egg_report.txt", ["fs.read", "fs.copy"])

    assert result.plan.steps[1].args["src"] == "/tmp/data/nested/egg_report.txt"


def test_canonicalizer_appends_text_readback_and_marks_added_step() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.mkdir", "args": {"path": "reports"}},
                {"id": 2, "action": "fs.write", "args": {"path": "reports/summary.json", "content": "{\"ok\":true}"}},
            ]
        }
    )

    result = canonicalize_plan(
        plan,
        "Write reports/summary.json and return it",
        ["fs.mkdir", "fs.write", "fs.read"],
    )

    assert [step.id for step in result.plan.steps] == [1, 2, 3]
    assert result.plan.steps[-1].action == "fs.read"
    assert result.plan.steps[-1].args["path"] == "reports/summary.json"
    assert result.plan.steps[-1].args[CANONICALIZER_ADDED_ARG] is True


def test_canonicalizer_repairs_write_content_from_single_upstream_alias() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = {'studies': 10, 'series': 20}"},
                    "output": "json_summary",
                },
                {
                    "id": 2,
                    "action": "fs.write",
                    "input": ["json_summary"],
                    "args": {"path": "reports/dicom_summary.json", "content": "{\"studies\": 0, \"series\": 0}"},
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Write and return the summary", ["python.exec", "fs.write", "fs.read"])

    assert result.plan.steps[1].args["content"] == {"$ref": "json_summary", "path": "json"}


def test_canonicalizer_repairs_existing_write_ref_to_python_exec_textual_path() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = {'studies': 10, 'series': 20}"},
                    "output": "summary_json",
                },
                {
                    "id": 2,
                    "action": "fs.write",
                    "input": ["summary_json"],
                    "args": {"path": "reports/summary.json", "content": {"$ref": "summary_json"}},
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Write and return the summary", ["python.exec", "fs.write", "fs.read"])

    assert result.plan.steps[1].args["content"] == {"$ref": "summary_json", "path": "json"}


def test_canonicalizer_rewrites_structured_fs_write_into_python_writer() -> None:
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

    result = canonicalize_plan(plan, "Write reports/histogram.json", ["fs.write", "python.exec"])

    assert [step.id for step in result.plan.steps] == [1]
    assert result.plan.steps[0].action == "python.exec"
    assert result.plan.steps[0].args["inputs"]["path"] == "reports/histogram.json"
    assert result.plan.steps[0].args["inputs"]["content"] == {"histogram": {"2": 500}}
    assert "_json_dumps_safe" in result.plan.steps[0].args["code"]
    assert result.plan.steps[0].args[CANONICALIZER_WRITE_PATH_ARG] == "reports/histogram.json"


def test_canonicalizer_rewrites_structured_ref_write_and_appends_readback() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "result = {'histogram': {'2': 500}}"},
                    "output": "histogram_result",
                },
                {
                    "id": 2,
                    "action": "fs.write",
                    "input": ["histogram_result"],
                    "args": {"path": "reports/histogram.json", "content": {"$ref": "histogram_result", "path": "histogram"}},
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Write reports/histogram.json and return it", ["python.exec", "fs.write", "fs.read"])

    assert [step.id for step in result.plan.steps] == [1, 2, 3]
    assert result.plan.steps[1].action == "fs.write"
    assert result.plan.steps[1].input == ["histogram_result"]
    assert result.plan.steps[1].args["content"] == {"$ref": "histogram_result", "path": "json"}
    assert result.plan.steps[2].action == "fs.read"
    assert result.plan.steps[2].args["path"] == "reports/histogram.json"
    assert result.plan.steps[2].args[CANONICALIZER_ADDED_ARG] is True


def test_canonicalizer_skips_binary_readback() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.write", "args": {"path": "reports/image.bin", "content": "0101"}},
            ]
        }
    )

    result = canonicalize_plan(plan, "Write reports/image.bin and return it", ["fs.write", "fs.read"])

    assert [step.id for step in result.plan.steps] == [1]


def test_canonicalizer_is_idempotent_and_preserves_existing_step_order() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {"id": 1, "action": "fs.mkdir", "args": {"path": "reports"}},
                {"id": 2, "action": "fs.write", "args": {"path": "reports/summary.json", "content": "{\"ok\":true}"}},
            ]
        }
    )

    first = canonicalize_plan(plan, "Write reports/summary.json and return it", ["fs.mkdir", "fs.write", "fs.read"])
    second = canonicalize_plan(first.plan, "Write reports/summary.json and return it", ["fs.mkdir", "fs.write", "fs.read"])

    assert [step.id for step in first.plan.steps] == [1, 2, 3]
    assert first.plan.model_dump() == second.plan.model_dump()
    assert second.repairs == []


def test_canonicalizer_repair_budget_exhaustion_fails_cleanly() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "sql.query",
                    "args": {"database": "dicom", "query": "SELECT name FROM patient"},
                    "output": "rows",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["rows"],
                    "args": {
                        "inputs": {"rows": {"$ref": "rows", "path": "rows"}},
                        "code": "result = {'csv': ','.join(row['name'] for row in inputs['rows'])}",
                    },
                    "output": "csv",
                },
            ]
        }
    )

    try:
        canonicalize_plan(plan, "Query patient names", ["sql.query", "python.exec"], repair_budget=0)
    except ValueError as exc:
        assert "repair budget exceeded" in str(exc).lower()
    else:
        raise AssertionError("Expected canonicalizer repair budget failure.")


def test_canonicalizer_does_not_merge_python_steps_with_io() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "steps": [
                {
                    "id": 1,
                    "action": "python.exec",
                    "args": {"code": "text = fs.read('alpha.txt'); result = {'text': text}"},
                    "output": "first_result",
                },
                {
                    "id": 2,
                    "action": "python.exec",
                    "input": ["first_result"],
                    "args": {
                        "inputs": {"previous": {"$ref": "first_result", "path": "text"}},
                        "code": "result = {'length': len(inputs['previous'])}",
                    },
                    "output": "second_result",
                },
            ]
        }
    )

    result = canonicalize_plan(plan, "Read alpha.txt and compute its length", ["python.exec"])

    assert [step.id for step in result.plan.steps] == [1, 2]
    assert [step.action for step in result.plan.steps] == ["python.exec", "python.exec"]
