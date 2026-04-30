from __future__ import annotations

from scripts.evaluate_live_100_runtime import _classify


def test_eval_classifier_treats_no_data_as_warning_not_failure() -> None:
    status, issues = _classify(
        "No matching records were found for filters: partition=totalseg, state=COMPLETED.",
        http_status=200,
        transport_error="",
    )

    assert status == "warn"
    assert issues == ["data_unavailable"]


def test_eval_classifier_keeps_optional_schema_absence_out_of_sql_runtime_failure() -> None:
    status, issues = _classify(
        "The requested optional schema concept body part/anatomical region is not present in the SQL catalog.",
        http_status=200,
        transport_error="",
    )

    assert status == "warn"
    assert issues == ["data_unavailable"]


def test_eval_classifier_fails_raw_json_and_references() -> None:
    assert _classify('{"queue_count": 10}', http_status=200, transport_error="") == (
        "fail",
        ["formatting_presentation"],
    )
    assert _classify("Reference path not found: rows", http_status=200, transport_error="") == (
        "fail",
        ["dataflow_reference"],
    )


def test_eval_classifier_fails_actual_tool_domain_mismatch() -> None:
    status, issues = _classify(
        "System process requests must use shell.exec; SLURM queue/jobs do not answer OS process requests.",
        http_status=200,
        transport_error="",
    )

    assert status == "fail"
    assert issues == ["tool_domain"]


def test_eval_classifier_marks_capability_unavailable_as_warning() -> None:
    status, issues = _classify(
        "SLURM accounting capability appears unavailable on this machine.",
        http_status=200,
        transport_error="",
    )

    assert status == "warn"
    assert issues == ["capability_unavailable", "data_unavailable"]
