from __future__ import annotations

from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.response_renderer import ResponseRenderContext, render_agent_response


def _sql_history() -> list[StepLog]:
    return [
        StepLog(
            step=ExecutionStep(
                id=1,
                action="sql.query",
                args={"database": "dicom", "query": 'SELECT COUNT(*) AS count_value FROM flathr."Patient";'},
                output="sql_result",
            ),
            result={"database": "dicom", "row_count": 1, "rows": [{"count_value": 28102}]},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=2, action="runtime.return", args={"value": {"$ref": "sql_result", "path": "rows.0.count_value"}}),
            result={"value": 28102, "output": "28102"},
            success=True,
        ),
    ]


def _slurm_result() -> dict:
    return {
        "results": {
            "cluster_summary": {
                "metric_group": "cluster_summary",
                "payload": {
                    "queue_count": 195,
                    "running_jobs": 8,
                    "pending_jobs": 187,
                    "node_count": 18,
                    "problematic_nodes": 6,
                    "gpu_available": True,
                    "total_gpus": 32,
                },
            }
        },
        "coverage": {"covered_requests": ["r1"], "missing_requests": []},
        "slurm_semantic_frame": {"requests": ["internal"]},
    }


def test_sql_count_response_includes_result_query_and_execution_table() -> None:
    rendered = render_agent_response(
        28102,
        execution_events=_sql_history(),
        context=ResponseRenderContext(source_action="sql.query", output_mode="count"),
    )

    assert "## Summary" in rendered.markdown
    assert "## Summary" in rendered.markdown
    assert "The SQL query returned a count of 28,102." in rendered.markdown
    assert "## Result" in rendered.markdown
    assert "Count: 28,102" in rendered.markdown
    assert "## Query Used" in rendered.markdown
    assert "```sql" in rendered.markdown
    assert "\n\n```sql\n" in rendered.markdown
    assert 'FROM flathr."Patient"' in rendered.markdown
    assert 'SELECT COUNT(*) AS count_value\nFROM flathr."Patient";' in rendered.markdown
    assert "| `Tool` | `sql.query` |" in rendered.markdown
    assert "| `Database` | `dicom` |" in rendered.markdown
    assert "runtime.return" not in rendered.markdown


def test_short_sql_query_display_pretty_prints_across_logical_lines() -> None:
    query = "SELECT COUNT(*) AS count_value FROM patients"
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"database": "dicom", "query": query}, output="sql_result"),
            result={"database": "dicom", "row_count": 1, "rows": [{"count_value": 2}]},
            success=True,
        )
    ]

    rendered = render_agent_response(2, execution_events=history, context=ResponseRenderContext(source_action="sql.query", output_mode="count"))

    assert "## Query Used" in rendered.markdown
    assert "SELECT COUNT(*) AS count_value\nFROM patients" in rendered.markdown
    assert "SELECT COUNT(*) AS count_value FROM patients" not in rendered.markdown


def test_long_sql_query_display_wraps_readably_and_preserves_postgres_quotes() -> None:
    query = (
        'SELECT p."PatientID", p."PatientName", p."PatientBirthDate", '
        'COUNT(s."StudyInstanceUID") AS study_count, MAX(s."StudyDate") AS last_study_date '
        'FROM flathr."Patient" p '
        'JOIN flathr."Study" s ON s."PatientID" = p."PatientID" '
        'WHERE p."PatientID" IS NOT NULL '
        'GROUP BY p."PatientID", p."PatientName", p."PatientBirthDate" '
        'HAVING COUNT(s."StudyInstanceUID") > 10 '
        'ORDER BY study_count DESC '
        'LIMIT 100;'
    )
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"database": "dicom", "query": query}, output="sql_result"),
            result={"database": "dicom", "row_count": 1, "rows": [{"PatientID": "p1", "study_count": 12}]},
            success=True,
        )
    ]

    rendered = render_agent_response(
        [{"PatientID": "p1", "study_count": 12}],
        execution_events=history,
        context=ResponseRenderContext(source_action="sql.query"),
    )

    assert "## Query Used" in rendered.markdown
    assert 'flathr."Patient"' in rendered.markdown
    assert '"PatientID"' in rendered.markdown
    assert '\nFROM flathr."Patient" p' in rendered.markdown
    assert '\nJOIN flathr."Study" s ON s."PatientID" = p."PatientID"' in rendered.markdown
    assert "\nGROUP BY" in rendered.markdown
    assert "\nORDER BY" in rendered.markdown


def test_sql_query_display_does_not_split_clause_words_inside_string_literals() -> None:
    query = "SELECT 'FROM patients WHERE hidden' AS literal_value FROM audit_log WHERE message = 'ORDER BY text'"
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="sql.query", args={"database": "dicom", "query": query}, output="sql_result"),
            result={"database": "dicom", "row_count": 1, "rows": [{"literal_value": "FROM patients WHERE hidden"}]},
            success=True,
        )
    ]

    rendered = render_agent_response(
        [{"literal_value": "FROM patients WHERE hidden"}],
        execution_events=history,
        context=ResponseRenderContext(source_action="sql.query"),
    )

    assert "SELECT 'FROM patients WHERE hidden' AS literal_value" in rendered.markdown
    assert "\nFROM audit_log" in rendered.markdown
    assert "\nWHERE message = 'ORDER BY text'" in rendered.markdown


def test_lifecycle_text_is_hidden_in_user_mode() -> None:
    events = [
        {"event_type": "planner.started", "payload": {}},
        {"event_type": "planner.completed", "payload": {}},
        {"event_type": "validator.started", "payload": {}},
        {"event_type": "validator.completed", "payload": {"result": {"success": True}}},
        *_sql_history(),
    ]

    rendered = render_agent_response(28102, execution_events=events, context=ResponseRenderContext(source_action="sql.query", output_mode="count"))

    assert "Thinking..." not in rendered.markdown
    assert "Plan ready" not in rendered.markdown
    assert "Validating" not in rendered.markdown
    assert "Validation passed" not in rendered.markdown


def test_debug_mode_includes_compact_debug_metadata() -> None:
    rendered = render_agent_response(
        28102,
        execution_events=_sql_history(),
        metadata={"planning_mode": "deterministic_intent"},
        context=ResponseRenderContext(mode="debug", source_action="sql.query", output_mode="count", show_debug_metadata=True),
    )

    assert "## Debug Metadata" in rendered.markdown
    assert "deterministic_intent" in rendered.markdown


def test_slurm_response_includes_clean_commands() -> None:
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.metrics", args={"metric_group": "cluster_summary"}, output="cluster_summary"),
            result={"metric_group": "cluster_summary", "payload": {"queue_count": 195, "running_jobs": 8, "pending_jobs": 187}},
            success=True,
        ),
        StepLog(step=ExecutionStep(id=2, action="runtime.return"), result={"value": _slurm_result()}, success=True),
    ]

    rendered = render_agent_response(_slurm_result(), execution_events=history, context=ResponseRenderContext(source_action="slurm.metrics"))

    assert "## Summary" in rendered.markdown
    assert "## SLURM Cluster Status" in rendered.markdown
    assert "\n# " not in rendered.markdown
    assert "Running: 8" in rendered.markdown
    assert "## Commands Used" in rendered.markdown
    assert "squeue -h -o" in rendered.markdown
    assert "sinfo -h -N -o" in rendered.markdown
    assert "coverage" not in rendered.markdown
    assert "slurm_semantic_frame" not in rendered.markdown


def test_long_slurm_command_display_wraps_in_commands_used() -> None:
    history = [
        StepLog(
            step=ExecutionStep(
                id=1,
                action="slurm.accounting_aggregate",
                args={
                    "partition": "totalseg",
                    "state": "COMPLETED",
                    "user": "airflow",
                    "metric": "average_elapsed",
                    "start": "2026-04-22 00:00:00",
                    "end": "2026-04-29 00:00:00",
                },
                output="runtime_stats",
            ),
            result={
                "result_kind": "accounting_aggregate",
                "metric": "average_elapsed",
                "partition": "totalseg",
                "job_count": 364,
                "value_human": "10m 53s",
            },
            success=True,
        )
    ]

    rendered = render_agent_response(
        {"value_human": "10m 53s"},
        execution_events=history,
        context=ResponseRenderContext(source_action="slurm.accounting_aggregate"),
    )

    assert "## Commands Used" in rendered.markdown
    assert "sacct -X -P" in rendered.markdown
    assert "--partition=totalseg" in rendered.markdown
    assert " \\\n" in rendered.markdown


def test_slurm_queue_rows_are_preserved_when_runtime_return_is_row_list() -> None:
    queue_result = {
        "jobs": [
            {
                "job_id": "1",
                "user": "alice",
                "state": "RUNNING",
                "partition": "hpc",
                "name": "train",
                "time": "1:00",
                "nodes": "1",
                "reason": "node-a",
            }
        ],
        "count": 1,
        "total_count": 1,
        "returned_count": 1,
        "limit": 100,
        "truncated": False,
        "filters": {"state": "RUNNING"},
        "group_by": None,
        "grouped": None,
    }
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={"state": "RUNNING", "limit": 100}, output="running_jobs"),
            result=queue_result,
            success=True,
        ),
        StepLog(step=ExecutionStep(id=2, action="runtime.return"), result={"value": queue_result["jobs"], "output": "..."}, success=True),
    ]

    rendered = render_agent_response(queue_result["jobs"], execution_events=history, context=ResponseRenderContext(source_action="slurm.queue"))

    assert "## Running SLURM Jobs" in rendered.markdown
    assert "| 1 | alice | RUNNING | hpc | train | 1:00 | 1 | node-a |" in rendered.markdown
    assert "## Commands Used" in rendered.markdown
    assert rendered.markdown.index("## Running SLURM Jobs") < rendered.markdown.index("## Commands Used")


def test_filesystem_aggregate_response_includes_operation_details() -> None:
    result = {
        "path": "/tmp/workspace",
        "pattern": "*.mp4",
        "recursive": True,
        "file_count": 12,
        "total_size_bytes": 4512331776,
        "display_size": "4.2 GB",
        "matches": [],
    }
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="fs.aggregate", args={"path": "/tmp/workspace", "pattern": "*.mp4", "recursive": True}),
            result=result,
            success=True,
        )
    ]

    rendered = render_agent_response(result, execution_events=history, context=ResponseRenderContext(source_action="fs.aggregate"))

    assert "Found 12 files totaling 4.2 GB." in rendered.markdown
    assert "## Operation" in rendered.markdown
    assert "| `Path` | `/tmp/workspace` |" in rendered.markdown
    assert "| `Pattern` | `*.mp4` |" in rendered.markdown


def test_large_json_is_not_dumped_in_user_mode() -> None:
    result = {f"field_{index}": {"nested": list(range(20))} for index in range(30)}

    rendered = render_agent_response(result, context=ResponseRenderContext())

    assert not rendered.markdown.lstrip().startswith("{")
    assert "Raw output is available in debug/raw mode" in rendered.markdown


def test_insight_layer_can_be_disabled_for_legacy_summary_path() -> None:
    rendered = render_agent_response(
        28102,
        execution_events=_sql_history(),
        context=ResponseRenderContext(source_action="sql.query", output_mode="count", enable_insight_layer=False),
    )

    assert "## Summary" not in rendered.markdown
    assert "Count: 28,102" in rendered.markdown
