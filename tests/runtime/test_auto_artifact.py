from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.auto_artifact import AutoArtifactMaterializer
from aor_runtime.runtime.presentation import PresentationContext, present_result


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": "postgresql+psycopg://example/db"},
        sql_default_database="dicom",
        **overrides,
    )


def _sql_log(row_count: int) -> StepLog:
    rows = [{"PatientID": f"MRN{index:04d}", "study_count": index} for index in range(row_count)]
    return StepLog(
        step=ExecutionStep(
            id=1,
            action="sql.query",
            args={"database": "dicom", "query": 'SELECT "PatientID" FROM flathr."Patient"'},
            output="patient_rows",
        ),
        result={"database": "dicom", "row_count": row_count, "rows": rows},
        success=True,
    )


def test_large_sql_list_is_materialized_to_csv_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    final_output = {"content": "inline rows", "artifacts": [], "metadata": {"goal": "list patients in dicom"}}

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="list patients in dicom",
        history=[_sql_log(51)],
        final_output=final_output,
    )

    assert result.applied is True
    assert result.artifact is not None
    assert result.artifact.rows_written == 51
    assert result.artifact.path.endswith(".csv")
    assert Path(result.artifact.path).is_file()
    content = Path(result.artifact.path).read_text()
    assert "PatientID,study_count" in content
    assert "MRN0050,50" in content
    assert "Rows written: **51**" in result.final_output["content"]
    assert "Source: **SQL query rows**" in result.final_output["content"]
    assert f"]({result.artifact.path})" in result.final_output["content"]
    assert 'SELECT "PatientID"' in result.final_output["content"]


def test_threshold_is_strictly_greater_than_configured_limit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="list patients in dicom",
        history=[_sql_log(50)],
        final_output={"content": "inline rows", "artifacts": [], "metadata": {}},
    )

    assert result.applied is False
    assert result.reason == "below_threshold"


def test_scalar_count_prompt_does_not_create_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="count patients in dicom",
        history=[_sql_log(100)],
        final_output={"content": "100", "artifacts": [], "metadata": {}},
    )

    assert result.applied is False
    assert result.reason == "contract_scalar"


def test_explicit_export_does_not_create_duplicate_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    history = [
        _sql_log(100),
        StepLog(
            step=ExecutionStep(id=2, action="fs.write", args={"path": "patients.csv", "content": "PatientID\n1"}, output="written_file"),
            result={"path": str(tmp_path / "patients.csv"), "bytes_written": 11},
            success=True,
        ),
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="list patients in dicom and save to patients.csv",
        history=history,
        final_output={"content": "saved", "artifacts": [str(tmp_path / "patients.csv")], "metadata": {}},
    )

    assert result.applied is False
    assert result.reason == "explicit_file_artifact"


def test_slurm_collection_is_materialized_without_raw_json_final(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    jobs = [{"job_id": str(index), "state": "PENDING"} for index in range(60)]
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={"state": "PENDING"}, output="pending_jobs"),
            result={"jobs": jobs, "count": 60, "total_count": 60, "returned_count": 60},
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="list pending jobs in slurm",
        history=history,
        final_output={"content": "raw job rows", "artifacts": [], "metadata": {}},
    )

    assert result.applied is True
    assert result.artifact is not None
    assert result.artifact.source_tool == "slurm.queue"
    assert "Rows written: **60**" in result.final_output["content"]
    assert "Source: **SLURM queue jobs**" in result.final_output["content"]
    assert "job_id,state" in Path(result.artifact.path).read_text()


def test_small_slurm_grouped_queue_counts_use_group_count_not_job_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={"group_by": "partition", "limit": None}, output="jobs_by_partition"),
            result={
                "jobs": [],
                "count": 292,
                "total_count": 292,
                "returned_count": 292,
                "limit": None,
                "truncated": False,
                "filters": {"partition": None},
                "group_by": "partition",
                "grouped": {"slicer": 33, "totalseg": 259},
            },
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="count of jobs in each slurm partition",
        history=history,
        final_output={"content": "inline grouped table", "artifacts": [], "metadata": {}},
    )

    assert result.applied is False
    assert result.reason == "below_threshold"
    assert result.metadata == {"row_count": 2}


def test_large_slurm_grouped_queue_counts_materialize_by_group_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    grouped = {f"partition_{index}": index for index in range(51)}
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={"group_by": "partition", "limit": None}, output="jobs_by_partition"),
            result={
                "jobs": [],
                "count": 1000,
                "total_count": 1000,
                "returned_count": 1000,
                "limit": None,
                "truncated": False,
                "filters": {"partition": None},
                "group_by": "partition",
                "grouped": grouped,
            },
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="count jobs by partition in slurm",
        history=history,
        final_output={"content": "inline grouped table", "artifacts": [], "metadata": {}},
    )

    assert result.applied is True
    assert result.artifact is not None
    assert result.artifact.rows_written == 51
    assert result.artifact.presentation_count == 51
    assert result.artifact.source_count == 1000
    assert result.artifact.source_tool == "slurm.queue"
    assert result.artifact.source_field == "grouped"
    assert "Source: **SLURM queue grouped counts**" in result.final_output["content"]
    content = Path(result.artifact.path).read_text()
    assert "group,count" in content
    assert "partition_50,50" in content


def test_large_parseable_shell_table_is_materialized_to_csv_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stdout = "USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND\n" + "\n".join(
        f"user{index} {1000 + index} {index % 10}.0 0.1 100 50 ? S 10:00 0:01 command-{index}"
        for index in range(51)
    )
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="shell.exec", args={"command": "ps aux --sort=-%cpu"}, output="processes"),
            result={"stdout": stdout, "stderr": "", "returncode": 0, "truncated": False},
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="list all cpu processes",
        history=history,
        final_output={"content": "inline table", "artifacts": [], "metadata": {}},
    )

    assert result.applied is True
    assert result.artifact is not None
    assert result.artifact.source_tool == "shell.exec"
    assert result.artifact.rows_written == 51
    assert "Source: **system shell output**" in result.final_output["content"]
    content = Path(result.artifact.path).read_text()
    assert "USER,PID,%CPU,%MEM,VSZ,RSS,TTY,STAT,START,TIME,COMMAND" in content
    assert "command-50" in content


def test_small_slurm_accounting_aggregate_uses_group_count_not_source_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    history = [
        StepLog(
            step=ExecutionStep(
                id=1,
                action="slurm.accounting_aggregate",
                args={"metric": "average_elapsed", "group_by": "partition"},
                output="partition_runtimes",
            ),
            result={
                "result_kind": "accounting_aggregate",
                "metric": "average_elapsed",
                "group_by": "partition",
                "job_count": 485,
                "total_count": 485,
                "returned_count": 485,
                "groups": [
                    {"key": "slicer", "job_count": 170, "average_elapsed_human": "6m 2s"},
                    {"key": "totalseg", "job_count": 315, "average_elapsed_human": "11m 30s"},
                ],
            },
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="Give me the average times for how long a job takes in each of the partitions in slurm",
        history=history,
        final_output={"content": "inline table", "artifacts": [], "metadata": {}},
    )

    assert result.applied is False
    assert result.reason == "below_threshold"
    assert result.metadata == {"row_count": 2}


def test_large_slurm_accounting_aggregate_materializes_by_group_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    groups = [
        {"key": f"partition_{index}", "job_count": index + 1, "average_elapsed_human": "1m"}
        for index in range(51)
    ]
    history = [
        StepLog(
            step=ExecutionStep(
                id=1,
                action="slurm.accounting_aggregate",
                args={"metric": "average_elapsed", "group_by": "partition"},
                output="partition_runtimes",
            ),
            result={
                "result_kind": "accounting_aggregate",
                "metric": "average_elapsed",
                "group_by": "partition",
                "job_count": 1000,
                "groups": groups,
            },
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="list average runtimes by partition in slurm",
        history=history,
        final_output={"content": "inline table", "artifacts": [], "metadata": {}},
    )

    assert result.applied is True
    assert result.artifact is not None
    assert result.artifact.presentation_count == 51
    assert result.artifact.source_count == 1000
    assert result.final_output["metadata"]["source_count"] == 1000
    assert "source contained **1,000** records" in result.final_output["content"]
    assert "partition_50" in Path(result.artifact.path).read_text()


def test_small_slurm_accounting_aggregate_renders_inline_groups_table() -> None:
    presented = present_result(
        {
            "result_kind": "accounting_aggregate",
            "metric": "average_elapsed",
            "group_by": "partition",
            "job_count": 485,
            "average_elapsed_human": "8m 7s",
            "groups": [
                {"key": "slicer", "job_count": 170, "average_elapsed_human": "6m 2s"},
                {"key": "totalseg", "job_count": 315, "average_elapsed_human": "11m 30s"},
            ],
        },
        PresentationContext(mode="user", source_action="slurm.accounting_aggregate"),
    )

    assert "## Groups" in presented.markdown
    assert "| slicer | 170 | 6m 2s |" in presented.markdown
    assert "| totalseg | 315 | 11m 30s |" in presented.markdown


def test_status_summary_prompt_does_not_auto_artifact_incidental_collections(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    jobs = [{"job_id": str(index), "state": "PENDING"} for index in range(60)]
    history = [
        StepLog(
            step=ExecutionStep(id=1, action="slurm.queue", args={}, output="queue"),
            result={"jobs": jobs, "count": 60, "total_count": 60},
            success=True,
        )
    ]

    result = AutoArtifactMaterializer(settings).maybe_materialize(
        goal="what is the status of the slurm cluster",
        history=history,
        final_output={"content": "cluster status", "artifacts": [], "metadata": {}},
    )

    assert result.applied is False
    assert result.reason == "not_list_or_table_goal"


def test_auto_artifact_directory_must_stay_in_workspace(tmp_path: Path) -> None:
    settings = _settings(tmp_path, auto_artifact_dir="../outside")

    try:
        AutoArtifactMaterializer(settings).maybe_materialize(
            goal="list patients in dicom",
            history=[_sql_log(51)],
            final_output={"content": "", "artifacts": [], "metadata": {}},
        )
    except ValueError as exc:
        assert "inside the workspace root" in str(exc) or "path traversal" in str(exc)
    else:
        raise AssertionError("Expected unsafe auto-artifact directory to be rejected")
