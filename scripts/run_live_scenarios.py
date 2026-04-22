#!/usr/bin/env python3
import contextlib
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwebui_gateway import PlannerGateway

try:
    import psycopg2 as _postgres_driver
except ImportError:  # pragma: no cover - fallback for local environments
    import psycopg as _postgres_driver


SPEC_PATH = REPO_ROOT / "agent_library" / "specs" / "ops_assistant_llm.yml"
REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
RUNS_DIR = REPO_ROOT / "artifacts" / "runtime_runs_shakeout"

MYDB_DSN = os.getenv("SQL_DATABASE_URL", "postgresql://admin:admin123@127.0.0.1:5432/mydb")
DICOM_DSN = os.getenv("SQL_DICOM_MOCK_DATABASE_URL", "postgresql://admin:admin123@127.0.0.1:5432/dicom_mock")

MOCK_SLURM_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os
    import sys

    CMD = os.path.basename(sys.argv[0])
    ARGS = sys.argv[1:]

    NODES = [
        {"name": "node-a", "state": "idle", "partition": "hpc*", "gpus": "gpu:0"},
        {"name": "node-b", "state": "mixed", "partition": "hpc*", "gpus": "gpu:0"},
        {"name": "node-c", "state": "allocated", "partition": "hpc*", "gpus": "gpu:0"},
        {"name": "gpu-a", "state": "idle", "partition": "gpu", "gpus": "gpu:4"},
    ]
    JOBS = [
        {"id": "101", "user": "vinith", "state": "PENDING", "partition": "hpc", "name": "align"},
        {"id": "102", "user": "vinith", "state": "RUNNING", "partition": "hpc", "name": "train"},
        {"id": "103", "user": "manju", "state": "RUNNING", "partition": "gpu", "name": "render"},
        {"id": "104", "user": "vinith", "state": "PENDING", "partition": "gpu", "name": "sim"},
    ]

    def arg_value(flag, default=""):
        if flag in ARGS:
            index = ARGS.index(flag)
            if index + 1 < len(ARGS):
                return ARGS[index + 1]
        for item in ARGS:
            if item.startswith(flag + "="):
                return item.split("=", 1)[1]
        return default

    def has(flag):
        return flag in ARGS

    def out(text=""):
        sys.stdout.write(text)
        if text and not text.endswith("\\n"):
            sys.stdout.write("\\n")
        raise SystemExit(0)

    def err(text, code=1):
        sys.stderr.write(text)
        if text and not text.endswith("\\n"):
            sys.stderr.write("\\n")
        raise SystemExit(code)

    if CMD == "sinfo":
        if has("--help"):
            out("usage: sinfo [-N] [-h] [-o FORMAT] [-p PARTITION]")
        partition = arg_value("-p", "")
        fmt = arg_value("-o", "")
        rows = NODES
        if partition:
            rows = [row for row in rows if row["partition"].rstrip("*") == partition]
        if fmt == "%N|%T":
            out("\\n".join(f"{row['name']}|{row['state']}" for row in rows))
        if fmt == "%N|%T|%P":
            out("\\n".join(f"{row['name']}|{row['state']}|{row['partition']}" for row in rows))
        err(f"unsupported sinfo args: {' '.join(ARGS)}")

    if CMD == "squeue":
        if has("--help"):
            out("usage: squeue [-h] [-u USER] [-p PARTITION] [-t STATES] [-o FORMAT]")
        rows = JOBS
        user = arg_value("-u", "")
        part = arg_value("-p", "")
        states = arg_value("-t", "")
        if user:
            rows = [row for row in rows if row["user"] == user]
        if part:
            rows = [row for row in rows if row["partition"] == part]
        if states:
            allowed = {item.strip().upper() for item in states.split(",") if item.strip()}
            rows = [row for row in rows if row["state"].upper() in allowed]
        out("\\n".join(f"{row['id']}|{row['user']}|{row['state']}|{row['partition']}|{row['name']}" for row in rows))

    err(f"unsupported command {CMD}")
    """
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    question: str
    expected_agents: tuple[str, ...]
    expected_fragments: Callable[[dict[str, Any]], list[str]]
    min_step_count: int = 1


def _connect(dsn: str):
    return _postgres_driver.connect(dsn)


def _query_scalar(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> Any:
    with contextlib.closing(_connect(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None


def _query_strings(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[str]:
    with contextlib.closing(_connect(dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [str(row[0]) for row in cur.fetchall()]


def _repo_root_python_files() -> list[str]:
    return sorted(path.name for path in REPO_ROOT.iterdir() if path.is_file() and path.suffix == ".py")


def _build_ground_truth() -> dict[str, Any]:
    root_py_files = _repo_root_python_files()
    dicom_tables = _query_strings(
        DICOM_DSN,
        """
        select table_name
        from information_schema.tables
        where table_schema = %s
        order by table_name
        """,
        ("dicom",),
    )
    mydb_multi_study_patients = int(
        _query_scalar(
            MYDB_DSN,
            """
            select count(*)
            from (
                select patient_id
                from dicom.studies
                group by patient_id
                having count(*) > 2
            ) counts
            """,
        )
        or 0
    )
    dicom_patient_count = int(_query_scalar(DICOM_DSN, "select count(*) from dicom.patients") or 0)
    return {
        "repo_python_files": root_py_files,
        "repo_python_count": len(root_py_files),
        "repo_python_first_five": root_py_files[:5],
        "dicom_tables": dicom_tables,
        "dicom_table_count": len(dicom_tables),
        "dicom_patient_count": dicom_patient_count,
        "mydb_multi_study_patient_count": mydb_multi_study_patients,
        "slurm_pending_count_vinith": 2,
        "slurm_pending_job_ids_vinith": ["101", "104"],
        "mixed_count_difference": abs(dicom_patient_count - len(root_py_files)),
    }


def _scenarios() -> list[Scenario]:
    return [
        Scenario(
            scenario_id="shell_root_python_inventory",
            question="In the repository root, list the first five Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["repo_python_count"]), *ctx["repo_python_first_five"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_dicom_schema_inventory",
            question="In the dicom_mock database, list the tables in the dicom schema alphabetically and tell me the total count.",
            expected_agents=("sql_runner_dicom_mock",),
            expected_fragments=lambda ctx: [str(ctx["dicom_table_count"]), *ctx["dicom_tables"]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="db_grouped_patient_count",
            question="In the mydb database, how many patients have more than 2 studies?",
            expected_agents=("sql_runner_mydb",),
            expected_fragments=lambda ctx: [str(ctx["mydb_multi_study_patient_count"])],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="slurm_pending_jobs",
            question="In the Slurm cluster, how many pending jobs does vinith have, and list the job IDs.",
            expected_agents=("slurm_runner_cluster",),
            expected_fragments=lambda ctx: [str(ctx["slurm_pending_count_vinith"]), *ctx["slurm_pending_job_ids_vinith"]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="mixed_db_shell_counts",
            question=(
                "In the dicom_mock database count patients, and in the repository root count Python files, "
                "then report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_patient_count"]),
                str(ctx["repo_python_count"]),
                str(ctx["mixed_count_difference"]),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="mixed_slurm_db_shell_counts",
            question=(
                "Count pending Slurm jobs for vinith, count patients in the dicom_mock database, "
                "and count Python files in the repository root. Report all three counts in one answer."
            ),
            expected_agents=("slurm_runner_cluster", "sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["slurm_pending_count_vinith"]),
                str(ctx["dicom_patient_count"]),
                str(ctx["repo_python_count"]),
            ],
            min_step_count=3,
        ),
    ]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MockSlurmGateway:
    def __init__(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix="openfabric_live_slurm_")
        self.root = Path(self._tempdir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.port = _find_free_port()
        self.log_path = self.root / "gateway.log"
        self._log_file = self.log_path.open("w+", encoding="utf-8")
        self.process: subprocess.Popen[str] | None = None
        self._write_mock_toolchain()

    def _write_mock_toolchain(self):
        script_path = self.root / "slurm_mock.py"
        script_path.write_text(MOCK_SLURM_SCRIPT, encoding="utf-8")
        script_path.chmod(0o755)
        for name in ("sinfo", "squeue"):
            target = self.bin_dir / name
            target.symlink_to(script_path)

    def start(self):
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(REPO_ROOT)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "dep_agent_library.slurm_gateway_agent.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_for_health()

    def _wait_for_health(self, timeout_seconds: float = 15.0):
        deadline = time.time() + timeout_seconds
        url = f"http://127.0.0.1:{self.port}/health"
        last_error = ""
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"Mock Slurm gateway exited early.\n{self.logs()}")
            try:
                response = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import json, sys, urllib.request;"
                            "payload=json.loads(urllib.request.urlopen(sys.argv[1], timeout=2).read().decode());"
                            "print(json.dumps(payload))"
                        ),
                        url,
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if response.returncode == 0:
                    payload = json.loads(response.stdout)
                    if payload.get("ok"):
                        return
                last_error = response.stderr.strip() or response.stdout.strip()
            except Exception as exc:  # pragma: no cover - best-effort startup loop
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for mock Slurm gateway. Last error: {last_error}\n{self.logs()}")

    def logs(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if not self._log_file.closed:
            self._log_file.close()
        self._tempdir.cleanup()


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _extract_run_id(events: list[dict[str, Any]]) -> str | None:
    for item in reversed(events):
        payload = item.get("payload")
        if isinstance(payload, dict):
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
    return None


def _evaluate_scenario(
    scenario: Scenario,
    answer: str,
    inspection: dict[str, Any] | None,
    events: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    normalized_answer = _normalize_text(answer)
    for fragment in scenario.expected_fragments(context):
        if isinstance(fragment, (list, tuple, set)):
            normalized_options = [_normalize_text(option) for option in fragment]
            if not any(option and option in normalized_answer for option in normalized_options):
                problems.append(f"Missing expected answer fragment: {list(fragment)}")
            continue
        if _normalize_text(fragment) not in normalized_answer:
            problems.append(f"Missing expected answer fragment: {fragment}")

    clarification_events = [item for item in events if item.get("event") == "clarification.required"]
    if clarification_events:
        problems.append("Run escalated to clarification.required.")

    summary = inspection.get("summary") if isinstance(inspection, dict) else {}
    if summary.get("status") != "completed":
        problems.append(f"Run status was {summary.get('status')!r}, expected 'completed'.")
    if int(summary.get("step_count") or 0) < scenario.min_step_count:
        problems.append(
            f"Run only recorded {summary.get('step_count')} step(s), expected at least {scenario.min_step_count}."
        )
    agents = summary.get("agents") if isinstance(summary.get("agents"), list) else []
    for agent_name in scenario.expected_agents:
        if agent_name not in agents:
            problems.append(f"Expected agent {agent_name!r} was not recorded in the run summary.")

    return {
        "passed": not problems,
        "problems": problems,
        "status": summary.get("status"),
        "step_count": summary.get("step_count"),
        "agents": agents,
    }


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_store_dir = RUNS_DIR / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    context = _build_ground_truth()
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_store_dir": str(run_store_dir),
        "ground_truth": copy.deepcopy(context),
        "scenarios": [],
    }

    gateway = None
    mock_slurm = MockSlurmGateway()
    exit_code = 0
    try:
        mock_slurm.start()
        os.environ["OPENFABRIC_RUN_STORE_DIR"] = str(run_store_dir)
        os.environ["SLURM_GATEWAY_HOST"] = "127.0.0.1"
        os.environ["SLURM_GATEWAY_PORT"] = str(mock_slurm.port)
        os.environ["SLURM_GATEWAY_SCHEME"] = "http"

        gateway = PlannerGateway(str(SPEC_PATH), timeout_seconds=300)

        try:
            for scenario in _scenarios():
                print(f"[SCENARIO] {scenario.scenario_id}")
                events: list[dict[str, Any]] = []

                def on_event(event_name: str, payload: dict[str, Any], _depth: int):
                    if event_name in {"workflow.result", "answer.final", "clarification.required", "validation.progress"}:
                        events.append({"event": event_name, "payload": copy.deepcopy(payload)})

                started_at = time.time()
                answer = ""
                inspection = None
                error = None
                try:
                    answer = gateway.ask(scenario.question, on_event=on_event)
                    run_id = _extract_run_id(events)
                    if run_id:
                        inspection = gateway.engine.inspect_run(run_id)
                    else:
                        error = "No run_id was observed in emitted events."
                except Exception as exc:  # pragma: no cover - live shakeout path
                    error = f"{type(exc).__name__}: {exc}"
                    run_id = _extract_run_id(events)
                    if run_id:
                        with contextlib.suppress(Exception):
                            inspection = gateway.engine.inspect_run(run_id)
                duration_ms = int((time.time() - started_at) * 1000)

                evaluation = (
                    _evaluate_scenario(scenario, answer, inspection, events, context)
                    if error is None
                    else {"passed": False, "problems": [error], "status": None, "step_count": None, "agents": []}
                )
                if not evaluation["passed"]:
                    exit_code = 1

                scenario_report = {
                    "scenario_id": scenario.scenario_id,
                    "question": scenario.question,
                    "duration_ms": duration_ms,
                    "answer": answer,
                    "events": events,
                    "run_id": _extract_run_id(events),
                    "inspection_summary": inspection.get("summary") if isinstance(inspection, dict) else None,
                    "evaluation": evaluation,
                }
                report["scenarios"].append(scenario_report)

                status = "PASS" if evaluation["passed"] else "FAIL"
                print(f"  {status} in {duration_ms} ms")
                if scenario_report["run_id"]:
                    print(f"  run_id: {scenario_report['run_id']}")
                if evaluation["problems"]:
                    for problem in evaluation["problems"]:
                        print(f"  problem: {problem}")
        except KeyboardInterrupt:  # pragma: no cover - live operator interrupt
            exit_code = 1
            report["interrupted"] = True

    finally:
        if gateway is not None:
            gateway.shutdown()
        mock_slurm.close()

    passed_count = len([item for item in report["scenarios"] if item["evaluation"]["passed"]])
    report["summary"] = {
        "scenario_count": len(report["scenarios"]),
        "passed_count": passed_count,
        "failed_count": len(report["scenarios"]) - passed_count,
        "all_passed": passed_count == len(report["scenarios"]),
    }
    report_path = REPORTS_DIR / "live_scenario_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    print(f"[REPORT] {report_path}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
