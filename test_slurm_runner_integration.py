import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parent


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
    HISTORY = [
        {"JobID": "201", "JobName": "seg", "User": "vinith", "State": "COMPLETED", "Partition": "hpc", "Elapsed": "00:10:00", "End": "2026-04-20T01:10:00"},
        {"JobID": "202", "JobName": "reg", "User": "vinith", "State": "COMPLETED", "Partition": "hpc", "Elapsed": "00:20:00", "End": "2026-04-20T02:20:00"},
        {"JobID": "203", "JobName": "qc", "User": "manju", "State": "FAILED", "Partition": "gpu", "Elapsed": "00:05:00", "End": "2026-04-20T03:05:00"},
        {"JobID": "204", "JobName": "dose", "User": "vinith", "State": "COMPLETED", "Partition": "gpu", "Elapsed": "00:15:00", "End": "2026-04-20T04:15:00"},
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
        if fmt == "%P|%a|%l|%D|%t|%N":
            partitions = {
                "hpc*": {"avail": "up", "limit": "infinite", "nodes": "3", "state": "mixed", "names": "node-a,node-b,node-c"},
                "gpu": {"avail": "up", "limit": "infinite", "nodes": "1", "state": "idle", "names": "gpu-a"},
            }
            items = partitions.items()
            if partition:
                items = [(name, meta) for name, meta in items if name.rstrip("*") == partition]
            out("\\n".join(f"{name}|{meta['avail']}|{meta['limit']}|{meta['nodes']}|{meta['state']}|{meta['names']}" for name, meta in items))
        if fmt == "%P|%t|%D|%G":
            partitions = [
                "hpc*|mixed|3|gpu:0",
                "gpu|idle|1|gpu:4",
            ]
            if partition:
                partitions = [row for row in partitions if row.split("|", 1)[0].rstrip("*") == partition]
            out("\\n".join(partitions))
        if fmt == "%t|%D":
            out("idle|2\\nmixed|1\\nallocated|1")
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

    if CMD == "sacct":
        if has("--help"):
            out("usage: sacct [-X] [-P] [-n] [--partition=PART] [--state=STATE] [--format=FIELDS]")
        rows = HISTORY
        user = arg_value("-u", "")
        part = arg_value("--partition", "")
        states = arg_value("--state", "")
        if user:
            rows = [row for row in rows if row["User"] == user]
        if part:
            rows = [row for row in rows if row["Partition"] == part]
        if states:
            allowed = {item.strip().upper() for item in states.split(",") if item.strip()}
            rows = [row for row in rows if row["State"].upper() in allowed]
        fmt = arg_value("--format", "JobID,JobName,User,State,Partition,Elapsed,End")
        fields = [field.strip() for field in fmt.split(",") if field.strip()]
        lines = ["|".join(str(row.get(field, "")) for field in fields) for row in rows]
        if has("-n"):
            out("\\n".join(lines))
        out("|".join(fields) + ("\\n" + "\\n".join(lines) if lines else ""))

    if CMD == "scontrol":
        if has("--help"):
            out("usage: scontrol show job <jobid> | hold|release|requeue|resume|suspend <jobid>")
        if len(ARGS) >= 3 and ARGS[0] == "show" and ARGS[1] == "job":
            jobid = ARGS[2]
            out(f"JobId={jobid} JobName=mockjob UserId=vinith State=RUNNING Partition=hpc Nodes=1")
        if len(ARGS) >= 2 and ARGS[0] in {"hold", "release", "requeue", "resume", "suspend"}:
            out(f"Updated job {ARGS[1]} with action {ARGS[0]}")
        err(f"unsupported scontrol args: {' '.join(ARGS)}")

    if CMD == "scancel":
        if has("--help"):
            out("usage: scancel <jobid>")
        if ARGS:
            out(f"Cancelled job {ARGS[0]}")
        err("scancel requires a job id")

    if CMD == "sshare":
        if has("--help"):
            out("usage: sshare [-u USER] [--User USER]")
        user = arg_value("-u", "") or arg_value("--User", "") or "vinith"
        out("Account|User|RawShares|NormShares|RawUsage|EffectvUsage|FairShare\\nroot|%s|1|0.500000|10|0.200000|2.500000" % user)

    if CMD == "sreport":
        if has("--help"):
            out("usage: sreport ...")
        out("Cluster|Login|Proper|Used\\ndefault|vinith|100|12")

    if CMD == "sacctmgr":
        if has("--help"):
            out("usage: sacctmgr ...")
        out("Account|User\\nroot|vinith")

    if CMD == "sdiag":
        out("Main schedule statistics (mock)")

    if CMD == "sprio":
        out("JOBID PRIORITY\\n102 1000")

    if CMD == "sstat":
        out("JobID AveCPU\\n102.batch 00:10:00")

    err(f"unsupported command {CMD}")
    """
)


RUNNER_SUBPROCESS_SCRIPT = textwrap.dedent(
    """\
    import json
    import sys

    from agent_library.common import EventRequest
    import agent_library.agents.slurm_runner as runner

    mode = sys.argv[1]

    def _mock_llm_process_result(task, command, stdout, stderr):
        text = (stdout or stderr or "").strip()
        return text, ""

    runner._llm_process_result = _mock_llm_process_result

    if mode == "force_fairshare_fallback":
        runner._llm_select_slurm_strategy = lambda task, context: {
            "primitive_id": "fallback_only",
            "selection_reason": "mocked fallback selector",
            "parameters": {},
            "fallback_command": "sshare",
            "fallback_args": ["-u", "vinith"],
            "fallback_reason": "mocked fairshare fallback",
        }

    request = json.loads(sys.stdin.read())
    response = runner.handle_event(EventRequest(**request))
    print(json.dumps(response, ensure_ascii=True, default=str))
    """
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _module_available_in_fresh_interpreter(name: str) -> bool:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)",
            name,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


class SlurmRunnerMockGatewayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name
            for name in ("fastapi", "pydantic", "requests", "uvicorn")
            if not _module_available_in_fresh_interpreter(name)
        ]
        if missing:
            raise unittest.SkipTest(f"Integration test requires installed packages: {', '.join(missing)}")

        cls._tempdir = tempfile.TemporaryDirectory(prefix="openfabric_mock_slurm_")
        cls._root = Path(cls._tempdir.name)
        cls._bin_dir = cls._root / "bin"
        cls._bin_dir.mkdir(parents=True, exist_ok=True)
        cls._write_mock_toolchain()

        cls._gateway_port = _find_free_port()
        cls._gateway_url = f"http://127.0.0.1:{cls._gateway_port}"
        cls._gateway_log_path = cls._root / "gateway.log"
        cls._gateway_log = cls._gateway_log_path.open("w+", encoding="utf-8")

        env = os.environ.copy()
        env["PATH"] = f"{cls._bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            if env.get("PYTHONPATH")
            else str(REPO_ROOT)
        )
        cls._gateway_env = env

        cls._gateway_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "dep_agent_library.slurm_gateway_agent.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls._gateway_port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=cls._gateway_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            cls._wait_for_gateway()
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "_gateway_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_file = getattr(cls, "_gateway_log", None)
        if log_file is not None and not log_file.closed:
            log_file.close()
        tempdir = getattr(cls, "_tempdir", None)
        if tempdir is not None:
            tempdir.cleanup()

    @classmethod
    def _write_mock_toolchain(cls):
        mock_script = cls._root / "slurm_mock.py"
        mock_script.write_text(MOCK_SLURM_SCRIPT, encoding="utf-8")
        mock_script.chmod(0o755)
        for name in ("sinfo", "squeue", "sacct", "scontrol", "scancel", "sshare", "sreport", "sacctmgr", "sdiag", "sprio", "sstat"):
            target = cls._bin_dir / name
            target.symlink_to(mock_script)

    @classmethod
    def _wait_for_gateway(cls, timeout_seconds: float = 15.0):
        deadline = time.time() + timeout_seconds
        last_error = ""
        while time.time() < deadline:
            if cls._gateway_process.poll() is not None:
                raise RuntimeError(f"Mock gateway exited early.\n{cls._gateway_logs()}")
            try:
                payload = _http_json(f"{cls._gateway_url}/health")
                if payload.get("ok"):
                    return
            except Exception as exc:  # pragma: no cover - best-effort startup loop
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for mock gateway. Last error: {last_error}\n{cls._gateway_logs()}")

    @classmethod
    def _gateway_logs(cls) -> str:
        try:
            return cls._gateway_log_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _invoke_runner(self, request: dict, *, mode: str = "") -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            if env.get("PYTHONPATH")
            else str(REPO_ROOT)
        )
        env["SLURM_GATEWAY_HOST"] = "127.0.0.1"
        env["SLURM_GATEWAY_PORT"] = str(self._gateway_port)
        env["OPENFABRIC_RAW_LOGS"] = "0"
        env["OPENFABRIC_DEBUG_LOGS"] = "0"
        env["OPENFABRIC_CONSOLE_EVENT_LOGS"] = "0"
        completed = subprocess.run(
            [sys.executable, "-c", RUNNER_SUBPROCESS_SCRIPT, mode],
            cwd=str(REPO_ROOT),
            env=env,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "Runner subprocess failed.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}\n"
                f"GATEWAY LOGS:\n{self._gateway_logs()}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            last_json_line = ""
            for line in reversed(completed.stdout.splitlines()):
                if line.strip().startswith("{"):
                    last_json_line = line.strip()
                    break
            if last_json_line:
                try:
                    return json.loads(last_json_line)
                except json.JSONDecodeError:
                    pass
            self.fail(f"Runner subprocess returned invalid JSON: {exc}\nOutput:\n{completed.stdout}")

    def test_mock_gateway_health_reports_mock_binaries(self):
        payload = _http_json(f"{self._gateway_url}/health")
        self.assertTrue(payload.get("ok"))
        self.assertIn("sinfo", payload.get("allowed_commands", []))
        self.assertTrue(payload.get("available_commands", {}).get("sinfo"))
        self.assertTrue(payload.get("available_commands", {}).get("squeue"))

    def test_runner_handles_deterministic_and_explicit_paths_against_mock_gateway(self):
        cases = [
            {
                "name": "node_inventory",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "how many nodes are currently in my slurm cluster and what is their state ?",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "how many nodes are currently in my slurm cluster and what is their state ?",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.cluster.node_inventory_summary",
                "field": "reduction_request",
                "contains": "slurm.node_inventory_summary",
            },
            {
                "name": "node_list",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "show node names in the Slurm cluster",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "show node names in the Slurm cluster",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.cluster.node_list",
                "field": "stdout",
                "contains": "node-a|idle|hpc*",
            },
            {
                "name": "partition_summary",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "show partition availability",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "show partition availability",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.cluster.partition_summary",
                "field": "stdout",
                "contains": "hpc*|up|infinite|3|mixed|node-a,node-b,node-c",
            },
            {
                "name": "pending_count",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "how many pending jobs are there for user vinith",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "how many pending jobs are there for user vinith",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.queue_count",
                "field": "reduction_request",
                "contains": "slurm.line_count",
            },
            {
                "name": "queue_list",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "show queued slurm jobs for user vinith",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "show queued slurm jobs for user vinith",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.queue_list",
                "field": "stdout",
                "contains": "101|vinith|PENDING|hpc|align",
            },
            {
                "name": "queue_breakdown",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "show the state breakdown of current jobs for user vinith",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "show the state breakdown of current jobs for user vinith",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.queue_state_breakdown",
                "field": "reduction_request",
                "contains": "slurm.state_breakdown",
            },
            {
                "name": "history_list",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "show failed jobs from yesterday",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "show failed jobs from yesterday",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.history_list",
                "field": "stdout",
                "contains": "203|qc|manju|FAILED|gpu|00:05:00|2026-04-20T03:05:00",
            },
            {
                "name": "elapsed_summary",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "how long did completed jobs in hpc partition take",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "how long did completed jobs in hpc partition take",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.elapsed_summary",
                "field": "reduction_request",
                "contains": "slurm.elapsed_summary",
            },
            {
                "name": "job_details",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "show job 102 details",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "show job 102 details",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.details",
                "field": "stdout",
                "contains": "JobId=102 JobName=mockjob UserId=vinith State=RUNNING Partition=hpc Nodes=1",
            },
            {
                "name": "cancel_job",
                "request": {
                    "event": "task.plan",
                    "payload": {
                        "task": "cancel job 104",
                        "target_agent": "slurm_runner_cluster",
                        "instruction": {
                            "operation": "query_from_request",
                            "question": "cancel job 104",
                        },
                    },
                },
                "execution_strategy": "deterministic",
                "primitive": "slurm.jobs.cancel",
                "field": "stdout",
                "contains": "Cancelled job 104",
            },
            {
                "name": "explicit_command",
                "request": {
                    "event": "slurm.query",
                    "payload": {"command": "squeue -u vinith"},
                },
                "execution_strategy": "explicit_command",
                "primitive": None,
                "field": "stdout",
                "contains": "101|vinith|PENDING|hpc|align",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                response = self._invoke_runner(case["request"])
                self.assertEqual(response["emits"][0]["event"], "slurm.result")
                payload = response["emits"][0]["payload"]
                self.assertEqual(payload.get("execution_strategy"), case["execution_strategy"])
                self.assertEqual(payload.get("deterministic_primitive"), case["primitive"])
                self.assertEqual(payload.get("returncode"), 0)
                self.assertIn(case["contains"], str(payload.get(case["field"]) or ""))

    def test_runner_uses_selector_fallback_command_against_mock_gateway(self):
        request = {
            "event": "task.plan",
            "payload": {
                "task": "show fairshare for user vinith",
                "target_agent": "slurm_runner_cluster",
                "instruction": {
                    "operation": "query_from_request",
                    "question": "show fairshare for user vinith",
                },
            },
        }
        response = self._invoke_runner(request, mode="force_fairshare_fallback")
        self.assertEqual(response["emits"][0]["event"], "slurm.result")
        payload = response["emits"][0]["payload"]
        self.assertEqual(payload.get("execution_strategy"), "selector_fallback_command")
        self.assertEqual(payload.get("deterministic_primitive"), "fallback_only")
        self.assertTrue(payload.get("fallback_used"))
        self.assertEqual(payload.get("returncode"), 0)
        self.assertIn("sshare -u vinith", str(payload.get("command") or ""))
        self.assertEqual(payload.get("reduction_request", {}).get("kind"), "pass_through")
        self.assertIn("root|vinith|1|0.500000|10|0.200000|2.500000", str(payload.get("stdout") or ""))


if __name__ == "__main__":
    unittest.main()
