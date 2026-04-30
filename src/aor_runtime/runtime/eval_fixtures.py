"""OpenFABRIC Runtime Module: aor_runtime.runtime.eval_fixtures

Purpose:
    Create deterministic fixture data and workspaces for capability and runtime evaluations.

Responsibilities:
    Coordinate LLM action plans, deterministic canonicalization, tool execution, output shaping, and session state.

Data flow / Interfaces:
    Consumes user goals, runtime settings, tool results, and session history; produces execution plans, events, and final Markdown.

Boundaries:
    Owns the deterministic safety boundary between LLM-proposed actions, executable tools, and user-visible output.
"""

from __future__ import annotations

import getpass
import json
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalFixturePayload(BaseModel):
    """Represent eval fixture payload within the OpenFABRIC runtime. It extends BaseModel.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by EvalFixturePayload.

    Data flow / Interfaces:
        Instances are created and consumed by planning, execution, validation, and presentation code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.runtime.eval_fixtures.EvalFixturePayload and related tests.
    """
    workspace_root: str
    run_store_path: str
    sql_databases: dict[str, str] = Field(default_factory=dict)
    sql_default_database: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)

    def settings_payload(self) -> dict[str, Any]:
        """Settings payload for EvalFixturePayload instances.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by planning, execution, validation, and presentation through EvalFixturePayload.settings_payload calls and related tests.
        """
        return {
            "workspace_root": self.workspace_root,
            "run_store_path": self.run_store_path,
            "sql_databases": dict(self.sql_databases),
            "sql_default_database": self.sql_default_database,
        }


NOTES_CONTENT = {
    "meeting_notes.txt": "agenda\nbudget\nfollow-up\n",
    "shopping.txt": "bread\nmilk\ntea\n",
    "poem.txt": "roses\nviolets\nstarlight\n",
    "todo.txt": "email\ncall\nlunch\n",
    "short.txt": "one\ntwo\n",
    "transform.txt": "quiet library\nLOUD VOICE\nmixed case words\nalpha beta\n",
}

FILE_TREE = {
    "alpha": {"a.txt": "a\n", "b.txt": "b\n", "ignore.md": "ignore\n", "nested/c.txt": "c\n"},
    "beta": {"first.txt": "first\n", "second.txt": "second\n", "ignore.md": "ignore\n", "nested/third.txt": "third\n"},
    "gamma": {"blue.txt": "blue\n", "red.txt": "red\n", "ignore.md": "ignore\n", "nested/green.txt": "green\n"},
    "delta": {"north.txt": "north\n", "south.txt": "south\n", "ignore.md": "ignore\n", "nested/east.txt": "east\n"},
    "epsilon": {"apple.txt": "apple\n", "banana.txt": "banana\n", "ignore.md": "ignore\n", "nested/citrus.txt": "citrus\n"},
    "zeta": {"oak.txt": "oak\n", "pine.txt": "pine\n", "ignore.md": "ignore\n", "nested/cedar.txt": "cedar\n"},
    "eta": {"sun.txt": "sun\n", "moon.txt": "moon\n", "ignore.md": "ignore\n", "nested/star.txt": "star\n"},
    "theta": {"lake.txt": "lake\n", "river.txt": "river\n", "ignore.md": "ignore\n", "nested/ocean.txt": "ocean\n"},
    "iota": {"bread.txt": "bread\n", "soup.txt": "soup\n", "ignore.md": "ignore\n", "nested/tea.txt": "tea\n"},
    "kappa": {"alpha.txt": "alpha\n", "beta.txt": "beta\n", "ignore.md": "ignore\n", "nested/gamma.txt": "gamma\n"},
    "lambda": {
        "clip1.mp4": "aaaa",
        "clip2.mp4": "bb",
        "notes.txt": "ignore\n",
        "nested/clip3.mp4": "hello",
    },
    "mu": {
        "readme.txt": "no mp4 files here\n",
        "nested/story.md": "still no video\n",
    },
}

SEARCH_TREE = {
    "pantry/cake.txt": "cinnamon sugar\n",
    "pantry/tea.txt": "ginger and cinnamon\n",
    "pantry/salt.txt": "sea salt\n",
    "pantry/notes.md": "cinnamon markdown\n",
    "journal/april.txt": "garden plans\n",
    "journal/may.txt": "garden party\n",
    "journal/june.md": "garden sketch\n",
    "stories/library.txt": "quiet library afternoon\n",
    "stories/park.txt": "sunny park walk\n",
    "travel/lisbon.txt": "harbor light\n",
    "travel/oslo.txt": "snow path\n",
    "travel/rome.txt": "museum and harbor\n",
    "kitchen/bread.txt": "fresh bread\n",
    "kitchen/spice.txt": "tea and cinnamon\n",
    "kitchen/soup.txt": "ginger soup\n",
    "kitchen/notes.md": "cinnamon markdown\n",
    "weekend/garden.txt": "weekend garden\n",
    "weekend/market.txt": "weekend market\n",
    "weekend/desk.txt": "weekday notes\n",
    "puzzle/logic.txt": "puzzle pieces\n",
    "puzzle/riddle.txt": "puzzle answer\n",
    "puzzle/plain.txt": "sunrise walk\n",
    "orchard/basket.txt": "orchard basket\n",
    "orchard/trees.txt": "orchard apple\n",
    "orchard/shed.txt": "garden tools\n",
    "lantern/day.txt": "morning light\n",
    "lantern/night.txt": "lantern glow\n",
    "lantern/festival.txt": "lantern song\n",
    "cafe/bill.txt": "table three\n",
    "cafe/menu.txt": "tea and toast\n",
    "cafe/notes.txt": "cinnamon bun\n",
}

FETCH_FIXTURES = {
    "example.html": '<html><head><title>Example Fixture</title><meta name="fixture" content="example"></head><body>example body</body></html>',
    "story.html": '<html><head><title>Story Fixture</title><meta name="fixture" content="story"></head><body>story body</body></html>',
    "museum.html": '<html><head><title>Museum Fixture</title><meta name="fixture" content="museum"></head><body>museum body</body></html>',
}


def rebuild_eval_workspace(workspace: Path) -> EvalFixturePayload:
    """Rebuild eval workspace for the surrounding runtime workflow.

    Inputs:
        Receives workspace for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures.rebuild_eval_workspace.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    notes_dir = workspace / "notes"
    for name, content in NOTES_CONTENT.items():
        _write(notes_dir / name, content)

    files_dir = workspace / "files"
    for directory, files in FILE_TREE.items():
        for relative_path, content in files.items():
            _write(files_dir / directory / relative_path, content)

    search_dir = workspace / "search"
    for relative_path, content in SEARCH_TREE.items():
        _write(search_dir / relative_path, content)

    outputs_dir = workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    writes_dir = workspace / "writes"
    writes_dir.mkdir(parents=True, exist_ok=True)

    fetch_dir = workspace / "fetch"
    for name, content in FETCH_FIXTURES.items():
        _write(fetch_dir / name, content)

    slurm_dir = workspace / "slurm"
    for name, content in _slurm_fixture_files().items():
        _write(slurm_dir / name, content)

    llm_intent_path = workspace / "llm_intent_responses.json"

    sql_path = workspace / "book_club.db"
    db = sqlite3.connect(sql_path)
    try:
        db.executescript(
            """
            CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL
            );
            CREATE TABLE "Patient" (
                "PatientID" TEXT PRIMARY KEY,
                "PatientName" TEXT NOT NULL,
                "PatientBirthDate" TEXT
            );
            CREATE TABLE "Study" (
                "StudyInstanceUID" TEXT PRIMARY KEY,
                "PatientID" TEXT NOT NULL,
                "Modality" TEXT,
                FOREIGN KEY("PatientID") REFERENCES "Patient"("PatientID")
            );
            INSERT INTO members(id, name, city) VALUES
                (1, 'Alice', 'Portland'),
                (2, 'Bob', 'Seattle'),
                (3, 'Carla', 'Portland');
            INSERT INTO books(id, title) VALUES
                (1, 'North Window'),
                (2, 'Evening Train');
            INSERT INTO "Patient"("PatientID", "PatientName", "PatientBirthDate") VALUES
                ('p1', 'Ada Alpha', '1940-01-01'),
                ('p2', 'Ben Beta', '1980-01-01'),
                ('p3', 'Cid Gamma', '2015-01-01');
            INSERT INTO "Study"("StudyInstanceUID", "PatientID", "Modality") VALUES
                ('s1', 'p1', 'CT'),
                ('s2', 'p1', 'MR'),
                ('s3', 'p2', 'CT');
            """
        )
        db.commit()
    finally:
        db.close()

    variables: dict[str, str] = {
        "workspace_root": str(workspace),
        "outputs_dir": str(outputs_dir),
        "writes_dir": str(writes_dir),
        "search_root": str(search_dir),
    }
    variables.update(_path_variables("notes", notes_dir, NOTES_CONTENT))
    variables.update(_directory_variables("files", files_dir, FILE_TREE))
    variables["files_lambda_tilde"] = _tilde_path(files_dir / "lambda")
    variables.update(
        {
            "search_pantry": str(search_dir / "pantry"),
            "search_journal": str(search_dir / "journal"),
            "search_stories": str(search_dir / "stories"),
            "search_travel": str(search_dir / "travel"),
            "search_kitchen": str(search_dir / "kitchen"),
            "search_weekend": str(search_dir / "weekend"),
            "search_puzzle": str(search_dir / "puzzle"),
            "search_orchard": str(search_dir / "orchard"),
            "search_lantern": str(search_dir / "lantern"),
            "search_cafe": str(search_dir / "cafe"),
        }
    )
    variables.update(
        {
            "fetch_example": (fetch_dir / "example.html").resolve().as_uri(),
            "fetch_story": (fetch_dir / "story.html").resolve().as_uri(),
            "fetch_museum": (fetch_dir / "museum.html").resolve().as_uri(),
            "slurm_fixture_dir": str(slurm_dir),
            "llm_intent_fixture_path": str(llm_intent_path),
            "slurm_llm_intent_fixture_path": str(llm_intent_path),
            "current_user": getpass.getuser(),
            "slurm_today_date": datetime.now().strftime("%Y-%m-%d"),
            "slurm_yesterday_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "slurm_last_week_date": (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"),
        }
    )
    _write(llm_intent_path, json.dumps(_llm_intent_fixture_payload(variables), indent=2) + "\n")

    return EvalFixturePayload(
        workspace_root=str(workspace),
        run_store_path=str(workspace / "runtime.db"),
        sql_databases={"book_club_db": f"sqlite:///{sql_path}"},
        sql_default_database="book_club_db",
        variables=variables,
    )


def render_template(value: Any, fixtures: EvalFixturePayload) -> Any:
    """Render template for the surrounding runtime workflow.

    Inputs:
        Receives value, fixtures for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures.render_template.
    """
    return _render_value(value, fixtures.variables)


def render_case_prompt(prompt: str, fixtures: EvalFixturePayload) -> str:
    """Render case prompt for the surrounding runtime workflow.

    Inputs:
        Receives prompt, fixtures for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures.render_case_prompt.
    """
    return str(render_template(prompt, fixtures))


def render_case_expected(expected: Any, fixtures: EvalFixturePayload) -> Any:
    """Render case expected for the surrounding runtime workflow.

    Inputs:
        Receives expected, fixtures for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures.render_case_expected.
    """
    return render_template(expected, fixtures)


def _path_variables(prefix: str, root: Path, files: dict[str, str]) -> dict[str, str]:
    """Handle the internal path variables helper path for this module.

    Inputs:
        Receives prefix, root, files for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._path_variables.
    """
    variables: dict[str, str] = {}
    for filename in files:
        stem = Path(filename).stem
        variables[f"{prefix}_{stem}"] = str(root / filename)
    return variables


def _directory_variables(prefix: str, root: Path, directories: dict[str, Any]) -> dict[str, str]:
    """Handle the internal directory variables helper path for this module.

    Inputs:
        Receives prefix, root, directories for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._directory_variables.
    """
    return {f"{prefix}_{name}": str(root / name) for name in directories}


def _tilde_path(path: Path) -> str:
    """Handle the internal tilde path helper path for this module.

    Inputs:
        Receives path for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._tilde_path.
    """
    resolved = path.resolve()
    home = Path.home().resolve()
    try:
        return "~/" + str(resolved.relative_to(home))
    except ValueError:
        return str(resolved)


def _render_value(value: Any, variables: dict[str, str]) -> Any:
    """Handle the internal render value helper path for this module.

    Inputs:
        Receives value, variables for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._render_value.
    """
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [_render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, variables) for key, item in value.items()}
    return value


def _write(path: Path, content: str) -> None:
    """Handle the internal write helper path for this module.

    Inputs:
        Receives path, content for this function; type hints and validators define accepted shapes.

    Returns:
        Returns None; side effects are limited to the local runtime operation described above.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _slurm_fixture_files() -> dict[str, str]:
    """Handle the internal slurm fixture files helper path for this module.

    Inputs:
        Uses module or instance state; no caller-supplied data parameters are required.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._slurm_fixture_files.
    """
    current_user = getpass.getuser()
    now = datetime.now().replace(microsecond=0)
    yesterday = now - timedelta(days=1)
    last_week = now - timedelta(days=6)

    squeue_lines = [
        f"12345|{current_user}|RUNNING|gpu|train-gpu|01:20:00|2|None",
        "12346|alice|PENDING|cpu|align|00:00:00|1|Resources",
        f"12347|{current_user}|PENDING|gpu|prep|00:00:00|1|Priority",
        "12348|carla|RUNNING|debug|notebook|00:10:00|1|None",
    ]
    sinfo_nodes_lines = [
        "slurm-worker-agatha|idle|gpu|64|256000|gpu:a100:4",
        "slurm-worker-bravo|alloc|gpu|64|256000|gpu:a100:4",
        "slurm-worker-charlie|mix|cpu|32|128000|(null)",
        "slurm-worker-delta|down|cpu|32|128000|(null)",
        "slurm-worker-echo|drain|cpu|32|128000|gpu:v100:1",
    ]
    sinfo_partition_lines = [
        "gpu|up|7-00:00:00|2|up|0/64/64/128|gpu:a100:8",
        "cpu|up|7-00:00:00|3|up|16/48/0/64|(null)",
        "debug|up|1-00:00:00|1|up|4/28/0/32|(null)",
    ]
    sacct_lines = [
        f"12340|{current_user}|COMPLETED|gpu|train-gpu|01:00:00|8|32G|{yesterday.strftime('%Y-%m-%dT08:00:00')}|{yesterday.strftime('%Y-%m-%dT08:10:00')}|{yesterday.strftime('%Y-%m-%dT09:10:00')}|0:0",
        f"12341|alice|FAILED|cpu|align|00:20:00|4|8G|{yesterday.strftime('%Y-%m-%dT10:00:00')}|{yesterday.strftime('%Y-%m-%dT10:05:00')}|{yesterday.strftime('%Y-%m-%dT10:25:00')}|1:0",
        f"12342|bob|CANCELLED|gpu|prep|00:00:30|1|4G|{now.strftime('%Y-%m-%dT07:30:00')}|{now.strftime('%Y-%m-%dT07:31:00')}|{now.strftime('%Y-%m-%dT07:31:30')}|0:15",
        f"12343|{current_user}|COMPLETED|debug|report|00:05:00|1|1G|{last_week.strftime('%Y-%m-%dT06:00:00')}|{last_week.strftime('%Y-%m-%dT06:01:00')}|{last_week.strftime('%Y-%m-%dT06:06:00')}|0:0",
    ]
    job_detail = (
        f"JobId=12345 JobName=train-gpu UserId={current_user}(1000) GroupId=ml(1000) "
        "Partition=gpu JobState=RUNNING NumNodes=2 Reason=None"
    )
    node_detail = (
        "NodeName=slurm-worker-agatha Arch=x86_64 CoresPerSocket=32 CPUAlloc=0 CPUEfctv=64 "
        "CPUTot=64 NodeAddr=slurm-worker-agatha Gres=gpu:a100:4 State=IDLE"
    )

    return {
        "squeue.txt": "\n".join(squeue_lines) + "\n",
        "sinfo_nodes.txt": "\n".join(sinfo_nodes_lines) + "\n",
        "sinfo_partitions.txt": "\n".join(sinfo_partition_lines) + "\n",
        "sacct.txt": "\n".join(sacct_lines) + "\n",
        "scontrol_job_12345.txt": job_detail + "\n",
        "scontrol_node_slurm-worker-agatha.txt": node_detail + "\n",
        "sacctmgr_show_cluster.txt": "clusterA|controlhost\n",
        "sacct_probe.txt": "12340\n12341\n",
    }


def _llm_intent_fixture_payload(variables: dict[str, str]) -> dict[str, dict[str, str]]:
    """Handle the internal llm intent fixture payload helper path for this module.

    Inputs:
        Receives variables for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by planning, execution, validation, and presentation code paths that import or call aor_runtime.runtime.eval_fixtures._llm_intent_fixture_payload.
    """
    current_user = getpass.getuser()
    shell_aggregate_prompt = f"using shell, how much space do mp4s take under {variables['files_lambda']}"
    shell_content_files_prompt = f"with shell, show files containing cinnamon under {variables['search_pantry']} as json"
    shell_content_lines_prompt = (
        f"with shell, show matching lines containing cinnamon under {variables['search_pantry']} as json"
    )
    shell_fetch_prompt = f"fetch the title of {variables['fetch_example']} using shell"
    return {
        "slurm": {
            "Is the cluster busy right now?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmMetricsIntent",
                    "confidence": 0.92,
                    "arguments": {"metric_group": "cluster_summary", "output_mode": "json"},
                    "reason": "The user wants a broad cluster load summary.",
                }
            ),
            "Are GPUs available?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmMetricsIntent",
                    "confidence": 0.91,
                    "arguments": {"metric_group": "gpu_summary", "output_mode": "json"},
                    "reason": "The user is asking about GPU availability.",
                }
            ),
            "What is going on with the scheduler?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmMetricsIntent",
                    "confidence": 0.83,
                    "arguments": {"metric_group": "queue_summary", "output_mode": "json"},
                    "reason": "The user wants a scheduler queue summary.",
                }
            ),
            "Are my jobs stuck?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmQueueIntent",
                    "confidence": 0.88,
                    "arguments": {"user": current_user, "state": "PENDING", "output_mode": "json"},
                    "reason": "Pending jobs are a safe interpretation of stuck jobs.",
                }
            ),
            "What failed recently?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmAccountingIntent",
                    "confidence": 0.89,
                    "arguments": {"state": "FAILED", "start": None, "end": None, "output_mode": "json"},
                    "reason": "Recent failures are best answered with accounting history.",
                }
            ),
            "How did jobs do yesterday?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmAccountingIntent",
                    "confidence": 0.9,
                    "arguments": {"start": None, "end": None, "output_mode": "json"},
                    "reason": "Yesterday job outcomes are a read-only accounting query.",
                }
            ),
            "Show me anything unhealthy in SLURM.": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmNodeStatusIntent",
                    "confidence": 0.8,
                    "arguments": {"state": "down", "output_mode": "json"},
                    "reason": "Down nodes are a safe unhealthy-cluster signal.",
                }
            ),
            "How much pressure are the partitions under?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmMetricsIntent",
                    "confidence": 0.84,
                    "arguments": {"metric_group": "partition_summary", "output_mode": "json"},
                    "reason": "Partition pressure is best represented with a partition summary.",
                }
            ),
            "Is slurmdbd healthy?": json.dumps(
                {
                    "matched": True,
                    "intent_type": "SlurmDBDHealthIntent",
                    "confidence": 0.94,
                    "arguments": {"output_mode": "json"},
                    "reason": "The user is explicitly asking for SLURMDBD health.",
                }
            ),
            "Cancel my job 123": json.dumps(
                {
                    "matched": False,
                    "intent_type": None,
                    "confidence": 0.0,
                    "arguments": {},
                    "reason": "The user requested a mutating SLURM operation.",
                }
            ),
        },
        "shell": {
            "check uptime": json.dumps(
                {
                    "matched": True,
                    "intent_type": "ShellInspectionIntent",
                    "confidence": 0.93,
                    "arguments": {"kind": "uptime", "output_mode": "text"},
                    "reason": "The user wants a read-only uptime summary.",
                }
            ),
            "show disk usage": json.dumps(
                {
                    "matched": True,
                    "intent_type": "ShellInspectionIntent",
                    "confidence": 0.92,
                    "arguments": {"kind": "disk_usage", "output_mode": "text"},
                    "reason": "The user wants a disk usage summary.",
                }
            ),
            "show memory usage": json.dumps(
                {
                    "matched": True,
                    "intent_type": "ShellInspectionIntent",
                    "confidence": 0.91,
                    "arguments": {"kind": "memory_summary", "output_mode": "text"},
                    "reason": "The user wants a memory usage summary.",
                }
            ),
            "show top processes": json.dumps(
                {
                    "matched": True,
                    "intent_type": "ShellInspectionIntent",
                    "confidence": 0.9,
                    "arguments": {"kind": "process_summary", "limit": 10, "output_mode": "text"},
                    "reason": "The user wants a short process summary.",
                }
            ),
            "show network summary": json.dumps(
                {
                    "matched": True,
                    "intent_type": "ShellInspectionIntent",
                    "confidence": 0.9,
                    "arguments": {"kind": "network_summary", "output_mode": "text"},
                    "reason": "The user wants an interface and address summary.",
                }
            ),
            "show listening ports": json.dumps(
                {
                    "matched": True,
                    "intent_type": "ShellInspectionIntent",
                    "confidence": 0.9,
                    "arguments": {"kind": "listening_ports", "output_mode": "text"},
                    "reason": "The user wants currently listening ports.",
                }
            ),
            shell_aggregate_prompt: json.dumps(
                {
                    "matched": True,
                    "intent_type": "FileAggregateIntent",
                    "confidence": 0.9,
                    "arguments": {
                        "path": variables["files_lambda"],
                        "pattern": "*.mp4",
                        "recursive": True,
                        "file_only": True,
                        "aggregate": "total_size",
                        "output_mode": "text",
                        "size_unit": "auto",
                    },
                    "reason": "The user is asking for a safe filesystem aggregate.",
                }
            ),
            shell_content_files_prompt: json.dumps(
                {
                    "matched": True,
                    "intent_type": "FileContentDiagnosticIntent",
                    "confidence": 0.88,
                    "arguments": {
                        "path": variables["search_pantry"],
                        "needle": "cinnamon",
                        "pattern": "*",
                        "recursive": True,
                        "result_kind": "files",
                        "output_mode": "json",
                    },
                    "reason": "The user wants matching files for a content search.",
                }
            ),
            shell_content_lines_prompt: json.dumps(
                {
                    "matched": True,
                    "intent_type": "FileContentDiagnosticIntent",
                    "confidence": 0.89,
                    "arguments": {
                        "path": variables["search_pantry"],
                        "needle": "cinnamon",
                        "pattern": "*",
                        "recursive": True,
                        "result_kind": "lines",
                        "output_mode": "json",
                    },
                    "reason": "The user wants matching lines from a content search.",
                }
            ),
            shell_fetch_prompt: json.dumps(
                {
                    "matched": True,
                    "intent_type": "FetchExtractIntent",
                    "confidence": 0.92,
                    "arguments": {"url": variables["fetch_example"], "extract": "title", "output_mode": "text"},
                    "reason": "The user wants a fetch-and-extract title operation.",
                }
            ),
        },
    }
