#!/usr/bin/env python3
import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_live_scenarios import (
    DICOM_DSN,
    MYDB_DSN,
    REPORTS_DIR,
    RUNS_DIR,
    SPEC_PATH,
    MockSlurmGateway,
    PlannerGateway,
    Scenario,
    _evaluate_scenario,
    _extract_run_id,
    _query_scalar,
    _query_strings,
)


def _sorted_file_names(path: Path, suffix: str | None = None, prefix: str | None = None) -> list[str]:
    items = []
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        if suffix is not None and entry.suffix.lower() != suffix.lower():
            continue
        if prefix is not None and not entry.name.startswith(prefix):
            continue
        items.append(entry.name)
    return sorted(items)


def _sorted_dir_names(path: Path) -> list[str]:
    return sorted(entry.name for entry in path.iterdir() if entry.is_dir())


def _substring_count(path: Path, needle: str) -> int:
    return path.read_text(encoding="utf-8").count(needle)


def _word_like_count(paths: list[Path], needle: str) -> int:
    total = 0
    token = needle.lower()
    for path in paths:
        total += path.read_text(encoding="utf-8").lower().count(token)
    return total


def _lines_in_file(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _command_output_lines(command: list[str]) -> list[str]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _command_output_text(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _build_ground_truth() -> dict[str, Any]:
    root_py_files = _sorted_file_names(REPO_ROOT, suffix=".py")
    root_md_files = _sorted_file_names(REPO_ROOT, suffix=".md")
    root_test_files = _sorted_file_names(REPO_ROOT, suffix=".py", prefix="test_")
    root_dirs = _sorted_dir_names(REPO_ROOT)
    agent_python_files = _sorted_file_names(REPO_ROOT / "agent_library" / "agents", suffix=".py")
    runtime_python_files = _sorted_file_names(REPO_ROOT / "runtime", suffix=".py")
    script_entries = sorted(entry.name for entry in (REPO_ROOT / "scripts").iterdir())
    script_python_files = _sorted_file_names(REPO_ROOT / "scripts", suffix=".py")
    spec_yaml_files = _sorted_file_names(REPO_ROOT / "agent_library" / "specs", suffix=".yml")
    openwebui_root_files = sorted(
        entry.name
        for entry in REPO_ROOT.iterdir()
        if entry.is_file() and "openwebui" in entry.name.lower()
    )
    docker_containers = _command_output_lines(["docker", "ps", "-a", "--format", "{{.Names}}"])
    docker_images = _command_output_lines(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    root_markdown_paths = [REPO_ROOT / name for name in root_md_files]
    mydb_tables = _query_strings(
        MYDB_DSN,
        """
        select table_name
        from information_schema.tables
        where table_schema = %s
        order by table_name
        """,
        ("dicom",),
    )
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
    return {
        "root_py_files": root_py_files,
        "root_py_count": len(root_py_files),
        "root_md_files": root_md_files,
        "root_md_count": len(root_md_files),
        "root_test_files": root_test_files,
        "root_test_count": len(root_test_files),
        "root_dirs": root_dirs,
        "root_dir_count": len(root_dirs),
        "agent_python_files": agent_python_files,
        "agent_python_count": len(agent_python_files),
        "runtime_python_files": runtime_python_files,
        "runtime_python_count": len(runtime_python_files),
        "script_entries": script_entries,
        "script_entry_count": len(script_entries),
        "script_python_files": script_python_files,
        "script_python_count": len(script_python_files),
        "runtime_entry_count": len(list((REPO_ROOT / "runtime").iterdir())),
        "spec_yaml_files": spec_yaml_files,
        "spec_yaml_count": len(spec_yaml_files),
        "assistant_spec_filename_count": len([name for name in spec_yaml_files if "assistant" in name.lower()]),
        "openwebui_root_files": openwebui_root_files,
        "openwebui_root_file_count": len(openwebui_root_files),
        "docker_containers": docker_containers,
        "docker_container_count": len(docker_containers),
        "docker_images": docker_images,
        "docker_image_count": len(docker_images),
        "git_branch": _command_output_text(["git", "branch", "--show-current"]),
        "last_commit_message": _command_output_text(["git", "log", "--format=%s", "-n", "1"]),
        "working_tree_clean": _command_output_text(
            ["bash", "-lc", "if git diff --quiet && git diff --cached --quiet; then echo true; else echo false; fi"]
        ),
        "version4_line_count": _lines_in_file(REPO_ROOT / "VERSION_4_PRIMITIVE_CATALOG.md"),
        "engine_task_plan_count": _substring_count(REPO_ROOT / "runtime" / "engine.py", "task.plan"),
        "gateway_planner_name_count": _substring_count(REPO_ROOT / "openwebui_gateway.py", "PlannerGateway"),
        "root_markdown_graph_count": _word_like_count(root_markdown_paths, "graph"),
        "mydb_tables": mydb_tables,
        "mydb_table_count": len(mydb_tables),
        "dicom_tables": dicom_tables,
        "dicom_table_count": len(dicom_tables),
        "mydb_patient_count": int(_query_scalar(MYDB_DSN, "select count(*) from dicom.patients") or 0),
        "mydb_study_count": int(_query_scalar(MYDB_DSN, "select count(*) from dicom.studies") or 0),
        "mydb_series_count": int(_query_scalar(MYDB_DSN, "select count(*) from dicom.series") or 0),
        "mydb_instance_count": int(_query_scalar(MYDB_DSN, "select count(*) from dicom.instances") or 0),
        "mydb_multi_study_patient_count": mydb_multi_study_patients,
        "dicom_patient_count": int(_query_scalar(DICOM_DSN, "select count(*) from dicom.patients") or 0),
        "dicom_study_count": int(_query_scalar(DICOM_DSN, "select count(*) from dicom.studies") or 0),
        "dicom_series_count": int(_query_scalar(DICOM_DSN, "select count(*) from dicom.series") or 0),
        "dicom_instance_count": int(_query_scalar(DICOM_DSN, "select count(*) from dicom.instances") or 0),
        "dicom_tag_count": int(_query_scalar(DICOM_DSN, "select count(*) from dicom.dicom_tags") or 0),
    }


def _shell_scenarios() -> list[Scenario]:
    return [
        Scenario(
            scenario_id="shell_01_root_python_inventory",
            question="Using the shell in the repository root, list the first five Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_py_count"]), *ctx["root_py_files"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_02_root_markdown_inventory",
            question="Using the shell in the repository root, list the Markdown files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_md_count"]), *ctx["root_md_files"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_03_root_test_inventory",
            question="Using the shell in the repository root, list the test Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_test_count"]), *ctx["root_test_files"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_04_agent_module_inventory",
            question="Using the shell, under agent_library/agents list the first five Python modules alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["agent_python_count"]), *ctx["agent_python_files"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_05_runtime_python_inventory",
            question="Using the shell, under runtime list the Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["runtime_python_count"]), *ctx["runtime_python_files"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_06_spec_inventory",
            question="Using the shell, in agent_library/specs list the YAML spec files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["spec_yaml_count"]), *ctx["spec_yaml_files"]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_07_scripts_inventory",
            question="Using the shell, in the scripts directory list the entries alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["script_entry_count"]), *ctx["script_entries"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_08_root_directory_inventory",
            question="Using the shell in the repository root, list the first five directories alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_dir_count"]), *ctx["root_dirs"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_09_openwebui_root_files",
            question="Using the shell in the repository root, count the files whose names contain openwebui and list them.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["openwebui_root_file_count"]), *ctx["openwebui_root_files"]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_10_root_python_vs_markdown",
            question="Using the shell in the repository root, count Python files and Markdown files, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["root_py_count"]),
                str(ctx["root_md_count"]),
                str(abs(ctx["root_py_count"] - ctx["root_md_count"])),
            ],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_11_agent_vs_spec_difference",
            question="Using the shell, count Python modules in agent_library/agents and YAML specs in agent_library/specs, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["agent_python_count"]),
                str(ctx["spec_yaml_count"]),
                str(abs(ctx["agent_python_count"] - ctx["spec_yaml_count"])),
            ],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_12_runtime_vs_test_difference",
            question="Using the shell, count Python files under runtime and test Python files in the repository root, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["runtime_python_count"]),
                str(ctx["root_test_count"]),
                str(abs(ctx["runtime_python_count"] - ctx["root_test_count"])),
            ],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_13_version4_line_count",
            question="Using the shell, how many lines are in VERSION_4_PRIMITIVE_CATALOG.md?",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["version4_line_count"])],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_14_engine_task_plan_count",
            question="Using the shell, in runtime/engine.py how many times does the string task.plan appear?",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["engine_task_plan_count"])],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_15_gateway_planner_name_count",
            question="Using the shell, in openwebui_gateway.py how many times does the string PlannerGateway appear?",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["gateway_planner_name_count"])],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_16_root_markdown_graph_count",
            question="Using the shell, across the Markdown files in the repository root how many times does the word graph appear?",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_markdown_graph_count"])],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_17_assistant_spec_filename_count",
            question="Using the shell, in agent_library/specs how many YAML spec filenames contain the word assistant?",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["assistant_spec_filename_count"])],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_18_script_python_inventory",
            question="Using the shell, in the scripts directory count the Python files and list the first five alphabetically.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["script_python_count"]), *ctx["script_python_files"][:3]],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_19_runtime_entry_and_python_counts",
            question="Using the shell, in runtime count the direct entries and the Python files, then report both counts.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(len(list((REPO_ROOT / 'runtime').iterdir()))),
                str(ctx["runtime_python_count"]),
            ],
            min_step_count=1,
        ),
        Scenario(
            scenario_id="shell_20_root_vs_agent_python_difference",
            question="Using the shell, count Python files in the repository root and in agent_library/agents, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["root_py_count"]),
                str(ctx["agent_python_count"]),
                str(abs(ctx["root_py_count"] - ctx["agent_python_count"])),
            ],
            min_step_count=1,
        ),
    ]


def _db_shell_scenarios() -> list[Scenario]:
    return [
        Scenario(
            scenario_id="db_shell_01_dicom_patients_vs_root_python",
            question=(
                "In the dicom_mock database count patients, and in the repository root using the shell count Python files, "
                "then report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_patient_count"]),
                str(ctx["root_py_count"]),
                str(abs(ctx["dicom_patient_count"] - ctx["root_py_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_02_mydb_patients_vs_root_python",
            question=(
                "In the mydb database count patients, and in the repository root using the shell count Python files, "
                "then report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_patient_count"]),
                str(ctx["root_py_count"]),
                str(abs(ctx["mydb_patient_count"] - ctx["root_py_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_03_multi_study_vs_root_python",
            question=(
                "In the mydb database count patients who have more than 2 studies, and in the repository root using the shell "
                "count Python files, then report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_multi_study_patient_count"]),
                str(ctx["root_py_count"]),
                str(abs(ctx["mydb_multi_study_patient_count"] - ctx["root_py_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_04_dicom_tables_and_root_markdown",
            question=(
                "In the dicom_mock database list the tables in the dicom schema alphabetically, and in the repository root using the shell "
                "count Markdown files. Report the table count, the Markdown count, and include the first few table names."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_table_count"]),
                str(ctx["root_md_count"]),
                *ctx["dicom_tables"][:3],
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_05_mydb_tables_vs_root_dirs",
            question=(
                "In the mydb database count the tables in the dicom schema, and in the repository root using the shell count directories. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_table_count"]),
                str(ctx["root_dir_count"]),
                str(abs(ctx["mydb_table_count"] - ctx["root_dir_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_06_dicom_tables_vs_specs",
            question=(
                "In the dicom_mock database count tables in the dicom schema, and in agent_library/specs using the shell count YAML specs. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_table_count"]),
                str(ctx["spec_yaml_count"]),
                str(abs(ctx["dicom_table_count"] - ctx["spec_yaml_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_07_mydb_studies_vs_scripts",
            question=(
                "In the mydb database count studies, and in the scripts directory using the shell count direct entries. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_study_count"]),
                str(ctx["script_entry_count"]),
                str(abs(ctx["mydb_study_count"] - ctx["script_entry_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_08_mydb_instances_vs_tests",
            question=(
                "In the mydb database count instances, and in the repository root using the shell count test Python files. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_instance_count"]),
                str(ctx["root_test_count"]),
                str(abs(ctx["mydb_instance_count"] - ctx["root_test_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_09_dicom_series_vs_agent_modules",
            question=(
                "In the dicom_mock database count series, and in agent_library/agents using the shell count Python modules. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_series_count"]),
                str(ctx["agent_python_count"]),
                str(abs(ctx["dicom_series_count"] - ctx["agent_python_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_10_dicom_instances_vs_runtime_python",
            question=(
                "In the dicom_mock database count instances, and under runtime using the shell count Python files. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_instance_count"]),
                str(ctx["runtime_python_count"]),
                str(abs(ctx["dicom_instance_count"] - ctx["runtime_python_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_11_dicom_tags_vs_specs",
            question=(
                "In the dicom_mock database count rows in dicom_tags, and in agent_library/specs using the shell count YAML spec files. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_tag_count"]),
                str(ctx["spec_yaml_count"]),
                str(abs(ctx["dicom_tag_count"] - ctx["spec_yaml_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_12_mydb_patients_vs_openwebui_files",
            question=(
                "In the mydb database count patients, and in the repository root using the shell count files whose names contain openwebui. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_patient_count"]),
                str(ctx["openwebui_root_file_count"]),
                str(abs(ctx["mydb_patient_count"] - ctx["openwebui_root_file_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_13_mydb_studies_vs_engine_task_plan",
            question=(
                "In the mydb database count studies, and using the shell count how many times task.plan appears in runtime/engine.py. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_study_count"]),
                str(ctx["engine_task_plan_count"]),
                str(abs(ctx["mydb_study_count"] - ctx["engine_task_plan_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_14_dicom_patients_and_root_markdown",
            question=(
                "In the dicom_mock database count patients, and in the repository root using the shell count Markdown files. "
                "Report both counts."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_patient_count"]),
                str(ctx["root_md_count"]),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_15_mydb_series_and_script_python",
            question=(
                "In the mydb database count series, and in the scripts directory using the shell count Python files. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_series_count"]),
                str(ctx["script_python_count"]),
                str(abs(ctx["mydb_series_count"] - ctx["script_python_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_16_dicom_table_list_and_test_count",
            question=(
                "In the dicom_mock database list the first four tables in the dicom schema alphabetically, and in the repository root using the shell "
                "count test Python files. Report the table count, the test count, and include the first few table names."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_table_count"]),
                str(ctx["root_test_count"]),
                *ctx["dicom_tables"][:3],
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_17_multi_study_vs_agent_modules",
            question=(
                "In the mydb database count patients who have more than 2 studies, and in agent_library/agents using the shell count Python modules. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_multi_study_patient_count"]),
                str(ctx["agent_python_count"]),
                str(abs(ctx["mydb_multi_study_patient_count"] - ctx["agent_python_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_18_dicom_studies_vs_root_dirs",
            question=(
                "In the dicom_mock database count studies, and in the repository root using the shell count directories. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["dicom_study_count"]),
                str(ctx["root_dir_count"]),
                str(abs(ctx["dicom_study_count"] - ctx["root_dir_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_19_mydb_instances_vs_root_python",
            question=(
                "In the mydb database count instances, and in the repository root using the shell count Python files. "
                "Report both counts and the difference."
            ),
            expected_agents=("sql_runner_mydb", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_instance_count"]),
                str(ctx["root_py_count"]),
                str(abs(ctx["mydb_instance_count"] - ctx["root_py_count"])),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="db_shell_20_dual_db_and_shell_counts",
            question=(
                "In the mydb database count patients, in the dicom_mock database count tables in the dicom schema, "
                "and in the repository root using the shell count Python files. Report all three counts in one answer."
            ),
            expected_agents=("sql_runner_mydb", "sql_runner_dicom_mock", "shell_runner"),
            expected_fragments=lambda ctx: [
                str(ctx["mydb_patient_count"]),
                str(ctx["dicom_table_count"]),
                str(ctx["root_py_count"]),
            ],
            min_step_count=3,
        ),
    ]


def _compound_shell_scenarios() -> list[Scenario]:
    return [
        Scenario(
            scenario_id="compound_shell_01_docker_inventory_shared_verb",
            question="list all docker containers and docke r images on this machines",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [*ctx["docker_containers"][:2], *ctx["docker_images"][:2]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_02_docker_count_difference",
            question="Count all docker containers and docker images on this machine, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["docker_container_count"]),
                str(ctx["docker_image_count"]),
                str(abs(ctx["docker_container_count"] - ctx["docker_image_count"])),
            ],
            min_step_count=3,
        ),
        Scenario(
            scenario_id="compound_shell_03_git_branch_and_last_commit",
            question="Show the current git branch and last commit message.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [ctx["git_branch"], ctx["last_commit_message"]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_04_git_branch_and_clean_state",
            question="Show the current git branch and whether the working tree is clean.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                ctx["git_branch"],
                ("true", "yes") if str(ctx["working_tree_clean"]).lower() == "true" else ("false", "no"),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_05_root_python_inventory",
            question="Using the shell in the repository root, list the first five Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_py_count"]), *ctx["root_py_files"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_06_root_markdown_inventory",
            question="Using the shell in the repository root, list the Markdown files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_md_count"]), *ctx["root_md_files"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_07_root_test_inventory",
            question="Using the shell in the repository root, list the test Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_test_count"]), *ctx["root_test_files"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_08_agent_module_inventory",
            question="Using the shell, under agent_library/agents list the first five Python modules alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["agent_python_count"]), *ctx["agent_python_files"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_09_runtime_python_inventory",
            question="Using the shell, under runtime list the Python files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["runtime_python_count"]), *ctx["runtime_python_files"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_10_spec_inventory",
            question="Using the shell, in agent_library/specs list the YAML spec files alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["spec_yaml_count"]), *ctx["spec_yaml_files"]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_11_scripts_inventory",
            question="Using the shell, in the scripts directory list the entries alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["script_entry_count"]), *ctx["script_entries"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_12_root_directory_inventory",
            question="Using the shell in the repository root, list the first five directories alphabetically and tell me the total count.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["root_dir_count"]), *ctx["root_dirs"][:3]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_13_openwebui_count_and_list",
            question="Using the shell in the repository root, count the files whose names contain openwebui and list them.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [str(ctx["openwebui_root_file_count"]), *ctx["openwebui_root_files"]],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_14_root_python_vs_markdown",
            question="Using the shell in the repository root, count Python files and Markdown files, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["root_py_count"]),
                str(ctx["root_md_count"]),
                str(abs(ctx["root_py_count"] - ctx["root_md_count"])),
            ],
            min_step_count=3,
        ),
        Scenario(
            scenario_id="compound_shell_15_agent_vs_spec_difference",
            question="Using the shell, count Python modules in agent_library/agents and YAML specs in agent_library/specs, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["agent_python_count"]),
                str(ctx["spec_yaml_count"]),
                str(abs(ctx["agent_python_count"] - ctx["spec_yaml_count"])),
            ],
            min_step_count=3,
        ),
        Scenario(
            scenario_id="compound_shell_16_runtime_vs_test_difference",
            question="Using the shell, count Python files under runtime and test Python files in the repository root, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["runtime_python_count"]),
                str(ctx["root_test_count"]),
                str(abs(ctx["runtime_python_count"] - ctx["root_test_count"])),
            ],
            min_step_count=3,
        ),
        Scenario(
            scenario_id="compound_shell_17_runtime_entries_and_python_counts",
            question="Using the shell, in runtime count the direct entries and the Python files, then report both counts.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["runtime_entry_count"]),
                str(ctx["runtime_python_count"]),
            ],
            min_step_count=2,
        ),
        Scenario(
            scenario_id="compound_shell_18_root_vs_agent_python_difference",
            question="Using the shell, count Python files in the repository root and in agent_library/agents, then report both counts and the difference.",
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["root_py_count"]),
                str(ctx["agent_python_count"]),
                str(abs(ctx["root_py_count"] - ctx["agent_python_count"])),
            ],
            min_step_count=3,
        ),
        Scenario(
            scenario_id="compound_shell_19_dual_string_counts_difference",
            question=(
                "Using the shell, in runtime/engine.py count how many times task.plan appears, and in "
                "openwebui_gateway.py count how many times PlannerGateway appears. Report both counts and the difference."
            ),
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["engine_task_plan_count"]),
                str(ctx["gateway_planner_name_count"]),
                str(abs(ctx["engine_task_plan_count"] - ctx["gateway_planner_name_count"])),
            ],
            min_step_count=3,
        ),
        Scenario(
            scenario_id="compound_shell_20_line_count_vs_markdown_token_count",
            question=(
                "Using the shell, how many lines are in VERSION_4_PRIMITIVE_CATALOG.md, and across the Markdown files "
                "in the repository root how many times does the word graph appear? Report both counts and the difference."
            ),
            expected_agents=("shell_runner",),
            expected_fragments=lambda ctx: [
                str(ctx["version4_line_count"]),
                str(ctx["root_markdown_graph_count"]),
                str(abs(ctx["version4_line_count"] - ctx["root_markdown_graph_count"])),
            ],
            min_step_count=3,
        ),
    ]


def _all_scenarios(selected_suites: set[str]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    if "shell" in selected_suites:
        scenarios.extend(_shell_scenarios())
    if "db-shell" in selected_suites:
        scenarios.extend(_db_shell_scenarios())
    if "compound-shell" in selected_suites:
        scenarios.extend(_compound_shell_scenarios())
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extended live OpenFabric shakeout scenarios.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=("shell", "db-shell", "compound-shell"),
        help="Scenario suite(s) to run. Defaults to both.",
    )
    args = parser.parse_args()

    selected_suites = set(args.suite or ["shell", "db-shell"])
    scenarios = _all_scenarios(selected_suites)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_store_dir = RUNS_DIR / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    context = _build_ground_truth()
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_store_dir": str(run_store_dir),
        "suites": sorted(selected_suites),
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
            for scenario in scenarios:
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
                except Exception as exc:
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
        except KeyboardInterrupt:
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
    report_path = REPORTS_DIR / "extended_live_scenario_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    print(f"[REPORT] {report_path}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
