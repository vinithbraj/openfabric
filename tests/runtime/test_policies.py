from __future__ import annotations

from aor_runtime.runtime.policies import DEFAULT_POLICIES, render_policy_text, select_policies


def test_select_policies_prefers_sql_for_structured_goal() -> None:
    schema = {
        "databases": [
            {
                "name": "clinical_db",
                "tables": [
                    {
                        "name": "patients",
                        "columns": [{"name": "name"}, {"name": "age"}],
                    }
                ],
            }
        ]
    }

    policies = select_policies("Get top 10 patients", ["sql.query", "python.exec"], schema)
    names = [policy.name for policy in policies]

    assert "sql_preference" in names
    assert "efficiency" in names
    assert "python_usage" not in names


def test_select_policies_prefers_filesystem_for_file_goal() -> None:
    policies = select_policies(
        "Copy the file notes.txt into backup.txt",
        ["fs.copy", "fs.read", "shell.exec", "python.exec"],
    )

    names = [policy.name for policy in policies]
    assert "filesystem_preference" in names
    assert "efficiency" in names


def test_select_policies_adds_python_usage_for_bulk_goal() -> None:
    policies = select_policies(
        "Copy all txt files from A to B",
        ["fs.copy", "fs.list", "python.exec", "shell.exec"],
    )

    names = [policy.name for policy in policies]
    assert "python_usage" in names
    assert "filesystem_preference" in names
    assert "efficiency" in names


def test_select_policies_does_not_force_python_for_simple_formatting_goal() -> None:
    policies = select_policies(
        "Get a list of all upgradable packages and extract the names as csv",
        ["shell.exec", "python.exec"],
    )

    names = [policy.name for policy in policies]
    assert "python_usage" not in names
    assert names == ["efficiency"]


def test_select_policies_honors_explicit_python_request() -> None:
    schema = {
        "databases": [
            {
                "name": "clinical_db",
                "tables": [
                    {
                        "name": "patients",
                        "columns": [{"name": "name"}, {"name": "age"}],
                    }
                ],
            }
        ]
    }

    policies = select_policies(
        "Using python, count how many patient names in clinical_db start with the letter C",
        ["sql.query", "python.exec"],
        schema,
    )

    names = [policy.name for policy in policies]
    assert "python_usage" in names
    assert "sql_preference" not in names


def test_select_policies_respects_allowed_tools() -> None:
    policies = select_policies("Query the patients table", ["fs.read", "shell.exec"])
    names = [policy.name for policy in policies]

    assert "sql_preference" not in names
    assert names == ["efficiency"]


def test_select_policies_keeps_remote_shell_goal_simple() -> None:
    policies = select_policies("Run uptime on node edge-1", ["shell.exec", "python.exec"])
    names = [policy.name for policy in policies]

    assert names == ["efficiency"]


def test_render_policy_text_includes_names_descriptions_and_rules() -> None:
    selected = [policy for policy in DEFAULT_POLICIES if policy.name in {"sql_preference", "filesystem_preference"}]
    text = render_policy_text(selected)

    assert "sql_preference: Prefer SQL for structured data queries" in text
    assert "- Use sql.query for filtering, aggregation, joins" in text
    assert "filesystem_preference: Use filesystem tools for file operations" in text
