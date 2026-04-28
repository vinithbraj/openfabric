from __future__ import annotations

from aor_runtime.core.contracts import ExecutionStep, StepLog
from aor_runtime.runtime.response_renderer import ResponseRenderContext, render_agent_response


def test_shell_response_shows_output_command_and_risk() -> None:
    log = StepLog(
        step=ExecutionStep(id=1, action="shell.exec", args={"command": "df -h"}),
        result={
            "command": "df -h",
            "node": "localhost",
            "stdout": "Filesystem Size Used Avail Use% Mounted on\n/dev/root 10G 5G 5G 50% /\n",
            "stderr": "",
            "returncode": 0,
            "risk": "safe_read_only",
            "policy_reason": "Read-only shell inspection command.",
        },
        success=True,
    )

    rendered = render_agent_response(
        "Filesystem Size Used Avail Use% Mounted on\n/dev/root 10G 5G 5G 50% /",
        execution_events=[log],
        context=ResponseRenderContext(mode="user"),
    )

    assert "## Result" in rendered.markdown
    assert "## Command Used" in rendered.markdown
    assert "```bash" in rendered.markdown
    assert "df -h" in rendered.markdown
    assert "safe_read_only" in rendered.markdown
    assert "Thinking..." not in rendered.markdown


def test_shell_refusal_is_formatted_as_result_without_command_section() -> None:
    rendered = render_agent_response(
        "Request Not Executed\n\nThe shell request was not executed: delete.",
        execution_events=[],
        context=ResponseRenderContext(mode="user"),
    )

    assert "Request Not Executed" in rendered.markdown
    assert "## Command Used" not in rendered.markdown
