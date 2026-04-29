from __future__ import annotations

from aor_runtime.runtime.output_envelope import normalize_tool_output, parse_shell_table
from aor_runtime.tools.text_format import format_data


def _ps_stdout(row_count: int = 2) -> str:
    return "USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND\n" + "\n".join(
        f"root {1000 + index} {index}.0 0.1 100 50 ? S 10:00 0:01 /usr/bin/proc-{index}"
        for index in range(row_count)
    )


def test_parse_shell_table_handles_ps_aux_command_column() -> None:
    rows = parse_shell_table(_ps_stdout())

    assert rows == [
        {
            "USER": "root",
            "PID": "1000",
            "%CPU": "0.0",
            "%MEM": "0.1",
            "VSZ": "100",
            "RSS": "50",
            "TTY": "?",
            "STAT": "S",
            "START": "10:00",
            "TIME": "0:01",
            "COMMAND": "/usr/bin/proc-0",
        },
        {
            "USER": "root",
            "PID": "1001",
            "%CPU": "1.0",
            "%MEM": "0.1",
            "VSZ": "100",
            "RSS": "50",
            "TTY": "?",
            "STAT": "S",
            "START": "10:00",
            "TIME": "0:01",
            "COMMAND": "/usr/bin/proc-1",
        },
    ]


def test_text_format_renders_parseable_shell_stdout_as_markdown_table() -> None:
    formatted = format_data(_ps_stdout(), "markdown")

    assert formatted["row_count"] == 2
    assert "| USER | PID | %CPU | %MEM | VSZ | RSS | TTY | STAT | START | TIME | COMMAND |" in formatted["content"]
    assert "| root | 1001 | 1.0 | 0.1 | 100 | 50 | ? | S | 10:00 | 0:01 | /usr/bin/proc-1 |" in formatted["content"]


def test_text_format_renders_nested_dict_as_markdown_not_json() -> None:
    formatted = format_data(
        {"cluster": {"queue_count": 172, "pending_jobs": 167}, "nodes": {"node_count": 6}},
        "markdown",
    )

    assert formatted["content"].startswith("| field | value |")
    assert "queue_count=172" in formatted["content"]
    assert not formatted["content"].lstrip().startswith("{")


def test_shell_output_envelope_normalizes_parseable_stdout_to_table() -> None:
    envelope = normalize_tool_output("shell.exec", {"stdout": _ps_stdout(3), "returncode": 0})

    assert envelope.kind == "table"
    assert envelope.presentation_count == 3
    assert envelope.source_tool == "shell.exec"
    assert envelope.source_field == "stdout"
