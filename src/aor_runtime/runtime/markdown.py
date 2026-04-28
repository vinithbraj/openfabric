from __future__ import annotations

from typing import Any, Literal, Sequence


MarkdownAlignment = Literal["left", "right", "center"]
SECTION_BREAK = "---"


def section(title: str, body: Sequence[str] | str | None = None) -> list[str]:
    lines = ["", "", "", f"## {str(title).strip()}", "", "", ""]
    body_lines = _as_lines(body)
    if body_lines:
        lines.extend(body_lines)
        lines.extend(["", "", ""])
    return lines


def code_block(language: str, text: str) -> list[str]:
    body = str(text or "").strip()
    return ["", "", "", "", f"```{language}", "", "", body, "", "", "```", "", "", "", ""]


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    alignments: Sequence[MarkdownAlignment] | None = None,
) -> list[str]:
    header_values = [cell(header) for header in headers]
    if not header_values:
        return []
    normalized_alignments = list(alignments or [])
    while len(normalized_alignments) < len(header_values):
        normalized_alignments.append("left")
    lines = [
        "| " + " | ".join(header_values) + " |",
        "| " + " | ".join(_alignment_marker(alignment) for alignment in normalized_alignments[: len(header_values)]) + " |",
    ]
    for row in rows:
        values = list(row)
        while len(values) < len(header_values):
            values.append(None)
        lines.append("| " + " | ".join(cell(value) for value in values[: len(header_values)]) + " |")
    return lines


def add_section_breaks(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    rendered: list[str] = []
    in_code_block = False
    seen_section = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block and line.startswith("## "):
            if seen_section:
                while rendered and rendered[-1] == "":
                    rendered.pop()
                rendered.extend(["", SECTION_BREAK, ""])
            seen_section = True
        rendered.append(line)
    return "\n".join(rendered).strip()


def cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    text = str(value).replace("|", "\\|").replace("\n", " ").strip()
    return text or "-"


def _as_lines(body: Sequence[str] | str | None) -> list[str]:
    if body is None:
        return []
    if isinstance(body, str):
        return [body] if body else []
    return [str(line) for line in body]


def _alignment_marker(alignment: str) -> str:
    if alignment == "right":
        return "---:"
    if alignment == "center":
        return ":---:"
    return "---"
