from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REQUIRED_HEADINGS = (
    "Purpose:",
    "Responsibilities:",
    "Data flow / Interfaces:",
    "Boundaries:",
)


def test_runtime_source_modules_have_standard_documentation_headers() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "aor_runtime"
    missing: list[str] = []

    for path in sorted(root.rglob("*.py")):
        module = ast.parse(path.read_text())
        docstring = ast.get_docstring(module) or ""
        if not docstring.startswith("OpenFABRIC Runtime Module:"):
            missing.append(f"{path.relative_to(root.parent.parent)}: missing OpenFABRIC module header")
            continue
        for heading in REQUIRED_HEADINGS:
            if heading not in docstring:
                missing.append(f"{path.relative_to(root.parent.parent)}: missing {heading}")

    assert not missing, "\n".join(missing)


def test_runtime_source_classes_and_functions_have_docstrings() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "aor_runtime"
    missing: list[str] = []

    for path in sorted(root.rglob("*.py")):
        module = ast.parse(path.read_text())
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node) or ""
                if not docstring:
                    missing.append(f"{path.relative_to(root.parent.parent)}:{node.lineno}: class {node.name} missing docstring")
                elif "Used by:" not in docstring:
                    missing.append(f"{path.relative_to(root.parent.parent)}:{node.lineno}: class {node.name} missing Used by")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node) or ""
                if not docstring:
                    missing.append(f"{path.relative_to(root.parent.parent)}:{node.lineno}: function {node.name} missing docstring")
                    continue
                for heading in ("Inputs:", "Returns:", "Used by:"):
                    if heading not in docstring:
                        missing.append(f"{path.relative_to(root.parent.parent)}:{node.lineno}: function {node.name} missing {heading}")

    assert not missing, "\n".join(missing)


def test_tracked_markdown_avoids_stale_architecture_phrases() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tracked_markdown = subprocess.check_output(["git", "ls-files", "*.md"], cwd=repo_root, text=True).splitlines()
    forbidden = (
        "Raw Planner Fallback",
        "GoalDecomposer",
        "capability-pack architecture is the active",
        "planner and decomposer model selection",
        "Deterministic SlurmCapabilityPack classify",
        "do not let the LLM emit tool calls directly",
    )
    violations: list[str] = []

    for relative in tracked_markdown:
        text = (repo_root / relative).read_text(errors="ignore")
        for phrase in forbidden:
            if phrase in text:
                violations.append(f"{relative}: contains stale phrase {phrase!r}")

    assert not violations, "\n".join(violations)
