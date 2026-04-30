from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from aor_runtime.config import Settings
from aor_runtime.core.contracts import ExecutionStep, StepLog, ToolSpec
from aor_runtime.runtime.openwebui_trace import OpenWebUITraceRenderer
from aor_runtime.runtime.response_renderer import ResponseRenderContext, render_agent_response
from aor_runtime.runtime.tool_surfaces import friendly_label_for_tool
from aor_runtime.runtime.validator import RuntimeValidator
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolRegistry, ToolResultModel
from aor_runtime.tools.factory import build_tool_registry


class FakeEchoTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        message: str

    class ToolResult(ToolResultModel):
        message: str

    def __init__(self) -> None:
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="fake.echo",
            description="Echo a message for registry surface tests.",
            arguments_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult(message=arguments.message)


def _settings_for_sqlite(tmp_path: Path) -> Settings:
    database_path = tmp_path / "dicom.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
        connection.commit()
    return Settings(
        workspace_root=tmp_path,
        run_store_path=tmp_path / "runtime.db",
        sql_databases={"dicom": f"sqlite:///{database_path}"},
        sql_default_database="dicom",
    )


def _step_log(action: str, result: dict[str, Any], *, args: dict[str, Any] | None = None, success: bool = True) -> StepLog:
    return StepLog(
        step=ExecutionStep(id=1, action=action, args=dict(args or {}), output="result"),
        result=result,
        success=success,
    )


def test_registered_generic_tool_validates_through_tool_registry(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    validator = RuntimeValidator(settings, tools=ToolRegistry([FakeEchoTool()]))

    validation, checks = validator.validate([_step_log("fake.echo", {"message": "hello"})])

    assert validation.success is True
    assert checks[0]["success"] is True
    assert "registered schema" in str(checks[0]["detail"])


def test_registered_generic_tool_result_shape_failure_is_clean(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    validator = RuntimeValidator(settings, tools=ToolRegistry([FakeEchoTool()]))

    validation, checks = validator.validate([_step_log("fake.echo", {"wrong": "shape"})])

    assert validation.success is False
    assert checks[0]["success"] is False
    assert "result schema mismatch" in str(checks[0]["detail"])


def test_unregistered_tool_still_fails_as_unknown_action(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")
    validator = RuntimeValidator(settings, tools=ToolRegistry([]))

    validation, checks = validator.validate([_step_log("fake.missing", {"message": "hello"})])

    assert validation.success is False
    assert checks[0]["detail"] == "unknown action"


def test_sql_validate_runtime_validation_and_presentation(tmp_path: Path) -> None:
    settings = _settings_for_sqlite(tmp_path)
    tool_registry = build_tool_registry(settings)
    result = tool_registry.invoke(
        "sql.validate",
        {"database": "dicom", "query": "SELECT COUNT(*) AS count_value FROM patients"},
    )
    history = [
        _step_log(
            "sql.validate",
            result,
            args={"database": "dicom", "query": "SELECT COUNT(*) AS count_value FROM patients"},
        ),
        StepLog(
            step=ExecutionStep(
                id=2,
                action="text.format",
                args={"source": {"$ref": "validation", "path": "explanation"}, "format": "markdown"},
                input=["validation"],
                output="formatted",
            ),
            result={"content": result["explanation"], "format": "markdown", "row_count": 1},
            success=True,
        ),
        StepLog(
            step=ExecutionStep(id=3, action="runtime.return", args={"value": result["explanation"]}),
            result={"value": result["explanation"], "output": result["explanation"]},
            success=True,
        ),
    ]

    validation, checks = RuntimeValidator(settings, tools=tool_registry).validate(history)
    rendered = render_agent_response(
        result["explanation"],
        execution_events=history,
        context=ResponseRenderContext(source_action="sql.validate"),
    )

    assert validation.success is True
    assert checks[0]["detail"] == "database=dicom valid=true"
    assert "## SQL Validation" in rendered.markdown
    assert "## Query Validated" in rendered.markdown
    assert "SELECT COUNT(*) AS count_value\nFROM patients" in rendered.markdown
    assert "unknown action" not in rendered.markdown


def test_sql_validate_openwebui_trace_uses_surface_label() -> None:
    renderer = OpenWebUITraceRenderer(mode="summary")
    started = renderer.render(
        {
            "event_type": "executor.step.started",
            "payload": {
                "step": {
                    "id": 1,
                    "action": "sql.validate",
                    "args": {"database": "dicom", "query": "SELECT COUNT(*) AS count_value FROM patients"},
                }
            },
        }
    )
    completed = renderer.render(
        {
            "event_type": "executor.step.completed",
            "payload": {
                "step": {"id": 1, "action": "sql.validate"},
                "success": True,
                "result": {"database": "dicom", "query": "SELECT 1", "valid": True, "explanation": "ok"},
            },
        }
    )

    assert friendly_label_for_tool("sql.validate") == "Validate SQL"
    assert started is not None and "Validate SQL" in started
    assert completed is not None and "SQL validation: valid" in completed
