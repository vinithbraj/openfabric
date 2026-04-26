from __future__ import annotations

from typing import Any, Literal

from aor_runtime.runtime.output_contract import OutputContract, normalize_output, render_output
from aor_runtime.core.contracts import ToolSpec
from aor_runtime.tools.base import BaseTool, ToolArgsModel, ToolResultModel

def runtime_return(
    value: Any,
    mode: Literal["text", "csv", "json", "count"] = "text",
    output_contract: dict[str, Any] | OutputContract | None = None,
) -> dict[str, Any]:
    contract = _coerce_contract(mode, output_contract)
    normalized = normalize_output(value, contract)
    output = render_output(normalized, contract)
    return {"value": normalized, "output": output}


class RuntimeReturnTool(BaseTool):
    class ToolArgs(ToolArgsModel):
        value: Any = None
        mode: Literal["text", "csv", "json", "count"] = "text"
        output_contract: OutputContract | None = None

    class ToolResult(ToolResultModel):
        value: Any = None
        output: str

    def __init__(self) -> None:
        self.args_model = self.ToolArgs
        self.result_model = self.ToolResult
        self.spec = ToolSpec(
            name="runtime.return",
            description="Internal tool for shaping a final deterministic return value.",
            arguments_schema={
                "type": "object",
                "properties": {
                    "value": {},
                    "mode": {"type": "string", "enum": ["text", "csv", "json", "count"]},
                    "output_contract": {"type": "object"},
                },
            },
        )

    def run(self, arguments: ToolArgs) -> ToolResult:
        return self.ToolResult.model_validate(runtime_return(arguments.value, arguments.mode, arguments.output_contract))


def _coerce_contract(
    mode: Literal["text", "csv", "json", "count"],
    output_contract: dict[str, Any] | OutputContract | None,
) -> OutputContract:
    if isinstance(output_contract, OutputContract):
        return output_contract
    if isinstance(output_contract, dict):
        payload = dict(output_contract)
        payload.setdefault("mode", mode)
        return OutputContract.model_validate(payload)
    return OutputContract(mode=mode)
