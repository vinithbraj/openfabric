"""OpenFABRIC Runtime Module: aor_runtime.tools.runtime_return

Purpose:
    Implement the internal final-output shaping tool.

Responsibilities:
    Normalize values into requested text, CSV, JSON-shaped, count, or contract-driven outputs for the final response boundary.

Data flow / Interfaces:
    Consumes resolved values from previous steps and returns both structured value and rendered output.

Boundaries:
    Normal user mode still passes through final presentation and raw-JSON rejection after this tool runs.
"""

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
    """Runtime return for the surrounding runtime workflow.

    Inputs:
        Receives value, mode, output_contract for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.runtime_return.runtime_return.
    """
    contract = _coerce_contract(mode, output_contract)
    normalized = normalize_output(value, contract)
    output = render_output(normalized, contract)
    return {"value": normalized, "output": output}


class RuntimeReturnTool(BaseTool):
    """Represent runtime return tool within the OpenFABRIC runtime. It extends BaseTool.

    Responsibilities:
        Encapsulates state, validation, or behavior owned by RuntimeReturnTool.

    Data flow / Interfaces:
        Instances are created and consumed by registered tool execution code paths according to type hints and validators.

    Used by:
        Used by callers of aor_runtime.tools.runtime_return.RuntimeReturnTool and related tests.
    """
    class ToolArgs(ToolArgsModel):
        """Represent tool args within the OpenFABRIC runtime. It extends ToolArgsModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolArgs.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.runtime_return.ToolArgs and related tests.
        """
        value: Any = None
        mode: Literal["text", "csv", "json", "count"] = "text"
        output_contract: OutputContract | None = None

    class ToolResult(ToolResultModel):
        """Represent tool result within the OpenFABRIC runtime. It extends ToolResultModel.

        Responsibilities:
            Encapsulates state, validation, or behavior owned by ToolResult.

        Data flow / Interfaces:
            Instances are created and consumed by registered tool execution code paths according to type hints and validators.

        Used by:
            Used by callers of aor_runtime.tools.runtime_return.ToolResult and related tests.
        """
        value: Any = None
        output: str

    def __init__(self) -> None:
        """Handle the internal initialize the object helper path for this module.

        Inputs:
            Uses module or instance state; no caller-supplied data parameters are required.

        Returns:
            Initializes the instance and returns None.

        Used by:
            Used by registered tool execution through RuntimeReturnTool.__init__ calls and related tests.
        """
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
        """Run for RuntimeReturnTool instances.

        Inputs:
            Receives arguments for this RuntimeReturnTool method; type hints and validators define accepted shapes.

        Returns:
            Returns the computed value described by the function name and type hints.

        Used by:
            Used by registered tool execution through RuntimeReturnTool.run calls and related tests.
        """
        return self.ToolResult.model_validate(runtime_return(arguments.value, arguments.mode, arguments.output_contract))


def _coerce_contract(
    mode: Literal["text", "csv", "json", "count"],
    output_contract: dict[str, Any] | OutputContract | None,
) -> OutputContract:
    """Handle the internal coerce contract helper path for this module.

    Inputs:
        Receives mode, output_contract for this function; type hints and validators define accepted shapes.

    Returns:
        Returns the computed value described by the function name and type hints.

    Used by:
        Used by registered tool execution code paths that import or call aor_runtime.tools.runtime_return._coerce_contract.
    """
    if isinstance(output_contract, OutputContract):
        return output_contract
    if isinstance(output_contract, dict):
        payload = dict(output_contract)
        payload.setdefault("mode", mode)
        return OutputContract.model_validate(payload)
    return OutputContract(mode=mode)
