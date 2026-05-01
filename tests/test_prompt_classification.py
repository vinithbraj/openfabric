from __future__ import annotations

from typing import Any

from agent_runtime.core.types import UserRequest
from agent_runtime.input_pipeline.decomposition import PromptClassification, classify_prompt


class FakeLLMClient:
    """Small fake LLM client that returns prompt-specific classification payloads."""

    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.last_prompt = ""
        self.last_schema: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_prompt = prompt
        self.last_schema = schema
        for raw_prompt, payload in self.payloads.items():
            if raw_prompt in prompt:
                return dict(payload)
        raise AssertionError(f"no fake payload configured for prompt: {prompt}")


def _client() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "list all files in this folder": {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["filesystem"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "This is a direct read-only filesystem task.",
            },
            "delete all logs older than 30 days": {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["filesystem"],
                "risk_level": "high",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "This is a destructive filesystem cleanup request.",
            },
            "show me the top 10 patients by study count": {
                "prompt_type": "simple_tool_task",
                "requires_tools": True,
                "likely_domains": ["sql"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "This is a read-only ranked aggregation query.",
            },
            "read this CSV and summarize it": {
                "prompt_type": "compound_tool_task",
                "requires_tools": True,
                "likely_domains": ["filesystem", "python_data"],
                "risk_level": "low",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "The request requires reading data and then summarizing it.",
            },
            "fix my code and run the tests": {
                "prompt_type": "complex_workflow",
                "requires_tools": True,
                "likely_domains": ["filesystem", "shell", "python_data"],
                "risk_level": "high",
                "needs_clarification": False,
                "clarification_question": None,
                "reason": "This combines code modification with execution and verification.",
            },
        }
    )


def test_classify_list_files_prompt() -> None:
    client = _client()

    classification = classify_prompt(UserRequest(raw_prompt="list all files in this folder"), client)

    assert isinstance(classification, PromptClassification)
    assert classification.prompt_type == "simple_tool_task"
    assert classification.likely_domains == ["filesystem"]
    assert "Return JSON only." in client.last_prompt
    assert "Do not produce commands, shell syntax, SQL, code, or executable plans." in client.last_prompt
    assert "list all files in this folder" in client.last_prompt
    assert client.last_schema is not None
    assert "prompt_type" in client.last_schema["properties"]


def test_classify_delete_logs_prompt() -> None:
    classification = classify_prompt(UserRequest(raw_prompt="delete all logs older than 30 days"), _client())

    assert classification.prompt_type == "simple_tool_task"
    assert classification.risk_level == "high"
    assert classification.requires_tools is True


def test_classify_top_patients_prompt() -> None:
    classification = classify_prompt(UserRequest(raw_prompt="show me the top 10 patients by study count"), _client())

    assert classification.prompt_type == "simple_tool_task"
    assert classification.likely_domains == ["sql"]
    assert classification.reason.startswith("This is a read-only")


def test_classify_csv_summary_prompt() -> None:
    classification = classify_prompt(UserRequest(raw_prompt="read this CSV and summarize it"), _client())

    assert classification.prompt_type == "compound_tool_task"
    assert classification.likely_domains == ["filesystem", "python_data"]


def test_classify_fix_code_prompt() -> None:
    classification = classify_prompt(UserRequest(raw_prompt="fix my code and run the tests"), _client())

    assert classification.prompt_type == "complex_workflow"
    assert classification.likely_domains == ["filesystem", "shell", "python_data"]
    assert classification.risk_level == "high"
