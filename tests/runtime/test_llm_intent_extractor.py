from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings
from aor_runtime.runtime.capabilities.slurm import SlurmMetricsIntent
from aor_runtime.runtime.llm_intent_extractor import LLMIntentExtractor


class FakeLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, *, system_prompt: str, user_prompt: str, model=None, temperature=None) -> str:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return self.responses.pop(0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(workspace_root=tmp_path, run_store_path=tmp_path / "runtime.db")


def test_extracts_valid_json_into_typed_intent(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.91, "arguments": {"metric_group": "cluster_summary", "output_mode": "json"}, "reason": "Broad cluster summary."}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is True
    assert isinstance(result.intent, SlurmMetricsIntent)
    assert result.intent.metric_group == "cluster_summary"
    assert result.intent.output_mode == "json"


def test_rejects_malformed_json(tmp_path: Path) -> None:
    llm = FakeLLM(['{"matched": true'])
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is False
    assert result.error is not None
    assert "malformed_json" in result.error


def test_rejects_unknown_intent_type(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "NotARealIntent", "confidence": 0.91, "arguments": {}, "reason": "wrong"}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is False
    assert result.error == "unknown_intent_type"


def test_rejects_invalid_arguments(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.91, "arguments": {"metric_group": "not_real"}, "reason": "bad"}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is False
    assert result.error is not None
    assert result.error.startswith("validation_error:")


def test_rejects_low_confidence(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.42, "arguments": {"metric_group": "cluster_summary"}, "reason": "too low"}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is False
    assert result.error == "low_confidence"


def test_rejects_tool_call_like_output(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": true, "intent_type": "SlurmMetricsIntent", "confidence": 0.91, "arguments": {"metric_group": "cluster_summary", "command": "squeue -h"}, "reason": "unsafe"}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is False
    assert result.error == "unsafe_arguments"


def test_rejects_execution_plan_like_output(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"steps": [{"id": 1, "action": "slurm.metrics", "args": {"metric_group": "cluster_summary"}}]}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    result = extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    assert result.matched is False
    assert result.error == "unsafe_payload_shape"


def test_prompt_does_not_expose_tools_or_shell_commands(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            '{"matched": false, "intent_type": null, "confidence": 0.0, "arguments": {}, "reason": "no_match"}'
        ]
    )
    extractor = LLMIntentExtractor(llm=llm, settings=_settings(tmp_path))

    extractor.extract_intent("Is the cluster busy right now?", "slurm", [SlurmMetricsIntent])

    system_prompt = llm.system_prompts[0]
    user_prompt = llm.user_prompts[0]
    combined = f"{system_prompt}\n{user_prompt}"
    assert "shell.exec" not in combined
    assert "python.exec" not in combined
    assert "slurm.queue" not in combined
    assert "squeue -h" not in combined
    assert "gateway" not in combined.lower()
