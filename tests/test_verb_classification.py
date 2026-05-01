from __future__ import annotations

from typing import Any

from agent_runtime.capabilities import build_default_registry
from agent_runtime.core.types import TaskFrame
from agent_runtime.input_pipeline.verb_classification import assign_semantic_verbs


class FakeLLMClient:
    """Fake structured LLM client for semantic verb assignment tests."""

    def __init__(self, payloads: dict[str, dict[str, Any]]) -> None:
        self.payloads = payloads
        self.last_prompt = ""
        self.last_schema: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_prompt = prompt
        self.last_schema = schema
        for marker, payload in self.payloads.items():
            if marker in prompt:
                return dict(payload)
        raise AssertionError(f"no fake payload configured for prompt: {prompt}")


def _task(task_id: str, description: str, *, constraints: dict[str, Any] | None = None) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        description=description,
        semantic_verb="unknown",
        object_type="unknown",
        intent_confidence=0.0,
        constraints=constraints or {},
        raw_evidence=description,
    )


def _client() -> FakeLLMClient:
    return FakeLLMClient(
        {
            "list files": {
                "assignments": [
                    {
                        "task_id": "task-list",
                        "semantic_verb": "search",
                        "object_type": "filesystem",
                        "intent_confidence": 0.94,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            },
            "remove files": {
                "assignments": [
                    {
                        "task_id": "task-remove",
                        "semantic_verb": "delete",
                        "object_type": "filesystem",
                        "intent_confidence": 0.93,
                        "risk_level": "medium",
                        "requires_confirmation": False,
                    }
                ]
            },
            "create report": {
                "assignments": [
                    {
                        "task_id": "task-report",
                        "semantic_verb": "create",
                        "object_type": "report",
                        "intent_confidence": 0.92,
                        "risk_level": "medium",
                        "requires_confirmation": False,
                    }
                ]
            },
            "run tests": {
                "assignments": [
                    {
                        "task_id": "task-tests",
                        "semantic_verb": "execute",
                        "object_type": "test_suite",
                        "intent_confidence": 0.96,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            },
            "summarize logs": {
                "assignments": [
                    {
                        "task_id": "task-logs",
                        "semantic_verb": "analyze",
                        "object_type": "logs",
                        "intent_confidence": 0.91,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            },
        }
    )


def test_assign_semantic_verbs_for_list_files() -> None:
    client = _client()
    registry = build_default_registry()

    tasks = assign_semantic_verbs([_task("task-list", "list files in this folder")], client, registry)

    assert tasks[0].semantic_verb in {"read", "search"}
    assert tasks[0].object_type == "filesystem.directory"
    assert tasks[0].risk_level == "low"
    assert "Do not generate commands, shell syntax, SQL, code, or executable plans." in client.last_prompt
    assert "read, search, create, update, delete, transform, analyze, summarize, compare, execute, render, unknown" in client.last_prompt
    assert "Choose object_type from this runtime-owned controlled vocabulary only:" in client.last_prompt


def test_assign_semantic_verbs_for_remove_files_forces_confirmation() -> None:
    tasks = assign_semantic_verbs(
        [_task("task-remove", "remove files older than a week")],
        _client(),
        build_default_registry(),
    )

    assert tasks[0].semantic_verb == "delete"
    assert tasks[0].requires_confirmation is True


def test_assign_semantic_verbs_for_create_report() -> None:
    tasks = assign_semantic_verbs(
        [_task("task-report", "create report from daily metrics")],
        _client(),
        build_default_registry(),
    )

    assert tasks[0].semantic_verb in {"create", "render"}
    assert tasks[0].object_type == "report"


def test_assign_semantic_verbs_for_run_tests_bumps_risk() -> None:
    tasks = assign_semantic_verbs(
        [_task("task-tests", "run tests for this project")],
        _client(),
        build_default_registry(),
    )

    assert tasks[0].semantic_verb == "execute"
    assert tasks[0].risk_level in {"medium", "high"}


def test_assign_semantic_verbs_for_summarize_logs() -> None:
    tasks = assign_semantic_verbs(
        [_task("task-logs", "summarize logs from today")],
        _client(),
        build_default_registry(),
    )

    assert tasks[0].semantic_verb in {"summarize", "analyze"}
    assert tasks[0].object_type == "unknown"
    assert tasks[0].risk_level == "low"


def test_assign_semantic_verbs_normalizes_free_form_memory_information() -> None:
    client = FakeLLMClient(
        {
            "free memory": {
                "assignments": [
                    {
                        "task_id": "task-memory",
                        "semantic_verb": "read",
                        "object_type": "memory information",
                        "intent_confidence": 0.95,
                        "risk_level": "low",
                        "requires_confirmation": False,
                    }
                ]
            }
        }
    )

    tasks = assign_semantic_verbs(
        [_task("task-memory", "retrieve free memory available on the system")],
        client,
        build_default_registry(),
        likely_domains=["system_administration", "operating_system", "system"],
    )

    assert tasks[0].object_type == "system.memory"
