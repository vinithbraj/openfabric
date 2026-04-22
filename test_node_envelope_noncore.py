import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.HTTPException = Exception
sys.modules.setdefault("fastapi", fastapi_stub)

requests_stub = types.ModuleType("requests")
requests_stub.post = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

from agent_library.common import EventRequest
from agent_library.agents.calculator import handle_event as calculator_handle_event
from agent_library.agents.filesystem import handle_event as filesystem_handle_event
from agent_library.agents.llm_operations_planner import (
    CAPABILITIES as LLM_PLANNER_CAPABILITIES,
    DEFAULT_ALLOWED_EVENTS,
    handle_event as llm_planner_handle_event,
)
from agent_library.agents.notifier import handle_event as notifier_handle_event
from agent_library.agents.operations_planner import handle_event as operations_planner_handle_event
from agent_library.agents.planner import handle_event as planner_handle_event
from agent_library.agents.retriever import handle_event as retriever_handle_event
from agent_library.agents.synthesizer import handle_event as synthesizer_handle_event


class NonCoreNodeEnvelopeTests(unittest.TestCase):
    def test_planner_emits_router_node_envelopes(self):
        response = planner_handle_event(EventRequest(event="user.ask", payload={"question": "research prime numbers"}))
        emits = {item["event"]: item["payload"] for item in response["emits"]}

        self.assertIn("research.query", emits)
        self.assertIn("task.plan", emits)
        self.assertEqual(emits["research.query"]["node"]["role"], "router")
        self.assertEqual(emits["research.query"]["node"]["task"], "research prime numbers")
        self.assertEqual(emits["research.query"]["node"]["operation"], "plan_request")
        self.assertEqual(emits["task.plan"]["node"]["request_event"], "user.ask")

    def test_operations_planner_emits_router_node_envelopes(self):
        response = operations_planner_handle_event(
            EventRequest(event="user.ask", payload={"question": "notify me deployment finished"})
        )
        notify_payload = next(item["payload"] for item in response["emits"] if item["event"] == "notify.send")

        self.assertEqual(notify_payload["node"]["agent"], "ops_planner")
        self.assertEqual(notify_payload["node"]["role"], "router")
        self.assertEqual(notify_payload["node"]["operation"], "plan_request")
        self.assertEqual(notify_payload["node"]["task"], "notify me deployment finished")

    def test_filesystem_emits_node_envelope_for_task_plan_reads(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            Path("README.md").write_text("hello world", encoding="utf-8")
            try:
                response = filesystem_handle_event(
                    EventRequest(
                        event="task.plan",
                        payload={
                            "task": "open README.md",
                            "original_task": "open README.md",
                            "step_id": "step1",
                            "run_id": "run1",
                            "target_agent": "filesystem",
                            "instruction": {"operation": "read_file", "path": "README.md"},
                        },
                    )
                )
            finally:
                os.chdir(original_cwd)

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["node"]["agent"], "filesystem")
        self.assertEqual(payload["node"]["role"], "filesystem")
        self.assertEqual(payload["node"]["operation"], "read_file")
        self.assertEqual(payload["node"]["step_id"], "step1")
        self.assertEqual(payload["node"]["run_id"], "run1")

    def test_notifier_emits_node_envelope_for_notifications(self):
        response = notifier_handle_event(
            EventRequest(
                event="task.plan",
                payload={
                    "task": "notify me deployment finished",
                    "original_task": "notify me deployment finished",
                    "instruction": {"operation": "send_notification", "message": "deployment finished"},
                },
            )
        )
        payload = response["emits"][0]["payload"]

        self.assertEqual(payload["node"]["agent"], "notifier")
        self.assertEqual(payload["node"]["role"], "notifier")
        self.assertEqual(payload["node"]["operation"], "send_notification")
        self.assertEqual(payload["node"]["task"], "notify me deployment finished")

    def test_retriever_emits_node_envelope_for_research_results(self):
        response = retriever_handle_event(EventRequest(event="research.query", payload={"query": "what is slurm"}))
        payload = response["emits"][0]["payload"]

        self.assertEqual(payload["node"]["agent"], "retriever")
        self.assertEqual(payload["node"]["role"], "executor")
        self.assertEqual(payload["node"]["operation"], "lookup_research")
        self.assertEqual(payload["node"]["task"], "what is slurm")

    def test_calculator_emits_node_envelope_for_task_results(self):
        with patch(
            "agent_library.agents.calculator._llm_preprocess",
            return_value={"processable": True, "function": "add", "operands": [2, 2], "reason": "explicit add request"},
        ):
            response = calculator_handle_event(
                EventRequest(event="task.plan", payload={"task": "what is 2 plus 2", "run_id": "run-calc"})
            )

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["node"]["agent"], "calculator")
        self.assertEqual(payload["node"]["role"], "executor")
        self.assertEqual(payload["node"]["operation"], "execute_task")
        self.assertEqual(payload["node"]["run_id"], "run-calc")

    def test_llm_operations_planner_emits_router_node_envelope_for_replans(self):
        original_capabilities = dict(LLM_PLANNER_CAPABILITIES)
        try:
            LLM_PLANNER_CAPABILITIES["agents"] = [
                {"name": "shell_runner", "subscribes_to": ["task.plan"], "emits": ["task.result"]}
            ]
            LLM_PLANNER_CAPABILITIES["available_events"] = sorted(DEFAULT_ALLOWED_EVENTS)
            with patch(
                "agent_library.agents.llm_operations_planner._llm_replan",
                return_value={
                    "replace_step_id": "step1",
                    "steps": [
                        {
                            "id": "step1_1",
                            "target_agent": "shell_runner",
                            "task": "retry the command",
                            "instruction": {"operation": "run_command", "command": "echo ok"},
                        }
                    ],
                },
            ), patch(
                "agent_library.agents.llm_operations_planner._llm_validate_plan_semantics",
                return_value={
                    "valid": True,
                    "reason": "Semantic validator accepted the replacement plan.",
                    "goal_coverage": "complete",
                    "decomposition": "good",
                    "user_action_alignment": "strong",
                    "issues": [],
                    "rewarded_paths": ["retry the failed command"],
                    "disallowed_paths": [],
                },
            ):
                response = llm_planner_handle_event(
                    EventRequest(
                        event="planner.replan.request",
                        payload={"task": "recover failed step", "step_id": "step1", "reason": "bad output"},
                    )
                )
        finally:
            LLM_PLANNER_CAPABILITIES.clear()
            LLM_PLANNER_CAPABILITIES.update(original_capabilities)

        payload = next(item["payload"] for item in response["emits"] if item["event"] == "planner.replan.result")
        self.assertEqual(payload["node"]["agent"], "ops_planner")
        self.assertEqual(payload["node"]["role"], "router")
        self.assertEqual(payload["node"]["operation"], "replan_step")
        self.assertEqual(payload["node"]["step_id"], "step1")
        self.assertEqual(payload["node"]["task"], "recover failed step")

    def test_synthesizer_emits_node_envelope_for_final_answers(self):
        with patch("agent_library.agents.synthesizer._synthesize", return_value="Final answer"):
            response = synthesizer_handle_event(
                EventRequest(
                    event="workflow.result",
                    payload={"task": "produce final answer", "run_id": "run-synth", "status": "completed", "steps": []},
                )
            )

        payload = response["emits"][0]["payload"]
        self.assertEqual(payload["answer"], "Final answer")
        self.assertEqual(payload["node"]["agent"], "synthesizer")
        self.assertEqual(payload["node"]["role"], "synthesizer")
        self.assertEqual(payload["node"]["operation"], "synthesize_answer")
        self.assertEqual(payload["node"]["run_id"], "run-synth")
        self.assertEqual(payload["node"]["task"], "produce final answer")


if __name__ == "__main__":
    unittest.main()
