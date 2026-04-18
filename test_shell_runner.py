import sys
import types
import unittest
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
sys.modules.setdefault("fastapi", fastapi_stub)

from agent_library.common import task_plan_context
from agent_library.agents.shell_runner import handle_event, _build_preprocess_prompt


class ShellRunnerDataFlowTests(unittest.TestCase):
    def test_task_plan_context_exposes_structured_context(self):
        context = task_plan_context(
            {
                "task": "check whether any listed patient is named Test",
                "original_task": "list patients, then check whether any listed patient is named Test",
                "target_agent": "shell_runner",
                "previous_step_result": {"rows": [{"PatientName": "Alice"}]},
            }
        )
        self.assertIn("Structured workflow context JSON:", context.execution_task)
        self.assertEqual(context.structured_context["previous_step_result"]["rows"][0]["PatientName"], "Alice")

    def test_preprocess_prompt_mentions_structured_input(self):
        prompt = _build_preprocess_prompt("check this list", {"rows": [{"PatientName": "Test"}]})
        self.assertIn("Structured workflow input JSON", prompt)
        self.assertIn("stdin", prompt)

    def test_explicit_shell_command_receives_instruction_input_on_stdin(self):
        captured = {}

        class _Completed:
            stdout = "true\n"
            stderr = ""
            returncode = 0

        def fake_run(_args, input=None, **_kwargs):
            captured["input"] = input
            return _Completed()

        payload = {
            "task": "check prior rows",
            "target_agent": "shell_runner",
            "instruction": {
                "operation": "run_command",
                "command": "python3 -c 'print(1)'",
                "input": [{"PatientName": "Test"}],
            },
        }
        with patch("agent_library.agents.shell_runner.subprocess.run", side_effect=fake_run):
            response = handle_event(types.SimpleNamespace(event="task.plan", payload=payload))

        self.assertEqual(captured["input"], '[{"PatientName": "Test"}]')
        emitted = response["emits"][0]
        self.assertEqual(emitted["event"], "shell.result")
        self.assertEqual(emitted["payload"]["stdout"], "true")

    def test_derived_shell_command_receives_structured_context_on_stdin(self):
        captured = {}

        class _Completed:
            stdout = "false\n"
            stderr = ""
            returncode = 0

        def fake_run(_args, input=None, **_kwargs):
            captured["input"] = input
            return _Completed()

        payload = {
            "task": "check whether anyone in this list is named Test",
            "original_task": "list patients, then check whether anyone in this list is named Test",
            "target_agent": "shell_runner",
            "previous_step_result": {
                "rows": [{"PatientName": "Alice"}],
                "step_id": "step1",
            },
        }
        with patch("agent_library.agents.shell_runner._derive_command_from_task", return_value="python3 -c 'print(0)'"), patch(
            "agent_library.agents.shell_runner.subprocess.run", side_effect=fake_run
        ):
            response = handle_event(types.SimpleNamespace(event="task.plan", payload=payload))

        self.assertIn('"current_step": "check whether anyone in this list is named Test"', captured["input"])
        self.assertIn('"previous_step_result"', captured["input"])
        emitted = response["emits"][0]
        self.assertEqual(emitted["event"], "shell.result")
        self.assertEqual(emitted["payload"]["stdout"], "false")


if __name__ == "__main__":
    unittest.main()
