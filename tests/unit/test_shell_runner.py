import sys
import types
import unittest
import json
from unittest.mock import patch


fastapi_stub = types.ModuleType("fastapi")


class _FastAPIStub:
    def post(self, *_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


fastapi_stub.FastAPI = _FastAPIStub
fastapi_stub.HTTPException = Exception


class _RequestStub:
    pass


fastapi_stub.Request = _RequestStub
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

        self.assertEqual(json.loads(captured["input"]), [{"PatientName": "Test"}])
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

    def test_find_permission_warnings_with_stdout_are_treated_as_partial_success(self):
        class _Completed:
            stdout = "./ok.sh\n"
            stderr = "find: './private': Permission denied\n"
            returncode = 1

        with patch("agent_library.agents.shell_runner.subprocess.run", return_value=_Completed()):
            response = handle_event(
                types.SimpleNamespace(
                    event="task.plan",
                    payload={
                        "task": "find all shell scripts in this repo",
                        "target_agent": "shell_runner",
                        "instruction": {
                            "operation": "run_command",
                            "command": 'find . -type f -name "*.sh"',
                        },
                    },
                )
            )

        emitted = response["emits"][0]
        self.assertEqual(emitted["event"], "shell.result")
        self.assertEqual(emitted["payload"]["returncode"], 0)
        self.assertEqual(emitted["payload"]["raw_returncode"], 1)
        self.assertIn("partial results", emitted["payload"]["detail"])

    def test_conda_remove_missing_environment_is_treated_as_idempotent_success(self):
        class _Completed:
            stdout = ""
            stderr = "EnvironmentLocationNotFound: Not a conda environment: /home/vinith/miniconda3/envs/vinith\n"
            returncode = 1

        with patch("agent_library.agents.shell_runner.subprocess.run", return_value=_Completed()):
            response = handle_event(
                types.SimpleNamespace(
                    event="task.plan",
                    payload={
                        "task": "remove conda environment named vinith with -y",
                        "target_agent": "shell_runner",
                        "instruction": {
                            "operation": "run_command",
                            "command": "conda env remove -n vinith -y",
                        },
                    },
                )
            )

        emitted = response["emits"][0]
        self.assertEqual(emitted["event"], "shell.result")
        self.assertEqual(emitted["payload"]["returncode"], 0)
        self.assertEqual(emitted["payload"]["raw_returncode"], 1)
        self.assertIn("already absent", emitted["payload"]["detail"])
        self.assertIn("already absent", emitted["payload"]["normalized_result"])

    def test_multiline_stdin_gets_trailing_newline_for_line_counts(self):
        captured = {}

        class _Completed:
            stdout = "3\n"
            stderr = ""
            returncode = 0

        def fake_run(_args, input=None, **_kwargs):
            captured["input"] = input
            return _Completed()

        with patch("agent_library.agents.shell_runner.subprocess.run", side_effect=fake_run):
            response = handle_event(
                types.SimpleNamespace(
                    event="task.plan",
                    payload={
                        "task": "count the previous lines",
                        "target_agent": "shell_runner",
                        "instruction": {
                            "operation": "run_command",
                            "command": "wc -l",
                            "input": "alpha\nbeta\ngamma",
                        },
                    },
                )
            )

        self.assertEqual(captured["input"], "alpha\nbeta\ngamma\n")
        emitted = response["emits"][0]
        self.assertEqual(emitted["payload"]["stdout"], "3")


if __name__ == "__main__":
    unittest.main()
