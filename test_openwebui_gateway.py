import sys
import types
import unittest
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


class _RequestStub:
    pass


fastapi_stub.Request = _RequestStub
sys.modules["fastapi"] = fastapi_stub

fastapi_responses_stub = types.ModuleType("fastapi.responses")


class _StreamingResponseStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _ResponseStub:
    def __init__(self, content=None, *args, **kwargs):
        self.content = content
        self.args = args
        self.kwargs = kwargs


fastapi_responses_stub.StreamingResponse = _StreamingResponseStub
fastapi_responses_stub.HTMLResponse = _ResponseStub
fastapi_responses_stub.JSONResponse = _ResponseStub
fastapi_responses_stub.PlainTextResponse = _ResponseStub
sys.modules["fastapi.responses"] = fastapi_responses_stub
sys.modules.pop("web_compat", None)

requests_stub = types.ModuleType("requests")
requests_stub.post = lambda *args, **kwargs: None
sys.modules["requests"] = requests_stub

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = _BaseModel
sys.modules["pydantic"] = pydantic_stub

uvicorn_stub = types.ModuleType("uvicorn")
sys.modules["uvicorn"] = uvicorn_stub

jsonschema_stub = types.ModuleType("jsonschema")
jsonschema_stub.validate = lambda instance, schema: None
sys.modules["jsonschema"] = jsonschema_stub

from openwebui_gateway import _format_progress, _should_use_gateway_direct_fallback, _stream_synthesis_parts


class OpenWebUIGatewayTests(unittest.TestCase):
    def test_progress_sections_use_colored_markdown_badges(self):
        thinking = _format_progress("user.ask", {"question": "list docker containers"}, 0)
        planning = _format_progress("plan.progress", {"message": "I found 1 action to run.", "steps": []}, 0)
        executing = _format_progress(
            "step.progress",
            {"stage": "started", "step_id": "step1", "target_agent": "shell_runner", "task": "list containers"},
            0,
        )
        self.assertIn("### 🟣 Thinking", thinking)
        self.assertIn("### 🟦 Plan", planning)
        self.assertIn("### 🟩 Running step", executing)

    def test_validation_progress_avoids_im_now_console_style_language(self):
        block = _format_progress(
            "validation.progress",
            {"stage": "option_started", "option_id": "option1", "option_label": "Primary plan"},
            0,
        )
        self.assertIn("### 🟨 Trying workflow option", block)
        self.assertNotIn("I’m now", block)
        self.assertNotIn("I'm now", block)

    def test_plan_progress_uses_readable_labels_not_backtick_console_labels(self):
        block = _format_progress(
            "plan.progress",
            {
                "message": "I found 1 action to run.",
                "options": [{"id": "option1", "label": "Primary plan", "step_count": 1}],
                "selected_option_id": "option1",
                "steps": [{"id": "step1", "target_agent": "sql_runner_dicom_mock", "task": "list all schemas in dicom_mock"}],
            },
            0,
        )
        self.assertIn("**Selected option:** option1", block)
        self.assertIn("- **step1** via **sql_runner_dicom_mock**:", block)
        self.assertNotIn("`selected`", block)

    def test_workflow_with_shell_output_uses_direct_fallback(self):
        payload = {
            "task": "list all docker containers",
            "status": "completed",
            "presentation": {"format": "markdown_table", "include_internal_steps": False},
            "steps": [
                {
                    "id": "step1",
                    "task": "list all docker containers",
                    "status": "completed",
                    "event": "shell.result",
                    "payload": {
                        "command": "docker ps -a",
                        "stdout": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db",
                        "stderr": "",
                        "returncode": 0,
                    },
                }
            ],
        }
        self.assertTrue(_should_use_gateway_direct_fallback("workflow.result", payload))

    def test_stream_synthesis_parts_bypasses_gateway_llm_for_shell_workflow(self):
        payload = {
            "task": "list all docker containers",
            "status": "completed",
            "presentation": {"format": "markdown_table", "include_internal_steps": False},
            "steps": [
                {
                    "id": "step1",
                    "task": "list all docker containers",
                    "status": "completed",
                    "event": "shell.result",
                    "payload": {
                        "command": "docker ps -a",
                        "stdout": "CONTAINER ID   IMAGE   NAMES\n123abc   postgres:16   postgres_db",
                        "stderr": "",
                        "returncode": 0,
                    },
                }
            ],
        }
        with patch("openwebui_gateway.requests.post", side_effect=AssertionError("gateway LLM should not be called")):
            answer = "".join(_stream_synthesis_parts("workflow.result", payload))
        self.assertIn("CONTAINER ID", answer)
        self.assertIn("postgres_db", answer)


if __name__ == "__main__":
    unittest.main()
