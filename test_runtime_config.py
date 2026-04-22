import os
import tempfile
import unittest
from unittest import mock

from runtime.runtime_config import (
    current_runtime_settings,
    describe_runtime_settings,
    get_runtime_setting,
    reset_runtime_settings,
    update_runtime_settings,
)
from runtime.runtime_config_ui import render_runtime_config_html


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_settings_update_and_reset_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")
            env = {
                "OPENFABRIC_RUNTIME_CONFIG_PATH": config_path,
                "OPENFABRIC_GATEWAY_STREAM_PROGRESS": "1",
                "OPENFABRIC_CONSOLE_EVENT_LOGS": "1",
                "OPENFABRIC_FULL_EVENT_LOGS": "0",
                "OPENFABRIC_DEBUG_LOGS": "1",
                "OPENFABRIC_RAW_LOGS": "1",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                initial = current_runtime_settings()
                self.assertTrue(initial["chat_progress_enabled"])
                self.assertTrue(initial["console_event_logs_enabled"])

                updated = update_runtime_settings(
                    {
                        "chat_progress_enabled": False,
                        "console_event_logs_enabled": False,
                    }
                )
                values = {item["id"]: item["value"] for item in updated["settings"]}
                self.assertFalse(values["chat_progress_enabled"])
                self.assertFalse(values["console_event_logs_enabled"])
                self.assertFalse(get_runtime_setting("chat_progress_enabled"))
                self.assertEqual(updated["config_path"], config_path)

                reset = reset_runtime_settings()
                reset_values = {item["id"]: item["value"] for item in reset["settings"]}
                self.assertTrue(reset_values["chat_progress_enabled"])
                self.assertTrue(reset_values["console_event_logs_enabled"])
                self.assertEqual(describe_runtime_settings()["config_path"], config_path)

    def test_render_runtime_config_html_contains_api_hooks(self):
        html = render_runtime_config_html()
        self.assertIn("Live Runtime Configuration", html)
        self.assertIn("/api/runtime-config", html)
        self.assertIn("/api/runtime-config/reset", html)
        self.assertIn("Open chat UI", html)


if __name__ == "__main__":
    unittest.main()
