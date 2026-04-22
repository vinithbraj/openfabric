import unittest
from unittest import mock

from codex_ui_gateway import create_app, render_codex_ui_html


class _RouteStub:
    def __init__(self, path):
        self.path = path


class _AppStub:
    def __init__(self):
        self.routes = []

    def get(self, path, **_kwargs):
        def decorator(func):
            self.routes.append(_RouteStub(path))
            return func

        return decorator


class CodexUiGatewayTests(unittest.TestCase):
    def test_render_codex_ui_html_contains_expected_ui_and_api_hooks(self):
        html = render_codex_ui_html()
        self.assertIn("Codex UI for OpenFabric", html)
        self.assertIn("/v1/models", html)
        self.assertIn("/v1/chat/completions", html)
        self.assertIn("openfabric_codex_ui_sessions_v1", html)
        self.assertIn("Codex-style chat for OpenFabric", html)
        self.assertIn("stream: true", html)
        self.assertIn("Runtime config", html)
        self.assertIn("/config", html)
        self.assertNotIn("status-dot", html)
        self.assertNotIn("typing-dot", html)

    def test_create_app_registers_ui_routes_once_on_shared_gateway_app(self):
        base_app = _AppStub()
        base_app.routes.append(_RouteStub("/v1/models"))

        with mock.patch("codex_ui_gateway.openwebui_gateway.create_app", return_value=base_app):
            app = create_app()
            paths = {route.path for route in app.routes}
            self.assertIn("/", paths)
            self.assertIn("/ui", paths)
            self.assertIn("/v1/models", paths)

            app_again = create_app()
            self.assertIs(app_again, app)
            self.assertEqual([route.path for route in app.routes].count("/"), 1)
            self.assertEqual([route.path for route in app.routes].count("/ui"), 1)


if __name__ == "__main__":
    unittest.main()
