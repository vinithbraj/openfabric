from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:
        pass

    class _ResponseStub:
        def __init__(self, content=None, *args, **kwargs):
            self.content = content
            self.args = args
            self.kwargs = kwargs

    class StreamingResponse(_ResponseStub):
        pass

    class HTMLResponse(_ResponseStub):
        pass

    class JSONResponse(_ResponseStub):
        pass

    class PlainTextResponse(_ResponseStub):
        pass

    class FastAPI:
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

        def on_event(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator
