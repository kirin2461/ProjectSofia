"""Встроенный HTTP-сервер (uvicorn) внутри лаунчера."""

import threading
import urllib.request

from fastapi import FastAPI

from app.server import BackendThread, find_free_port, wait_http


def test_find_free_port_is_bindable():
    port = find_free_port()
    assert 1024 < port < 65536
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_free_port_skips_busy_port():
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        assert find_free_port(preferred=taken) != taken


def test_backend_thread_serves_health():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    port = find_free_port()
    backend = BackendThread(app, host="127.0.0.1", port=port)
    backend.start()
    try:
        assert wait_http(f"http://127.0.0.1:{port}/health", timeout=30)
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
        assert b'"ok"' in body
    finally:
        backend.shutdown()
        backend.join(10)
    assert not backend.is_alive()


def test_backend_reports_ready_flag():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    port = find_free_port()
    backend = BackendThread(app, host="127.0.0.1", port=port)
    assert not backend.ready
    backend.start()
    try:
        assert backend.wait_ready(timeout=30) is True
    finally:
        backend.shutdown()
        backend.join(10)


def test_wait_http_times_out():
    assert wait_http("http://127.0.0.1:9/nothing", timeout=1.0) is False
