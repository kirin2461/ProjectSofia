"""HTTP-сервер приложения внутри процесса лаунчера (поток uvicorn)."""

import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import uvicorn


def find_free_port(preferred: Optional[int] = None, host: str = "127.0.0.1") -> int:
    """Свободный TCP-порт. Сначала пробуем preferred, иначе — любой."""
    if preferred:
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, preferred))
                return preferred
        except OSError:
            pass
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 60.0, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if 200 <= r.status < 500:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(interval)
    return False


class BackendThread(threading.Thread):
    """uvicorn в фоновом потоке: FastAPI-приложение живёт в лаунчере."""

    def __init__(self, app, host: str = "127.0.0.1", port: int = 0):
        super().__init__(name="sofia-backend", daemon=True)
        self.app = app
        self.host = host
        self.port = port
        self.ready = False
        self._server: Optional[uvicorn.Server] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def run(self):  # noqa: D401
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            log_config=None,
        )
        self._server = uvicorn.Server(config)
        try:
            self._server.run()
        finally:
            self.ready = False

    def wait_ready(self, timeout: float = 60.0) -> bool:
        ok = wait_http(f"{self.url}/health", timeout=timeout)
        self.ready = ok
        return ok

    def shutdown(self, timeout: float = 5.0):
        if self._server is not None:
            self._server.should_exit = True
