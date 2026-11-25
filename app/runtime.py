"""Оркестрация запуска Софии: модель → llama-server → FastAPI → окно.

Пользователь ничего не настраивает: лаунчер сам находит или скачивает
llama.cpp и GGUF-модель, подбирает порты и поднимает оба сервера.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

from app import bootstrap, paths
from app.llm import (
    LlamaServer,
    ModelInfo,
    free_vram_gb,
    gpu_name,
    has_nvidia_gpu,
    scan_models,
    total_ram_gb,
)
from app.server import BackendThread, find_free_port

logger = logging.getLogger(__name__)

Report = Callable[[str, Optional[int]], None]
Stage = Callable[[int], None]

# Этапы запуска — их показывает заставка (индекс передаётся в on_stage).
STAGES = (
    "Подготовка",
    "Оборудование",
    "Модель",
    "llama.cpp",
    "Загрузка модели в память",
    "Сервер Софии",
)

DEFAULT_CTX = 8192
LARGE_CTX = 16384
LLAMA_START_TIMEOUT = 300.0
BACKEND_START_TIMEOUT = 240.0


class StartupError(RuntimeError):
    """Ошибка запуска с человекочитаемым текстом."""


def pick_ctx(model: ModelInfo, vram_gb: float) -> int:
    """Контекст llama-server: большой, если модель помещается в VRAM с запасом."""
    if vram_gb and model.size_gb <= vram_gb * 0.6:
        return LARGE_CTX
    return DEFAULT_CTX


def ensure_log_handler(log_file: Path) -> logging.Handler:
    """Файл-лог ровно один раз: лаунчер уже мог подключить тот же файл.

    Второй FileHandler на тот же путь печатает каждую строку дважды,
    поэтому сначала ищем обработчик по имени файла.
    """
    root = logging.getLogger()
    target = Path(log_file).resolve()
    for handler in root.handlers:
        name = getattr(handler, "baseFilename", None)
        if name and Path(name).resolve() == target:
            return handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    return handler


class Runtime:
    """Всё, что нужно работающему приложению."""

    def __init__(self, base: Path, report: Report, on_stage: Optional[Stage] = None):
        self.base = Path(base)
        self.report = report
        self.on_stage = on_stage or (lambda _i: None)
        self.model: Optional[ModelInfo] = None
        self.server: Optional[LlamaServer] = None
        self.backend: Optional[BackendThread] = None
        self.models: List[ModelInfo] = []
        self.llama_port = 0
        self.api_port = 0
        self.provider = "llamacpp"
        self.gpu: Optional[str] = None
        self.vram_gb = 0.0
        self.ram_gb = 0.0
        self.threads = 4
        self._started = False

    # ── Свойства ────────────────────────────────────────────
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}/"

    @property
    def ready(self) -> bool:
        return self._started and self.backend is not None and self.backend.ready

    # ── Подготовка ──────────────────────────────────────────
    def prepare(self):
        base = self.base
        self.on_stage(0)
        self.report("Подготовка каталогов…", None)
        bootstrap.prepare_home(base)
        os.chdir(base)  # относительные пути из config/settings.py

        ensure_log_handler(paths.log_path(base))

        used = bootstrap.configure_bundled_caches()
        if used:
            logger.info("Встроенные модели: %s", ", ".join(used))

        self.on_stage(1)
        self._detect_hardware()
        self.on_stage(2)
        self._detect_models()
        self._start_llama()
        self._apply_env()
        self.on_stage(5)
        self.start_backend()
        self.on_stage(len(STAGES))

    def _detect_hardware(self):
        self.gpu = gpu_name()
        self.vram_gb = free_vram_gb()
        self.ram_gb = total_ram_gb()
        self.threads = bootstrap.pick_threads()
        if self.gpu:
            self.report(
                f"GPU: {self.gpu}, свободно {self.vram_gb:.1f} ГБ VRAM", None
            )
        else:
            self.report("GPU NVIDIA не найден — работа на CPU", None)

    def _detect_models(self):
        self.models = scan_models(bootstrap.model_search_dirs(self.base))
        self.report(f"Найдено моделей GGUF: {len(self.models)}", None)
        self.model = bootstrap.ensure_model(
            self.base, self.vram_gb, self.ram_gb, self.report
        )
        self.report(
            f"Модель: {self.model.name} ({self.model.size_gb:.1f} ГБ)", None
        )

    def _start_llama(self):
        self.on_stage(3)
        server_exe = bootstrap.ensure_llama_server(
            self.base, has_nvidia_gpu(), self.report
        )
        self.llama_port = find_free_port(8080)
        self.report(f"Запуск llama-server на порту {self.llama_port}…", None)
        self.server = LlamaServer(
            server_exe,
            self.model,
            self.llama_port,
            ctx=pick_ctx(self.model, self.vram_gb),
            ngl="auto",
            threads=self.threads,
            log_file=self.base / "logs" / "llama-server.log",
        )
        self.server.start()
        self.on_stage(4)
        self.report("Загрузка модели в память (может занять 1–3 минуты)…", None)
        if not self.server.wait_ready(LLAMA_START_TIMEOUT):
            raise StartupError(
                "llama-server не запустился за 5 минут. "
                "Подробности: logs/llama-server.log"
            )
        self.report("LLM готова", None)

    def _apply_env(self):
        self.api_port = self.api_port or find_free_port(8000)
        updates = bootstrap.build_env_updates(
            provider=self.provider,
            model=self.model.name,
            api_port=self.api_port,
            base_url=self.server.base_url if self.server else None,
            model_path=str(self.model.path),
        )
        for key, value in updates.items():
            os.environ[key] = value
        bootstrap.upsert_env(paths.env_path(self.base), updates)

    # ── Backend ─────────────────────────────────────────────
    def start_backend(self):
        self.report(f"Запуск сервера Софии на порту {self.api_port}…", None)
        app = self._import_app()
        self.backend = BackendThread(app, host="127.0.0.1", port=self.api_port)
        self.backend.start()
        if not self.backend.wait_ready(timeout=BACKEND_START_TIMEOUT):
            raise StartupError(
                "Сервер Софии не ответил на /health. Подробности: logs/sofia.log"
            )
        self._started = True
        self.report("Готово", None)

    @staticmethod
    def _import_app():
        import main as sofia_main  # noqa: PLC0415 — импорт после настройки env

        return sofia_main.app

    # ── Остановка ───────────────────────────────────────────
    def stop(self):
        if self.backend is not None:
            self.backend.shutdown()
            self.backend.join(10)
            self.backend = None
        if self.server is not None:
            self.server.stop()
            self.server = None
        self._started = False

    # ── Диагностика ─────────────────────────────────────────
    def diagnostics(self) -> List[tuple]:
        return [
            ("Python", True, sys.version.split()[0]),
            ("Папка данных", paths.is_writable(self.base), str(self.base)),
            ("GPU", bool(self.gpu), self.gpu or "нет NVIDIA"),
            ("VRAM свободно", self.vram_gb > 0, f"{self.vram_gb:.1f} ГБ"),
            ("RAM", True, f"{self.ram_gb:.1f} ГБ"),
            ("Модель", self.model is not None,
             str(self.model) if self.model else "не выбрана"),
            ("llama-server", bool(self.server), f"порт {self.llama_port}"),
            ("LLM отвечает", self.server is not None and self.server.alive(),
             self.server.base_url if self.server else "—"),
            ("Сервер Софии", self.ready, self.url if self.api_port else "—"),
        ]


def relaunch():
    """Перезапустить приложение (после смены модели)."""
    import subprocess

    if paths.is_frozen():
        subprocess.Popen([sys.executable, *sys.argv[1:]],
                         cwd=str(Path(sys.executable).parent))
    else:
        script = Path(__file__).resolve().parent.parent / "launcher.py"
        subprocess.Popen([sys.executable, str(script), *sys.argv[1:]],
                         cwd=str(script.parent))
