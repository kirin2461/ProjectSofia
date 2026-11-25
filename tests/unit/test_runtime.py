"""Тесты оркестрации запуска (app/runtime.py) без реальных серверов."""

from pathlib import Path

import pytest

from app.llm import ModelInfo
from app.runtime import DEFAULT_CTX, LARGE_CTX, Runtime, StartupError, pick_ctx


def _model(size_gb=4.0, layers=32):
    return ModelInfo(Path("m/model.gguf"), "brain", size_gb, layers, False)


def test_pick_ctx_small_model_gets_large_context():
    assert pick_ctx(_model(4.0), vram_gb=8.0) == LARGE_CTX


def test_pick_ctx_tight_vram_keeps_default():
    assert pick_ctx(_model(7.0), vram_gb=8.0) == DEFAULT_CTX


def test_pick_ctx_cpu_only():
    assert pick_ctx(_model(1.0), vram_gb=0.0) == DEFAULT_CTX


def test_ready_requires_started_and_backend(tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    assert not rt.ready
    rt._started = True
    assert not rt.ready


def test_url_points_to_loopback(tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    rt.api_port = 8123
    assert rt.url == "http://127.0.0.1:8123/"


def test_apply_env_writes_process_and_file(tmp_path, monkeypatch):
    from app.envfile import read_env

    for key in ("LLM_PROVIDER", "MODEL_BRAIN", "MODEL_EXTRACTOR", "API_PORT",
                "API_HOST", "SOFIA_MODEL_PATH", "LLM_BASE_URL"):
        monkeypatch.setenv(key, "")  # восстановится после теста
    rt = Runtime(tmp_path, lambda *_: None)
    rt.model = _model()
    rt.api_port = 8123
    rt._apply_env()

    import os

    assert os.environ["MODEL_BRAIN"] == "brain"
    assert os.environ["API_PORT"] == "8123"
    env = read_env(tmp_path / ".env")
    assert env["API_PORT"] == "8123"
    assert env["SOFIA_MODEL_PATH"] == str(rt.model.path)


def test_diagnostics_reports_missing_parts(tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    rows = rt.diagnostics()
    labels = [r[0] for r in rows]
    assert "Модель" in labels and "Сервер Софии" in labels
    model_row = next(r for r in rows if r[0] == "Модель")
    assert model_row[1] is False and model_row[2] == "не выбрана"


def test_stop_is_idempotent(tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    rt.stop()
    rt.stop()
    assert rt.server is None and rt.backend is None


def test_startup_error_is_runtime_error():
    assert issubclass(StartupError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise StartupError("нет модели")


def test_stages_are_named():
    from app.runtime import STAGES

    assert len(STAGES) == 6 and all(STAGES)


# ── Лог-файл: одна строка на событие ────────────────────────
@pytest.fixture
def clean_root_log():
    """Сохраняет и восстанавливает обработчики корневого логгера."""
    import logging

    root = logging.getLogger()
    saved = list(root.handlers)
    root.handlers = [h for h in saved if not isinstance(h, logging.FileHandler)]
    yield root
    for handler in list(root.handlers):
        if handler not in saved:
            handler.close()
    root.handlers = saved


def test_log_handler_added_once(tmp_path, clean_root_log):
    import logging

    from app.runtime import ensure_log_handler

    path = tmp_path / "logs" / "sofia.log"
    first = ensure_log_handler(path)
    second = ensure_log_handler(path)
    assert first is second
    files = [h for h in clean_root_log.handlers if isinstance(h, logging.FileHandler)]
    assert len(files) == 1


def test_log_handler_does_not_duplicate_launcher_file(tmp_path, clean_root_log):
    import logging

    from app.runtime import ensure_log_handler

    path = tmp_path / "sofia.log"
    launcher = logging.FileHandler(path, encoding="utf-8")  # как в launcher.py
    clean_root_log.addHandler(launcher)
    try:
        added = ensure_log_handler(path)
        assert added is launcher
        files = [h for h in clean_root_log.handlers
                 if isinstance(h, logging.FileHandler)]
        assert len(files) == 1

        # Одно событие — одна строка, хотя обработчиков «дважды» не появилось.
        logging.getLogger("sofia.test").warning("одна строка")
        assert path.read_text(encoding="utf-8").count("одна строка") == 1
    finally:
        clean_root_log.removeHandler(launcher)
        launcher.close()


def test_on_stage_defaults_to_noop(tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    rt.on_stage(3)  # не падает без колбэка
