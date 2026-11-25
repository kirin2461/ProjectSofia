"""Точка входа Софии: поднимает сервер, модель и показывает окно.

Запуск:
    python launcher.py            — из репозитория (dev)
    Sofia.exe                     — собранная версия

Пользователю ничего настраивать не нужно: всё, чего не хватает, лаунчер
скачивает сам (GGUF-модель, llama.cpp), а прогресс показывает заставка.
"""

import ctypes
import faulthandler
import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from app import paths

APP_TITLE = "София"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_fatal_log = None  # держим файл открытым: faulthandler пишет в него при падении


def _setup_logging(base: Path):
    global _fatal_log
    paths.ensure_dirs(base)
    handlers: list = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(paths.log_path(base), encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers,
                        force=True)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # SOFIA_LOG_LEVEL=DEBUG включает подробную трассировку шагов реплики
    # (полезно, когда падение в нативном коде не даёт трейсбэка).
    if os.environ.get("SOFIA_LOG_LEVEL", "").strip().upper() == "DEBUG":
        for name in ("sofia", "core", "app", "memory"):
            logging.getLogger(name).setLevel(logging.DEBUG)

    # Падение в нативном коде (Soar, CLIPS, torch) не даёт трейсбэка:
    # faulthandler пишет стек всех потоков в logs/sofia-fatal.log.
    try:
        _fatal_log = open(paths.log_path(base).with_name("sofia-fatal.log"),
                          "a", encoding="utf-8")
        faulthandler.enable(file=_fatal_log, all_threads=True)
    except OSError:
        faulthandler.enable()


def _single_instance(base: Path) -> QLockFile | None:
    """Второй экземпляр не запускаем: двум llama-server не хватит памяти."""
    lock = QLockFile(str(base / "sofia.lock"))
    lock.setStaleLockTime(30000)
    if lock.tryLock(100):
        return lock
    QMessageBox.information(None, APP_TITLE,
                            "София уже запущена — посмотрите значок в трее.")
    return None


def _app_user_model_id():
    if sys.platform != "win32":
        return
    try:  # без этого в панели задач иконка питона, а не Софии
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Sofia.Assistant.1"
        )
    except (AttributeError, OSError):
        pass


def main() -> int:
    base = paths.resolve_base_dir()
    os.environ.setdefault(paths.BASE_ENV, str(base))
    _setup_logging(base)
    _app_user_model_id()

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("Sofia")
    app.setQuitOnLastWindowClosed(False)

    lock = _single_instance(base)
    if lock is None:
        return 0

    from app.ui import Controller, apply_style  # только после QApplication

    apply_style(app)
    controller = Controller(app, base)
    app.aboutToQuit.connect(controller.shutdown)
    controller.start()
    QTimer.singleShot(0, controller.splash.raise_)

    code = app.exec()
    relaunch = bool(app.property("sofia_relaunch"))
    controller.shutdown()
    lock.unlock()
    if relaunch:
        from app.runtime import relaunch as do_relaunch

        do_relaunch()
    return code


if __name__ == "__main__":
    sys.exit(main())
