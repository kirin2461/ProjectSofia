"""Пути приложения: портативная установка и запись в файловую систему.

Лаунчер работает в двух режимах:
- dev   — запуск из репозитория (python launcher.py), база = корень проекта;
- frozen — собранный .exe, база = папка с .exe, если она доступна на запись,
  иначе %LOCALAPPDATA%\\Sofia (например, установка в Program Files).
"""

import os
import sys
from pathlib import Path

APP_NAME = "Sofia"
BASE_ENV = "SOFIA_HOME"

DATA_SUBDIRS = (
    "data",
    "data/library",
    "data/voice_samples",
    "data/game_profiles",
    "data/streams",
    "logs",
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    """Папка программы: рядом с .exe либо корень репозитория."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".sofia_write_test"
        probe.write_text("1", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def resolve_base_dir(env=None) -> Path:
    """Каталог, где живут .env, data/, logs/."""
    env = os.environ if env is None else env
    override = env.get(BASE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    here = install_dir()
    if is_writable(here):
        return here
    local = env.get("LOCALAPPDATA") or str(Path.home())
    return (Path(local) / APP_NAME).resolve()


def ensure_dirs(base: Path) -> Path:
    for rel in DATA_SUBDIRS:
        (base / rel).mkdir(parents=True, exist_ok=True)
    return base


def env_path(base: Path) -> Path:
    return base / ".env"


def data_dir(base: Path) -> Path:
    return base / "data"


def bin_dir(base: Path) -> Path:
    return base / "bin"


def model_dir(base: Path) -> Path:
    return base / "models"


def log_path(base: Path) -> Path:
    return base / "logs" / "sofia.log"


def bundled_dir() -> Path:
    """Ресурсы, упакованные PyInstaller (_MEIPASS)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return install_dir()
