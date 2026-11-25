"""Изоляция тестов от живых данных в data/.

Модули читают пути из config.settings при импорте, а memory_manager сразу
открывает Chroma. Поэтому подмена делается в pytest_configure — до сбора
тестов и до первого импорта config.settings.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA_DIR = (REPO_ROOT / "data").resolve()

# Ключи config.settings, указывающие в data/, → имя внутри временной data/
DATA_PATH_KEYS = {
    "MEMORY_DB_PATH": "memory_db",
    "VECTOR_DB_PATH": "vector_db",
    "LIBRARY_PATH": "library",
    "VOICE_SAMPLES_DIR": "voice_samples",
    "SOAR_CHUNKS_PATH": "soar_chunks.soar",
    "SAFETY_LOG_PATH": "safety_log.jsonl",
    "ACTION_LOG_PATH": "action_log.jsonl",
    "EMOTION_STATE_PATH": "emotion_state.json",
    "XP_PATH": "xp.json",
    "PROFILE_PATH": "profile.json",
    "STREAM_OUTPUT_DIR": "streams",
    "GAME_PROFILES_DIR": "game_profiles",
}

_state = {"home": None, "owned": False, "chroma_paths": []}


def is_inside_real_data(path) -> bool:
    try:
        Path(path).resolve().relative_to(REAL_DATA_DIR)
        return True
    except ValueError:
        return False


def _make_home(config) -> Path:
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        return factory.mktemp("sofia_home", numbered=True)
    _state["owned"] = True
    return Path(tempfile.mkdtemp(prefix="sofia_home_"))


def _guard_chroma():
    """Запомнить все пути PersistentClient и не дать открыть реальную data/."""
    try:
        import chromadb
    except ImportError:
        return
    original = chromadb.PersistentClient

    def guarded(path="./chroma", *args, **kwargs):
        _state["chroma_paths"].append(str(path))
        if is_inside_real_data(path):
            raise RuntimeError(f"тест открывает живую Chroma: {path}")
        return original(path, *args, **kwargs)

    chromadb.PersistentClient = guarded


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    home = _make_home(config)
    data = home / "data"
    data.mkdir(parents=True, exist_ok=True)
    _state["home"] = home
    os.environ["SOFIA_HOME"] = str(home)
    for key, name in DATA_PATH_KEYS.items():
        os.environ[key] = str(data / name)
    _guard_chroma()

    import memory.feedback
    import modules.skills
    memory.feedback.DEFAULT_PATH = data / "feedback.jsonl"
    modules.skills.DEFAULT_PATH = data / "skills.jsonl"


def pytest_unconfigure(config):
    if _state["owned"] and _state["home"] is not None:
        shutil.rmtree(_state["home"], ignore_errors=True)


@pytest.fixture(scope="session")
def sofia_home() -> Path:
    """Временный каталог приложения (вместо корня репозитория)."""
    return _state["home"]


@pytest.fixture(scope="session")
def chroma_opened_paths() -> list:
    """Пути, по которым за сессию открывался chromadb.PersistentClient."""
    return _state["chroma_paths"]
