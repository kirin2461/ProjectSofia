"""Сторож: тесты не должны трогать живую data/ в корне репозитория."""

from pathlib import Path

from tests.conftest import DATA_PATH_KEYS, REAL_DATA_DIR, is_inside_real_data


def test_settings_paths_point_to_temp_home(sofia_home):
    import config.settings as settings

    for key in DATA_PATH_KEYS:
        value = getattr(settings, key)
        assert not is_inside_real_data(value), f"{key} = {value}"
        assert Path(value).resolve().is_relative_to(Path(sofia_home).resolve()), key


def test_memory_manager_chroma_is_not_live_data():
    import memory.memory_manager as mm

    assert not is_inside_real_data(mm.MEMORY_DB_PATH)
    assert not is_inside_real_data(mm.VECTOR_DB_PATH)


def test_chroma_never_opened_inside_real_data(chroma_opened_paths):
    import memory.memory_manager  # noqa: F401 — гарантирует хотя бы одно открытие

    bad = [p for p in chroma_opened_paths if is_inside_real_data(p)]
    assert not bad, f"Chroma открывалась в {REAL_DATA_DIR}: {bad}"


def test_guard_blocks_live_chroma():
    import chromadb
    import pytest

    assert chromadb.PersistentClient.__name__ == "guarded"
    with pytest.raises(RuntimeError):
        chromadb.PersistentClient(path=str(REAL_DATA_DIR / "memory_db"))


def test_default_jsonl_paths_isolated():
    import memory.feedback as feedback
    import modules.skills as skills
    from modules import browser_agent

    assert not is_inside_real_data(feedback.DEFAULT_PATH)
    assert not is_inside_real_data(skills.DEFAULT_PATH)
    assert not is_inside_real_data(browser_agent.profile_dir("sofia"))
    assert not is_inside_real_data(browser_agent.screens_dir())
