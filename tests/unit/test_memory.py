"""Юнит-тесты памяти (Phase 3-02). Fallback-стор или Chroma, без Ollama."""

import time

import pytest

import memory.memory_manager as mm


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Свежий fallback-стор во временной папке."""
    ep = mm._FallbackCollection(str(tmp_path / "ep"), "episodes")
    lib = mm._FallbackCollection(str(tmp_path / "lib"), "library")
    monkeypatch.setattr(mm, "episodes", ep)
    monkeypatch.setattr(mm, "books", lib)
    return ep, lib


def test_add_and_count(fresh):
    mm.add_episode("Привет, меня зовут Аня", {"session_id": "s1"})
    assert mm.count_episodes() == 1


def test_recall_finds_related(fresh):
    mm.add_episode("Я люблю программировать на Python", {"session_id": "s1"})
    mm.add_episode("Погода сегодня солнечная", {"session_id": "s1"})
    res = mm.recall_episodes("Python программирование", n_results=1)
    docs = res["documents"][0]
    assert docs and "Python" in docs[0]


def test_metadata_includes_emotion(fresh):
    mm.add_episode("тест", {"P": 0.5, "A": 0.1, "D": 0.0, "emotion": "радость"})
    got = mm.episodes.get()
    assert got["metadatas"][0]["emotion"] == "радость"


def test_list_tags_sanitized(fresh):
    mm.add_episode("тест", {"tags": ["dialog", "greeting"]})
    meta = mm.episodes.get()["metadatas"][0]
    assert isinstance(meta["tags"], str)
    assert "dialog" in meta["tags"]


def test_time_window_filter(fresh):
    mm.add_episode("старое", {"timestamp": time.time() - 100000})
    mm.add_episode("свежее сообщение", {"timestamp": time.time()})
    res = mm.recall_episodes("сообщение", n_results=5, time_window=3600)
    docs = res["documents"][0]
    assert all("старое" not in d for d in docs)


def test_summarize_old_marks(fresh):
    mm.add_episode("давнее", {"timestamp": time.time() - 10 * 86400})
    mm.add_episode("недавнее", {"timestamp": time.time()})
    n = mm.summarize_old(threshold_days=7)
    assert n == 1
    metas = mm.episodes.get()["metadatas"]
    summarized = [m for m in metas if m.get("summarized")]
    assert len(summarized) == 1


def test_hash_embed_deterministic():
    a = mm._hash_embed("привет мир")
    b = mm._hash_embed("привет мир")
    assert a == b
    assert len(a) == mm._EMBED_DIM
