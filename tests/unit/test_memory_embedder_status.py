"""Сбой e5 не должен молча подменять память пустой коллекцией __default.

Проверяем три обещания:
- падение модели — ERROR с причиной, а не WARNING «и поехали дальше»;
- тег коллекций остаётся e5s (эпизоды и книги не «исчезают»);
- разовый сбой не закрепляется до перезапуска: попытка повторяется.
"""

import logging

import pytest

from memory import embedder as emb
from memory import memory_manager as mm


class FakeEmbedder:
    tag = "e5s"
    model_path = "fake/model.onnx"
    model_name = "multilingual-e5-small"

    def name(self):
        return "local-e5s"

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def broken(monkeypatch):
    """Модель «не загружается», кэш эмбеддера чист."""
    def boom(*a, **k):
        raise FileNotFoundError("нет model_quantized.onnx")

    monkeypatch.setattr(emb, "LocalEmbedder", boom)
    monkeypatch.setattr(emb, "RETRY_SEC", 60.0)
    emb.reset_embedder()
    yield boom
    emb.reset_embedder()


@pytest.fixture
def mm_state():
    """Сохранить состояние бэкенда памяти после теста (другие тесты им живут)."""
    saved = {
        "_embedder": mm._embedder,
        "EMBED_TAG": mm.EMBED_TAG,
        "EMBED_MODE": mm.EMBED_MODE,
        "EMBED_REASON": mm.EMBED_REASON,
        "MEMORY_DEGRADED": mm.MEMORY_DEGRADED,
    }
    yield
    for key, value in saved.items():
        setattr(mm, key, value)


@pytest.fixture
def no_fallback(monkeypatch):
    monkeypatch.delenv("SOFIA_EMBED_FALLBACK", raising=False)


# ── эмбеддер ────────────────────────────────────────────────
def test_failure_is_error_with_reason(broken, caplog):
    with caplog.at_level(logging.ERROR, logger="memory.embedder"):
        assert emb.get_embedder() is None

    errors = " ".join(r.getMessage() for r in caplog.records
                      if r.levelno >= logging.ERROR)
    assert "векторная память недоступна" in errors
    assert "поиск по эпизодам отключён" in errors
    assert "model_quantized.onnx" in errors, "причина должна быть в тексте"
    assert "model_quantized.onnx" in emb.embedder_error()


def test_failure_is_retried_not_frozen(broken, monkeypatch):
    monkeypatch.setattr(emb, "RETRY_SEC", 0.0)
    assert emb.get_embedder() is None
    first = emb.embedder_attempts()
    assert emb.get_embedder() is None
    assert emb.embedder_attempts() == first + 1


def test_cooldown_does_not_hammer_the_model(broken):
    assert emb.get_embedder() is None
    attempts = emb.embedder_attempts()
    for _ in range(3):
        assert emb.get_embedder() is None
    assert emb.embedder_attempts() == attempts


def test_force_retry_recovers(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("сбой загрузки")

    monkeypatch.setattr(emb, "LocalEmbedder", boom)
    emb.reset_embedder()
    try:
        assert emb.get_embedder() is None
        assert emb.embedder_error()
        monkeypatch.setattr(emb, "LocalEmbedder", lambda *a, **k: FakeEmbedder())
        model = emb.get_embedder(force=True)
        assert isinstance(model, FakeEmbedder)
        assert emb.embedder_error() == ""
    finally:
        emb.reset_embedder()


# ── бэкенд памяти ───────────────────────────────────────────
def test_degraded_memory_keeps_e5s_collection(mm_state, no_fallback, monkeypatch, caplog):
    monkeypatch.setattr(mm, "get_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mm, "embedder_error", lambda: "нет файлов модели")

    with caplog.at_level(logging.ERROR):
        mode = mm.init_memory_backend()

    assert mode == "unavailable"
    assert mm.EMBED_TAG == "e5s", "нельзя переключаться на __default"
    assert mm._collection_name("episodes") == "episodes__e5s"
    assert mm._collection_name("library") == "library__e5s"

    status = mm.memory_status()
    assert status["degraded"] is True
    assert status["fallback"] is False
    assert status["search"] == "off"
    assert "нет файлов модели" in status["reason"]
    assert any("векторная память недоступна" in r.getMessage() for r in caplog.records)


def test_explicit_fallback_is_marked_in_state(mm_state, monkeypatch, caplog):
    monkeypatch.setenv("SOFIA_EMBED_FALLBACK", "ollama")
    monkeypatch.setattr(mm, "get_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mm, "embedder_error", lambda: "нет модели")

    class Alt:
        def name(self):
            return "ollama-bge-m3"

        def __call__(self, input):  # noqa: A002
            return [[0.0] * 8 for _ in input]

    monkeypatch.setattr(mm, "_make_embedder", lambda: Alt())

    with caplog.at_level(logging.WARNING):
        mode = mm.init_memory_backend()

    assert mode == "ollama"
    status = mm.memory_status()
    assert status["fallback"] is True and status["degraded"] is True
    assert status["tag"] == "default"
    assert any("РЕЗЕРВ" in r.getMessage() for r in caplog.records)


def test_unavailable_backend_does_not_read(mm_state, no_fallback, monkeypatch):
    monkeypatch.setattr(mm, "get_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mm, "embedder_error", lambda: "нет модели")
    mm.init_memory_backend()

    res = mm.recall_episodes("что-нибудь")
    assert res["documents"] == [[]]
    assert res["ids"] == [[]]


def test_unavailable_backend_does_not_write(mm_state, no_fallback, monkeypatch):
    monkeypatch.setattr(mm, "get_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mm, "embedder_error", lambda: "нет модели")
    mm.init_memory_backend()

    written = []

    class Recorder:
        def add(self, **kwargs):
            written.append(kwargs)

        def update(self, **kwargs):
            written.append(kwargs)

    monkeypatch.setattr(mm, "episodes", Recorder())

    assert mm.embeddings_ready() is False
    mm.add_episode("эпизод, который некуда векторизовать")
    assert written == []
    with pytest.raises(RuntimeError):
        mm.add_book_chunks(["id"], ["текст"], [{"source": "book.pdf"}])


def test_recovery_switches_back_to_e5_without_restart(mm_state, no_fallback, monkeypatch):
    monkeypatch.setattr(mm, "get_embedder", lambda *a, **k: None)
    monkeypatch.setattr(mm, "embedder_error", lambda: "нет модели")
    mm.init_memory_backend()
    assert mm.EMBED_MODE == "unavailable"

    # модель «починилась»: следующий поиск обязан это заметить сам
    monkeypatch.setattr(mm, "get_embedder", lambda *a, **k: FakeEmbedder())
    assert mm.embeddings_ready() is True
    assert mm.EMBED_MODE == "e5"
    assert mm.EMBED_TAG == "e5s"
    assert mm.memory_status()["degraded"] is False
