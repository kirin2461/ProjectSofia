"""Многоязычные эмбеддинги: размерность, нормировка, ранжирование по-русски."""

import math

import pytest

pytest.importorskip("onnxruntime")

from memory import embedder as emb  # noqa: E402

_MODEL = emb.get_embedder()
pytestmark = pytest.mark.skipif(_MODEL is None, reason="нет файлов модели эмбеддингов")


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_dimension():
    assert len(_MODEL.embed_query("второй закон Ньютона")) == emb.EMBED_DIM


def test_vectors_normalized():
    vec = _MODEL.embed_query("энтропия замкнутой системы")
    assert abs(math.sqrt(sum(v * v for v in vec)) - 1.0) < 1e-4


def test_passage_and_query_same_space():
    """Префиксы query:/passage: не должны разносить векторы по разным областям."""
    q = _MODEL.embed_query("Что такое энтропия?")
    docs = _MODEL.embed_documents(["Энтропия — мера беспорядка системы."])
    assert cosine(q, docs[0]) > 0.7


def test_russian_relevance_beats_unrelated():
    docs = [
        "Ускорение тела прямо пропорционально равнодействующей силе.",
        "Молярный объём идеального газа при нормальных условиях 22,4 литра.",
    ]
    vecs = _MODEL.embed_documents(docs)
    q = _MODEL.embed_query("как связаны сила и ускорение?")
    assert cosine(q, vecs[0]) > cosine(q, vecs[1])


def test_empty_input():
    assert _MODEL.embed_documents([]) == []


def test_batch_equals_single():
    """Квантованная модель не бит-в-бит при разном размере батча, но вектор тот же."""
    docs = [
        "Энтропия замкнутой системы не убывает при обратимых процессах.",
        "Второй закон Ньютона связывает силу, массу и ускорение.",
    ]
    batch = _MODEL.embed_documents(docs)
    single = _MODEL.embed_documents([docs[0]])[0]
    assert cosine(batch[0], single) > 0.99
    assert cosine(batch[0], single) > cosine(batch[0], batch[1])
