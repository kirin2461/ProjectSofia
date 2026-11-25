"""Поиск по книгам: переранжирование, пороги, цитаты."""

import pytest

from memory import book_search as bs


def hit(text, sim=0.8, page=10, chapter="Механика", title="Сивухин", span=1):
    return bs.BookHit(
        text=text, source="s.pdf", title=title, chapter=chapter,
        page_start=page, page_end=page + span, score=0.0, similarity=sim,
    )


def scored(h, value):
    h.score = value
    return h


def test_tokenize_drops_stopwords():
    tokens = bs.tokenize("Что такое второй закон Ньютона?")
    assert "что" not in tokens
    assert "такое" not in tokens
    assert any(t.startswith("ньютон") for t in tokens)


def test_tokenize_merges_word_forms():
    assert bs.tokenize("импульса") == bs.tokenize("импульс")
    assert bs.tokenize("уравнения") == bs.tokenize("уравнение")


def test_tokenize_empty():
    assert bs.tokenize("") == []


def test_rerank_prefers_term_overlap():
    relevant = hit("Энтропия замкнутой системы не убывает", sim=0.80)
    similar = hit("Молярный объём газа при нормальных условиях", sim=0.82)
    ranked = bs.rerank("энтропия", [similar, relevant])
    assert ranked[0].text.startswith("Энтропия")


def test_rerank_keeps_vector_order_without_terms():
    a, b = hit("первый", sim=0.9), hit("второй", sim=0.7)
    ranked = bs.rerank("и в на", [b, a])  # одни служебные слова
    assert ranked[0].similarity == 0.9


def test_select_drops_irrelevant_below_threshold():
    hits = [scored(hit("a", 0.80), 0.80), scored(hit("b", 0.40), 0.40)]
    picked = bs.select(hits, top_k=4, min_score=0.52, gap=0.5)
    assert [h.text for h in picked] == ["a"]


def test_select_drops_far_from_best():
    hits = [
        scored(hit("a", 0.86), 0.86),
        scored(hit("b", 0.60), 0.60),
        scored(hit("c", 0.55), 0.55),
    ]
    picked = bs.select(hits, top_k=4, min_score=0.52, gap=0.22)
    assert [h.text for h in picked] == ["a"]


def test_select_keeps_close_group():
    hits = [
        scored(hit("a", 0.86), 0.86),
        scored(hit("b", 0.82), 0.82),
        scored(hit("c", 0.70), 0.70),
    ]
    picked = bs.select(hits, top_k=4, min_score=0.52, gap=0.22)
    assert [h.text for h in picked] == ["a", "b", "c"]


def test_select_respects_top_k():
    hits = [scored(hit(str(i), 0.9), 0.9) for i in range(6)]
    assert len(bs.select(hits, top_k=2, min_score=0.1, gap=1.0)) == 2


def test_select_empty():
    assert bs.select([]) == []


def test_hits_from_chroma_maps_metadata():
    raw = {
        "documents": [["Второй закон Ньютона"]],
        "metadatas": [[{
            "source": "sivuhin1.pdf", "title": "Сивухин", "chapter": "Механика",
            "page_start": 45, "page_end": 46,
        }]],
        "distances": [[0.16]],
    }
    hits = bs.hits_from_chroma(raw, "второй закон")
    assert hits[0].citation() == "Сивухин, Механика, с. 45–46"
    assert abs(hits[0].similarity - 0.84) < 1e-6


def test_citation_without_chapter_duplicates():
    h = hit("текст", page=12, chapter="Сивухин", title="Сивухин", span=0)
    assert h.citation() == "Сивухин, с. 12"


def test_format_sources_numbered():
    text = bs.format_sources([hit("первый фрагмент", page=10, span=0),
                              hit("второй фрагмент", page=20, span=0)])
    assert text.startswith("[1] (Сивухин, Механика, с. 10)")
    assert "[2]" in text and "второй фрагмент" in text


def test_format_sources_empty():
    assert bs.format_sources([]) == ""


def test_search_books_without_index_returns_empty(monkeypatch):
    from memory import memory_manager

    monkeypatch.setattr(memory_manager, "count_books", lambda: 0)
    assert bs.search_books("энтропия") == []


def test_search_books_pipeline(monkeypatch):
    from memory import memory_manager

    monkeypatch.setattr(memory_manager, "count_books", lambda: 5)
    monkeypatch.setattr(
        memory_manager, "query_books",
        lambda q, n_results=3: {
            "documents": [["Энтропия замкнутой системы не убывает"]],
            "metadatas": [[{"source": "s.pdf", "title": "Сивухин",
                            "chapter": "Термодинамика", "page_start": 112}]],
            "distances": [[0.12]],
        },
    )
    hits = bs.search_books("почему энтропия не убывает")
    assert len(hits) == 1
    assert hits[0].page_start == 112


def test_search_books_survives_backend_error(monkeypatch):
    from memory import memory_manager

    def boom(*args, **kwargs):
        raise RuntimeError("chroma упала")

    monkeypatch.setattr(memory_manager, "count_books", lambda: 3)
    monkeypatch.setattr(memory_manager, "query_books", boom)
    assert bs.search_books("энтропия") == []
