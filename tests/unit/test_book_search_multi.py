"""Мультизапросный поиск, расширение до секции и метка секций при индексации."""

from memory import book_search as bs
from memory import library_ingest as li


def hit(text="текст про энтропию", source="a.pdf", page=10, score=0.8,
        sim=0.85, section="", chunk=-1):
    return bs.BookHit(text=text, source=source, title="Сивухин",
                      chapter="Глава 1", page_start=page, page_end=page,
                      score=score, similarity=sim,
                      section_id=section, chunk=chunk)


# ── слияние результатов нескольких запросов ─────────────────
def test_merge_dedupes_same_paragraph():
    a = [hit(score=0.7)]
    b = [hit(score=0.9, sim=0.95)]        # тот же абзац, другая оценка
    merged = bs.merge_candidates([a, b], "энтропия")
    assert len(merged) == 1
    assert merged[0].score >= 0.7


def test_merge_keeps_different_paragraphs():
    a = [hit(text="энтропия растёт", page=10)]
    b = [hit(text="тепло передаётся", page=20)]
    assert len(bs.merge_candidates([a, b], "энтропия")) == 2


def test_merge_reranks_by_terms():
    a = [hit(text="случайный текст про погоду", page=1, score=0.9, sim=0.9)]
    b = [hit(text="энтропия замкнутой системы не убывает", page=2, score=0.5, sim=0.6)]
    merged = bs.merge_candidates([a, b], "энтропия")
    assert "энтропия" in merged[0].text


def test_merge_empty():
    assert bs.merge_candidates([], "запрос") == []
    assert bs.merge_candidates([[], []], "запрос") == []


# ── расширение до секции ─────────────────────────────────────
def test_expand_uses_section_texts():
    h = hit(text="обрывок мысли", section="энтропия#0")
    out = bs.expand_sections(
        [h],
        fetch_section=lambda sid: ["начало секции", "обрывок мысли", "конец секции"],
    )
    assert out[0].matched_text == "обрывок мысли"
    assert out[0].text.startswith("начало секции")
    assert out[0].text.endswith("конец секции")


def test_expand_falls_back_to_neighbors_without_section():
    h = hit(text="середина", chunk=3)
    out = bs.expand_sections(
        [h],
        fetch_section=None,
        fetch_neighbors=lambda src, idx: ["чанк 2", "середина", "чанк 4"],
    )
    assert out[0].matched_text == "середина"
    assert "чанк 2" in out[0].text and "чанк 4" in out[0].text


def test_expand_respects_budget():
    h = hit(text="маленький фрагмент", section="s#0")
    big = ["я" * 900, "фрагмент", "я" * 900, "я" * 900]
    out = bs.expand_sections([h], fetch_section=lambda sid: big, max_chars=1000)
    assert len(out[0].text) <= 1100


def test_expand_noop_when_fetcher_missing():
    h = hit(text="как было")
    out = bs.expand_sections([h], fetch_section=None, fetch_neighbors=None)
    assert out[0].text == "как было" and out[0].matched_text == ""


def test_expand_survives_fetcher_error():
    def boom(_):
        raise RuntimeError("нет связи")

    h = hit(text="остаётся", section="s#0")
    out = bs.expand_sections([h], fetch_section=boom)
    assert out[0].text == "остаётся"


def test_expand_does_not_shrink_fragment():
    h = hit(text="длинный исходный фрагмент " * 10, section="s#0")
    out = bs.expand_sections([h], fetch_section=lambda sid: ["коротко"])
    assert out[0].text.startswith("длинный исходный")


def test_double_expansion_is_idempotent():
    h = hit(text="фрагмент", section="s#0")
    fetch = lambda sid: ["а", "фрагмент", "б"]
    bs.expand_sections([h], fetch_section=fetch)
    first = h.text
    bs.expand_sections([h], fetch_section=fetch)
    assert h.text == first


# ── чтение секций из коллекции ───────────────────────────────
class FakeCollection:
    def __init__(self, rows):
        self.rows = rows          # [(id, doc, meta)]
        self.calls = []

    def get(self, where=None, **kwargs):
        self.calls.append(where)
        docs, metas = [], []
        for _id, doc, meta in self.rows:
            if all(meta.get(k) == v for k, v in (where or {}).items()):
                docs.append(doc)
                metas.append(meta)
        return {"documents": docs, "metadatas": metas}


def test_section_texts_ordered_by_page():
    coll = FakeCollection([
        ("1", "вторая", {"section_id": "s#0", "page_start": 20, "chunk": 5}),
        ("2", "первая", {"section_id": "s#0", "page_start": 10, "chunk": 4}),
        ("3", "чужая", {"section_id": "other", "page_start": 30, "chunk": 6}),
    ])
    assert bs.section_texts(coll, "s#0") == ["первая", "вторая"]


def test_neighbor_texts_around_chunk():
    coll = FakeCollection([
        ("1", "до", {"source": "a.pdf", "chunk": 2}),
        ("2", "сам", {"source": "a.pdf", "chunk": 3}),
        ("3", "после", {"source": "a.pdf", "chunk": 4}),
        ("4", "далеко", {"source": "a.pdf", "chunk": 99}),
        ("5", "другая книга", {"source": "b.pdf", "chunk": 3}),
    ])
    assert bs.neighbor_texts(coll, "a.pdf", 3) == ["до", "сам", "после"]


def test_section_texts_tolerates_collection_without_where():
    class NoWhere:
        def get(self):
            return {"documents": ["a", "b"], "metadatas": [{}, {}]}

    assert bs.section_texts(NoWhere(), "s#0") == []


# ── метки секций при индексации ──────────────────────────────
def test_sections_grouped_by_chapter():
    chunks = [
        {"chapter": "Глава 1", "text": "a"},
        {"chapter": "Глава 1", "text": "b"},
        {"chapter": "Глава 2", "text": "c"},
    ]
    li._assign_sections(chunks)
    assert chunks[0]["section_id"] == chunks[1]["section_id"] != chunks[2]["section_id"]
    assert chunks[0]["section_id"].startswith("глава 1#")


def test_no_chapter_leaves_section_empty():
    chunks = [{"chapter": "", "text": "a"}, {"chapter": "", "text": "b"}]
    li._assign_sections(chunks)
    assert all(c["section_id"] == "" for c in chunks)


def test_sections_are_stable_for_same_chapter_name():
    a = [{"chapter": "Глава 1", "text": "x"}]
    b = [{"chapter": "Глава 1", "text": "y"}]
    li._assign_sections(a)
    li._assign_sections(b)
    assert a[0]["section_id"] == b[0]["section_id"]


# ── сборка запросов в search_books ───────────────────────────
def test_search_books_tries_all_queries(monkeypatch):
    from memory import memory_manager

    asked = []

    def fake_query(q, n_results=24):
        asked.append(q)
        return {"documents": [[f"энтропия: {q}"]],
                "metadatas": [[{"source": "a.pdf", "title": "Книга",
                                "chapter": "Гл. 1", "page_start": 5, "page_end": 5,
                                "section_id": "", "chunk": 0}]],
                "distances": [[0.1]]}

    monkeypatch.setattr(memory_manager, "count_books", lambda: 1)
    monkeypatch.setattr(memory_manager, "query_books", fake_query)
    monkeypatch.setattr(memory_manager, "books", None, raising=False)

    hits = bs.search_books("что такое энтропия", queries=["энтропия"], expand=False)
    assert len(asked) == 2 and hits
