"""Юнит-тесты чанкинга библиотеки (Phase 3-01). Без pymupdf/watchdog."""

import pytest

from memory.library_ingest import (
    chunk_text,
    _chunks_from_outline,
    smart_chunk_pdf,
)


class FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self, mode="text"):
        if mode == "dict":
            return {"blocks": []}
        return self._text


class FakeDoc:
    def __init__(self, text, toc=None, pages=None):
        self._text = text
        self._toc = toc or []
        self._pages = pages or [FakePage(text)]
        self.page_count = len(self._pages)

    def get_toc(self):
        return self._toc

    def __getitem__(self, i):
        return self._pages[i]


def test_chunk_text_overlap():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert all(len(c) <= 400 for c in chunks)
    # Перекрытие: конец первого == начало второго (50 символов)
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_no_infinite_loop_when_overlap_equals_size():
    chunks = chunk_text("abcdef", chunk_size=100, overlap=100)
    assert chunks == ["abcdef"]


def test_chunk_covers_whole_text():
    text = "x" * 5000
    chunks = chunk_text(text, chunk_size=800, overlap=80)
    assert "".join(chunks)[:5000] == text


def test_outline_chunking_numeric_toc_entries():
    # Реальный pymupdf: [level, title, page]; уровень — число
    doc = FakeDoc("", toc=[[1, "Глава 1", 1], [1, "Глава 2", 2]],
                  pages=[FakePage("текст первой главы " * 100),
                         FakePage("текст второй главы " * 100)])
    chunks = _chunks_from_outline(doc)
    assert chunks is not None
    chapters = {c["chapter"] for c in chunks}
    assert "Глава 1" in chapters and "Глава 2" in chapters
    assert all(c["page_start"] >= 1 for c in chunks)


def test_smart_chunk_fallback_fixed():
    doc = FakeDoc("Слово " * 500)
    chunks = smart_chunk_pdf(doc)
    assert chunks
    # Главы в книге нет — и выдумывать «Часть 1» нельзя: цитата должна
    # читаться как «Сивухин, с. 45», а не «Сивухин, Часть 1, с. 45».
    assert chunks[0]["chapter"] == ""
    assert chunks[0]["page_end"] == doc.page_count


def test_fallback_pages_are_real_not_one_to_last():
    pages = [FakePage("Страница " * 60) for _ in range(12)]
    doc = FakeDoc("", pages=pages)
    chunks = smart_chunk_pdf(doc)
    assert len(chunks) > 1
    # каждый чанк привязан к своему диапазону, а не ко всей книге
    assert all(c["page_start"] >= 1 and c["page_end"] <= 12 for c in chunks)
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] < 12
    assert [c["page_start"] for c in chunks] == sorted(c["page_start"] for c in chunks)
    assert all(c["page_start"] <= c["page_end"] for c in chunks)


def test_metadata_present():
    doc = FakeDoc("Текст " * 400)
    for ch in smart_chunk_pdf(doc):
        for key in ("text", "chapter", "page_start", "page_end"):
            assert key in ch
