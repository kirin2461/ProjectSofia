"""Сверка индекса книг с папкой: книга, удалённая при выключенном сервере.

on_deleted видит только удаление вживую. Без сверки чанки такой книги остаются
archived=False и продолжают выдаваться в ответах. Эмбеддинги при архивации
не удаляются — только флаг.
"""

import logging

import pytest

from memory import library_ingest as li
from memory import memory_manager as mm


@pytest.fixture
def col(tmp_path, monkeypatch):
    """Изолированный стор книг вместо боевой Chroma."""
    c = mm._FallbackCollection(str(tmp_path / "vec"), "library")
    monkeypatch.setattr(li, "books", c)
    monkeypatch.setattr(mm, "books", c)
    return c


def add_chunks(col, folder, name, n=2, archived=False):
    fhash = li._file_hash(str(folder / name))
    ids = [f"{fhash}_{i}" for i in range(n)]
    docs = [f"{name}: фрагмент номер {i}" for i in range(n)]
    metas = [{
        "source": name,
        "file_hash": fhash,
        "chunk": i,
        "archived": archived,
        "page_start": 1,
        "page_end": 1,
    } for i in range(n)]
    col.add(ids=ids, documents=docs, metadatas=metas)
    return ids


def test_missing_book_is_archived(tmp_path, col):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "present.pdf").write_bytes(b"%PDF-1.4")
    add_chunks(col, lib, "present.pdf")
    add_chunks(col, lib, "gone.pdf")

    summary = li.reconcile_library(str(lib))

    assert summary["archived"] == 2
    assert summary["missing"] == 1
    assert summary["missing_files"] == ["gone.pdf"]

    metas = col.get()["metadatas"]
    for meta in metas:
        assert meta["archived"] is (meta["source"] == "gone.pdf")


def test_nothing_is_deleted(tmp_path, col):
    lib = tmp_path / "library"
    lib.mkdir()
    add_chunks(col, lib, "gone.pdf", n=3)
    before = list(col._data["embeddings"])

    li.reconcile_library(str(lib))

    assert col.count() == 3, "чанки должны остаться"
    assert col._data["embeddings"] == before, "эмбеддинги не удаляем"


def test_reconcile_is_idempotent(tmp_path, col):
    lib = tmp_path / "library"
    lib.mkdir()
    add_chunks(col, lib, "gone.pdf")

    first = li.reconcile_library(str(lib))
    second = li.reconcile_library(str(lib))

    assert first["archived"] == 2
    assert second["archived"] == 0
    assert second["missing"] == 1


def test_already_archived_is_not_counted_again(tmp_path, col):
    lib = tmp_path / "library"
    lib.mkdir()
    add_chunks(col, lib, "old.pdf", archived=True)

    summary = li.reconcile_library(str(lib))

    assert summary["archived"] == 0
    assert summary["missing"] == 1
    assert summary["missing_files"] == ["old.pdf"]


def test_missing_folder_does_not_archive_the_whole_library(tmp_path, col):
    """Отключённый внешний диск — не повод считать все книги удалёнными."""
    lib = tmp_path / "library"
    add_chunks(col, lib, "book.pdf")

    summary = li.reconcile_library(str(lib))

    assert summary["archived"] == 0
    assert summary["skipped"]
    assert col.get()["metadatas"][0]["archived"] is False


def test_empty_index(tmp_path, col):
    lib = tmp_path / "library"
    lib.mkdir()

    summary = li.reconcile_library(str(lib))

    assert summary["archived"] == 0
    assert summary["missing"] == 0
    assert summary["chunks"] == 0


def test_scan_library_reconciles_before_indexing(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "new.pdf").write_bytes(b"%PDF")
    calls = []

    def fake_reconcile(path=li.LIBRARY_PATH):
        calls.append(str(path))
        return {"archived": 0, "missing": 0, "missing_files": []}

    monkeypatch.setattr(li, "reconcile_library", fake_reconcile)
    monkeypatch.setattr(li, "ingest_file", lambda path: False)

    assert li.scan_library(str(lib)) == 0
    assert calls == [str(lib)]


def test_scan_library_survives_reconcile_failure(tmp_path, monkeypatch, caplog):
    lib = tmp_path / "library"
    lib.mkdir()
    indexed = []

    def boom(path=li.LIBRARY_PATH):
        raise RuntimeError("Chroma недоступна")

    monkeypatch.setattr(li, "reconcile_library", boom)
    monkeypatch.setattr(li, "ingest_file", lambda path: indexed.append(path) or False)

    with caplog.at_level(logging.ERROR):
        assert li.scan_library(str(lib)) == 0
    assert any("сверк" in r.getMessage().lower() for r in caplog.records)


def test_ingest_reports_write_failure(tmp_path, monkeypatch, caplog):
    """Нет эмбеддингов — индексация не должна выглядеть успешной."""
    fitz = pytest.importorskip("pymupdf")

    lib = tmp_path / "library"
    lib.mkdir()
    path = lib / "book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Newton second law: a = F / m. " * 20)
    doc.save(str(path))
    doc.close()

    def refuse(ids, docs, metas):
        raise RuntimeError("эмбеддинги недоступны")

    monkeypatch.setattr(li.memory_manager, "add_book_chunks", refuse)

    with caplog.at_level(logging.ERROR):
        assert li.ingest_file(str(path)) is False

    messages = " ".join(r.getMessage().lower() for r in caplog.records)
    assert "эмбеддинг" in messages or "индекс" in messages
