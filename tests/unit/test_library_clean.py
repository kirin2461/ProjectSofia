"""Чистка текста книг и OCR-фолбэк в индексаторе библиотеки."""

from pathlib import Path

import pytest

from memory import library_ingest as li


class FakePage:
    def __init__(self, text, pixmap=None):
        self._text = text
        self._pixmap = pixmap

    def get_text(self, mode="text"):
        if mode == "dict":
            return {"blocks": []}
        return self._text

    def get_pixmap(self, dpi=300):
        if self._pixmap is None:
            raise RuntimeError("нет растра")
        return self._pixmap


class FakePixmap:
    def tobytes(self, fmt="png"):
        return b"\x89PNG-fake"


class FakeDoc:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def __getitem__(self, i):
        return self._pages[i]


def test_dehyphenate():
    assert li.dehyphenate("физи-\nка") == "физика"
    assert li.dehyphenate("физи\u00ad\nка") == "физика"
    assert li.dehyphenate("обычный текст") == "обычный текст"


def test_clean_pages_drops_running_headers():
    pages = [
        "Сивухин, том 1\nВторой закон Ньютона\nнаходится в главе 2",
        "Сивухин, том 1\nЗакон сохранения импульса\nсохраняется",
        "Сивухин, том 1\nЭнтропия\nне убывает",
        "Сивухин, том 1\nТензор энергии-импульса\nописывает поле",
    ]
    cleaned = li.clean_pages(pages)
    assert all("Сивухин" not in p for p in cleaned)
    assert "Второй закон Ньютона" in cleaned[0]


def test_clean_pages_drops_page_numbers():
    pages = ["Второй закон\n- 42 -", "Третий закон\n43", "Четвёртый\n14"]
    cleaned = li.clean_pages(pages)
    assert all("42" not in p and "43" not in p for p in cleaned)
    assert "Второй закон" in cleaned[0]


def test_clean_pages_keeps_numbers_inside_text():
    pages = ["давление достигает 42 кПа при 300 К", "ещё страница текста"]
    cleaned = li.clean_pages(pages)
    assert "42" in cleaned[0]


def test_clean_pages_collapses_spaces():
    cleaned = li.clean_pages(["слово    слово\t\tслово"])
    assert "слово слово слово" in cleaned[0]


def test_page_chunks_keep_real_page_numbers():
    pages = ["Страница один. " * 40, "Страница два. " * 40, "Страница три. " * 40]
    chunks = li._page_chunks(pages)
    assert chunks
    assert all(ch["text"].strip() for ch in chunks)
    # каждая страница не больше чанка — значит каждая страница отдельным чанком
    assert [ch["page_start"] for ch in chunks] == [1, 2, 3]
    assert all(ch["page_start"] == ch["page_end"] for ch in chunks)
    assert "один" not in chunks[1]["text"]


def test_page_chunks_join_small_pages_with_range():
    pages = ["а" * 200, "б" * 200, "в" * 200]
    chunks = li._page_chunks(pages)
    assert len(chunks) == 1
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 3


def test_page_chunks_split_long_page():
    pages = ["слово " * 900]
    chunks = li._page_chunks(pages)
    assert len(chunks) > 1
    assert all(ch["page_start"] == 1 and ch["page_end"] == 1 for ch in chunks)


def test_page_chunks_skip_blank_pages():
    chunks = li._page_chunks(["   ", "", "текст страницы"])
    assert len(chunks) == 1
    assert chunks[0]["page_start"] == 3


def test_page_chunks_no_chapter_placeholder():
    """Пустая глава лучше, чем ложная «Часть 1» в цитате."""
    chunks = li._page_chunks(["немного текста"])
    assert chunks[0]["chapter"] == ""


def test_clean_pages_normalizes_typographic_spaces():
    cleaned = li.clean_pages(["Второй\u00a0закон\u2009Ньютона"])
    assert cleaned[0] == "Второй закон Ньютона"
    assert "\u00a0" not in cleaned[0]


def test_clean_pages_empty():
    assert li.clean_pages([]) == []


def test_read_pages_uses_ocr_for_scans(monkeypatch):
    monkeypatch.setattr(li, "ocr_page", lambda page, dpi=0, lang="": "распознанный текст страницы")
    doc = FakeDoc([FakePage("")])

    pages = li.read_pages(doc, ocr=True)

    assert "распознанный" in pages[0]


def test_read_pages_skips_ocr_when_text_exists(monkeypatch):
    calls = []
    monkeypatch.setattr(li, "ocr_page", lambda page, **kw: calls.append(1) or "")
    doc = FakeDoc([FakePage("нормальный текстовый слой страницы, достаточно длинный")])

    li.read_pages(doc, ocr=True)

    assert calls == []


def test_ocr_page_without_tesseract_returns_empty(monkeypatch):
    monkeypatch.setattr(li, "find_tesseract", lambda: None)
    monkeypatch.setattr(li, "_ocr_warned", False)
    page = FakePage("", pixmap=FakePixmap())

    assert li.ocr_page(page) == ""


def test_find_tesseract_prefers_bundled_build(tmp_path, monkeypatch):
    """В exe Tesseract лежит внутри сборки (_MEIPASS/bin/tesseract)."""
    from app import paths

    bundled = tmp_path / "_internal"
    (bundled / "bin" / "tesseract").mkdir(parents=True)
    exe = bundled / "bin" / "tesseract" / "tesseract.exe"
    exe.write_bytes(b"MZ")

    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(paths, "bundled_dir", lambda: bundled)
    monkeypatch.setattr(paths, "install_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(li.shutil, "which", lambda name: r"C:\Windows\tesseract.exe")

    assert li.find_tesseract() == str(exe)


def test_find_tesseract_env_wins(tmp_path, monkeypatch):
    custom = tmp_path / "custom" / "tesseract.exe"
    custom.parent.mkdir()
    custom.write_bytes(b"MZ")
    monkeypatch.setenv("TESSERACT_CMD", str(custom))

    assert li.find_tesseract() == str(custom)


def test_find_tesseract_none_anywhere(monkeypatch):
    from app import paths

    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(paths, "bundled_dir", lambda: Path(r"C:\nowhere\a"))
    monkeypatch.setattr(paths, "install_dir", lambda: Path(r"C:\nowhere\b"))
    monkeypatch.setattr(li.shutil, "which", lambda name: None)

    assert li.find_tesseract() is None


def test_ocr_page_runs_tesseract(monkeypatch):
    monkeypatch.setattr(li, "find_tesseract", lambda: r"C:\fake\tesseract.exe")

    class Result:
        returncode = 0
        stdout = b"raspoznanno"
        stderr = b""

    monkeypatch.setattr(li.subprocess, "run", lambda *a, **k: Result())
    page = FakePage("", pixmap=FakePixmap())

    assert li.ocr_page(page) == "raspoznanno"


def test_ocr_page_reports_failure(monkeypatch):
    monkeypatch.setattr(li, "find_tesseract", lambda: r"C:\fake\tesseract.exe")

    class Result:
        returncode = 1
        stdout = b""
        stderr = b"no tessdata"

    monkeypatch.setattr(li.subprocess, "run", lambda *a, **k: Result())
    page = FakePage("", pixmap=FakePixmap())

    assert li.ocr_page(page) == ""


def make_pdf(path, text, font=None):
    """Создать PDF с текстом. Кириллица требует шрифта с её глифами."""
    fitz = pytest.importorskip("pymupdf")
    doc = fitz.open()
    page = doc.new_page()
    if font:
        page.insert_font(fontname="body", fontfile=font)
        page.insert_text((72, 72), text, fontname="body", fontsize=10)
    else:
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


CYR_FONT = "C:/Windows/Fonts/arial.ttf"
needs_cyr = pytest.mark.skipif(
    not Path(CYR_FONT).exists(), reason="нет шрифта с кириллицей"
)


def test_ingest_skips_unchanged_file(tmp_path, monkeypatch):
    path = tmp_path / "book.pdf"
    make_pdf(path, "Newton second law: a = F/m. " * 20)

    stamp = li._file_stamp(str(path))

    class FakeCollection:
        def get(self, where=None):
            return {"ids": ["1"], "metadatas": [{"stamp": stamp}], "documents": ["x"]}

    monkeypatch.setattr(li, "books", FakeCollection())
    added = []
    monkeypatch.setattr(li.memory_manager, "add_book_chunks",
                        lambda *a: added.append(a))

    assert li.ingest_file(str(path)) is False
    assert added == []


@needs_cyr
def test_ingest_pdf_indexes_chunks(tmp_path, monkeypatch):
    path = tmp_path / "book.pdf"
    make_pdf(
        path,
        "Второй закон Ньютона: ускорение равно силе, делённой на массу. " * 30,
        CYR_FONT,
    )

    class FakeCollection:
        def get(self, where=None):
            return {"ids": [], "metadatas": [], "documents": []}

        def delete(self, where=None, ids=None):
            return None

    monkeypatch.setattr(li, "books", FakeCollection())
    added = []
    monkeypatch.setattr(li.memory_manager, "add_book_chunks",
                        lambda ids, docs, metas: added.append((ids, docs, metas)))

    assert li.ingest_file(str(path)) is True
    assert added, "чанки не записаны"
    ids, docs, metas = added[0]
    assert len(ids) == len(docs) == len(metas)
    assert metas[0]["source"] == "book.pdf"
    assert "Ньютон" in docs[0]
    assert metas[0]["stamp"] == li._file_stamp(str(path))
