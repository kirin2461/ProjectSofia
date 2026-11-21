"""Phase-3-01: Watcher и индексатор библиотеки (PDF → Chroma).

- Наблюдает за data/library/ (watchdog)
- Умный чанкинг: по outline → по размеру шрифта → фикс. чанки с overlap
- on_created / on_modified → (пере)индексация, on_deleted → архивация

Фоновый поток, не блокирует основной. Битые PDF — try/except с логом.
"""

import hashlib
import logging
import os
import threading
import time
from typing import List, Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # watchdog не установлен — watcher недоступен
    FileSystemEventHandler = object  # type: ignore
    Observer = None  # type: ignore

from config.settings import LIBRARY_PATH, LIBRARY_CHUNK_SIZE, LIBRARY_CHUNK_OVERLAP
from memory.memory_manager import books

logger = logging.getLogger(__name__)

try:
    import pymupdf as fitz  # pymupdf ≥ 1.24 (старый алиас fitz устарел)
except ImportError:  # pragma: no cover
    try:
        import fitz  # старые сборки pymupdf
    except ImportError:
        fitz = None


# ── Чанкинг ─────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = LIBRARY_CHUNK_SIZE,
    overlap: int = LIBRARY_CHUNK_OVERLAP,
) -> List[str]:
    """Фиксированные чанки с перекрытием (fallback)."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    # Перекрытие не может быть >= размера чанка, иначе шаг станет 1
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def _chunks_from_outline(doc) -> Optional[List[dict]]:
    """Чанки по оглавлению (toc), если есть."""
    toc = doc.get_toc()  # [[level, title, page], ...]
    if not toc:
        return None

    result = []
    for i, (_level, title, page) in enumerate(toc):
        start_page = max(0, page - 1)
        end_page = (toc[i + 1][2] - 1) if i + 1 < len(toc) else doc.page_count
        end_page = min(doc.page_count, max(end_page, start_page + 1))
        text = "\n".join(
            doc[p].get_text() for p in range(start_page, end_page)
        ).strip()
        if not text:
            continue
        for sub in chunk_text(text):
            result.append({
                "text": sub,
                "chapter": title,
                "page_start": start_page + 1,
                "page_end": end_page,
            })
    return result or None


def _chunks_from_headings(doc) -> Optional[List[dict]]:
    """Чанки по заголовкам (размер шрифта > 14pt и жирный)."""
    sections = []
    current = {"title": "Начало", "page_start": 1, "text": []}
    try:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if not txt:
                            continue
                        is_heading = (
                            span.get("size", 0) > 14
                            and bool(span.get("flags", 0) & 16)  # bold
                        )
                        if is_heading and len("".join(current["text"])) > 200:
                            sections.append(current)
                            current = {
                                "title": txt,
                                "page_start": page_num + 1,
                                "text": [],
                            }
                        else:
                            current["text"].append(txt + " ")
            current["page_end"] = page_num + 1
        sections.append(current)
    except Exception as e:  # noqa: BLE001
        logger.debug("heading-chunking failed: %s", e)
        return None

    result = []
    for sec in sections:
        full = "".join(sec["text"]).strip()
        if not full:
            continue
        for sub in chunk_text(full):
            result.append({
                "text": sub,
                "chapter": sec["title"],
                "page_start": sec["page_start"],
                "page_end": sec.get("page_end", sec["page_start"]),
            })
    return result if len(result) > 1 else None


def smart_chunk_pdf(doc) -> List[dict]:
    """Умный чанкинг: outline → заголовки → фикс. чанки."""
    result = _chunks_from_outline(doc)
    if result:
        logger.info("Чанкинг по оглавлению: %d чанков", len(result))
        return result

    result = _chunks_from_headings(doc)
    if result:
        logger.info("Чанкинг по заголовкам: %d чанков", len(result))
        return result

    full = "\n".join(doc[p].get_text() for p in range(doc.page_count))
    chunks = [
        {"text": c, "chapter": f"Часть {i + 1}", "page_start": 1,
         "page_end": doc.page_count}
        for i, c in enumerate(chunk_text(full))
    ]
    logger.info("Чанкинг фиксированный: %d чанков", len(chunks))
    return chunks


# ── Индексация ──────────────────────────────────────────────
def _file_hash(path: str) -> str:
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()


def _remove_file_chunks(path: str):
    """Удалить все чанки файла из коллекции."""
    try:
        books.delete(where={"file_hash": _file_hash(path)})
    except Exception as e:  # noqa: BLE001
        logger.debug("remove chunks: %s", e)


def ingest_file(path: str):
    """Проиндексировать PDF-файл."""
    if not path.lower().endswith(".pdf"):
        return  # TODO: epub, txt, docx
    if fitz is None:
        logger.error("pymupdf не установлен — индексация недоступна")
        return

    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        logger.error("Битый PDF %s: %s", path, e)
        return

    meta = doc.metadata or {}
    title = meta.get("title") or os.path.basename(path)
    author = meta.get("author") or "неизвестен"
    source = os.path.basename(path)
    fhash = _file_hash(path)

    # Переиндексация: сначала убрать старые чанки
    _remove_file_chunks(path)

    chunks = smart_chunk_pdf(doc)
    if not chunks:
        logger.warning("Нет текста в %s", source)
        return

    ids, docs, metas = [], [], []
    for i, ch in enumerate(chunks):
        ids.append(f"{fhash}_{i}")
        docs.append(ch["text"])
        metas.append({
            "source": source,
            "title": title,
            "author": author,
            "chapter": ch["chapter"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "file_hash": fhash,
            "chunk": i,
            "archived": False,
            "indexed_at": time.time(),
        })

    # Пакетная вставка
    books.add(ids=ids, documents=docs, metadatas=metas)
    logger.info("Проиндексирован %s: %d чанков", source, len(chunks))


def archive_file(path: str):
    """Пометить чанки файла archived=True (не удалять физически)."""
    fhash = _file_hash(path)
    try:
        existing = books.get(where={"file_hash": fhash})
    except Exception as e:  # noqa: BLE001
        logger.debug("archive get: %s", e)
        return
    ids = existing.get("ids", [])
    metas = existing.get("metadatas", [])
    docs = existing.get("documents", [])
    for i, doc_id in enumerate(ids):
        meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
        meta["archived"] = True
        books.update(
            ids=[doc_id],
            metadatas=[meta],
            documents=[docs[i]] if i < len(docs) else None,
        )
    if ids:
        logger.info("Архивирован %s: %d чанков", os.path.basename(path), len(ids))


# ── Watcher ─────────────────────────────────────────────────
class LibraryHandler(FileSystemEventHandler):
    def __init__(self):
        self._debounce = {}

    def _should_process(self, path: str) -> bool:
        """Защита от дублей событий (debounce 1 сек)."""
        now = time.time()
        last = self._debounce.get(path, 0)
        if now - last < 1.0:
            return False
        self._debounce[path] = now
        return True

    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            threading.Thread(
                target=ingest_file, args=(event.src_path,), daemon=True
            ).start()

    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            threading.Thread(
                target=ingest_file, args=(event.src_path,), daemon=True
            ).start()

    def on_deleted(self, event):
        if not event.is_directory:
            archive_file(event.src_path)


def start_watcher(library_path: str = LIBRARY_PATH):
    """Запустить watcher на папку с книгами (фоновый поток)."""
    os.makedirs(library_path, exist_ok=True)
    if Observer is None:
        logger.warning("watchdog не установлен — watcher библиотеки отключён")
        return None
    observer = Observer()
    observer.schedule(LibraryHandler(), library_path, recursive=False)
    observer.start()
    logger.info("Library watcher запущен: %s", library_path)
    return observer
