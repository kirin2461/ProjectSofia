"""Phase-3-01: Watcher и индексатор библиотеки (PDF → Chroma).

- Наблюдает за data/library/ (watchdog) + досматривает папку при старте
- Умный чанкинг: по outline → по размеру шрифта → фикс. чанки с overlap
- Чистка: переносы, колонтитулы, номера страниц (иначе мусор в эмбеддингах)
- OCR страниц без текстового слоя (Tesseract, если доступен)
- on_created / on_modified → (пере)индексация, on_deleted → архивация

Фоновый поток, не блокирует основной. Битые PDF — try/except с логом.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:  # watchdog не установлен — watcher недоступен
    FileSystemEventHandler = object  # type: ignore
    Observer = None  # type: ignore

from config.settings import LIBRARY_PATH, LIBRARY_CHUNK_SIZE, LIBRARY_CHUNK_OVERLAP
from memory import memory_manager
from memory.memory_manager import books

logger = logging.getLogger(__name__)

try:
    import pymupdf as fitz  # pymupdf ≥ 1.24 (старый алиас fitz устарел)
except ImportError:  # pragma: no cover
    try:
        import fitz  # старые сборки pymupdf
    except ImportError:
        fitz = None

OCR_MIN_CHARS = 24      # меньше символов на странице — считаем сканом
OCR_DPI = 300
OCR_LANG = "rus+eng"
BATCH = 64              # сколько чанков пишем в Chroma за раз

_ocr_warned = False


# ── Чистка текста ───────────────────────────────────────────
_HYPHEN_RE = re.compile(r"([а-яёa-z])[-\u00ad]\s*\n\s*([а-яёa-z])", re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r"^\s*[-–—|.]?\s*\d{1,4}\s*[-–—|.]?\s*$")
_SPACES_RE = re.compile(r"[ \t\u00a0]{2,}")
# Типографские пробелы: в книгах их много, для модели это мусор
_WEIRD_SPACES_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")


def dehyphenate(text: str) -> str:
    """Склеивает перенесённые слова: «физи-\\nка» → «физика»."""
    return _HYPHEN_RE.sub(r"\1\2", text or "")


def clean_pages(pages: List[str], min_repeat_ratio: float = 0.3) -> List[str]:
    """Убирает номера страниц, колонтитулы и лишние пробелы.

    Колонтитул — короткая строка, повторяющаяся на заметной части страниц
    (название книги сверху, «Глава 4» снизу). В эмбеддингах такие строки
    дают ложные совпадения: все чанки книги начинают «быть похожи».
    """
    if not pages:
        return []

    pages = [_WEIRD_SPACES_RE.sub(" ", p or "") for p in pages]
    pages = [dehyphenate(p) for p in pages]
    counts: Counter = Counter()
    for p in pages:
        for line in {ln.strip() for ln in p.splitlines() if ln.strip()}:
            counts[line] += 1

    limit = max(3, int(len(pages) * min_repeat_ratio))
    repeated = {ln for ln, c in counts.items() if c >= limit and len(ln) <= 80}

    cleaned = []
    for p in pages:
        kept = []
        for line in p.splitlines():
            s = line.strip()
            if s and (s in repeated or _PAGE_NUM_RE.match(s)):
                continue
            kept.append(_SPACES_RE.sub(" ", line).rstrip())
        text = "\n".join(kept).strip()
        cleaned.append(re.sub(r"\n{3,}", "\n\n", text))
    return cleaned


# ── OCR ─────────────────────────────────────────────────────
def find_tesseract() -> Optional[str]:
    """Ищет Tesseract: env → внутрь сборки (bin/tesseract) → рядом с exe → PATH."""
    env = os.environ.get("TESSERACT_CMD")
    if env and Path(env).exists():
        return env
    from app.paths import bundled_dir, install_dir
    for root in (bundled_dir(), install_dir()):
        for cand in (
            root / "bin" / "tesseract" / "tesseract.exe",
            root / "bin" / "tesseract.exe",
            root / "tesseract" / "tesseract.exe",
        ):
            if cand.exists():
                return str(cand)
    return shutil.which("tesseract")


def ocr_page(page, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    """Распознаёт страницу-скан. Пустая строка, если Tesseract недоступен."""
    global _ocr_warned
    exe = find_tesseract()
    if not exe:
        if not _ocr_warned:
            logger.warning(
                "OCR недоступен: не найден Tesseract. Страницы-сканы останутся "
                "без текста. Положите сборку в bin/tesseract или задайте TESSERACT_CMD."
            )
            _ocr_warned = True
        return ""
    if not hasattr(page, "get_pixmap"):
        return ""

    try:
        png = page.get_pixmap(dpi=dpi).tobytes("png")
    except Exception as e:  # noqa: BLE001
        logger.debug("OCR render failed: %s", e)
        return ""

    env = os.environ.copy()
    tessdata = Path(exe).parent / "tessdata"
    if tessdata.exists():
        env["TESSDATA_PREFIX"] = str(tessdata)

    try:
        proc = subprocess.run(
            [exe, "stdin", "stdout", "-l", lang, "--psm", "3"],
            input=png,
            capture_output=True,
            timeout=180,
            env=env,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("OCR вызов не удался: %s", e)
        return ""

    if proc.returncode != 0:
        logger.debug(
            "OCR код %s: %s", proc.returncode,
            proc.stderr.decode("utf-8", "replace")[:200],
        )
        return ""
    return proc.stdout.decode("utf-8", "replace")


def read_pages(doc, ocr: bool = True) -> List[str]:
    """Текст всех страниц (с OCR для сканов) + чистка."""
    pages = []
    for i in range(doc.page_count):
        try:
            page = doc[i]
            text = page.get_text() or ""
        except Exception as e:  # noqa: BLE001
            logger.debug("страница %d: %s", i, e)
            text = ""
        if ocr and len(text.strip()) < OCR_MIN_CHARS:
            recognized = ocr_page(page)
            if recognized.strip():
                text = recognized
        pages.append(text)
    return clean_pages(pages)


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


def _chunks_from_outline(doc, pages: Optional[List[str]] = None) -> Optional[List[dict]]:
    """Чанки по оглавлению (toc), если есть."""
    toc = doc.get_toc()  # [[level, title, page], ...]
    if not toc:
        return None
    if pages is None:
        pages = read_pages(doc, ocr=False)

    result = []
    for i, (_level, title, page) in enumerate(toc):
        start_page = max(0, int(page) - 1)
        end_page = (int(toc[i + 1][2]) - 1) if i + 1 < len(toc) else len(pages)
        end_page = min(len(pages), max(end_page, start_page + 1))
        text = "\n".join(pages[start_page:end_page]).strip()
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


def _heading_marks(doc) -> List[tuple]:
    """Позиции заголовков по шрифту: [(индекс страницы, текст)]."""
    marks = []
    try:
        for p in range(doc.page_count):
            blocks = doc[p].get_text("dict").get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = (span.get("text") or "").strip()
                        if not txt or len(txt) > 90:
                            continue
                        if span.get("size", 0) > 14 and bool(span.get("flags", 0) & 16):
                            marks.append((p, txt))
    except Exception as e:  # noqa: BLE001
        logger.debug("heading scan failed: %s", e)
        return []
    return marks


def _chunks_from_headings(doc, pages: List[str]) -> Optional[List[dict]]:
    """Чанки по заголовкам (размер шрифта > 14pt и жирный)."""
    marks = _heading_marks(doc)
    if not marks:
        return None

    # Раздел начинается с заголовка и тянется до следующего
    sections = [("Начало", 0)]
    for page_idx, title in marks:
        if page_idx > sections[-1][1]:
            sections.append((title, page_idx))

    result = []
    for i, (title, start) in enumerate(sections):
        end = sections[i + 1][1] if i + 1 < len(sections) else len(pages)
        text = "\n".join(pages[start:end]).strip()
        if not text:
            continue
        for sub in chunk_text(text):
            result.append({
                "text": sub,
                "chapter": title,
                "page_start": start + 1,
                "page_end": max(start + 1, end),
            })
    return result if len(result) > 1 else None


def _page_chunks(pages: List[str], doc=None) -> List[dict]:
    """Fallback: чанки собираются постранично, чтобы цитата указывала страницу.

    Раньше весь текст книги склеивался в одну строку, и у всех чанков были
    page_start=1, page_end=последняя — ссылка «с. 1–500» бесполезна.
    """
    result: List[dict] = []
    buf: List[str] = []
    buf_pages: List[int] = []

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            for sub in chunk_text(text):
                result.append({
                    "text": sub,
                    "chapter": "",
                    "page_start": buf_pages[0],
                    "page_end": buf_pages[-1],
                })
        buf.clear()
        buf_pages.clear()

    for idx, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        if buf and sum(len(p) for p in buf) + len(page) > LIBRARY_CHUNK_SIZE:
            flush()
        buf.append(page)
        buf_pages.append(idx)
    flush()
    return result


def _assign_sections(chunks: List[dict]) -> None:
    """Помечает чанки, относящиеся к одной секции (главе).

    Нужно для расширения найденного абзаца до целой секции: ссылка «Сивухин,
    с. 112» остаётся, но модель получает связный кусок, а не обрывок мысли.
    Секция — непрерывный участок с одинаковым названием главы; без названия
    (постраничный чанкинг) остаётся пустой section_id, и поиск расширяет
    фрагмент соседними чанками.
    """
    run_start, run_title = 0, None
    total = len(chunks)

    def flush(end: int, title) -> None:
        if not title:
            return
        base = re.sub(r"\s+", " ", str(title)).strip().lower()[:80]
        for k in range(run_start, end):
            chunks[k]["section_id"] = f"{base}#{run_start}"

    for i, ch in enumerate(chunks):
        title = (ch.get("chapter") or "").strip()
        if title != (run_title or ""):
            flush(i, run_title)
            run_start, run_title = i, title
        ch.setdefault("section_id", "")
    flush(total, run_title)


def smart_chunk_pdf(doc, ocr: bool = True) -> List[dict]:
    """Умный чанкинг: outline → заголовки → фикс. чанки."""
    pages = read_pages(doc, ocr=ocr)
    if not any(p.strip() for p in pages):
        return []

    result = _chunks_from_outline(doc, pages)
    if result:
        logger.info("Чанкинг по оглавлению: %d чанков", len(result))
        return result

    result = _chunks_from_headings(doc, pages)
    if result:
        logger.info("Чанкинг по заголовкам: %d чанков", len(result))
        return result

    chunks = _page_chunks(pages)
    logger.info("Чанкинг постраничный: %d чанков", len(chunks))
    return chunks


# ── Индексация ──────────────────────────────────────────────
def _file_hash(path: str) -> str:
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()


def _file_stamp(path: str) -> str:
    """Отпечаток версии файла: по нему пропускаем неизменённые книги."""
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except OSError:
        return ""


def _remove_file_chunks(path: str):
    """Удалить все чанки файла из коллекции."""
    try:
        books.delete(where={"file_hash": _file_hash(path)})
    except Exception as e:  # noqa: BLE001
        logger.debug("remove chunks: %s", e)


def ingest_file(path: str, force: bool = False) -> bool:
    """Проиндексировать PDF. True — если файл действительно обработан."""
    if not path.lower().endswith(".pdf"):
        return False  # TODO: epub, txt, docx
    if fitz is None:
        logger.error("pymupdf не установлен — индексация недоступна")
        return False

    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        logger.error("Битый PDF %s: %s", path, e)
        return False

    source = os.path.basename(path)
    fhash = _file_hash(path)
    stamp = _file_stamp(path)

    if not force:
        try:
            same = books.get(where={"file_hash": fhash})
            metas = same.get("metadatas") or []
            if metas and all(m and m.get("stamp") == stamp for m in metas):
                logger.debug("Без изменений, пропуск: %s", source)
                return False
        except Exception:  # noqa: BLE001
            pass

    meta = doc.metadata or {}
    title = meta.get("title") or source
    author = meta.get("author") or "неизвестен"

    # Переиндексация: сначала убрать старые чанки
    _remove_file_chunks(path)

    chunks = smart_chunk_pdf(doc)
    if not chunks:
        logger.warning("Нет текста в %s", source)
        return False

    _assign_sections(chunks)

    try:
        for start in range(0, len(chunks), BATCH):
            part = chunks[start:start + BATCH]
            ids, docs, metas = [], [], []
            for j, ch in enumerate(part):
                idx = start + j
                ids.append(f"{fhash}_{idx}")
                docs.append(ch["text"])
                metas.append({
                    "source": source,
                    "title": title,
                    "author": author,
                    "chapter": ch["chapter"],
                    "section_id": ch.get("section_id", ""),
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "file_hash": fhash,
                    "chunk": idx,
                    "archived": False,
                    "stamp": stamp,
                    "indexed_at": time.time(),
                })
            memory_manager.add_book_chunks(ids, docs, metas)
    except Exception as e:  # noqa: BLE001
        # Нет векторов / Chroma недоступна — файл не проиндексирован.
        # Молчать нельзя: иначе watcher будет «успешно» пропускать книги.
        logger.error("Индексация %s прервана: %s", source, e)
        return False

    logger.info("Проиндексирован %s: %d чанков", source, len(chunks))
    return True


def _chunk_pages(col, page_size: int = 512, max_chunks: int = 200_000):
    """Страницы чанков без эмбеддингов: их не читаем и не трогаем."""
    offset, paginated = 0, True
    while offset < max_chunks:
        try:
            page = col.get(limit=page_size, offset=offset,
                           include=["documents", "metadatas"])
        except TypeError:  # fallback-стор не умеет страницы
            paginated = False
            page = col.get()
        ids = page.get("ids") or []
        if not ids:
            return
        yield page
        if not paginated or len(ids) < page_size:
            return
        offset += len(ids)


def reconcile_library(library_path: str = LIBRARY_PATH) -> dict:
    """Сверить индекс с папкой и заархивировать чанки исчезнувших файлов.

    on_deleted ловит удаление только при живом watcher: книга, удалённая при
    выключенном сервере, иначе остаётся активной навсегда и продолжает
    выдаваться в ответах. Здесь индекс приводится к состоянию папки.

    Эмбеддинги и тексты не удаляются — только archived=True, данные могут
    понадобиться, если книгу вернут. Если папки нет вовсе (не подключён
    диск), сверка пропускается: иначе одна неудачная проверка заархивировала
    бы всю библиотеку.
    """
    summary = {
        "archived": 0,
        "missing": 0,
        "missing_files": [],
        "indexed_files": 0,
        "chunks": 0,
        "skipped": "",
    }
    folder = Path(library_path)
    if not folder.exists():
        summary["skipped"] = f"папка не найдена: {folder}"
        logger.warning("Библиотека: сверка индекса пропущена — %s", summary["skipped"])
        return summary

    present = {p.name.lower() for p in folder.iterdir() if p.is_file()}

    # source → активные (не archived) чанки этого файла
    groups: dict = {}
    try:
        for page in _chunk_pages(books):
            ids = page.get("ids") or []
            metas = page.get("metadatas") or []
            summary["chunks"] += len(ids)
            for i, doc_id in enumerate(ids):
                meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
                source = str(meta.get("source") or "")
                group = groups.setdefault(source, [])
                if source and not meta.get("archived"):
                    group.append((doc_id, meta))
    except Exception as e:  # noqa: BLE001
        logger.error("Библиотека: сверка индекса не удалась: %s", e)
        summary["skipped"] = f"ошибка чтения индекса: {e}"
        return summary

    summary["indexed_files"] = len([s for s in groups if s])

    for source in sorted(groups):
        active = groups[source]
        if not source or source.lower() in present:
            continue
        summary["missing"] += 1
        summary["missing_files"].append(source)
        for start in range(0, len(active), 256):
            part = active[start:start + 256]
            meta_batch = [{**meta, "archived": True} for _, meta in part]
            try:
                books.update(ids=[doc_id for doc_id, _ in part],
                             metadatas=meta_batch)
            except Exception as e:  # noqa: BLE001
                logger.error("Библиотека: не удалось архивировать %s: %s", source, e)
                break
            summary["archived"] += len(part)

    if summary["missing"]:
        logger.warning(
            "Библиотека: %d книг нет на диске (%s) — %d чанков помечены archived",
            summary["missing"], ", ".join(summary["missing_files"][:5]),
            summary["archived"],
        )
    else:
        logger.info("Библиотека: сверка индекса — всё на месте (%d файлов, %d чанков)",
                    summary["indexed_files"], summary["chunks"])
    return summary


def scan_library(library_path: str = LIBRARY_PATH) -> int:
    """Сверяет индекс с папкой и индексирует новые PDF. Число новых книг."""
    folder = Path(library_path)
    if not folder.exists():
        return 0

    try:
        reconcile_library(str(folder))
    except Exception as e:  # noqa: BLE001
        logger.error("Библиотека: сверка индекса сорвалась: %s", e)

    if not memory_manager.embeddings_ready():
        logger.error(
            "Библиотека: индексация отключена — векторная память недоступна (%s)",
            memory_manager.EMBED_REASON or "нет эмбеддингов",
        )
        return 0

    processed = 0
    for pdf in sorted(folder.glob("*.pdf")):
        try:
            if ingest_file(str(pdf)):
                processed += 1
        except Exception as e:  # noqa: BLE001
            logger.error("Ошибка индексации %s: %s", pdf.name, e)
    if processed:
        logger.info("Библиотека: обработано книг — %d", processed)
    return processed


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
