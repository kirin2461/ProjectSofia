"""Индексация книг из data/library/."""

import os
import hashlib
import fitz  # pymupdf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from memory.memory_manager import books


class LibraryHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            ingest_file(event.src_path)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 80):
    """Разбиение текста на чанки."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def ingest_file(path: str):
    """Проиндексировать файл."""
    if not path.endswith(".pdf"):
        return  # TODO: epub, txt, docx

    text = parse_pdf(path)
    chunks = chunk_text(text)
    file_hash = hashlib.md5(path.encode()).hexdigest()

    for i, chunk in enumerate(chunks):
        doc_id = f"{file_hash}_{i}"
        books.add(
            ids=[doc_id],
            documents=[chunk],
            metadatas=[{"source": os.path.basename(path), "chunk": i}],
        )


def start_watcher(library_path: str = "data/library"):
    """Запустить watcher на папку с книгами."""
    os.makedirs(library_path, exist_ok=True)
    observer = Observer()
    observer.schedule(LibraryHandler(), library_path, recursive=False)
    observer.start()
    return observer
