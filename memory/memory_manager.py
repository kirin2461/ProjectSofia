"""Управление памятью: эпизодическая + библиотека."""

import chromadb
from config.settings import MEMORY_DB_PATH, VECTOR_DB_PATH

# Эпизодическая память
episodic_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
episodes = episodic_client.get_or_create_collection("episodes")

# Библиотека (книги)
library_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
books = library_client.get_or_create_collection("library")


def add_episode(text: str, metadata: dict):
    """Добавить эпизод диалога."""
    import hashlib
    doc_id = hashlib.md5(text.encode()).hexdigest()
    episodes.add(ids=[doc_id], documents=[text], metadatas=[metadata])


def recall_episodes(query: str, n_results: int = 3):
    """Поиск по эпизодической памяти."""
    return episodes.query(query_texts=[query], n_results=n_results)


def recall_library(query: str, n_results: int = 3):
    """Поиск по библиотеке."""
    return books.query(query_texts=[query], n_results=n_results)
