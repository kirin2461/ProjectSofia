"""Phase-3-02: Управление памятью — эпизодическая + библиотека (Chroma).

Эмбеддинги через bge-m3 (Ollama). Если Ollama недоступна (например, на VDS),
Chroma использует встроенную модель по умолчанию. Если недоступен и Chroma —
используется лёгкий JSON-стор с hash-эмбеддингами (dev/test).

Публичный API:
    add_episode(text, metadata) -> id
    recall_episodes(query, n_results, time_window)
    recall_library(query, n_results)
    summarize_old(threshold_days) -> int
    count_episodes() -> int
"""

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import (
    MEMORY_DB_PATH,
    VECTOR_DB_PATH,
    MODEL_EMBED,
    OLLAMA_HOST,
    EPISODES_PER_RECALL,
)

logger = logging.getLogger(__name__)

_EMBED_DIM = 256


def _hash_embed(text: str) -> List[float]:
    """Детерминированный hash-эмбеддинг (char trigram), offline-fallback."""
    vec = [0.0] * _EMBED_DIM
    low = text.lower()
    grams = [low[i:i + 3] for i in range(max(1, len(low) - 2))]
    for gram in grams:
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        vec[h % _EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class OllamaEmbeddingFunction:
    """Chroma-совместимая embedding-функция через Ollama (bge-m3)."""

    def __init__(self, model: str = MODEL_EMBED, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def name(self) -> str:
        return f"ollama-{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        import requests

        vectors = []
        for text in input:
            try:
                r = requests.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60,
                )
                r.raise_for_status()
                vectors.append(r.json()["embedding"])
            except Exception:  # noqa: BLE001
                vectors.append(_hash_embed(text))
        return vectors


def _make_embedder():
    """Пытается использовать bge-m3; при ошибке — None (Chroma default)."""
    try:
        import requests

        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        logger.info("Память: эмбеддинги через %s", MODEL_EMBED)
        return OllamaEmbeddingFunction()
    except Exception:  # noqa: BLE001
        logger.info("Ollama недоступна — hash/дефолтные эмбеддинги (dev/тесты).")
        return None


# ═══════════════════════════════════════════════════════════
# Fallback-стор (если chromadb не установлен)
# ═══════════════════════════════════════════════════════════
class _FallbackCollection:
    """Минимальная замена Chroma collection: JSON + cosine similarity."""

    def __init__(self, path: str, name: str):
        self._path = Path(path) / f"{name}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    def _save(self):
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False), encoding="utf-8"
        )

    def add(self, ids, documents, metadatas=None, embeddings=None):
        for i, doc_id in enumerate(ids):
            if doc_id in self._data["ids"]:
                idx = self._data["ids"].index(doc_id)
                self.update([doc_id],
                            metadatas=[metadatas[i]] if metadatas else None,
                            documents=[documents[i]])
                continue
            self._data["ids"].append(doc_id)
            self._data["documents"].append(documents[i])
            self._data["metadatas"].append((metadatas or [{}] * len(ids))[i] or {})
            vec = (embeddings[i] if embeddings else _hash_embed(documents[i]))
            self._data["embeddings"].append(vec)
        self._save()

    def update(self, ids, metadatas=None, documents=None, **kwargs):
        for i, doc_id in enumerate(ids):
            if doc_id not in self._data["ids"]:
                continue
            idx = self._data["ids"].index(doc_id)
            if metadatas and metadatas[i] is not None:
                self._data["metadatas"][idx] = metadatas[i]
            if documents and documents[i] is not None:
                self._data["documents"][idx] = documents[i]
                self._data["embeddings"][idx] = _hash_embed(documents[i])
        self._save()

    def delete(self, where=None, ids=None):
        if ids:
            keep = [i for i, x in enumerate(self._data["ids"]) if x not in ids]
        else:
            keep = [i for i in range(len(self._data["ids"]))
                    if not _match(self._data["metadatas"][i], where)]
        for key in ("ids", "documents", "metadatas", "embeddings"):
            self._data[key] = [self._data[key][i] for i in keep]
        self._save()

    def get(self, where=None):
        idx = [i for i in range(len(self._data["ids"]))
               if _match(self._data["metadatas"][i], where)]
        return {
            "ids": [self._data["ids"][i] for i in idx],
            "documents": [self._data["documents"][i] for i in idx],
            "metadatas": [self._data["metadatas"][i] for i in idx],
        }

    def count(self) -> int:
        return len(self._data["ids"])

    def query(self, query_texts=None, n_results=3, where=None,
              query_embeddings=None, **kwargs):
        q = (query_texts[0] if query_texts else "") or ""
        qv = _hash_embed(q)
        scored = []
        for i in range(len(self._data["ids"])):
            if not _match(self._data["metadatas"][i], where):
                continue
            ev = self._data["embeddings"][i]
            score = sum(a * b for a, b in zip(qv, ev))
            scored.append((score, i))
        scored.sort(reverse=True)
        top = [i for _, i in scored[:n_results]]
        return {
            "ids": [[self._data["ids"][i] for i in top]],
            "documents": [[self._data["documents"][i] for i in top]],
            "metadatas": [[self._data["metadatas"][i] for i in top]],
            "distances": [[1 - s for s, _ in scored[:n_results]]],
        }


def _match(meta: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    for key, cond in where.items():
        val = (meta or {}).get(key)
        if isinstance(cond, dict):
            if "$gte" in cond and not (val is not None and val >= cond["$gte"]):
                return False
            if "$lt" in cond and not (val is not None and val < cond["$lt"]):
                return False
            if "$eq" in cond and val != cond["$eq"]:
                return False
        elif val != cond:
            return False
    return True


def _sanitize_metadata(metadata: dict) -> dict:
    """Chroma принимает только str/int/float/bool; списки → строка."""
    clean = {}
    for k, v in (metadata or {}).items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = ", ".join(str(x) for x in v)
        elif v is None:
            continue
        else:
            clean[k] = str(v)
    return clean


# ═══════════════════════════════════════════════════════════
# Инициализация коллекций
# ═══════════════════════════════════════════════════════════
_hash_embedder = _make_embedder()

try:  # локальные многоязычные векторы (основной путь, работает offline)
    from memory.embedder import get_embedder

    _embedder = get_embedder()
except Exception:  # noqa: BLE001
    _embedder = None

# Если локальной модели нет — остаётся старый путь (Ollama bge-m3 / Chroma default)
_embedder = _embedder or _hash_embedder
EMBED_TAG = getattr(_embedder, "tag", "default")


def _collection_name(base: str) -> str:
    """Имя коллекции зависит от модели эмбеддингов: разные векторы не смешиваем."""
    return f"{base}__{EMBED_TAG}"


def _embed_passages(texts: List[str]) -> Optional[List[List[float]]]:
    if _embedder is None or not hasattr(_embedder, "embed_documents"):
        return None
    return _embedder.embed_documents(texts)


def _embed_query(text: str) -> Optional[List[float]]:
    if _embedder is None or not hasattr(_embedder, "embed_query"):
        return None
    return _embedder.embed_query(text)


try:
    import chromadb

    _collection_kwargs = {"embedding_function": _embedder} if _embedder else {}
    _local_kwargs = (
        {"embedding_function": None, "metadata": {"hnsw:space": "cosine"}}
        if hasattr(_embedder, "tag")
        else _collection_kwargs
    )
    episodic_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
    library_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    episodes = episodic_client.get_or_create_collection(
        _collection_name("episodes"), **_local_kwargs
    )
    books = library_client.get_or_create_collection(
        _collection_name("library"), **_local_kwargs
    )
    BACKEND = "chroma"
except ImportError:
    logger.warning("chromadb не установлен — JSON fallback-стор (dev/тесты)")
    episodic_client = None
    library_client = None
    episodes = _FallbackCollection(MEMORY_DB_PATH, "episodes")
    books = _FallbackCollection(VECTOR_DB_PATH, "library")
    BACKEND = "fallback"


# ═══════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════
def _episode_id(text: str, timestamp: float) -> str:
    raw = f"{text}{timestamp}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def add_episode(text: str, metadata: Optional[dict] = None) -> str:
    """Добавить эпизод диалога. Возвращает его id."""
    metadata = dict(metadata or {})
    ts = metadata.get("timestamp", time.time())
    metadata["timestamp"] = ts
    doc_id = _episode_id(text, ts)
    extra = {}
    vecs = _embed_passages([text]) if BACKEND == "chroma" else None
    if vecs:
        extra["embeddings"] = vecs
    episodes.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[_sanitize_metadata(metadata)],
        **extra,
    )
    return doc_id


def _is_fallback(col) -> bool:
    return isinstance(col, _FallbackCollection)


def _query(col, query: str, n_results: int, where: Optional[dict] = None):
    """Поиск: векторный, если коллекция это умеет, иначе текстовый (dev/тесты)."""
    if BACKEND == "chroma" and not _is_fallback(col):
        qv = _embed_query(query)
        if qv:
            return col.query(query_embeddings=[qv], n_results=n_results, where=where)
    return col.query(query_texts=[query], n_results=n_results, where=where)


def recall_episodes(
    query: str,
    n_results: int = EPISODES_PER_RECALL,
    time_window: Optional[float] = None,
):
    """Семантический поиск по эпизодам. time_window — в секундах."""
    where = None
    if time_window is not None:
        cutoff = time.time() - time_window
        where = {"timestamp": {"$gte": cutoff}}

    try:
        return _query(episodes, query, n_results, where)
    except Exception as e:  # noqa: BLE001
        logger.warning("recall_episodes failed: %s", e)
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}


def query_books(query: str, n_results: int = 3, where: Optional[dict] = None):
    """Кандидаты из библиотеки для последующего переранжирования."""
    try:
        return _query(books, query, n_results, where or {"archived": False})
    except Exception as e:  # noqa: BLE001
        logger.warning("query_books failed: %s", e)
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}


def recall_library(query: str, n_results: int = 3):
    """Семантический поиск по библиотеке (не archived)."""
    return query_books(query, n_results=n_results)


def count_books() -> int:
    try:
        return books.count()
    except Exception:  # noqa: BLE001
        return 0


def add_book_chunks(ids, documents, metadatas) -> None:
    """Записать чанки книги (векторы считаем сами — они многоязычные)."""
    extra = {}
    vecs = _embed_passages(list(documents)) if BACKEND == "chroma" else None
    if vecs:
        extra["embeddings"] = vecs
    books.add(
        ids=list(ids),
        documents=list(documents),
        metadatas=[_sanitize_metadata(m) for m in metadatas],
        **extra,
    )


def migrate_legacy(limit: int = 2000) -> int:
    """Перенести старые эпизоды под текущую модель эмбеддингов.

    До появления многоязычных векторов память писалась эмбеддингами
    all-MiniLM-L6-v2 (англоязычная модель) — на русском поиск по ним почти
    случаен. Сами реплики целы, поэтому их достаточно пересчитать.
    """
    if BACKEND != "chroma" or not hasattr(_embedder, "tag"):
        return 0
    try:
        if episodes.count() > 0:
            return 0
        legacy = episodic_client.get_or_create_collection("episodes")
        total = legacy.count()
    except Exception as e:  # noqa: BLE001
        logger.debug("migrate_legacy: %s", e)
        return 0
    if not total:
        return 0

    moved, offset = 0, 0
    while offset < min(total, limit):
        try:
            page = legacy.get(limit=256, offset=offset)
        except Exception as e:  # noqa: BLE001
            logger.warning("migrate_legacy get: %s", e)
            break
        ids = page.get("ids") or []
        docs = page.get("documents") or []
        metas = page.get("metadatas") or []
        if not ids:
            break
        try:
            episodes.add(
                ids=ids,
                documents=docs,
                metadatas=[_sanitize_metadata(m or {}) for m in metas],
                embeddings=_embed_passages(docs),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("migrate_legacy add: %s", e)
            break
        moved += len(ids)
        offset += len(ids)

    if moved:
        logger.info("Память: перенесено %d эпизодов на %s", moved, EMBED_TAG)
    return moved



def summarize_old(threshold_days: int = 7) -> int:
    """Пометить эпизоды старше threshold_days как summarized."""
    cutoff = time.time() - threshold_days * 86400
    try:
        old = episodes.get(where={"timestamp": {"$lt": cutoff}})
    except Exception as e:  # noqa: BLE001
        logger.warning("summarize_old failed: %s", e)
        return 0

    ids = old.get("ids", [])
    metas = old.get("metadatas", [])
    docs = old.get("documents", [])
    updated = 0
    for i, doc_id in enumerate(ids):
        meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
        if meta.get("summarized"):
            continue
        meta["summarized"] = True
        episodes.update(
            ids=[doc_id],
            metadatas=[_sanitize_metadata(meta)],
            documents=[docs[i]] if i < len(docs) else None,
        )
        updated += 1
    return updated


def count_episodes() -> int:
    try:
        return episodes.count()
    except Exception:  # noqa: BLE001
        return 0
