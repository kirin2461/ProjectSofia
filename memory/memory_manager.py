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

    def query(self, query_texts, n_results=3, where=None, **kwargs):
        q = query_texts[0] if query_texts else ""
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
_embedder = _make_embedder()

try:
    import chromadb

    _collection_kwargs = {"embedding_function": _embedder} if _embedder else {}
    episodic_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
    library_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    episodes = episodic_client.get_or_create_collection("episodes", **_collection_kwargs)
    books = library_client.get_or_create_collection("library", **_collection_kwargs)
    BACKEND = "chroma"
except ImportError:
    logger.warning("chromadb не установлен — JSON fallback-стор (dev/тесты)")
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
    episodes.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[_sanitize_metadata(metadata)],
    )
    return doc_id


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
        return episodes.query(query_texts=[query], n_results=n_results, where=where)
    except Exception as e:  # noqa: BLE001
        logger.warning("recall_episodes failed: %s", e)
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}


def recall_library(query: str, n_results: int = 3):
    """Семантический поиск по библиотеке (не archived)."""
    try:
        return books.query(
            query_texts=[query], n_results=n_results, where={"archived": False}
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("recall_library failed: %s", e)
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}


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
