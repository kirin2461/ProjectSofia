"""Phase-3-02: Управление памятью — эпизодическая + библиотека (Chroma).

Эмбеддинги — локальная multilingual-e5-small (ONNX, offline). Если она не
поднялась, это ERROR и отключение поиска/записи по эпизодам: тег коллекций
остаётся e5s, накопленная память не «исчезает». Резерв (Ollama bge-m3) —
только явно: SOFIA_EMBED_FALLBACK=ollama, коллекции __default, статус degraded.
Без chromadb — лёгкий JSON-стор с hash-эмбеддингами (dev/test).

Публичный API:
    add_episode(text, metadata) -> id
    recall_episodes(query, n_results, time_window)
    recall_library(query, n_results)
    summarize_old(threshold_days) -> int
    count_episodes() -> int
    memory_status() -> dict     (для /state: видно, что память деградировала)
"""

import hashlib
import json
import logging
import math
import os
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
from memory.embedder import EMBED_TAG as _E5_TAG
from memory.embedder import embedder_error, get_embedder

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
            except Exception as e:  # noqa: BLE001
                # Молча подмешивать hash-векторы в чужое пространство нельзя:
                # поиск по такой коллекции станет случайным.
                raise RuntimeError(
                    f"Ollama {self.model} не ответила: {e}"
                ) from e
        return vectors


def _make_embedder():
    """Ollama bge-m3 — только как явный резерв (SOFIA_EMBED_FALLBACK=ollama)."""
    try:
        import requests

        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        logger.info("Память: эмбеддинги через Ollama %s", MODEL_EMBED)
        return OllamaEmbeddingFunction()
    except Exception:  # noqa: BLE001
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
_embedder = None
EMBED_TAG = _E5_TAG          # суффикс коллекций; e5s — пока резерв не включён явно
EMBED_MODE = "unavailable"   # e5 | ollama | hash | unavailable
EMBED_REASON = ""
MEMORY_DEGRADED = True

_WARNED: set = set()


def _warn_once(kind: str, message: str) -> None:
    """Одна и та же деградация не должна заливать лог на каждый эпизод."""
    if kind in _WARNED:
        return
    _WARNED.add(kind)
    logger.error("Память: %s", message)


def _fallback_mode() -> str:
    """Явный резерв из настроек: off (по умолчанию) | ollama."""
    return (os.environ.get("SOFIA_EMBED_FALLBACK") or "off").strip().lower()


def _collection_name(base: str) -> str:
    """Имя коллекции зависит от модели эмбеддингов: разные векторы не смешиваем."""
    return f"{base}__{EMBED_TAG}"


def init_memory_backend() -> str:
    """Выбрать эмбеддер и тег коллекций. Возвращает режим.

    Норма — только e5. Падение e5 не переключает память на другую коллекцию:
    тег остаётся e5s, поиск и запись отключаются громко (ERROR + статус).
    Резерв включается лишь явно (SOFIA_EMBED_FALLBACK=ollama), живёт в
    коллекциях __default и всегда помечен degraded — за нормальную работу
    он не выдаётся.
    """
    global _embedder, EMBED_TAG, EMBED_MODE, EMBED_REASON, MEMORY_DEGRADED

    model = get_embedder()
    if model is not None:
        _embedder = model
        EMBED_TAG = getattr(model, "tag", _E5_TAG)
        EMBED_MODE = "e5"
        EMBED_REASON = ""
        MEMORY_DEGRADED = False
        _WARNED.clear()
        logger.info("Память: эмбеддинги %s (%s)",
                    getattr(model, "model_name", model.name()), model.model_path)
        return EMBED_MODE

    reason = embedder_error() or "модель эмбеддингов не загружена"

    if _fallback_mode() == "ollama":
        alt = _make_embedder()
        if alt is not None:
            _embedder = alt
            EMBED_TAG = "default"
            EMBED_MODE = "ollama"
            EMBED_REASON = reason
            MEMORY_DEGRADED = True
            _WARNED.clear()
            logger.warning(
                "Память: РЕЗЕРВ — эмбеддинги Ollama %s, коллекции __default. "
                "e5 недоступна: %s. Накопленная память e5s не видна, пока модель "
                "не поднимется (проверьте /state).",
                MODEL_EMBED, reason,
            )
            return EMBED_MODE
        logger.error("Память: резерв Ollama недоступен, e5 тоже: %s", reason)

    _embedder = None
    EMBED_TAG = _E5_TAG
    EMBED_MODE = "unavailable"
    EMBED_REASON = reason
    MEMORY_DEGRADED = True
    _WARNED.clear()
    logger.error(
        "Память: векторная память недоступна (%s); коллекции %s и %s не тронуты, "
        "новые эпизоды не записываются, поиск отключён",
        reason, _collection_name("episodes"), _collection_name("library"),
    )
    return EMBED_MODE


def _try_recover() -> bool:
    """Поднять e5 без перезапуска (get_embedder сам держит паузу между попытками).

    Возвращаемся только из «unavailable»: явный резерв не переключаем на лету,
    иначе коллекции поменяются под работающими запросами.
    """
    if EMBED_MODE != "unavailable":
        return False
    if get_embedder() is None:
        return False
    init_memory_backend()
    if EMBED_MODE == "e5":
        logger.warning("Память: эмбеддинги восстановлены, поиск снова работает")
        return True
    return False


def embeddings_ready() -> bool:
    """Можно ли векторизовать текст: без этого нет ни записи, ни поиска."""
    if BACKEND != "chroma":
        return True
    if EMBED_MODE in ("e5", "ollama"):
        return True
    return _try_recover()


def memory_status() -> dict:
    """Состояние памяти для /state: деградация должна быть видна, а не угадываться."""
    return {
        "backend": BACKEND,
        "embeddings": EMBED_MODE,
        "tag": EMBED_TAG,
        "degraded": bool(MEMORY_DEGRADED),
        "fallback": EMBED_MODE in ("ollama", "hash"),
        "search": "off" if EMBED_MODE == "unavailable" else "on",
        "reason": EMBED_REASON,
        "collections": {
            "episodes": _collection_name("episodes"),
            "library": _collection_name("library"),
        },
        "episodes": count_episodes(),
        "books": count_books(),
    }


def _embed_passages(texts: List[str]) -> Optional[List[List[float]]]:
    if not embeddings_ready():
        return None
    if not hasattr(_embedder, "embed_documents"):
        return None
    return _embedder.embed_documents(texts)


def _embed_query(text: str) -> Optional[List[float]]:
    if not embeddings_ready():
        return None
    if not hasattr(_embedder, "embed_query"):
        return None
    return _embedder.embed_query(text)


init_memory_backend()

try:
    import chromadb

    if EMBED_MODE in ("e5", "unavailable"):
        # Векторы считаем сами (префиксы query:/passage:), Chroma их не трогает.
        # Тег тот же e5s и при недоступной модели — старые эпизоды не «исчезают».
        _collection_kwargs = {
            "embedding_function": None,
            "metadata": {"hnsw:space": "cosine"},
        }
    elif EMBED_MODE == "ollama":
        # Резерв: векторы считает сама Chroma через переданную функцию.
        _collection_kwargs = {"embedding_function": _embedder}
    else:
        _collection_kwargs = {}

    episodic_client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
    library_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    episodes = episodic_client.get_or_create_collection(
        _collection_name("episodes"), **_collection_kwargs
    )
    books = library_client.get_or_create_collection(
        _collection_name("library"), **_collection_kwargs
    )
    BACKEND = "chroma"
except ImportError:
    logger.warning("chromadb не установлен — JSON fallback-стор (dev/тесты)")
    episodic_client = None
    library_client = None
    episodes = _FallbackCollection(MEMORY_DB_PATH, "episodes")
    books = _FallbackCollection(VECTOR_DB_PATH, "library")
    BACKEND = "fallback"
    EMBED_MODE = "hash"
    EMBED_TAG = "default"
    MEMORY_DEGRADED = True
    EMBED_REASON = EMBED_REASON or "chromadb не установлен — hash-эмбеддинги (dev/тесты)"


# ═══════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════
def _episode_id(text: str, timestamp: float) -> str:
    raw = f"{text}{timestamp}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def add_episode(text: str, metadata: Optional[dict] = None) -> str:
    """Добавить эпизод диалога. Возвращает его id.

    Если векторов нет, эпизод не пишется (текст без вектора в e5-коллекцию
    не положить), но и падать в потоке ответа нельзя: пишем ERROR один раз.
    """
    metadata = dict(metadata or {})
    ts = metadata.get("timestamp", time.time())
    metadata["timestamp"] = ts
    doc_id = _episode_id(text, ts)
    if not embeddings_ready():
        _warn_once("write", f"эпизод не сохранён — {EMBED_REASON or 'нет эмбеддингов'}")
        return doc_id
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


def _empty_hits(with_distances: bool = False) -> dict:
    out = {"documents": [[]], "metadatas": [[]], "ids": [[]]}
    if with_distances:
        out["distances"] = [[]]
    return out


def _query(col, query: str, n_results: int, where: Optional[dict] = None,
           with_distances: bool = False):
    """Поиск: векторный, если коллекция это умеет, иначе текстовый (dev/тесты)."""
    if BACKEND == "chroma" and not _is_fallback(col):
        qv = _embed_query(query)
        if qv:
            return col.query(query_embeddings=[qv], n_results=n_results, where=where)
        if EMBED_MODE in ("e5", "unavailable"):
            # Без вектора запроса искать по e5-коллекции нечем — молча не ищем.
            _warn_once("search",
                       f"поиск отключён — {EMBED_REASON or 'нет эмбеддингов'}")
            return _empty_hits(with_distances)
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
        return _empty_hits()


def query_books(query: str, n_results: int = 3, where: Optional[dict] = None):
    """Кандидаты из библиотеки для последующего переранжирования."""
    try:
        return _query(books, query, n_results, where or {"archived": False},
                      with_distances=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("query_books failed: %s", e)
        return _empty_hits(with_distances=True)


def recall_library(query: str, n_results: int = 3):
    """Семантический поиск по библиотеке (не archived)."""
    return query_books(query, n_results=n_results)


def count_books() -> int:
    try:
        return books.count()
    except Exception:  # noqa: BLE001
        return 0


def add_book_chunks(ids, documents, metadatas) -> None:
    """Записать чанки книги (векторы считаем сами — они многоязычные).

    Без векторов запись невозможна — бросаем, чтобы индексатор не отчитался
    «проиндексировано» там, где на самом деле ничего не сохранено.
    """
    if not embeddings_ready():
        raise RuntimeError(f"эмбеддинги недоступны: {EMBED_REASON or 'нет модели'}")
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


def _legacy_page(col, offset: int, size: int = 256) -> dict:
    try:
        return col.get(limit=size, offset=offset,
                       include=["documents", "metadatas"])
    except TypeError:  # fallback-стор не умеет страницы
        return col.get()


def _text_key(text: str) -> str:
    return " ".join((text or "").split())


def _existing_ids(ids: List[str]) -> set:
    """id, которые уже есть в новой коллекции (id старых записей другие, но бывают совпадения)."""
    try:
        got = episodes.get(ids=list(ids))
        return set(got.get("ids") or [])
    except TypeError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.debug("migrate_legacy ids: %s", e)
        return set()
    try:
        got = episodes.get()
    except Exception as e:  # noqa: BLE001
        logger.debug("migrate_legacy ids: %s", e)
        return set()
    return {i for i in (got.get("ids") or []) if i in set(ids)}


def _existing_texts(max_docs: int = 50000) -> set:
    """Тексты, уже лежащие в новой коллекции: у домиграции не должно быть дублей."""
    texts, offset, paginated = set(), 0, True
    while offset < max_docs:
        try:
            page = episodes.get(limit=1000, offset=offset, include=["documents"])
        except TypeError:
            paginated = False
            try:
                page = episodes.get()
            except Exception as e:  # noqa: BLE001
                logger.debug("migrate_legacy texts: %s", e)
                break
        except Exception as e:  # noqa: BLE001
            logger.debug("migrate_legacy texts: %s", e)
            break
        docs = page.get("documents") or []
        texts.update(_text_key(d) for d in docs if d)
        if not paginated or len(docs) < 1000:
            break
        offset += 1000
    return texts


def migrate_legacy(limit: int = 2000, dry_run: bool = False) -> int:
    """Донести старые эпизоды до текущей модели, не создавая дублей.

    До появления многоязычных векторов память писалась эмбеддингами
    all-MiniLM-L6-v2 (англоязычная модель) — на русском поиск по ним почти
    случаен. Сами реплики целы, поэтому их достаточно пересчитать.

    Раньше функция выходила, если новая коллекция непуста, — и 92 старых
    эпизода оставались в `episodes` навсегда. Теперь перенос идемпотентен:
    пропускаются id, уже лежащие в новой коллекции, и тексты, которые там
    уже есть (у старых записей другие id, но тот же текст).

    dry_run=True — посчитать, ничего не записывая.
    """
    if BACKEND != "chroma" or EMBED_MODE != "e5":
        return 0
    try:
        legacy = episodic_client.get_or_create_collection("episodes")
        total = legacy.count()
    except Exception as e:  # noqa: BLE001
        logger.debug("migrate_legacy: %s", e)
        return 0
    if not total:
        return 0

    known_texts = _existing_texts()
    moved, offset = 0, 0
    while offset < min(total, limit):
        try:
            page = _legacy_page(legacy, offset)
        except Exception as e:  # noqa: BLE001
            logger.warning("migrate_legacy get: %s", e)
            break
        ids = page.get("ids") or []
        if not ids:
            break
        docs = page.get("documents") or []
        metas = page.get("metadatas") or []
        known = _existing_ids(ids)

        pick = []
        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            if not text or doc_id in known or _text_key(text) in known_texts:
                continue
            pick.append(i)

        if pick:
            new_docs = [docs[i] for i in pick]
            if dry_run:
                moved += len(pick)
            else:
                try:
                    episodes.add(
                        ids=[ids[i] for i in pick],
                        documents=new_docs,
                        metadatas=[
                            _sanitize_metadata(metas[i] if i < len(metas) and metas[i] else {})
                            for i in pick
                        ],
                        embeddings=_embed_passages(new_docs),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("migrate_legacy add: %s", e)
                    break
                moved += len(pick)
                known_texts.update(_text_key(d) for d in new_docs)
        offset += len(ids)

    if moved:
        if dry_run:
            logger.info("Память: домиграция перенесла бы %d эпизодов (dry-run)", moved)
        else:
            logger.info("Память: перенесено %d эпизодов на %s", moved, EMBED_TAG)
    else:
        logger.debug("Память: домиграция нечего переносить (legacy=%d)", total)
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
