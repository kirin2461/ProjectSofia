"""Поиск по библиотеке книг: вектор + лексика, с цитатами.

Зачем переранжирование: у многоязычных эмбеддингов близкие расстояния у
похожих по теме фрагментов (0.15 против 0.19), и чистый векторный top-3
часто тащит соседнюю главу. Терминологический запрос («энтропия», «тензор
энергии-импульса») надёжнее ловится по словам, поэтому итоговая оценка —
смесь косинуса и взвешенного по idf перекрытия терминов.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

# Служебные слова: не несут темы, только шумят в перекрытии
_STOP = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между", "это", "какие",
    "такое", "такой", "такая", "такие", "который", "которая", "которые",
    "чем", "чему", "про", "без", "надо", "нужно", "пожалуйста", "скажи",
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "what",
    "how", "why", "does", "do", "for", "with", "on", "at", "by", "be",
}


def _stem(token: str) -> str:
    """Грубое усечение окончаний: «импульса»/«импульс» → «импульс».

    Полноценный стеммер не нужен: для ранжирования достаточно свести
    словоформы к общему префиксу, а ложные склейки редки и не мешают.
    """
    if len(token) <= 6:
        return token
    return token[:6]


def tokenize(text: str) -> List[str]:
    """Значимые токены: нижний регистр, без служебных слов, со стеммингом."""
    return [
        _stem(w) for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP
    ]


@dataclass
class BookHit:
    """Найденный фрагмент книги."""

    text: str
    source: str
    title: str
    chapter: str
    page_start: int
    page_end: int
    score: float
    similarity: float
    section_id: str = ""
    chunk: int = -1
    matched_text: str = ""     # исходный фрагмент до расширения до секции

    def citation(self) -> str:
        """Короткая ссылка вида «Сивухин, т. 1, с. 112»."""
        book = (self.title or self.source or "книга").strip()
        parts = [book]
        if self.chapter and self.chapter not in book:
            parts.append(self.chapter.strip())
        if self.page_start:
            pages = str(self.page_start)
            if self.page_end and self.page_end != self.page_start:
                pages = f"{self.page_start}–{self.page_end}"
            parts.append(f"с. {pages}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "chapter": self.chapter,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": round(self.score, 3),
            "similarity": round(self.similarity, 3),
            "citation": self.citation(),
            "section_id": self.section_id,
            "chunk": self.chunk,
            "expanded": bool(self.matched_text),
        }


def _idf(pool_tokens: Sequence[Sequence[str]]) -> Dict[str, float]:
    n = max(1, len(pool_tokens))
    df: Dict[str, int] = {}
    for toks in pool_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1 + n / max(1, c)) for t, c in df.items()}


def rerank(query: str, candidates: List[BookHit]) -> List[BookHit]:
    """Смешивает косинус с перекрытием терминов (idf), возвращает по убыванию."""
    if not candidates:
        return []

    q_terms = tokenize(query)
    if not q_terms:
        # Запрос без значимых слов — доверяем только вектору
        return sorted(candidates, key=lambda h: -h.similarity)

    pool_tokens = [tokenize(c.text) for c in candidates]
    idf = _idf(pool_tokens)
    total = sum(idf.get(t, 0.0) for t in q_terms) or 1.0

    q_norm = " ".join(_WORD_RE.findall(query.lower()))
    for hit, toks in zip(candidates, pool_tokens):
        present = set(toks)
        overlap = sum(idf.get(t, 0.0) for t in q_terms if t in present) / total
        phrase = 1.0 if len(q_norm) > 12 and q_norm in hit.text.lower() else 0.0
        hit.score = 0.45 * hit.similarity + 0.45 * overlap + 0.10 * phrase

    return sorted(candidates, key=lambda h: -h.score)


def hits_from_chroma(result: dict, query: str = "") -> List[BookHit]:
    """Преобразует ответ Chroma в список BookHit."""
    docs = _flat(result.get("documents"))
    metas = _flat(result.get("metadatas"))
    dists = _flat(result.get("distances"))

    hits: List[BookHit] = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        meta = metas[i] if i < len(metas) and metas[i] else {}
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 1.0
        hits.append(
            BookHit(
                text=str(doc),
                source=str(meta.get("source") or ""),
                title=str(meta.get("title") or ""),
                chapter=str(meta.get("chapter") or ""),
                page_start=int(meta.get("page_start") or 0),
                page_end=int(meta.get("page_end") or 0),
                score=0.0,
                similarity=max(0.0, min(1.0, 1.0 - dist)),
                section_id=str(meta.get("section_id") or ""),
                chunk=int(meta.get("chunk") if meta.get("chunk") is not None else -1),
            )
        )
    return rerank(query, hits) if query else hits


def _flat(value) -> list:
    """[[a, b]] → [a, b] (формат Chroma: список на каждый запрос)."""
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value


def select(
    hits: List[BookHit],
    top_k: int = 4,
    min_score: float = 0.52,
    max_chars: int = 2600,
    gap: float = 0.22,
) -> List[BookHit]:
    """Отбрасывает нерелевантное и ограничивает суммарный объём контекста.

    gap — насколько фрагмент может отставать от лучшего. На реальных книгах
    релевантные абзацы идут плотной группой, а «соседняя глава» отстаёт
    заметно; без этого порога 4B-модель получает солянку из разных разделов.
    """
    if not hits:
        return []
    best = hits[0].score
    picked: List[BookHit] = []
    budget = max_chars
    for hit in hits:
        if len(picked) >= top_k:
            break
        if hit.score < min_score and hit.similarity < 0.88:
            continue
        if hit.score < best - gap:
            continue
        if budget - len(hit.text) < 0 and picked:
            continue
        picked.append(hit)
        budget -= len(hit.text)
    return picked


def format_sources(hits: List[BookHit], max_chars: int = 2600) -> str:
    """Блок для промпта: фрагменты с подписями-цитатами."""
    if not hits:
        return ""
    parts = []
    used = 0
    for i, hit in enumerate(hits, 1):
        body = hit.text.strip()
        if used + len(body) > max_chars and parts:
            break
        used += len(body)
        parts.append(f"[{i}] ({hit.citation()})\n{body}")
    return "\n\n".join(parts)


def merge_candidates(groups: List[List[BookHit]], query: str = "") -> List[BookHit]:
    """Объединяет результаты нескольких запросов без повторов.

    Ключ — источник, страница и первые значимые слова: один и тот же абзац,
    найденный двумя формулировками, должен остаться одним фрагментом.
    """
    best: Dict[tuple, BookHit] = {}
    for group in groups or []:
        for hit in group or []:
            key = (hit.source, hit.page_start, " ".join(tokenize(hit.text)[:40]))
            current = best.get(key)
            if current is None:
                best[key] = hit
                continue
            current.similarity = max(current.similarity, hit.similarity)
            if hit.score > current.score:
                current.score = hit.score
    merged = list(best.values())
    return rerank(query, merged) if query else merged


def expand_sections(
    hits: List[BookHit],
    fetch_section=None,
    fetch_neighbors=None,
    max_chars: int = 1800,
) -> List[BookHit]:
    """Достраивает найденный фрагмент до секции (или соседних чанков).

    Ответ на «почему» и «выведи» требует связного куска: один абзац часто
    начинается с середины мысли. Исходный фрагмент сохраняется в
    `matched_text`, чтобы было видно, что именно совпало с вопросом.
    """
    for hit in hits:
        if hit.matched_text:
            continue
        pieces: List[str] = []
        if hit.section_id and fetch_section is not None:
            try:
                pieces = [str(t) for t in (fetch_section(hit.section_id) or []) if t]
            except Exception as e:  # noqa: BLE001
                logger.debug("секция %s недоступна: %s", hit.section_id, e)
        if not pieces and hit.chunk >= 0 and fetch_neighbors is not None:
            try:
                pieces = [str(t) for t in
                          (fetch_neighbors(hit.source, hit.chunk) or []) if t]
            except Exception as e:  # noqa: BLE001
                logger.debug("соседние чанки недоступны: %s", e)
        if not pieces:
            continue

        joined: List[str] = []
        used = 0
        for piece in pieces:
            piece = piece.strip()
            if not piece or piece in joined:
                continue
            if joined and used + len(piece) > max_chars:
                break
            joined.append(piece)
            used += len(piece)
        text = "\n".join(joined)
        if len(text) > len(hit.text):
            hit.matched_text = hit.text
            hit.text = text
    return hits


def _collection_get(collection, where: Dict) -> dict:
    """get() по метаданным: у Chroma и у запасного хранилища разные сигнатуры."""
    try:
        return collection.get(where=where) or {}
    except TypeError:
        return collection.get() or {}


def section_texts(collection, section_id: str, max_chunks: int = 12) -> List[str]:
    """Тексты всех чанков секции в порядке страниц."""
    got = _collection_get(collection, {"section_id": section_id})
    metas = got.get("metadatas") or []
    docs = got.get("documents") or []
    pairs = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        if str(meta.get("section_id") or "") != section_id:
            continue
        pairs.append((int(meta.get("page_start") or 0),
                      int(meta.get("chunk") or 0), str(doc or "")))
    pairs.sort()
    return [doc for _, _, doc in pairs[:max_chunks] if doc]


def neighbor_texts(collection, source: str, chunk: int, radius: int = 1) -> List[str]:
    """Чанки вокруг найденного: контекст, когда секция неизвестна."""
    got = _collection_get(collection, {"source": source})
    metas = got.get("metadatas") or []
    docs = got.get("documents") or []
    pairs = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        if str(meta.get("source") or "") != source:
            continue
        idx = int(meta.get("chunk") if meta.get("chunk") is not None else -1)
        if abs(idx - chunk) <= radius:
            pairs.append((idx, str(doc or "")))
    pairs.sort()
    return [doc for _, doc in pairs if doc]


def search_books(
    query: str,
    top_k: int = 4,
    pool: int = 24,
    min_score: float = 0.52,
    queries: Optional[List[str]] = None,
    expand: bool = True,
) -> List[BookHit]:
    """Полный цикл: несколько формулировок → Chroma → слияние → переранжирование.

    `queries` — дополнительные формулировки того же вопроса (см.
    core/query_rewrite): каждая даёт свой пул кандидатов, а слияние убирает
    повторы до отбора, чтобы бюджет контекста не тратился на один и тот же
    абзац дважды.
    """
    try:
        from memory import memory_manager

        if memory_manager.count_books() == 0:
            return []
        all_queries = [query] + [q for q in (queries or []) if q and q != query]
        groups = []
        for q in all_queries[:3]:
            raw = memory_manager.query_books(q, n_results=max(pool, top_k))
            groups.append(hits_from_chroma(raw))
        hits = merge_candidates(groups, query)
        try:
            from memory import reranker
            if reranker.available():
                hits = reranker.rerank(hits, query)
        except Exception as e:  # noqa: BLE001
            logger.debug("реранкер: %s", e)
        picked = select(hits, top_k=top_k, min_score=min_score)
        if expand and picked:
            collection = getattr(memory_manager, "books", None)
            if collection is not None:
                picked = expand_sections(
                    picked,
                    fetch_section=lambda sid: section_texts(collection, sid),
                    fetch_neighbors=lambda src, idx: neighbor_texts(collection, src, idx),
                )
        return picked
    except Exception as e:  # noqa: BLE001
        logger.warning("search_books failed: %s", e)
        return []
