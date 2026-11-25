"""Запрос пользователя → формулировки для поиска по книгам.

Зачем: человек спрашивает бытовым языком («почему чай остывает?»), а в учебнике
написано «теплообмен с окружающей средой». Векторный поиск на такой разрыв
реагирует слабо, а несколько формулировок — дешёвый способ поднять попадание
без обучения чего-либо.

Переписывание здесь эвристическое и мгновенное: LLM-вариант тратил бы секунды
на каждом ходу ради небольшого прироста. Зато учитывается контекст: короткий
вопрос-продолжение («а подробнее?») склеивается с предыдущим сообщением —
иначе поиск ищет «подробнее» по всей книге.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Вопросительные слова и связки: в книге их нет, в поиске они только шумят
_LEAD = re.compile(
    r"^\s*(?:(?:а|и|но|так)\s+)?(?:(?:что|кто|как|где|когда|почему|зачем|"
    r"какой|какая|какие|каков|сколько|чем|чему|про|о|об|насчёт|насчет)\s+)+",
    re.IGNORECASE,
)
_TAIL = re.compile(r"[?!.\s]+$")
_PAREN = re.compile(r"\([^)]*\)")
# Связки внутри вопроса: «что такое X» → «X», «что это значит»
_FILLER = re.compile(
    r"^\s*(?:(?:такое|такой|такая|такие|это|значит|означает|является|"
    r"называется|бывает)\s+)+",
    re.IGNORECASE,
)

# Продолжения, которые сами по себе не несут темы
_DEICTIC = re.compile(
    r"^\s*(?:а\s+)?(?:это|этот|эта|эти|такое|такой|подробнее|подробней|ещё|еще|"
    r"дальше|почему|а\s+как|и\s+что|поясни|объясни|расскажи\s+подробнее)\b",
    re.IGNORECASE,
)

MAX_QUERY = 180
MAX_QUERIES = 3


def is_followup(text: str) -> bool:
    """Короткий вопрос-продолжение, который без контекста бессмысленен."""
    t = (text or "").strip()
    if len(t.split()) <= 4 and _DEICTIC.search(t):
        return True
    return bool(re.fullmatch(r"\s*(?:а|и)?\s*(?:почему|зачем|как|где)\??\s*", t, re.I))


def keywords(text: str) -> str:
    """Убирает вопросительные слова, скобки и знаки: остаются термины."""
    t = _PAREN.sub(" ", text or "")
    for _ in range(3):                 # «что такое», «а это что такое»
        before = t
        t = _LEAD.sub("", t)
        t = _FILLER.sub("", t)
        if t == before:
            break
    t = _TAIL.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_QUERY]


def build_queries(text: str, prev_user: str = "") -> List[str]:
    """1–3 поисковые формулировки: исходник, термины и (для продолжений) контекст."""
    text = (text or "").strip()
    if not text:
        return []
    out: List[str] = [text[:MAX_QUERY]]

    if prev_user and is_followup(text):
        merged = f"{keywords(prev_user)} {keywords(text)}".strip()
        if merged:
            out.append(merged[:MAX_QUERY])

    kw = keywords(text)
    if kw and len(kw) >= 3 and kw != text:
        out.append(kw)

    # Повторы и почти-повторы не нужны: каждый лишний запрос — лишний эмбеддинг
    seen, unique = set(), []
    for q in out:
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[:MAX_QUERIES]


def describe(queries: Optional[List[str]]) -> str:
    """Строка для лога: сколько формулировок реально ушло в поиск."""
    if not queries:
        return "0"
    return f"{len(queries)}: " + " | ".join(q[:60] for q in queries)
