"""Сжатие контекста: сводка старой части диалога вместо скользящего окна.

Зачем: llama-server переиспользует KV-кеш по совпадению префикса запроса.
Пока история только дописывается справа, каждый ход считается почти даром
(единицы новых токенов). Но если просто выбрасывать старые сообщения из
начала очереди, префикс меняется на каждом ходу и кеш сбрасывается целиком.

Поэтому вместо окна — редкое сжатие: старая половина диалога превращается
в короткую сводку, которая живёт в системном блоке. Префикс остаётся
стабильным между сжатиями, а факты не теряются.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

from config.settings import (
    CONTEXT_BUDGET_TOKENS,
    LLM_API,
    LLM_API_KEY,
    MODEL_EXTRACTOR_ACTIVE,
    NUM_CTX,
)

logger = logging.getLogger(__name__)

COMPACT_TRIGGER_MESSAGES = 8   # при скольки сообщениях пора сжимать
COMPACT_KEEP_MESSAGES = 4      # сколько последних оставить дословно
SUMMARY_MAX_CHARS = 900

_client = None


def _llm_client():
    """Клиент для сводок (ленивый, чтобы не мешать импорту в тестах)."""
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(base_url=LLM_API, api_key=LLM_API_KEY or "none")
    return _client


def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: для русского ~2.5 символа на токен.

    Точный токенизатор здесь не нужен: порог сжатия берётся с запасом в
    треть контекста, а ошибка оценки в 20% ничего не ломает.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.5) + 4)


def messages_tokens(messages: Iterable[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def budget_tokens() -> int:
    """Сколько токенов отдаём истории диалога."""
    if CONTEXT_BUDGET_TOKENS and CONTEXT_BUDGET_TOKENS > 0:
        return int(CONTEXT_BUDGET_TOKENS)
    return min(6000, max(2000, int(NUM_CTX * 0.35)))


def should_compact(
    messages: List[dict],
    summary: str = "",
    trigger: int = COMPACT_TRIGGER_MESSAGES,
    budget: Optional[int] = None,
) -> bool:
    """Пора ли сжимать: по числу сообщений или по объёму."""
    if len(messages) >= trigger:
        return True
    limit = budget if budget is not None else budget_tokens()
    return messages_tokens(messages) + estimate_tokens(summary) > limit


def heuristic_summary(messages: List[dict], limit: int = SUMMARY_MAX_CHARS) -> str:
    """Резервная сводка без модели: роли + начала реплик.

    Нужна, когда модель занята или ответила ошибкой: терять факты из середины
    диалога нельзя, а выбросить сообщения — именно то, чего мы избегаем.
    """
    parts = []
    for m in messages:
        role = "Пользователь" if m.get("role") == "user" else "София"
        text = re.sub(r"\s+", " ", str(m.get("content", ""))).strip()
        if not text:
            continue
        cut = 90 if role == "Пользователь" else 70
        if len(text) > cut:
            text = text[:cut].rstrip() + "…"
        parts.append(f"- {role}: {text}")
    joined = "\n".join(parts)
    if len(joined) > limit:
        joined = joined[:limit].rstrip() + "…"
    return joined


def _summary_prompt(messages: List[dict], previous: str) -> str:
    dialog = "\n".join(
        f"{'Пользователь' if m.get('role') == 'user' else 'София'}: {m.get('content')}"
        for m in messages
    )
    head = (
        "Сожми историю диалога в короткую сводку на русском, 3–6 пунктов:\n"
        "- кто пользователь и что о нём известно (имя, профессия, интересы);\n"
        "- что обсуждали и к чему пришли;\n"
        "- важные имена, числа, формулы, названия;\n"
        "- незакрытые вопросы.\n"
        "Пиши тезисами, без вступлений и без выдумок. Не больше 120 слов.\n"
    )
    if previous.strip():
        head += f"\nПредыдущая сводка (учти её, не повторяй дословно):\n{previous.strip()}\n"
    return head + f"\nДиалог:\n{dialog}"


async def summarize_history(
    messages: List[dict],
    previous: str = "",
    timeout: float = 25.0,
) -> str:
    """Сводка через модель; при любой ошибке — резервный вариант."""
    if not messages:
        return previous
    try:
        import asyncio

        r = await asyncio.wait_for(
            _llm_client().chat.completions.create(
                model=MODEL_EXTRACTOR_ACTIVE,
                messages=[{"role": "user", "content": _summary_prompt(messages, previous)}],
                extra_body={
                    "num_ctx": 4096,
                    "num_predict": 220,
                    "think": False,
                    "cache_prompt": True,
                },
            ),
            timeout=timeout,
        )
        text = (r.choices[0].message.content or "").strip()
        if len(text) >= 40:
            return text[:SUMMARY_MAX_CHARS]
        logger.debug("сводка слишком короткая (%d симв.), беру резервную", len(text))
    except Exception as e:  # noqa: BLE001
        logger.warning("сводка диалога не удалась: %s", e)
    fallback = heuristic_summary(messages)
    return (previous + "\n" + fallback).strip()[:SUMMARY_MAX_CHARS] if previous else fallback


async def maybe_compact(
    session,
    trigger: int = COMPACT_TRIGGER_MESSAGES,
    keep: int = COMPACT_KEEP_MESSAGES,
    budget: Optional[int] = None,
) -> dict:
    """Сжимает старую часть истории, если она распухла.

    Возвращает статистику для лога и /state. Ничего не делает, если сжимать
    нечего — обычный случай на коротком диалоге.
    """
    import collections

    messages = list(session.history)
    before_tokens = messages_tokens(messages) + estimate_tokens(getattr(session, "summary", ""))
    stat = {"compacted": False, "messages": len(messages), "tokens": before_tokens}

    if not should_compact(messages, getattr(session, "summary", ""), trigger, budget):
        return stat

    old = messages[:-keep] if len(messages) > keep else []
    if len(old) < 2:
        return stat

    summary = await summarize_history(old, getattr(session, "summary", ""))
    session.summary = summary
    max_history = getattr(session, "max_history", max(len(messages), trigger))
    session.history = collections.deque(messages[-keep:], maxlen=max_history)
    session.compactions = getattr(session, "compactions", 0) + 1

    after_tokens = messages_tokens(list(session.history)) + estimate_tokens(summary)
    stat.update(
        compacted=True,
        messages=len(session.history),
        tokens=after_tokens,
        before_tokens=before_tokens,
        saved=before_tokens - after_tokens,
    )
    logger.info(
        "Контекст сжат: %d → %d сообщений, ~%d → ~%d токенов",
        len(messages), stat["messages"], before_tokens, after_tokens,
    )
    return stat
