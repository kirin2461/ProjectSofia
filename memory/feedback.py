"""Оценки ответов: единственный честный сигнал, что «стало хуже».

Модель не учится на этих файлах, и притворяться иначе не нужно. Зато:
- видно, на каких вопросах она проваливается (по стратегии и по наличию книг);
- накопленный перекос превращается в одну строку подсказки в промпте
  («на вопросы с книгами часто жалуются — проверяй цитаты»), а не в догадку.

Формат — JSONL, по строке на оценку: дописывать дёшево, читать можно хвостом,
битые строки не ломают статистику (их просто пропускаем).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/feedback.jsonl")
MIN_FOR_ADVICE = 4          # меньше — это шум, а не сигнал
BAD_RATE = 0.4              # доля 👎, после которой стоит вмешаться


def _path(path: Optional[Path | str]) -> Path:
    return Path(path) if path else DEFAULT_PATH


def record(
    rating: int,
    question: str,
    answer: str = "",
    *,
    strategy: str = "",
    sources: int = 0,
    model: str = "",
    note: str = "",
    path: Optional[Path | str] = None,
) -> dict:
    """Дописывает оценку: 1 — понравилось, −1 — нет, 0 — нейтрально."""
    rating = 1 if rating > 0 else (-1 if rating < 0 else 0)
    entry = {
        "ts": time.time(),
        "rating": rating,
        "question": (question or "")[:400],
        "answer": (answer or "")[:400],
        "strategy": (strategy or "")[:60],
        "sources": int(sources or 0),
        "model": (model or "")[:60],
        "note": (note or "")[:200],
    }
    target = _path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:  # диск заполнен / файл занят — терять ответ из-за этого нельзя
        logger.warning("feedback не записан: %s", e)
    return entry


def read(path: Optional[Path | str] = None, limit: int = 0) -> List[dict]:
    """Все оценки (или последние `limit`), битые строки пропускаются."""
    target = _path(path)
    if not target.exists():
        return []
    out: List[dict] = []
    try:
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and "rating" in item:
                    out.append(item)
    except OSError as e:
        logger.warning("feedback не прочитан: %s", e)
        return []
    return out[-limit:] if limit else out


def latest(items: List[dict]) -> List[dict]:
    """Одна оценка на ответ — последняя.

    Пользователь передумал (👍 → 👎) — в файле две строки, а голос один.
    Ответ узнаём по паре (вопрос, ответ); строки без текста ответа
    (старые и ручные записи) опознать нельзя — они считаются каждая.
    """
    keyed: Dict[tuple, dict] = {}
    for n, item in enumerate(items):
        answer = str(item.get("answer") or "")
        key = (str(item.get("question") or ""), answer) if answer else ("#", n)
        keyed.pop(key, None)        # порядок — по последней оценке
        keyed[key] = item
    return list(keyed.values())


def stats(path: Optional[Path | str] = None) -> dict:
    """Счёт по оценкам и разбивка по стратегиям."""
    items = latest(read(path))
    up = sum(1 for i in items if i.get("rating", 0) > 0)
    down = sum(1 for i in items if i.get("rating", 0) < 0)
    by_strategy: Dict[str, Dict[str, int]] = {}
    with_sources = {"up": 0, "down": 0}
    for item in items:
        key = str(item.get("strategy") or "—")
        slot = by_strategy.setdefault(key, {"up": 0, "down": 0})
        if item.get("rating", 0) > 0:
            slot["up"] += 1
            if item.get("sources"):
                with_sources["up"] += 1
        elif item.get("rating", 0) < 0:
            slot["down"] += 1
            if item.get("sources"):
                with_sources["down"] += 1
    total = up + down
    return {
        "total": len(items),
        "up": up,
        "down": down,
        "good_rate": round(up / total, 3) if total else 0.0,
        "by_strategy": by_strategy,
        "with_sources": with_sources,
    }


def advice(path: Optional[Path | str] = None) -> str:
    """Строка-подсказка для промпта; пустая, если сигнала мало.

    Смысл — поправить самую частую жалобу, а не переписать личность: одна
    короткая поправка, которую модель может применить к следующему ответу.
    """
    items = latest(read(path))
    if len(items) < MIN_FOR_ADVICE:
        return ""
    st = stats(path)
    if st["down"] < MIN_FOR_ADVICE or st["down"] / max(1, st["down"] + st["up"]) < BAD_RATE:
        return ""

    worst, worst_rate = "", 0.0
    for name, slot in st["by_strategy"].items():
        n = slot["up"] + slot["down"]
        if n < 2:
            continue
        rate = slot["down"] / n
        if rate > worst_rate:
            worst, worst_rate = name, rate

    if st["with_sources"]["down"] > st["with_sources"]["up"]:
        return ("Пользователь часто недоволен ответами по книгам: цитируй только "
                "то, что есть во фрагментах, и не пересказывай их своими словами "
                "как дословную цитату.")
    if worst and worst_rate >= BAD_RATE:
        if "greet" in worst:
            return "Приветствия пользователю кажутся слишком длинными: отвечай короче."
        return ("Пользователь часто недоволен ответами на вопросы этого типа: "
                "отвечай короче и по существу, без вступлений.")
    return ""


def describe(st: Optional[dict] = None, path: Optional[Path | str] = None) -> str:
    """Одна строка для лога и панели."""
    data = st if st is not None else stats(path)
    if not data.get("total"):
        return "оценок пока нет"
    return f"👍 {data['up']} / 👎 {data['down']} (доля хороших {data['good_rate']:.0%})"
