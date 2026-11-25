"""Профиль пользователя: факты о нём, а не сжатая проза диалога.

Зачем: сводка контекста — это компрессия, и она теряет именно то, что делает
ассистента «умнее»: имя, профессию, предпочтения, запреты. Сводку пересчитывают
и сокращают, факты должны жить отдельно, со своей уверенностью и источником.

Файл: `data/profile.json`. Запись атомарная: битый профиль не должен
превращаться в потерю всего файла.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MAX_FACTS = 200
MAX_LEN = 300
_STOP = {"и", "в", "во", "не", "на", "с", "как", "что", "это", "я", "у", "по",
         "к", "о", "об", "для", "мне", "мой", "моя", "мы", "ты", "вы"}
_WORD_RE = re.compile(r"[0-9a-zа-яё]+")


def _tokens(text: str) -> set:
    """Значимые слова со грубым стеммингом: «энтропия»/«энтропию» — одно."""
    words = _WORD_RE.findall((text or "").lower().replace("ё", "е"))
    return {w if len(w) <= 6 else w[:6] for w in words if w not in _STOP}


def similar(a: str, b: str) -> float:
    """Пересечение значимых слов / меньшее множество (Жаккар по минимуму)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def is_useful(text: str) -> bool:
    """Отсеивает то, что фактом о пользователе не является."""
    t = " ".join((text or "").split())
    if not (6 <= len(t) <= MAX_LEN):
        return False
    if t.endswith("?") or "?" in t[:3]:
        return False
    low = t.lower()
    for noise in ("не знаю", "не помню", "без понятия", "привет", "спасибо",
                  "пожалуйста", "как дела"):
        if low.startswith(noise) or low == noise:
            return False
    # Факт должен что-то утверждать: одно слово — не факт
    return len(_tokens(t)) >= 2


@dataclass
class Fact:
    text: str
    confidence: float = 0.6
    source: str = ""          # откуда взят: turn-id, реплика
    key: str = ""             # «профессия», «имя» — для замены старых значений

    def to_dict(self) -> dict:
        return asdict(self)


class Profile:
    """Хранилище фактов с дедупликацией и приоритетом по уверенности."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._facts: List[Fact] = []
        self._load()

    # ── чтение/запись ───────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("facts") if isinstance(raw, dict) else raw
            for item in items or []:
                if isinstance(item, dict) and item.get("text"):
                    self._facts.append(Fact(
                        text=str(item["text"]),
                        confidence=float(item.get("confidence") or 0.6),
                        source=str(item.get("source") or ""),
                        key=str(item.get("key") or ""),
                    ))
        except Exception as e:  # noqa: BLE001
            logger.warning("профиль не прочитан (%s) — начинаем с пустого", e)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"facts": [f.to_dict() for f in self._facts]}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("профиль не сохранён: %s", e)
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ── изменения ───────────────────────────────────────────
    def add(self, text: str, key: str = "", confidence: float = 0.6,
            source: str = "") -> bool:
        """Добавить факт. False — если это не факт или он уже есть."""
        t = " ".join((text or "").split())
        if not is_useful(t):
            return False
        with self._lock:
            for fact in self._facts:
                if similar(fact.text, t) >= 0.8:
                    # уточнение того же факта: оставляем более уверенную версию
                    if confidence > fact.confidence:
                        fact.text, fact.confidence = t, confidence
                        fact.source = source or fact.source
                    self._save()
                    return False
                if key and fact.key == key:
                    fact.text, fact.confidence = t, confidence
                    fact.source = source or fact.source
                    self._save()
                    return True
            self._facts.append(Fact(t, confidence, source, key))
            self._trim()
            self._save()
            return True

    def forget(self, needle: str) -> int:
        """Удаляет факты, где встречается подстрока. Возвращает число."""
        n = _norm(needle)
        with self._lock:
            before = len(self._facts)
            self._facts = [f for f in self._facts if n not in _norm(f.text)]
            removed = before - len(self._facts)
            if removed:
                self._save()
            return removed

    def clear(self) -> None:
        with self._lock:
            self._facts = []
            self._save()

    # ── чтение ──────────────────────────────────────────────
    def facts(self, n: int = 8, query: str = "") -> List[Fact]:
        """Топ фактов: при запросе — сначала релевантные, потом уверенные."""
        if not self._facts:
            return []
        if not query:
            ordered = sorted(self._facts, key=lambda f: -f.confidence)
            return ordered[:n]
        scored = [(0.7 * similar(f.text, query) + 0.3 * f.confidence, f)
                  for f in self._facts]
        scored.sort(key=lambda p: -p[0])
        return [f for _, f in scored[:n]]

    def as_prompt(self, n: int = 8, query: str = "") -> str:
        chosen = self.facts(n=n, query=query)
        if not chosen:
            return ""
        lines = [f"- {f.text}" for f in chosen]
        return "Что известно о пользователе:\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self._facts)

    def _trim(self) -> None:
        if len(self._facts) <= MAX_FACTS:
            return
        self._facts.sort(key=lambda f: -f.confidence)
        del self._facts[MAX_FACTS:]


def _norm(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def make_profile(path: Path | str) -> Profile:
    return Profile(path)
