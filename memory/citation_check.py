"""Проверка цитат в готовом ответе против выданных фрагментов книг.

Зачем: RAG защищает от выдумок только на входе. Модель может взять реальный
абзац и «процитировать» фразу, которой в нём нет, — по смыслу похоже, дословно
нет. Такую цитату видно текстовым сравнением, и это единственный способ
отличить опору на источник от правдоподобного пересказа.

Что проверяем: фрагменты в кавычках «…» и "…". Формулы и числа не проверяются —
нормализация оставляет только буквы и цифры, и запись вида «F = ma» в кавычках
сравнивать бессмысленно.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

MIN_QUOTE = 12          # короче — это термин, а не цитата
MAX_QUOTE = 400
EXACT_RATIO = 0.92      # столько от цитаты должно совпасть подряд

_QUOTE_RE = re.compile(r"«([^»]{2,%d})»|\"([^\"]{2,%d})\"" % (MAX_QUOTE, MAX_QUOTE))
_LETTERS_RE = re.compile(r"[^0-9a-zа-яё]+")


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, только буквы и цифры через один пробел."""
    t = (text or "").lower().replace("ё", "е")
    return _LETTERS_RE.sub(" ", t).strip()


def find_quotes(answer: str) -> List[str]:
    """Все цитаты из ответа (в порядке появления), без повторов."""
    seen, out = set(), []
    for m in _QUOTE_RE.finditer(answer or ""):
        quote = (m.group(1) or m.group(2) or "").strip()
        key = normalize(quote)
        if len(key) < MIN_QUOTE or key in seen:
            continue
        seen.add(key)
        out.append(quote)
    return out


@dataclass
class CitationReport:
    """Итог проверки: какие цитаты подтверждены, какие — нет."""

    checked: int = 0
    verified: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    sources_empty: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing

    def to_dict(self) -> Dict[str, object]:
        return {
            "checked": self.checked,
            "verified": len(self.verified),
            "missing": self.missing,
            "sources_empty": self.sources_empty,
            "ok": self.ok,
        }


def _best_coverage(quote: str, source: str) -> float:
    """Доля цитаты, найденная подряд в источнике (0..1)."""
    if not quote or not source:
        return 0.0
    matcher = SequenceMatcher(None, quote, source, autojunk=False)
    match = matcher.find_longest_match(0, len(quote), 0, len(source))
    return match.size / len(quote)


def verify(answer: str, sources: str, ratio: float = EXACT_RATIO) -> CitationReport:
    """Сверяет цитаты ответа с текстом фрагментов (sources — блок промпта)."""
    quotes = find_quotes(answer)
    src_norm = normalize(sources)
    report = CitationReport(checked=len(quotes), sources_empty=not src_norm)
    if not quotes:
        return report

    for quote in quotes:
        q_norm = normalize(quote)
        if not q_norm:
            continue
        if q_norm in src_norm:
            report.verified.append(quote)
            continue
        # Цитата может быть склеена из двух абзацев или слегка пересказана.
        # Проверяем по кускам: 40 символов достаточно, чтобы отличить
        # дословную фразу от придуманной.
        head = q_norm[:80]
        tail = q_norm[-80:]
        if (len(head) >= 40 and head in src_norm) or (len(tail) >= 40 and tail in src_norm):
            report.verified.append(quote)
            continue
        if _best_coverage(q_norm, src_norm) >= ratio:
            report.verified.append(quote)
        else:
            report.missing.append(quote)
    return report


def sources_from_hits(hits) -> str:
    """Блок источников из BookHit-ов: то же, что видела модель."""
    parts = []
    for i, hit in enumerate(hits or [], 1):
        parts.append(f"[{i}] ({hit.citation()})\n{hit.text.strip()}")
    return "\n\n".join(parts)


def describe(report: Optional[CitationReport]) -> str:
    """Одна строка для лога и панели: без шума, когда всё в порядке."""
    if report is None or not report.checked:
        return ""
    if report.ok:
        return f"цитаты подтверждены: {len(report.verified)}"
    return (f"не подтверждено цитат: {len(report.missing)} "
            f"из {report.checked}")
