"""Метрики качества поиска по книгам: без них «стало умнее» — вопрос веры.

Считаем три вещи, которые видит пользователь:

- **попадание** — нашёлся ли ожидаемый фрагмент (книга + страница) в выдаче;
- **отказ** — молчит ли она на вопросах, которых в книгах нет;
- **цитата** — упоминает ли ответ ту книгу и страницу, откуда взял.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

# Как выглядит честный отказ: модель говорит, что в книгах этого нет
_ABSTAIN = (
    "в книгах этого нет",
    "в книге этого нет",
    "в книгах нет",
    "в книге нет",
    "нет в книге",
    "нет в книгах",
    "не нашла в книг",
    "не нашёл в книг",
    "не нашла в книгах",
    "не нашёл в книгах",
    "не содержится в книг",
    "книги об этом",
    "в предоставленных фрагментах",
)

_WORD_RE = re.compile(r"[0-9a-zа-яё]+")
_DASH = re.compile(r"[–—-]")


def norm(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower().replace("ё", "е")))


def is_abstain(answer: str) -> bool:
    """Ответ честно признаёт отсутствие знаний в книгах."""
    a = " ".join((answer or "").lower().replace("ё", "е").split())
    return any(marker in a for marker in _ABSTAIN)


def pages_mentioned(answer: str) -> List[int]:
    """Все номера страниц, упомянутые в ответе: «с. 112», «с. 112–114»."""
    numbers: List[int] = []
    for match in re.finditer(r"(?:с\.|стр\.|страниц[аы]?)\s*(\d{1,4})", answer or "",
                             re.IGNORECASE):
        numbers.append(int(match.group(1)))
    for match in re.finditer(r"(\d{1,4})\s*[–—-]\s*(\d{1,4})", answer or ""):
        a, b = int(match.group(1)), int(match.group(2))
        if 0 < a <= b <= 3000:
            numbers.extend([a, b])
    return numbers


def cites(answer: str, book: str, pages: Sequence[int] = (), tol: int = 2) -> bool:
    """Ссылается ли ответ на нужную книгу и одну из нужных страниц.

    tol — допуск: соседняя страница в цитате не считается ошибкой, потому что
    чанк может начинаться на предыдущей.
    """
    a = norm(answer)
    if book and norm(book) not in a:
        return False
    if not pages:
        return True
    mentioned = pages_mentioned(answer)
    if not mentioned:
        return False
    return any(abs(p - m) <= tol for p in pages for m in mentioned)


def hit_matches(hit, book: str = "", pages: Sequence[int] = (), tol: int = 2) -> bool:
    """Совпадает ли найденный фрагмент с ожидаемым (книга + страницы)."""
    if book:
        haystack = norm(f"{getattr(hit, 'title', '')} {getattr(hit, 'source', '')}")
        if norm(book) not in haystack:
            return False
    if not pages:
        return True
    start = int(getattr(hit, "page_start", 0) or 0)
    end = int(getattr(hit, "page_end", start) or start)
    return any(start - tol <= p <= end + tol for p in pages)


def rank_of_expected(hits: Sequence, case: dict, tol: int = 2) -> int:
    """Индекс первого ожидаемого фрагмента (−1, если не найден)."""
    book = case.get("book", "")
    pages = case.get("pages") or []
    for i, hit in enumerate(hits or []):
        if hit_matches(hit, book, pages, tol=tol):
            return i
    return -1


def case_row(case: dict, hits: Sequence) -> Dict[str, object]:
    """Строка отчёта по одному вопросу."""
    should_find = bool(case.get("should_find", True))
    rank = rank_of_expected(hits, case) if should_find else -1
    return {
        "id": case.get("id") or case.get("question", "")[:40],
        "should_find": should_find,
        "hits": len(hits or []),
        "rank": rank,
        "found": rank >= 0 if should_find else not hits,
        "top_score": round(float(getattr(hits[0], "score", 0.0)), 3) if hits else 0.0,
    }


def aggregate(rows: Iterable[dict]) -> Dict[str, float]:
    """Сводка: recall@k по положительным, доля отказов по отрицательным, MRR."""
    rows = list(rows)
    positive = [r for r in rows if r["should_find"]]
    negative = [r for r in rows if not r["should_find"]]
    found = [r for r in positive if r["rank"] >= 0]
    recall = len(found) / len(positive) if positive else 1.0
    mrr = (sum(1.0 / (r["rank"] + 1) for r in found) / len(positive)) if positive else 1.0
    abstain = (sum(1 for r in negative if r["found"]) / len(negative)) if negative else 1.0
    return {
        "cases": float(len(rows)),
        "positive": float(len(positive)),
        "negative": float(len(negative)),
        "recall": round(recall, 3),
        "mrr": round(mrr, 3),
        "abstain": round(abstain, 3),
    }


def load_cases(lines: Iterable[str]) -> List[dict]:
    """Разбирает JSONL: пропускает пустые строки, падает на битом JSON."""
    import json

    cases = []
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"строка {lineno}: {e}") from e
        if not case.get("question"):
            raise ValueError(f"строка {lineno}: нет поля question")
        cases.append(case)
    return cases


def format_table(rows: Sequence[dict], summary: Optional[dict] = None) -> str:
    """Компактная таблица для консоли."""
    out = [f"{'id':<22} {'нужно':<6} {'найдено':<8} {'ранг':>5} {'очки':>6}"]
    for r in rows:
        rank = "—" if r["rank"] < 0 else str(r["rank"] + 1)
        out.append(f"{str(r['id'])[:22]:<22} "
                   f"{'да' if r['should_find'] else 'нет':<6} "
                   f"{'да' if r['found'] else 'нет':<8} {rank:>5} "
                   f"{r['top_score']:>6.3f}")
    if summary:
        out.append("")
        out.append(f"случаев {int(summary['cases'])}: recall {summary['recall']:.0%}, "
                   f"MRR {summary['mrr']:.2f}, отказы {summary['abstain']:.0%}")
    return "\n".join(out)
