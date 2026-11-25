"""Черновик golden set из проиндексированных книг.

Скрипт берёт фрагменты из библиотеки (равномерно по книгам и страницам),
просит модель придумать к каждому вопрос, который задал бы живой человек, и
пишет JSONL в формате `scripts/eval_rag.py`: книга + страницы ответа. Плюс
набор вопросов, ответа на которые в книгах быть не должно.

Это черновик: просмотрите файл глазами и удалите неудачные вопросы — golden
set ценен ровно настолько, насколько ему можно верить.

    .venv/Scripts/python.exe scripts/make_golden.py --per-book 8
    .venv/Scripts/python.exe scripts/eval_rag.py --golden eval/golden.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.jsonx import first_json  # noqa: E402

logger = logging.getLogger("make_golden")

PROMPT = (
    "Ниже фрагмент из книги. Придумай ОДИН вопрос, который мог бы задать "
    "обычный человек и на который этот фрагмент отвечает. Не копируй фразы "
    "из текста дословно, не упоминай «фрагмент» и «книгу», не задавай вопрос "
    "«о чём этот текст». Вопрос — одна строка, до 120 символов.\n\n"
    "Верни ровно JSON: {\"вопрос\": \"...\"}\n\n"
    "Фрагмент:\n"
)

# Вопросы, ответа на которые в учебниках нет: поиск должен промолчать
NEGATIVES = [
    "рецепт борща со сметаной",
    "курс биткоина на завтра",
    "какая погода будет в выходные",
    "как пройти первый уровень в Minecraft",
    "лучший сериал этого года",
    "сколько стоит такси до аэропорта",
]

MIN_CHUNK = 300
MAX_QUESTION = 160


def clean(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_records(collection=None, limit: int = 4000) -> List[dict]:
    """Фрагменты с метаданными: текст, книга, страницы."""
    if collection is None:
        from memory import memory_manager

        collection = memory_manager.books
    try:
        got = collection.get(limit=limit, include=["documents", "metadatas"])
    except TypeError:
        got = collection.get()
    docs = got.get("documents") or []
    metas = got.get("metadatas") or [{}] * len(docs)
    out = []
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        if meta.get("archived"):
            continue
        text = clean(doc)
        if len(text) < MIN_CHUNK:
            continue
        out.append({
            "text": text,
            "book": str(meta.get("title") or meta.get("source") or ""),
            "page": int(meta.get("page_start") or 0),
            "page_end": int(meta.get("page_end") or meta.get("page_start") or 0),
        })
    return out


def pick(records: Iterable[dict], per_book: int, seed: int = 7) -> List[dict]:
    """Равномерно по книге: начало, середина, конец — а не первые N абзацев."""
    by_book: Dict[str, List[dict]] = {}
    for r in records:
        by_book.setdefault(r["book"], []).append(r)
    rng = random.Random(seed)
    chosen: List[dict] = []
    for book in sorted(by_book):
        items = sorted(by_book[book], key=lambda r: r["page"])
        if len(items) <= per_book:
            chosen.extend(items)
            continue
        step = len(items) / per_book
        for i in range(per_book):
            lo = int(i * step)
            hi = max(lo + 1, int((i + 1) * step))
            chosen.append(items[rng.randrange(lo, hi)])
    return chosen


def to_question(raw: str) -> Optional[str]:
    """Ответ модели → вопрос или None, если брак."""
    obj = first_json(raw or "")
    q = ""
    if isinstance(obj, dict):
        q = str(obj.get("вопрос") or obj.get("question") or "")
    q = clean(q)
    if not q or len(q) > MAX_QUESTION or len(q) < 8:
        return None
    low = q.lower()
    if "фрагмент" in low or "этот текст" in low:
        return None
    return q


def _book_key(book: str) -> str:
    """Короткий кусок названия: eval сравнивает по вхождению подстроки."""
    words = clean(book).split()
    return " ".join(words[:4])


def build(chat: Callable[[str], str], records: Iterable[dict]) -> List[dict]:
    cases: List[dict] = []
    seen = set()
    for n, rec in enumerate(records, 1):
        try:
            q = to_question(chat(PROMPT + rec["text"][:1800]))
        except Exception as e:  # noqa: BLE001
            logger.info("модель не ответила: %s", e)
            continue
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        pages = sorted({p for p in (rec["page"], rec["page_end"]) if p})
        cases.append({
            "id": f"auto-{n}",
            "question": q,
            "book": _book_key(rec["book"]),
            "pages": pages,
            "note": "черновик: проверьте глазами",
        })
    return cases


def negatives(n: int) -> List[dict]:
    return [
        {"id": f"gap-{i + 1}", "question": q, "should_find": False}
        for i, q in enumerate(NEGATIVES[:max(0, n)])
    ]


def to_jsonl(cases: Iterable[dict]) -> str:
    return "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"


def make_chat(model: str) -> Callable[[str], str]:
    from openai import OpenAI

    from config.settings import LLM_API, LLM_API_KEY

    client = OpenAI(base_url=LLM_API, api_key=LLM_API_KEY or "none")

    def chat(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=120,
            extra_body={"think": False},
        )
        return r.choices[0].message.content or ""

    return chat


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="черновик golden set из книг")
    ap.add_argument("--out", default="eval/golden.jsonl")
    ap.add_argument("--per-book", type=int, default=8)
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--model", default=None)
    ap.add_argument("--force", action="store_true", help="перезаписать существующий файл")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} уже есть — проверенный набор не перезаписываю (--force)",
              file=sys.stderr)
        return 1
    records = read_records()
    if not records:
        print("нет книг в индексе: сначала положите PDF в data/library", file=sys.stderr)
        return 1

    from config.settings import MODEL_BRAIN

    chosen = pick(records, args.per_book)
    cases = build(make_chat(args.model or MODEL_BRAIN), chosen)
    cases += negatives(args.negatives)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_jsonl(cases), encoding="utf-8")
    print(f"вопросов: {len(cases)} → {out}. Просмотрите файл перед прогоном eval.")
    return 0 if len(cases) > args.negatives else 1


if __name__ == "__main__":
    raise SystemExit(main())
