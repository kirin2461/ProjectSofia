"""Сбор датасета для стилевой LoRA: тексты книг → пары «вопрос → ответ».

Это подготовка, а не обучение: скрипт берёт фрагменты уже проиндексированных
книг, просит модель сформулировать по ним вопрос и ответ в нужном тоне и
складывает результат в JSONL для QLoRA (см. docs/LORA_STYLE.md).

Зачем так: учить стилю на «сырых» абзацах нельзя — модель начнёт копировать
книгу вместо ответа. Пары «вопрос → ответ по фрагменту» дают и содержание, и
регистр речи.

    .venv/Scripts/python.exe scripts/build_style_dataset.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.jsonx import first_json  # noqa: E402

logger = logging.getLogger("lora_dataset")

PROMPT = (
    "Ниже фрагмент из книги. Придумай ОДИН короткий вопрос, на который этот "
    "фрагмент отвечает, и ответь на него в том же стиле: спокойно, по делу, "
    "с обращением на «ты», без вступлений вроде «Отличный вопрос». Ссылайся "
    "на источник словами «в книге», не выдумывай фактов за пределами текста.\n\n"
    "Верни ровно JSON: {\"вопрос\": \"...\", \"ответ\": \"...\"}\n\n"
    "Фрагмент:\n"
)

MIN_ANSWER = 80
MAX_ANSWER = 1200


def clean(text: str) -> str:
    """Нормализация для дедупликации и записи: без типографских пробелов."""
    text = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def _extract_json(text: str) -> Optional[dict]:
    """Первый JSON-объект из ответа модели (разбор — в core/jsonx)."""
    return first_json(text or "")


def to_pair(raw: str) -> Optional[Dict[str, str]]:
    """Ответ модели → пара {question, answer} или None, если брак."""
    obj = _extract_json(raw)
    if not obj:
        return None
    q = clean(str(obj.get("вопрос") or obj.get("question") or ""))
    a = clean(str(obj.get("ответ") or obj.get("answer") or ""))
    if not q or not a:
        return None
    if not (MIN_ANSWER <= len(a) <= MAX_ANSWER):
        return None
    if len(q) > 300:
        return None
    return {"question": q, "answer": a}


def make_prompt(chunk: str) -> str:
    """Промпт без .format: в тексте есть фигурные скобки JSON-примера."""
    return PROMPT + chunk[:2500]


def make_chat(model: str) -> Callable[[str], str]:
    """Живая модель через OpenAI-совместимый llama-server."""
    from openai import OpenAI

    from config.settings import LLM_API, LLM_API_KEY

    client = OpenAI(base_url=LLM_API, api_key=LLM_API_KEY or "none")

    def chat(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
            extra_body={"cache_prompt": True, "think": False},
        )
        return r.choices[0].message.content or ""

    return chat


def read_chunks(limit: int, min_chars: int = 400, collection=None) -> List[str]:
    """Фрагменты книг из Chroma (самые длинные — информативнее).

    Запасное хранилище в `memory_manager` не знает `limit` — вызываем без него.
    """
    if collection is None:
        from memory import memory_manager

        collection = memory_manager.books
    try:
        got = collection.get(limit=600)
    except TypeError:
        got = collection.get()
    docs = got.get("documents") or []
    chunks = [clean(d) for d in docs if d and len(clean(d)) >= min_chars]
    chunks.sort(key=len, reverse=True)
    return chunks[:limit]


def build(chat: Callable[[str], str], chunks: Iterable[str]) -> List[Dict[str, str]]:
    """Пары без повторов: одинаковый вопрос в датасете бесполезен."""
    seen_q, seen_a = set(), set()
    pairs: List[Dict[str, str]] = []
    for chunk in chunks:
        raw = chat(make_prompt(chunk))
        pair = to_pair(raw)
        if not pair:
            logger.debug("брак модели, пропуск")
            continue
        key_q = pair["question"].lower()[:80]
        key_a = pair["answer"].lower()[:80]
        if key_q in seen_q or key_a in seen_a:
            continue
        seen_q.add(key_q)
        seen_a.add(key_a)
        pairs.append(pair)
    return pairs


def to_jsonl(pairs: Iterable[Dict[str, str]]) -> str:
    """Формат инструкций: messages — его ждут и trl, и chat-шаблоны."""
    lines = []
    for p in pairs:
        lines.append(json.dumps({
            "messages": [
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": p["answer"]},
            ]
        }, ensure_ascii=False))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="датасет для стилевой LoRA")
    ap.add_argument("--out", default="data/lora/style.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="сколько фрагментов взять")
    ap.add_argument("--model", default=None, help="модель для генерации пар")
    ap.add_argument("--dry-run", action="store_true", help="только собрать фрагменты")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    chunks = read_chunks(args.limit)
    if not chunks:
        print("нет книг в индексе: сначала положите PDF в data/library", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"фрагментов: {len(chunks)}, средняя длина "
              f"{sum(map(len, chunks)) // len(chunks)} симв.")
        return 0

    from config.settings import MODEL_BRAIN

    pairs = build(make_chat(args.model or MODEL_BRAIN), chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_jsonl(pairs), encoding="utf-8")
    print(f"пар: {len(pairs)} → {out}")
    return 0 if pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
