"""Прогон golden set по библиотеке: recall@k, отказы, MRR.

Запуск (нужен проиндексированный `data/library`):

    .venv/Scripts/python.exe scripts/eval_rag.py --golden eval/golden.jsonl
    .venv/Scripts/python.exe scripts/eval_rag.py --min-recall 0.6   # как гейт в CI

Файл golden.jsonl — по одной строке на вопрос:

    {"id": "thermo-1", "question": "Что говорит книга про энтропию?",
     "book": "Сивухин", "pages": [112]}
    {"id": "gap-1", "question": "рецепт борща", "should_find": false}

`should_find: false` — вопрос, ответа на который в книгах быть не должно:
проверяет, что поиск молчит, а не тащит случайный абзац.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics  # noqa: E402


def evaluate(golden: Path, top_k: int = 4, min_score: float = 0.52
             ) -> Tuple[List[dict], dict]:
    """Считает строки отчёта и сводку по всем вопросам golden set."""
    from memory.book_search import search_books

    cases = metrics.load_cases(golden.read_text(encoding="utf-8").splitlines())
    rows = []
    for case in cases:
        hits = search_books(case["question"], top_k=top_k, min_score=min_score)
        rows.append(metrics.case_row(case, hits))
    return rows, metrics.aggregate(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="оценка поиска по книгам")
    ap.add_argument("--golden", default="eval/golden.jsonl")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=0.52)
    ap.add_argument("--min-recall", type=float, default=0.0,
                    help="ниже этого — код возврата 1 (для CI)")
    ap.add_argument("--json", action="store_true", help="машинный вывод")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    golden = Path(args.golden)
    if not golden.exists():
        print(f"нет файла {golden}: возьмите eval/golden.example.jsonl "
              f"и замените вопросы на свои", file=sys.stderr)
        return 2

    rows, summary = evaluate(golden, top_k=args.top_k, min_score=args.min_score)
    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False))
    else:
        print(metrics.format_table(rows, summary))

    if summary["recall"] < args.min_recall:
        print(f"recall {summary['recall']:.0%} ниже порога "
              f"{args.min_recall:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
