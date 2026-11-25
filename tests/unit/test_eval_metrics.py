"""Метрики RAG: попадание в страницу, честный отказ, ссылка в ответе."""

import json
from pathlib import Path

import pytest

from eval import metrics


class Hit:
    def __init__(self, title="Сивухин", page=112, score=0.7, source="s.pdf"):
        self.title = title
        self.page_start = page
        self.page_end = page
        self.score = score
        self.source = source


# ── цитата в ответе ─────────────────────────────────────────
def test_cites_book_and_page():
    assert metrics.cites("В книге сказано (Сивухин, с. 112).", "Сивухин", [112])


def test_cites_tolerates_neighbour_page():
    assert metrics.cites("(Сивухин, с. 113)", "Сивухин", [112], tol=2)


def test_cites_fails_on_other_page():
    assert not metrics.cites("(Сивухин, с. 40)", "Сивухин", [112], tol=2)


def test_cites_fails_without_citation():
    assert not metrics.cites("Энтропия не убывает.", "Сивухин", [112])


def test_cites_page_range_in_answer():
    assert metrics.cites("(Сивухин, с. 112–114)", "Сивухин", [113])


def test_pages_mentioned_parses_forms():
    assert 45 in metrics.pages_mentioned("см. стр. 45")
    assert 45 in metrics.pages_mentioned("страница 45")


def test_cites_ignores_year_as_page():
    """«в 2024 году» не должно считаться ссылкой на страницу 2024."""
    assert metrics.pages_mentioned("это было в 2024 году") == []


# ── отказ ───────────────────────────────────────────────────
@pytest.mark.parametrize("answer", [
    "В книгах этого нет, но из общих знаний скажу так.",
    "В книге нет такого определения.",
    "Я не нашла в книгах подтверждения.",
    "В предоставленных фрагментах об этом ничего нет.",
])
def test_abstain_detected(answer):
    assert metrics.is_abstain(answer)


@pytest.mark.parametrize("answer", [
    "Энтропия замкнутой системы не убывает.",
    "Скорость — производная радиус-вектора по времени.",
    "Согласно Сивухину, это второй закон.",
])
def test_answer_with_content_is_not_abstain(answer):
    assert not metrics.is_abstain(answer)


# ── совпадение фрагмента с ожиданием ────────────────────────
def test_hit_matches_by_title_and_page():
    assert metrics.hit_matches(Hit(), "Сивухин", [112])
    assert not metrics.hit_matches(Hit(page=50), "Сивухин", [112])


def test_hit_matches_other_book():
    assert not metrics.hit_matches(Hit(title="Ландау"), "Сивухин", [112])


def test_hit_matches_page_range():
    hit = Hit(page=112)
    hit.page_end = 115
    assert metrics.hit_matches(hit, "Сивухин", [114])


def test_hit_without_pages_expectation():
    assert metrics.hit_matches(Hit(), "Сивухин", [])


# ── строки и сводка ─────────────────────────────────────────
def test_case_row_positive_found():
    row = metrics.case_row({"id": "a", "question": "q", "book": "Сивухин", "pages": [112]},
                           [Hit(page=40, score=0.9), Hit(page=112, score=0.7)])
    assert row["rank"] == 1 and row["found"] and row["top_score"] == 0.9


def test_case_row_positive_missing():
    row = metrics.case_row({"id": "a", "question": "q", "book": "Сивухин", "pages": [112]},
                           [Hit(page=40)])
    assert row["rank"] == -1 and not row["found"]


def test_case_row_negative_counts_silence_as_success():
    ok = metrics.case_row({"question": "рецепт борща", "should_find": False}, [])
    bad = metrics.case_row({"question": "рецепт борща", "should_find": False}, [Hit()])
    assert ok["found"] and not bad["found"]


def test_aggregate_math():
    rows = [
        {"should_find": True, "rank": 0, "found": True},
        {"should_find": True, "rank": 2, "found": True},
        {"should_find": True, "rank": -1, "found": False},
        {"should_find": False, "rank": -1, "found": True},
        {"should_find": False, "rank": -1, "found": False},
    ]
    summary = metrics.aggregate(rows)
    assert summary["recall"] == round(2 / 3, 3)
    assert summary["mrr"] == round((1 + 1 / 3) / 3, 3)
    assert summary["abstain"] == 0.5


def test_aggregate_without_cases():
    summary = metrics.aggregate([])
    assert summary["recall"] == 1.0 and summary["cases"] == 0.0


# ── golden set ──────────────────────────────────────────────
def test_load_cases_skips_blank_lines(tmp_path):
    cases = metrics.load_cases(['{"question": "a"}', "", '{"question": "b"}'])
    assert [c["question"] for c in cases] == ["a", "b"]


def test_load_cases_rejects_broken_json():
    with pytest.raises(ValueError, match="строка 1"):
        metrics.load_cases(['{"question": '])


def test_load_cases_requires_question():
    with pytest.raises(ValueError, match="question"):
        metrics.load_cases(['{"id": "x"}'])


def test_example_golden_file_is_valid():
    path = Path(__file__).resolve().parents[2] / "eval" / "golden.example.jsonl"
    cases = metrics.load_cases(path.read_text(encoding="utf-8").splitlines())
    assert cases
    assert any(c.get("should_find") is False for c in cases)
    for case in cases:
        assert isinstance(case["question"], str) and case["question"]


def test_format_table_has_header_and_summary():
    rows = [{"id": "a", "should_find": True, "rank": 0, "found": True, "top_score": 0.8}]
    text = metrics.format_table(rows, metrics.aggregate(rows))
    assert "id" in text.splitlines()[0] and "recall" in text
