"""Переоценка ответа (👍 → 👎) — один голос, а не два; без текста оценку не принимаем."""

import asyncio
import json

import pytest
from fastapi import HTTPException

import main
from memory import feedback


def test_changed_mind_counts_once(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(1, "q", "a", path=path)
    feedback.record(-1, "q", "a", path=path)
    st = feedback.stats(path)
    assert st["total"] == 1 and st["up"] == 0 and st["down"] == 1


def test_file_keeps_every_click(tmp_path):
    """Формат JSONL прежний: история кликов в файле не теряется."""
    path = tmp_path / "f.jsonl"
    feedback.record(1, "q", "a", path=path)
    feedback.record(-1, "q", "a", path=path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert [i["rating"] for i in feedback.read(path)] == [1, -1]


def test_different_answers_are_separate_votes(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(1, "q", "ответ 1", path=path)
    feedback.record(-1, "q", "ответ 2", path=path)
    st = feedback.stats(path)
    assert st["up"] == 1 and st["down"] == 1


def test_lines_without_answer_are_not_merged(tmp_path):
    """Старые записи без ответа не склеиваем — их не опознать."""
    path = tmp_path / "f.jsonl"
    for rating in (1, 1, -1):
        feedback.record(rating, "q", path=path)
    assert feedback.stats(path)["total"] == 3


def test_flip_flopping_does_not_trigger_advice(tmp_path):
    """Один ответ, восемь переключений — это не «часто недоволен»."""
    path = tmp_path / "f.jsonl"
    for _ in range(4):
        feedback.record(-1, "q", "a", strategy="answer", sources=3, path=path)
        feedback.record(1, "q", "a", strategy="answer", sources=3, path=path)
    assert feedback.stats(path)["total"] == 1
    assert feedback.advice(path) == ""


def test_last_rating_wins_in_strategy_split(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(-1, "q", "a", strategy="answer", sources=2, path=path)
    feedback.record(1, "q", "a", strategy="answer", sources=2, path=path)
    st = feedback.stats(path)
    assert st["by_strategy"]["answer"] == {"up": 1, "down": 0}
    assert st["with_sources"] == {"up": 1, "down": 0}


# ── /api/feedback ───────────────────────────────────────────
@pytest.mark.parametrize("payload", [
    {"rating": 1},
    {"rating": 1, "question": "q"},
    {"rating": 1, "answer": "a"},
    {"rating": 1, "question": "  ", "answer": "a"},
    None,
])
def test_api_rejects_feedback_without_text(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(main.feedback, "DEFAULT_PATH", tmp_path / "f.jsonl")
    monkeypatch.setattr(main, "_LAST_TURN", {"question": "старый", "answer": "ход"})
    with pytest.raises(HTTPException) as err:
        asyncio.run(main.api_feedback(payload))
    assert err.value.status_code == 400
    assert not (tmp_path / "f.jsonl").exists()


def test_api_records_explicit_turn(tmp_path, monkeypatch):
    path = tmp_path / "f.jsonl"
    monkeypatch.setattr(main.feedback, "DEFAULT_PATH", path)
    monkeypatch.setattr(main, "_trace", lambda *a, **kw: None)
    monkeypatch.setattr(main, "_LAST_TURN", {"question": "q", "answer": "a",
                                             "strategy": "уточни", "sources": 2})
    resp = asyncio.run(main.api_feedback({"rating": -1, "question": "q", "answer": "a"}))
    body = json.loads(resp.body)
    assert body["entry"]["strategy"] == "уточни" and body["entry"]["sources"] == 2
    again = json.loads(asyncio.run(
        main.api_feedback({"rating": 1, "question": "q", "answer": "a"})).body)
    assert again["stats"]["total"] == 1
    assert feedback.stats(path)["up"] == 1 and feedback.stats(path)["down"] == 0
