"""Оценки ответов: запись, статистика и подсказка для промпта."""

import json

from memory import feedback


def test_record_and_read_roundtrip(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(1, "что такое энтропия?", "энтропия — функция состояния",
                    strategy="answer", sources=3, path=path)
    items = feedback.read(path)
    assert len(items) == 1
    assert items[0]["rating"] == 1
    assert items[0]["sources"] == 3
    assert items[0]["question"].startswith("что такое")


def test_rating_is_normalized(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(5, "q", path=path)
    feedback.record(-7, "q", path=path)
    feedback.record(0, "q", path=path)
    assert [i["rating"] for i in feedback.read(path)] == [1, -1, 0]


def test_read_missing_file_is_empty(tmp_path):
    assert feedback.read(tmp_path / "нет.jsonl") == []


def test_broken_lines_are_skipped(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"rating": 1}\nне json\n\n{"rating": -1}\n', encoding="utf-8")
    assert [i["rating"] for i in feedback.read(path)] == [1, -1]


def test_record_creates_parent_dir(tmp_path):
    path = tmp_path / "a" / "b" / "f.jsonl"
    feedback.record(1, "q", path=path)
    assert path.exists()


def test_record_survives_unwritable_path(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("занято", encoding="utf-8")
    # путь-файл вместо каталога: запись не должна бросать исключение
    entry = feedback.record(1, "q", path=target / "f.jsonl")
    assert entry["rating"] == 1


def test_text_is_truncated(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(1, "в" * 1000, "о" * 1000, note="п" * 1000, path=path)
    item = feedback.read(path)[0]
    assert len(item["question"]) == 400
    assert len(item["answer"]) == 400
    assert len(item["note"]) == 200


def test_stats_counts(tmp_path):
    path = tmp_path / "f.jsonl"
    for rating in (1, 1, -1, 1):
        feedback.record(rating, "q", strategy="answer", path=path)
    st = feedback.stats(path)
    assert st["up"] == 3 and st["down"] == 1 and st["total"] == 4
    assert st["good_rate"] == 0.75
    assert st["by_strategy"]["answer"] == {"up": 3, "down": 1}


def test_stats_empty(tmp_path):
    st = feedback.stats(tmp_path / "нет.jsonl")
    assert st["total"] == 0 and st["good_rate"] == 0.0


def test_stats_splits_by_sources(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(-1, "q", sources=3, path=path)
    feedback.record(1, "q", sources=0, path=path)
    st = feedback.stats(path)
    assert st["with_sources"] == {"up": 0, "down": 1}


def test_advice_empty_without_data(tmp_path):
    assert feedback.advice(tmp_path / "нет.jsonl") == ""


def test_advice_silent_when_ratings_positive(tmp_path):
    path = tmp_path / "f.jsonl"
    for _ in range(6):
        feedback.record(1, "q", path=path)
    assert feedback.advice(path) == ""


def test_advice_mentions_citations_when_book_answers_fail(tmp_path):
    path = tmp_path / "f.jsonl"
    for _ in range(5):
        feedback.record(-1, "q", strategy="answer", sources=3, path=path)
    advice = feedback.advice(path)
    assert "цитируй" in advice.lower()


def test_advice_short_for_greetings(tmp_path):
    path = tmp_path / "f.jsonl"
    for _ in range(5):
        feedback.record(-1, "привет", strategy="op_greet", sources=0, path=path)
    assert "короче" in feedback.advice(path)


def test_advice_needs_minimum_volume(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(-1, "q", path=path)
    feedback.record(-1, "q", path=path)
    assert feedback.advice(path) == ""


def test_describe_formats_counts(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(1, "q", path=path)
    feedback.record(-1, "q", path=path)
    text = feedback.describe(path=path)
    assert "👍 1" in text and "50%" in text
    assert feedback.describe(path=tmp_path / "нет.jsonl") == "оценок пока нет"


def test_records_are_valid_jsonl(tmp_path):
    path = tmp_path / "f.jsonl"
    feedback.record(1, "q", path=path)
    feedback.record(-1, "q", path=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) for line in lines)
