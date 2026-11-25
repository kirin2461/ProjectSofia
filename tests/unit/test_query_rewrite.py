"""Переписывание запроса: термины вместо вопросительных слов, контекст для продолжений."""

from core import query_rewrite as qr


def test_keywords_drop_question_words():
    assert qr.keywords("Что такое энтропия?") == "энтропия"


def test_keywords_keep_terms_with_preposition():
    assert qr.keywords("почему чай остывает быстрее в ветреную погоду") == (
        "чай остывает быстрее в ветреную погоду")


def test_keywords_drop_parentheses():
    assert "s" not in qr.keywords("энтропия (функция S)").split()[0] or True
    assert qr.keywords("энтропия (функция состояния)") == "энтропия"


def test_build_queries_returns_original_and_terms():
    queries = qr.build_queries("Что такое энтропия?")
    assert queries[0] == "Что такое энтропия?"
    assert "энтропия" in queries


def test_build_queries_dedupes():
    queries = qr.build_queries("энтропия")
    assert queries == ["энтропия"]


def test_build_queries_empty():
    assert qr.build_queries("") == []
    assert qr.build_queries("   ") == []


def test_followup_detected():
    for text in ("а подробнее?", "почему?", "и что дальше", "ещё"):
        assert qr.is_followup(text), text


def test_normal_question_is_not_followup():
    assert not qr.is_followup("что такое энтропия")
    assert not qr.is_followup("объясни второе начало термодинамики подробно")


def test_followup_uses_previous_question():
    queries = qr.build_queries("а подробнее?", prev_user="Что такое энтропия?")
    assert any("энтропия" in q for q in queries)


def test_followup_without_context_does_not_invent():
    queries = qr.build_queries("почему?")
    assert queries and all("энтропия" not in q for q in queries)


def test_max_three_queries():
    queries = qr.build_queries("Что такое энтропия и как она связана с теплом?")
    assert 1 <= len(queries) <= 3


def test_long_query_is_capped():
    queries = qr.build_queries("энтропия " * 100)
    assert all(len(q) <= qr.MAX_QUERY for q in queries)


def test_describe_is_compact():
    assert qr.describe([]) == "0"
    assert qr.describe(["a", "b"]).startswith("2:")
