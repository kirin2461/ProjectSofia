"""Профиль: факты о пользователе, дедупликация, приоритет релевантности."""

import json

from memory import profile as prof


def make(tmp_path):
    return prof.Profile(tmp_path / "profile.json")


def test_add_and_read(tmp_path):
    p = make(tmp_path)
    assert p.add("Пользователя зовут Алексей") is True
    assert len(p) == 1
    assert "Алексей" in p.as_prompt()


def test_duplicate_is_not_stored_twice(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь работает инженером-физиком")
    p.add("Пользователь работает инженером-физиком")
    assert len(p) == 1


def test_near_duplicate_keeps_more_confident_version(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь увлекается астрономией и телескопами", confidence=0.5)
    p.add("Пользователь увлекается астрономией и телескопами", confidence=0.9)
    assert len(p) == 1
    assert p.facts()[0].confidence == 0.9


def test_same_key_replaces_old_value(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь работает в школе", key="работа", confidence=0.5)
    p.add("Пользователь работает в университете", key="работа", confidence=0.9)
    assert len(p) == 1
    assert "университете" in p.facts()[0].text


def test_not_useful_texts_are_rejected(tmp_path):
    p = make(tmp_path)
    for text in ("", "не знаю", "Спасибо!", "как дела?", "привет"):
        assert p.add(text) is False
    assert len(p) == 0


def test_persistence_roundtrip(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь живёт в Новосибирске", confidence=0.8)
    q = prof.Profile(tmp_path / "profile.json")
    assert len(q) == 1
    assert q.facts()[0].confidence == 0.8


def test_corrupt_file_gives_empty_profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{это не json", encoding="utf-8")
    p = prof.Profile(path)
    assert len(p) == 0
    assert p.add("Пользователь любит физику твёрдого тела") is True
    assert json.loads(path.read_text(encoding="utf-8"))["facts"]


def test_forget_removes_matching_facts(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь любит кошек и собак")
    p.add("Пользователь изучает квантовую механику")
    assert p.forget("кошек") == 1
    assert len(p) == 1 and "квантовую" in p.facts()[0].text


def test_clear(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь пишет диссертацию по оптике")
    p.clear()
    assert len(p) == 0 and p.as_prompt() == ""


def test_relevant_fact_comes_first(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь любит кошек", confidence=0.95)
    p.add("Пользователь изучает термодинамику и энтропию", confidence=0.5)
    top = p.facts(n=1, query="что такое энтропия")
    assert "энтропию" in top[0].text


def test_prompt_lists_facts_as_bullets(tmp_path):
    p = make(tmp_path)
    p.add("Пользователь изучает физику")
    prompt = p.as_prompt()
    assert prompt.startswith("Что известно о пользователе:") and prompt.count("- ") == 1


def test_fact_cap_keeps_most_confident(tmp_path):
    p = make(tmp_path)
    for i in range(prof.MAX_FACTS + 20):
        p.add(f"Пользователь увлекается темой номер {i}", confidence=0.1)
    p.add("Пользователь пишет книгу по термодинамике", confidence=0.99)
    assert len(p) <= prof.MAX_FACTS
    assert any("книгу по термодинамике" in f.text for f in p.facts(n=5))


def test_long_text_rejected(tmp_path):
    p = make(tmp_path)
    assert p.add("я " * 400) is False


def test_similarity_is_direction_agnostic():
    assert prof.similar("физика твёрдого тела", "физика твёрдого тела и оптика") == 1.0
    assert prof.similar("физика", "кулинария") == 0.0
