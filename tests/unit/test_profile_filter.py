"""Фильтр профиля: служебные строки экстрактора не должны попадать в промпт.

Строки взяты дословно из data/profile.json (11 фактов, все source="extractor").
Файл не переписываем: профиль фильтрует их при чтении.
"""

import json
from pathlib import Path

import pytest

from memory import profile as prof

REAL_JUNK = [
    "**Цель:** Получить рекомендацию по направлению обучения.",
    '**Суть запроса:** Просьба предоставить информацию о "Беишарском полигоне".',
    "**Цель:** Узнать, что это за объект.",
    '**Вопрос:** "Привет, как аиби (aibi) работает на деле?"',
    '**Тема:** Принцип работы "айби" (вероятно, имеется в виду какой-то конкретный AI/система).',
    '**Вопрос:** "Что сейчас происходит?" (или "Что происходит?")',
    "Ключевые факты:",
    "**Запрос:** Курс доллара к рублю.",
    "**Действие:** Поиск (Google).",
    "**Статус:** Требуется актуальная информация.",
    "**Имя:** (Не указано пользователем)",
]


def junk_profile_file(tmp_path, facts=None):
    """Копия структуры реального профиля: 11 мусорных фактов от экстрактора."""
    items = facts if facts is not None else REAL_JUNK
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "facts": [
            {"text": t, "confidence": 0.55, "source": "extractor", "key": ""}
            for t in items
        ]
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ── реальные строки ─────────────────────────────────────────
@pytest.mark.parametrize("text", REAL_JUNK)
def test_every_real_extractor_line_is_rejected(text):
    assert prof.is_useful(text) is False, text


def test_no_real_junk_reaches_the_prompt(tmp_path):
    path = junk_profile_file(tmp_path)
    p = prof.Profile(path)

    assert p.facts() == []
    assert p.as_prompt(8) == ""
    assert p.as_prompt(8, query="что такое энтропия") == ""


def test_loading_does_not_rewrite_the_file(tmp_path):
    path = junk_profile_file(tmp_path)
    before = path.read_text(encoding="utf-8")

    prof.Profile(path)

    assert path.read_text(encoding="utf-8") == before


def test_add_refuses_the_same_junk(tmp_path):
    p = prof.Profile(tmp_path / "p.json")
    for text in REAL_JUNK:
        assert p.add(text, source="extractor") is False
    assert len(p) == 0


# ── правила фильтра ─────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Ключевые факты:",
    "Профиль:",
    "**Тема:**",
    "Факты о пользователе :",
    "Цель:** Получить рекомендацию",       # main.lstrip("*") съедает открывающие **
])
def test_headers_without_value_are_rejected(text):
    assert prof.is_useful(text) is False


@pytest.mark.parametrize("text", [
    '**Вопрос:** "Что сейчас происходит?"',
    '"Что сейчас происходит?"',
    "Что происходит?)",
    "«Что происходит?»",
    "Пользователь любит физику? наверное",
])
def test_questions_with_quotes_and_brackets_are_rejected(text):
    assert prof.is_useful(text) is False


@pytest.mark.parametrize("text", [
    "(Не указано пользователем)",
    "**Имя:** (Не указано пользователем)",
    "Имя: не указано",
    "Профессия: (не указана)",
    "Возраст: —",
])
def test_placeholder_values_are_rejected(text):
    assert prof.is_useful(text) is False


@pytest.mark.parametrize("text", [
    "Пользователя зовут Алексей",
    "Пользователь работает инженером-физиком в Новосибирске",
    "**Профессия:** инженер-физик",
    "Пользователь изучает физику: термодинамику и оптику",
    "Любимая книга пользователя — «Курс общей физики» Сивухина",
])
def test_real_facts_survive(text):
    assert prof.is_useful(text) is True, text


# ── правдоподобная смесь ────────────────────────────────────
def test_clean_facts_survive_next_to_noise(tmp_path):
    path = junk_profile_file(tmp_path, REAL_JUNK + [
        "Пользователя зовут Алексей",
        "Пользователь пишет диссертацию по оптике",
    ])
    p = prof.Profile(path)

    prompt = p.as_prompt(8)
    assert "Алексей" in prompt
    assert "оптике" in prompt
    assert "Ключевые факты" not in prompt
    assert "(Не указано" not in prompt
    assert "Цель" not in prompt


def test_noisy_but_useful_line_is_not_dropped(tmp_path):
    """Строка со значением — факт, даже если у неё есть метка."""
    path = junk_profile_file(tmp_path, ["Профессия: инженер-физик"])
    p = prof.Profile(path)
    assert [f.text for f in p.facts()] == ["Профессия: инженер-физик"]


def test_on_disk_file_has_no_usable_extractor_facts():
    """Сторож на живом файле: известные служебные строки отсеиваются."""
    path = Path(__file__).resolve().parents[2] / "data" / "profile.json"
    if not path.exists():
        pytest.skip("нет data/profile.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("facts") if isinstance(raw, dict) else raw
    texts = [str(item.get("text") or "") for item in items or []]
    for text in texts:
        if text in REAL_JUNK:
            assert prof.is_useful(text) is False, text

    p = prof.Profile(path)
    prompt = p.as_prompt(8, query="что такое энтропия")
    for marker in ("Ключевые факты", "(Не указано", "**Цель:**", "**Вопрос:**"):
        assert marker not in prompt
