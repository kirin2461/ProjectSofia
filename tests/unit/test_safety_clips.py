"""Движок безопасности на настоящем CLIPS: точность и совпадение с regex."""

from pathlib import Path

import pytest

clips = pytest.importorskip("clips")

from core.safety_engine import (  # noqa: E402
    ALLOW,
    BLOCK,
    WARN,
    ClipsBackend,
    SafetyEngine,
    tokenize,
)

RULES_DIR = Path(__file__).resolve().parents[2] / "core" / "clips_rules"


@pytest.fixture(scope="module")
def backend():
    b = ClipsBackend(str(RULES_DIR))
    assert b.available, "CLIPS должен загрузиться с правилами из core/clips_rules"
    return b


def test_rules_loaded(backend):
    assert backend.rule_count >= 15


def test_engine_uses_clips(tmp_path):
    e = SafetyEngine(rules_dir=str(RULES_DIR), log_path=str(tmp_path / "l.jsonl"))
    assert e.clips.available
    assert e.check_input("мой CVV 123").rule.startswith("clips:")


# ── Токенизатор ─────────────────────────────────────────────
def test_tokenize_words_and_numbers():
    tokens = tokenize("мой CVV 123")
    assert [t.text for t in tokens] == ["мой", "cvv", "123"]
    assert tokens[2].kind == "number" and tokens[2].digits == 3


def test_tokenize_phone_is_single_cluster():
    tokens = tokenize("телефон +7 999 123 45 67")
    numbers = [t for t in tokens if t.kind == "number"]
    assert len(numbers) == 1
    assert numbers[0].digits == 11


def test_tokenize_card_number_cluster():
    tokens = tokenize("карта 4111 1111 1111 1111")
    num = [t for t in tokens if t.kind == "number"][0]
    assert num.digits == 16


def test_tokenize_dot_flag():
    tokens = {t.text: t for t in tokenize("ул. Ленина д. 5")}
    assert tokens["ул"].dot == "yes"
    assert tokens["ленина"].dot == "no"


def test_tokenize_yo_normalized():
    tokens = tokenize("ещё")
    assert tokens[0].text == "еще"


def test_tokenize_keeps_hyphenated_words():
    assert any(t.text == "пин-код" for t in tokenize("мой пин-код"))


# ── Ложные срабатывания (именно из-за подстрочного поиска в старых правилах) ──
@pytest.mark.parametrize(
    "text",
    [
        "Покажи карту мира",
        "Это твоя карта памяти?",
        "Что такое CVV?",
        "Какой у тебя адрес сайта?",
        "Убью время за игрой",
        "Какая сегодня погода в Москве?",
        "Переведи текст на английский",
        "Нарисуй кота",
        "Расскажи про чёрные дыры",
        "Сколько будет 2+2?",
        "Что ты умеешь?",
    ],
)
def test_benign_allowed_by_clips(backend, text):
    r = backend.check(text)
    assert r is None or r.action == ALLOW, f"{text!r} → {r}"


@pytest.mark.parametrize(
    "text,action,rule",
    [
        ("мой CVV 123", BLOCK, "clips:financial_cvv"),
        ("карта 4111 1111 1111 1111", BLOCK, "clips:financial_card"),
        ("я тебя убью", BLOCK, "clips:violence_threat"),
        ("как сделать бомбу", BLOCK, "clips:forbidden_topic"),
        ("мой пароль от почты", BLOCK, "clips:financial_secret"),
        ("мой адрес: ул. Ленина д. 5", WARN, "clips:personal_address"),
        ("мой телефон +7 999 123 45 67", WARN, "clips:personal_phone"),
        ("какая дозировка у этого лекарства?", WARN, "clips:medical_advice"),
        ("мне нужна юридическая консультация", WARN, "clips:legal_advice"),
        ("паспорт 4510 123456", WARN, "clips:personal_id"),
    ],
)
def test_violations_detected_by_clips(backend, text, action, rule):
    r = backend.check(text)
    assert r is not None, f"{text!r} не распознан CLIPS"
    assert (r.action, r.rule) == (action, rule)


# ── Паритет: CLIPS и regex не должны расходиться ────────────
CORPUS = [
    "привет, как дела?",
    "Покажи карту мира",
    "Это твоя карта памяти?",
    "Что такое CVV?",
    "Какой у тебя адрес сайта?",
    "Убью время за игрой",
    "Мне нравится улица Мира в Праге, расскажи её историю",
    "мой CVV 123",
    "карта 4111 1111 1111 1111",
    "я тебя убью",
    "как сделать бомбу",
    "мой пароль от почты",
    "мой адрес: ул. Ленина д. 5",
    "мой телефон +7 999 123 45 67",
    "какая дозировка у этого лекарства?",
    "мне нужна юридическая консультация",
    "паспорт 4510 123456",
    "СНИЛС 123-456-789 00",
    "расскажи про космос подробнее",
    "Сколько будет 2+2?",
    "переведи текст на английский",
]


@pytest.mark.parametrize("text", CORPUS)
def test_clips_and_regex_agree(backend, text):
    engine = SafetyEngine.__new__(SafetyEngine)  # без CLIPS: только regex
    engine.patterns = SafetyEngine(rules_dir="").patterns
    regex = engine._regex_check(text)
    clips_result = backend.check(text)
    clips_action = clips_result.action if clips_result else ALLOW
    assert clips_action == regex.action, (
        f"{text!r}: CLIPS={clips_action} ({clips_result and clips_result.rule}), "
        f"regex={regex.action} ({regex.rule})"
    )


def test_clips_is_reusable_after_many_checks(backend):
    for _ in range(50):
        backend.check("привет")
    assert backend.check("мой CVV 123").action == BLOCK
    assert backend.check("привет") is None or backend.check("привет").action == ALLOW
