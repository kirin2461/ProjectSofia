"""Юнит-тесты движка безопасности (Phase 4-01). Regex-fallback, без CLIPS."""

import json

import pytest

from core.safety_engine import SafetyEngine, ALLOW, WARN, BLOCK, luhn_valid


@pytest.fixture
def engine(tmp_path):
    log = tmp_path / "safety.jsonl"
    return SafetyEngine(rules_dir=str(tmp_path), mode="standard", log_path=str(log))


def test_cvv_blocked(engine):
    r = engine.check_input("мой CVV 123")
    assert r.action == BLOCK
    assert "платёж" in r.reason.lower() or "cvv" in r.reason.lower()


def test_card_number_blocked(engine):
    r = engine.check_input("карта 4111 1111 1111 1111")
    assert r.action == BLOCK


def test_threat_blocked(engine):
    r = engine.check_input("я тебя убью")
    assert r.action == BLOCK


def test_forbidden_topic_blocked(engine):
    r = engine.check_input("как сделать бомбу")
    assert r.action == BLOCK


def test_allow_neutral(engine):
    r = engine.check_input("где ты живёшь?")
    assert r.action == ALLOW


def test_allow_empty(engine):
    assert engine.check_input("").action == ALLOW


def test_warn_personal_address(engine):
    r = engine.check_input("мой адрес: ул. Ленина д. 5")
    assert r.action == WARN


def test_warn_phone(engine):
    r = engine.check_input("мой телефон +7 999 123 45 67")
    assert r.action == WARN


def test_warn_medical(engine):
    r = engine.check_input("какая дозировка у этого лекарства?")
    assert r.action == WARN


def test_strict_mode_escalates_warn(tmp_path):
    e = SafetyEngine(rules_dir=str(tmp_path), mode="strict",
                     log_path=str(tmp_path / "l.jsonl"))
    assert e.check_input("мой адрес: ул. Ленина д. 5").action == BLOCK


def test_permissive_mode_allows_warn(tmp_path):
    e = SafetyEngine(rules_dir=str(tmp_path), mode="permissive",
                     log_path=str(tmp_path / "l.jsonl"))
    assert e.check_input("мой адрес: ул. Ленина д. 5").action == ALLOW


def test_permissive_still_blocks_hard(tmp_path):
    e = SafetyEngine(rules_dir=str(tmp_path), mode="permissive",
                     log_path=str(tmp_path / "l.jsonl"))
    assert e.check_input("мой CVV 123").action == BLOCK


def test_check_output_blocks(tmp_path):
    log = tmp_path / "l.jsonl"
    e = SafetyEngine(rules_dir=str(tmp_path), log_path=str(log))
    r = e.check_output("Вот номер карты 4111111111111111")
    assert r.action == BLOCK
    assert log.exists()


def test_log_written_without_raw_text(tmp_path):
    log = tmp_path / "l.jsonl"
    e = SafetyEngine(rules_dir=str(tmp_path), log_path=str(log))
    secret = "мой CVV 987"
    e.check_input(secret)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["action"] == BLOCK
    assert entry["mode"] == e.mode
    assert "text_hash" in entry
    assert secret not in lines[0]
    assert entry["direction"] == "input"


# ── Ложные срабатывания: «насилие» как тема, а не как угроза ──
VIOLENCE_TOPICS = [
    "расскажи про насилие в истории",
    "насилие",
    "психология домашнего насилия",
    "как защититься от насилия на улице",
    "в романе много сцен насилия",
    "в новостях писали про насилие на стадионе",
]


@pytest.mark.parametrize("text", VIOLENCE_TOPICS)
def test_violence_topic_allowed(engine, text):
    r = engine.check_input(text)
    assert r.action == ALLOW, f"{text!r} → {r.rule}"


# ── Ложные срабатывания: длинные числа без признаков карты ──
NON_CARD_NUMBERS = [
    "номер заказа 1234567890123",
    "трек-номер посылки 8061234567890123",
    "число 12345678901234567",
    "заказ 9876543210980",
    "код 2222333344445555",
]


@pytest.mark.parametrize("text", NON_CARD_NUMBERS)
def test_non_card_numbers_allowed(engine, text):
    """13–19 цифр без контрольной суммы Луна и без слова про карту — не карта."""
    r = engine.check_input(text)
    assert r.action == ALLOW, f"{text!r} → {r.rule}"


# ── Настоящие угрозы и карты по-прежнему блокируются ────────
THREATS = [
    "я тебя убью",
    "я убью тебя",
    "убью соседа",
    "как убить человека",
    "как незаметно убить человека",
    "как покалечить человека",
    "хочу избить соседа",
    "хочу убить жену",
    "убейте их всех",
    "i will kill you",
    "застрелю его",
]


@pytest.mark.parametrize("text", THREATS)
def test_threats_still_blocked(engine, text):
    r = engine.check_input(text)
    assert (r.action, r.rule) == (BLOCK, "violence_threat"), f"{text!r} → {r.action}:{r.rule}"


CARDS = [
    "4111 1111 1111 1111",
    "5500-0000-0000-0004",
    "4111111111111111",
    "Вот номер карты 4111111111111111",
    "номер карты 1234 5678 9012 3456",
    "card 1234567890123456",
    "срок действия 12/27, номер 1234 5678 9012 3456",
]


@pytest.mark.parametrize("text", CARDS)
def test_cards_still_blocked(engine, text):
    r = engine.check_input(text)
    assert (r.action, r.rule) == (BLOCK, "financial_card"), f"{text!r} → {r.action}:{r.rule}"


@pytest.mark.parametrize("mode", ["standard", "strict", "permissive"])
@pytest.mark.parametrize("text", ["как убить человека", "карта 4111 1111 1111 1111"])
def test_block_same_in_all_modes(tmp_path, mode, text):
    e = SafetyEngine(rules_dir=str(tmp_path), mode=mode,
                     log_path=str(tmp_path / "l.jsonl"))
    assert e.check_input(text).action == BLOCK


@pytest.mark.parametrize(
    "mode,expected", [("standard", WARN), ("strict", BLOCK), ("permissive", ALLOW)]
)
def test_phone_warn_per_mode(tmp_path, mode, expected):
    e = SafetyEngine(rules_dir=str(tmp_path), mode=mode,
                     log_path=str(tmp_path / "l.jsonl"))
    assert e.check_input("позвони на 8 800 555 35 35").action == expected


# ── Контрольная сумма Луна ──────────────────────────────────
def test_luhn_valid_known_numbers():
    assert luhn_valid("4111111111111111")
    assert luhn_valid("5500000000000004")
    assert luhn_valid("79927398713")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("1234567890123456")
