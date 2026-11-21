"""Юнит-тесты движка безопасности (Phase 4-01). Regex-fallback, без CLIPS."""

import json

import pytest

from core.safety_engine import SafetyEngine, ALLOW, WARN, BLOCK


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
