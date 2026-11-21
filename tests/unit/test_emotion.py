"""Юнит-тесты PAD-движка эмоций (Phase 4-02). Без GPU/LLM."""

import math
import time

import pytest

from core.emotion_engine import EmotionEngine, EMOTIONS


def test_initial_neutral():
    e = EmotionEngine()
    assert e.P == e.A == e.D == 0.0
    assert e.dominant_emotion() == "нейтрально"


def test_update_from_text_joy():
    e = EmotionEngine()
    emotion = e.update_from_text("Спасибо, это здорово!")
    assert emotion == "радость"
    assert e.P > 0
    assert e.A > 0


def test_update_from_text_insult_raises_arousal_drops_pleasure():
    e = EmotionEngine()
    e.update_from_text("Ты дурак и идиот")
    assert e.P < 0
    assert e.A > 0
    assert e.dominant_emotion() == "гнев"


def test_update_from_text_neutral_returns_none():
    e = EmotionEngine()
    assert e.update_from_text("Сегодня вторник") is None
    assert e.P == 0.0


def test_update_from_event():
    e = EmotionEngine()
    e.update_from_event("hp_low")
    assert e.P < 0
    assert e.A > 0
    e2 = EmotionEngine()
    e2.update_from_event("победа")
    assert e2.P > 0 and e2.D > 0


def test_decay_returns_to_baseline():
    e = EmotionEngine(decay_lambda=0.0003)
    e.set_pad(0.8, 0.6, 0.4)
    # ~10 часов
    e.decay(dt_seconds=36000)
    assert abs(e.P) < 0.01
    assert abs(e.A) < 0.01
    assert abs(e.D) < 0.01


def test_decay_math_matches_formula():
    lam = 0.001
    e = EmotionEngine(decay_lambda=lam)
    e.set_pad(1.0, 0.0, 0.0)
    dt = 600
    e.decay(dt_seconds=dt)
    expected = 1.0 * math.exp(-lam * dt) + 0.0 * (1 - math.exp(-lam * dt))
    assert e.P == pytest.approx(expected, abs=1e-6)


def test_decay_zero_dt_no_change():
    e = EmotionEngine()
    e.set_pad(0.5, 0.5, 0.5)
    e.decay(dt_seconds=0)
    assert e.P == 0.5


def test_prompt_modifier_energetic():
    e = EmotionEngine()
    e.set_pad(0.0, 0.9, 0.0)
    mod = e.to_prompt_modifier()
    assert mod["temperature"] > 0.7
    assert "энергично" in mod["tone_instruction"]


def test_prompt_modifier_compassion():
    e = EmotionEngine()
    e.set_pad(-0.9, 0.0, 0.0)
    assert "сочувственно" in e.to_prompt_modifier()["tone_instruction"]


def test_prompt_modifier_confident():
    e = EmotionEngine()
    e.set_pad(0.0, 0.0, 0.8)
    assert "уверенно" in e.to_prompt_modifier()["tone_instruction"]


def test_clamping():
    e = EmotionEngine()
    e.set_pad(5.0, -5.0, 2.0)
    assert e.P == 1.0 and e.A == -1.0 and e.D == 1.0


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "emo.json"
    e = EmotionEngine(decay_lambda=0.0)
    e.set_pad(0.3, 0.4, 0.5)
    e.save(str(path))

    e2 = EmotionEngine(decay_lambda=0.0)
    e2.load(str(path))
    assert e2.P == pytest.approx(0.3)
    assert e2.A == pytest.approx(0.4)
    assert e2.D == pytest.approx(0.5)


def test_all_eight_emotions_defined():
    expected = {"радость", "грусть", "гнев", "страх",
                "доверие", "отвращение", "удивление", "ожидание"}
    assert expected.issubset(EMOTIONS.keys())


def test_to_dict_shape():
    d = EmotionEngine().to_dict()
    assert set(d) == {"P", "A", "D", "dominant_emotion", "confidence"}
