"""Бюджет истории: слот минус ответ минус системная часть промпта.

Замер на живом сервере (типичный ход): персона 220 + сводка 259 + книги 873 +
веб 861 + история 5159 = 7372 токена без вопроса и ответа. Старый бюджет
(6000) такую историю пропускал, хотя в слот 8192 всё вместе не влезает.
"""

import asyncio
import types

import pytest

from config.settings import CONTEXT_BUDGET_TOKENS, NUM_PREDICT
from core import cognitive_cycle
from core import context as ctx


def msg(role, text):
    return {"role": role, "content": text}


# ── Бюджет ─────────────────────────────────────────────────
def test_budget_leaves_room_for_answer_and_system():
    slot = ctx.NUM_CTX
    budget = ctx.budget_tokens()
    assert 0 < budget < slot
    assert slot - budget >= NUM_PREDICT
    assert slot - budget >= ctx.ANSWER_RESERVE_TOKENS + ctx.SYSTEM_RESERVE_TOKENS


def test_budget_for_real_slot_8192():
    budget = ctx.budget_tokens(8192)
    # Восемь килобайт слота: ответ 512 и системная часть ~2.6k не отдаём истории
    assert 4000 <= budget <= 5500


def test_budget_grows_with_slot():
    assert ctx.budget_tokens(16384) - ctx.budget_tokens(8192) == 16384 - 8192


def test_budget_has_floor_for_tiny_slot():
    assert ctx.budget_tokens(2048) == ctx.MIN_BUDGET_TOKENS


def test_explicit_budget_wins(monkeypatch):
    monkeypatch.setattr(ctx, "CONTEXT_BUDGET_TOKENS", 1234)
    assert ctx.budget_tokens(8192) == 1234


def test_measured_turn_is_over_budget():
    """Реальный ход (история 5159 + сводка 259) обязан уйти в сжатие."""
    history_and_summary = 5159 + 259
    assert ctx.budget_tokens(8192) < history_and_summary


def test_compact_triggers_on_measured_history():
    # ~5159 токенов истории на русском: 5159 * 3.0 симв./токен
    long_history = [msg("user", "слово " * 3200)]
    assert ctx.should_compact(long_history, budget=ctx.budget_tokens(8192))


# ── Оценка токенов ─────────────────────────────────────────
def test_estimate_zero_for_empty():
    assert ctx.estimate_tokens("") == 0


def test_estimate_prose_close_to_measured():
    """Персона: 699 символов, замер токенизатором — 214 токенов."""
    tokens = ctx.estimate_tokens(cognitive_cycle.SYSTEM_PERSONA)
    assert 200 <= tokens <= 290


def test_estimate_does_not_understate_dense_text():
    """Цифры, латиница, разметка: старая формула 2.5 симв./токен занижала."""
    dense = '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}'
    assert ctx.estimate_tokens(dense) >= len(dense) / 2.5


def test_estimate_dense_higher_than_prose_per_char():
    """Один и тот же объём: цифры дороже связного русского текста."""
    prose = "внутренняя энергия зависит только от температуры"
    dense = "82,68 96,41 26.11.2026 8192 32768 https://www.cbr.ru/x"
    prose = (prose + " ") * 3  # выравниваем длину
    assert len(dense) <= len(prose)
    assert ctx.estimate_tokens(dense) > ctx.estimate_tokens(prose[:len(dense)])


def test_estimate_grows_with_text():
    assert ctx.estimate_tokens("привет") < ctx.estimate_tokens("привет " * 500)


def test_estimate_never_zero_for_text():
    assert ctx.estimate_tokens(" ") >= 1


@pytest.mark.parametrize("slot", [4096, 8192, 16384, 32768])
def test_budget_fits_slot(slot):
    assert ctx.budget_tokens(slot) + ctx.ANSWER_RESERVE_TOKENS <= slot


# ── Сводка просит столько же, по-рабочему ──────────────────
def test_summary_request_uses_max_tokens(monkeypatch):
    seen = {}

    async def fake_create(**kwargs):
        seen.update(kwargs)
        message = types.SimpleNamespace(content="сводка " * 10)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )

    fake = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(ctx, "_llm_client", lambda: fake)

    asyncio.run(ctx.summarize_history([msg("user", "Обсуждали энтропию")]))

    assert seen["max_tokens"] == ctx.SUMMARY_MAX_TOKENS
    body = seen.get("extra_body") or {}
    assert "num_predict" not in body and "num_ctx" not in body and "think" not in body
    assert body["cache_prompt"] is True
