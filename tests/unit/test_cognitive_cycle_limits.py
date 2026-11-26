"""Лимит ответа уходит в create() рабочим полем max_tokens.

llama-server игнорирует и num_predict, и num_ctx: проба с num_predict=5 дала
459 токенов, а max_tokens=5 и n_predict=5 — ровно 5 с finish_reason=length.
Поля think в сборке нет вовсе, а deprecated-варнинг на enable_thinking
приходит из --chat-template-kwargs. Поэтому каждый шаг цикла обязан
передавать max_tokens, а мёртвых полей в запросе быть не должно.
"""

import asyncio
import types

import pytest

from config.settings import NUM_PREDICT
from core import cognitive_cycle

DEAD_FIELDS = ("num_predict", "num_ctx", "think")


class FakeClient:
    """Подмена AsyncOpenAI: запоминает kwargs каждого create()."""

    def __init__(self, content: str = "ок"):
        self.requests = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )
        self._content = content

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        message = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )

    async def _stream(self):
        for piece in ("ответ", " готов"):
            delta = types.SimpleNamespace(content=piece)
            yield types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta)]
            )


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(cognitive_cycle, "client", client)
    return client


def run(coro):
    return asyncio.run(coro)


def answer():
    async def take():
        return [t async for t in cognitive_cycle.generate_response("монолог", "вопрос")]
    return take()


def last(client) -> dict:
    return client.requests[-1]


def extra(request: dict) -> dict:
    return request.get("extra_body") or {}


def all_steps(fake):
    """Прогнать все четыре шага цикла и вернуть их запросы."""
    run(cognitive_cycle.extract_facts("привет"))
    run(cognitive_cycle.monologue("", [], "привет"))
    run(answer())
    run(cognitive_cycle.critic("ответ", "факты"))
    return list(fake.requests)


# ── Свои лимиты у каждой роли ──────────────────────────────
def test_extractor_limits_answer(fake):
    run(cognitive_cycle.extract_facts("привет"))
    assert last(fake)["max_tokens"] == 256


def test_monologue_limits_answer(fake):
    run(cognitive_cycle.monologue("факты", [], "как дела?"))
    assert last(fake)["max_tokens"] == 128


def test_answer_uses_configured_limit(fake):
    run(answer())
    assert last(fake)["max_tokens"] == NUM_PREDICT


def test_critic_limits_answer(fake):
    run(cognitive_cycle.critic("ответ", "факты"))
    assert last(fake)["max_tokens"] == 256


def test_roles_keep_their_own_limits(fake):
    """Монолог короче ответа: это мысль, а не эссе."""
    run(cognitive_cycle.monologue("", [], "привет"))
    monologue_limit = last(fake)["max_tokens"]
    run(cognitive_cycle.extract_facts("привет"))
    extractor_limit = last(fake)["max_tokens"]
    run(answer())
    answer_limit = last(fake)["max_tokens"]
    assert monologue_limit < extractor_limit
    assert extractor_limit <= answer_limit


# ── Мёртвые поля не отправляются ───────────────────────────
def test_no_dead_fields_in_extra_body(fake):
    for request in all_steps(fake):
        body = extra(request)
        for field in DEAD_FIELDS:
            assert field not in body, f"{field} сервер игнорирует"
            assert field not in request, f"{field} сервер игнорирует"


def test_cache_prompt_still_sent(fake):
    """cache_prompt проверен на сервере (cache_n=21) — его не трогаем."""
    for request in all_steps(fake):
        assert extra(request)["cache_prompt"] is True


# ── Стриминг не пострадал ──────────────────────────────────
def test_streaming_answer_returns_tokens(fake):
    assert run(answer()) == ["ответ", " готов"]
    assert last(fake)["stream"] is True
