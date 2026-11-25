"""Когнитивный цикл: монолог получает подсказку Soar, но не диктует ответ."""

import asyncio
import types

import pytest

from core import cognitive_cycle


class FakeClient:
    """Подмена AsyncOpenAI: запоминает запрос, возвращает заготовку."""

    def __init__(self, content: str = "монолог"):
        self.requests = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )
        self._content = content

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(cognitive_cycle, "client", client)
    return client


def run(coro):
    return asyncio.run(coro)


def prompt_of(client) -> str:
    messages = client.requests[-1]["messages"]
    return "\n".join(m["content"] for m in messages)


def test_strategy_reaches_monologue_prompt(fake):
    hint = "Сообщение непонятно без контекста: задай один короткий уточняющий вопрос."
    run(cognitive_cycle.monologue("факты", [], "как это?", strategy=hint))
    assert hint in prompt_of(fake)


def test_without_strategy_no_line(fake):
    run(cognitive_cycle.monologue("факты", [], "расскажи про космос"))
    assert "Выбранная стратегия" not in prompt_of(fake)


def test_strategy_does_not_become_system_rule(fake):
    """Подсказка живёт в контексте пользователя, а не в системной персоне."""
    run(cognitive_cycle.monologue("", [], "привет", strategy="поздоровайся"))
    system = fake.requests[-1]["messages"][0]
    assert system["role"] == "system"
    assert "поздоровайся" not in system["content"]


def test_history_precedes_context(fake):
    history = [
        {"role": "user", "content": "первая реплика"},
        {"role": "assistant", "content": "первый ответ"},
    ]
    run(cognitive_cycle.monologue("", [], "третья", history=history))
    roles = [m["role"] for m in fake.requests[-1]["messages"]]
    assert roles[:3] == ["system", "user", "assistant"]
    assert "третья" in fake.requests[-1]["messages"][-1]["content"]


def test_failed_extractor_returns_empty(monkeypatch):
    class Boom:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        async def _create(self, **kwargs):
            raise RuntimeError("сервер недоступен")

    monkeypatch.setattr(cognitive_cycle, "client", Boom())
    assert run(cognitive_cycle.extract_facts("привет")) == ""


# ── Интернет: данные идут в хвост промпта, а не в системный блок ──
WEB = "[1] cbr.ru — Курс доллара\nКурс доллара 82,68 рубля на сегодня."


class StreamClient(FakeClient):
    """Подмена AsyncOpenAI со стримингом: два токена в ответе."""

    async def _create(self, **kwargs):
        self.requests.append(kwargs)

        async def tokens():
            for piece in ("ответ", " готов"):
                delta = types.SimpleNamespace(content=piece)
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])

        return tokens()


@pytest.fixture
def stream(monkeypatch):
    client = StreamClient()
    monkeypatch.setattr(cognitive_cycle, "client", client)
    return client


def _collect(**kwargs) -> "asyncio.Future":
    async def take():
        return [t async for t in cognitive_cycle.generate_response(
            "монолог", "вопрос", **kwargs)]
    return take()


def test_web_block_goes_to_user_tail(stream):
    run(_collect(web=WEB))
    messages = stream.requests[-1]["messages"]
    assert WEB in messages[-1]["content"]
    assert "cbr.ru" not in messages[0]["content"]


def test_web_does_not_change_system_prompt(stream):
    run(_collect())
    without = stream.requests[-1]["messages"][0]["content"]
    run(_collect(web=WEB))
    with_web = stream.requests[-1]["messages"][0]["content"]
    assert with_web == without


def test_answer_instruction_asks_to_name_site(stream):
    run(_collect(web=WEB))
    tail = stream.requests[-1]["messages"][-1]["content"]
    assert "источник" in tail.lower()


def test_monologue_sees_web_data(fake):
    run(cognitive_cycle.monologue("", [], "какой курс?", web=WEB))
    assert WEB in prompt_of(fake)


def test_monologue_tells_model_how_to_ask_for_search(fake):
    run(cognitive_cycle.monologue("", [], "какой курс?"))
    assert "ПОИСК:" in prompt_of(fake)


def test_monologue_forbids_pointless_search(fake):
    """Приветствие и вопросы о ней — не повод идти в интернет."""
    run(cognitive_cycle.monologue("", [], "привет, как тебя зовут?"))
    prompt = prompt_of(fake)
    assert "НЕ ищи интернет" in prompt
    assert "2–6 слов" in prompt
