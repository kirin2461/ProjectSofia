"""Сжатие контекста: сводка вместо скользящего окна."""

import asyncio
import collections
import types

import pytest

from core import context as ctx


class FakeSession:
    def __init__(self, messages, summary="", max_history=10):
        self.history = collections.deque(messages, maxlen=max_history)
        self.summary = summary
        self.compactions = 0
        self.max_history = max_history


def msg(role, text):
    return {"role": role, "content": text}


def dialog(n):
    return [msg("user", f"u{i}") if i % 2 == 0 else msg("assistant", f"a{i}")
            for i in range(n)]


def test_estimate_tokens_zero_for_empty():
    assert ctx.estimate_tokens("") == 0


def test_estimate_tokens_grows_with_text():
    assert ctx.estimate_tokens("привет") < ctx.estimate_tokens("привет " * 500)


def test_should_compact_by_message_count():
    assert ctx.should_compact(dialog(8))


def test_should_compact_by_budget():
    long_messages = [msg("user", "слово " * 400) for _ in range(3)]
    assert ctx.should_compact(long_messages, budget=200)


def test_should_not_compact_short_dialog():
    assert not ctx.should_compact([msg("user", "привет")], budget=5000)


def test_budget_tokens_positive():
    assert ctx.budget_tokens() >= 2000


def test_heuristic_summary_marks_roles():
    text = ctx.heuristic_summary([
        msg("user", "Меня зовут Аня"),
        msg("assistant", "Привет, Аня"),
    ])
    assert "Пользователь: Меня зовут Аня" in text
    assert "София: Привет, Аня" in text


def test_heuristic_summary_respects_limit():
    text = ctx.heuristic_summary([msg("user", "а" * 2000)], limit=100)
    assert len(text) <= 101


def test_compact_keeps_tail_and_sets_summary(monkeypatch):
    async def fake_summary(messages, previous="", timeout=0):
        return "Сводка: обсуждали физику"

    monkeypatch.setattr(ctx, "summarize_history", fake_summary)
    session = FakeSession(dialog(8))

    stat = asyncio.run(ctx.maybe_compact(session))

    assert stat["compacted"] is True
    assert [m["content"] for m in session.history] == ["u4", "a5", "u6", "a7"]
    assert session.summary.startswith("Сводка")
    assert session.compactions == 1
    assert stat["saved"] > 0


def test_compact_is_noop_on_short_dialog(monkeypatch):
    called = []

    async def fake_summary(*args, **kwargs):
        called.append(1)
        return "x"

    monkeypatch.setattr(ctx, "summarize_history", fake_summary)
    session = FakeSession([msg("user", "привет")])

    stat = asyncio.run(ctx.maybe_compact(session))

    assert stat["compacted"] is False
    assert called == []
    assert session.summary == ""


def test_compact_merges_previous_summary(monkeypatch):
    seen = {}

    async def fake_summary(messages, previous="", timeout=0):
        seen["previous"] = previous
        return "новая сводка"

    monkeypatch.setattr(ctx, "summarize_history", fake_summary)
    session = FakeSession(dialog(8), summary="старая сводка")

    asyncio.run(ctx.maybe_compact(session))

    assert seen["previous"] == "старая сводка"
    assert session.summary == "новая сводка"


def test_compact_does_not_exceed_max_history(monkeypatch):
    async def fake_summary(messages, previous="", timeout=0):
        return "сводка"

    monkeypatch.setattr(ctx, "summarize_history", fake_summary)
    session = FakeSession(dialog(8), max_history=6)
    asyncio.run(ctx.maybe_compact(session))
    assert len(session.history) <= 6


def test_summarizer_falls_back_when_llm_fails(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("сервер недоступен")

    fake = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=boom)
        )
    )
    monkeypatch.setattr(ctx, "_llm_client", lambda: fake)

    text = asyncio.run(ctx.summarize_history([msg("user", "Меня зовут Аня")]))

    assert "Аня" in text
    assert "Пользователь" in text


def test_short_llm_answer_uses_fallback(monkeypatch):
    async def too_short(*args, **kwargs):
        message = types.SimpleNamespace(content="ок")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    fake = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=too_short)
        )
    )
    monkeypatch.setattr(ctx, "_llm_client", lambda: fake)

    text = asyncio.run(ctx.summarize_history([msg("user", "Обсуждали энтропию")]))

    assert "энтропию" in text


def test_summary_prompt_includes_previous():
    prompt = ctx._summary_prompt([msg("user", "привет")], "ранее говорили о физике")
    assert "ранее говорили о физике" in prompt
    assert "Пользователь: привет" in prompt
