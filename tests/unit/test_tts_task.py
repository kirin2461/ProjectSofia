"""Озвучка — задача сессии: новый ответ отменяет старую, отмена не рвёт кадр."""

import asyncio
import types

import main


class FakeWS:
    def __init__(self, gate=None):
        self.frames = []
        self.gate = gate

    async def send_bytes(self, data):
        if self.gate is not None:
            await self.gate.wait()
        self.frames.append(data)


def synth_echo(chunk):
    return chunk.encode("utf-8")


def long_text(tag, n=40):
    return " ".join(f"{tag} предложение {i}." for i in range(n))


def test_new_reply_cancels_previous_speech(monkeypatch):
    def slow(chunk):
        import time
        time.sleep(0.02)
        return chunk.encode("utf-8")

    monkeypatch.setattr(main, "tts_engine", types.SimpleNamespace(synthesize=slow))
    session = main.SessionState("s")
    ws = FakeWS()

    async def scenario():
        first = main._start_tts(session, ws, long_text("Первое"))
        await asyncio.sleep(0.03)
        second = main._start_tts(session, ws, "Второй ответ.")
        await second
        await asyncio.sleep(0.05)
        return first, second

    first, second = asyncio.run(scenario())
    assert first.cancelled()
    assert second.done() and not second.cancelled()
    assert session.tts_task is second
    spoken = b"".join(ws.frames).decode("utf-8")
    assert "Второй ответ." in spoken
    assert spoken.count("Первое") < 40


def test_finished_speech_is_not_reported_as_cancelled(monkeypatch):
    monkeypatch.setattr(main, "tts_engine", types.SimpleNamespace(synthesize=synth_echo))
    session = main.SessionState("s")
    ws = FakeWS()

    async def scenario():
        await main._start_tts(session, ws, "Коротко.")
        return main._cancel_tts(session)

    assert asyncio.run(scenario()) is False
    assert ws.frames == [b"\xd0\x9a\xd0\xbe\xd1\x80\xd0\xbe\xd1\x82\xd0\xba\xd0\xbe."]


def test_cancel_during_send_finishes_the_frame(monkeypatch):
    """Отмена посреди send_bytes: кадр дописан целиком, следующих нет."""
    monkeypatch.setattr(main, "tts_engine", types.SimpleNamespace(synthesize=synth_echo))
    session = main.SessionState("s")

    async def scenario():
        gate = asyncio.Event()
        ws = FakeWS(gate)
        task = main._start_tts(session, ws, long_text("Кадр", 10))
        await asyncio.sleep(0.05)          # первый кадр уже ждёт в send_bytes
        assert main._cancel_tts(session) is True
        gate.set()
        await asyncio.sleep(0.05)
        return ws, task

    ws, task = asyncio.run(scenario())
    assert task.cancelled()
    assert len(ws.frames) == 1
    assert ws.frames[0].decode("utf-8").startswith("Кадр предложение 0.")


def test_send_errors_do_not_escape(monkeypatch):
    class DeadWS:
        async def send_bytes(self, data):
            raise RuntimeError("соединение закрыто")

    monkeypatch.setattr(main, "tts_engine", types.SimpleNamespace(synthesize=synth_echo))
    session = main.SessionState("s")

    async def scenario():
        task = main._start_tts(session, DeadWS(), "Раз. Два.")
        await task
        return task

    task = asyncio.run(scenario())
    assert task.done() and task.exception() is None


def test_cancel_without_task_is_noop():
    session = main.SessionState("s")
    assert main._cancel_tts(session) is False
    assert main._cancel_tts(None) is False
