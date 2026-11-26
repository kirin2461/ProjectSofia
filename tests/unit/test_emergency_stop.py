"""Экстренная остановка: контроллер блокируется, голос глохнет, ответ честный.

Только mock-контроллер и подменённый WebSocket — ни мышь, ни клавиатура
системы не трогаются.
"""

import asyncio
import json
import types

import pytest

import main
from modules.game_control import (
    ActionPlan,
    ActionSafety,
    CognitiveActions,
    GameController,
    MockController,
    ReactiveActions,
)


class FakeWS:
    def __init__(self):
        self.texts = []
        self.frames = []

    async def send_text(self, text):
        self.texts.append(text)

    async def send_bytes(self, data):
        self.frames.append(data)

    def last(self):
        return json.loads(self.texts[-1])


# ── контроллеры ─────────────────────────────────────────────
def test_mock_controller_blocks_actions_after_stop():
    c = MockController()
    c.click(1, 2)
    c.emergency_stop()
    c.click(3, 4)
    c.key_press("w")
    c.move_mouse(5, 6)
    c.type_text("abc")
    assert [a for a, _ in c.actions] == ["click"]
    assert c.stopped is True


def test_mock_controller_resume_unblocks():
    c = MockController()
    c.emergency_stop()
    c.resume()
    c.key_press("e")
    assert c.actions == [("key_press", {"key": "e"})]


class FakeBackend:
    """Подмена pydirectinput: пишет вызовы, ничего не нажимает."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **kw: self.calls.append((name, a))


def make_game_controller():
    gc = GameController.__new__(GameController)
    gc.use_directinput = True
    gc._backend = FakeBackend()
    gc._screen_size = (1920, 1080)
    gc._held_keys = []
    gc._mouse_down = False
    return gc


def test_game_controller_releases_held_input_on_stop():
    gc = make_game_controller()
    gc._held_keys = ["shift", "w"]          # остановка посреди key_combo
    gc._mouse_down = True                   # и посреди drag
    gc.emergency_stop()
    calls = gc._backend.calls
    assert ("keyUp", ("w",)) in calls and ("keyUp", ("shift",)) in calls
    assert ("mouseUp", ()) in calls
    assert gc._held_keys == [] and gc._mouse_down is False


def test_game_controller_ignores_actions_while_stopped():
    gc = make_game_controller()
    gc.emergency_stop()
    gc._backend.calls.clear()
    gc.click(10, 10)
    gc.key_press("e")
    gc.key_combo(["ctrl", "s"])
    gc.drag((0, 0), (5, 5))
    gc.type_text("hi")
    assert gc._backend.calls == []


def test_reactive_stop_stops_controller():
    c = MockController()
    ra = ReactiveActions(c)
    ra.track_target((0, 0, 10, 10))
    ra.emergency_stop()
    assert ra._target is None and c.stopped
    ra.track_target((0, 0, 10, 10))
    assert [a for a, _ in c.actions] == ["move_mouse"]


def test_action_safety_rejects_until_resume():
    s = ActionSafety()
    c = MockController()
    s.emergency_stop(c)
    assert s.validate(ActionPlan("press_key", key="e")) is False
    assert c.stopped
    s.resume(c)
    assert s.validate(ActionPlan("press_key", key="e")) is True
    assert not c.stopped


def test_plan_is_interrupted_by_stop():
    c = MockController()
    ca = CognitiveActions(c)
    plans = ca.parse_plan("press a\npress b\npress c")
    original = ca.execute_plan

    def execute_then_stop(plan):
        original(plan)
        if plan.key == "a":
            c.emergency_stop()

    ca.execute_plan = execute_then_stop
    ca.execute_plans(plans)
    assert [kw["key"] for _, kw in c.actions] == ["a"]


# ── команда по WebSocket ────────────────────────────────────
def test_ws_emergency_stop_blocks_controller(monkeypatch):
    c = MockController()
    monkeypatch.setattr(main, "game", c)
    ws = FakeWS()

    asyncio.run(main._handle_command(ws, {"type": "game", "action": "emergency_stop"}))
    reply = ws.last()
    assert reply["status"] == "stopped" and reply["controller"] is True
    assert c.stopped

    asyncio.run(main._handle_command(ws, {"type": "game", "action": "click", "x": 1, "y": 2}))
    assert ws.last()["status"] == "blocked"
    assert c.actions == []

    asyncio.run(main._handle_command(ws, {"type": "game", "action": "resume"}))
    assert ws.last()["status"] == "resumed"
    asyncio.run(main._handle_command(ws, {"type": "game", "action": "press_key", "key": "e"}))
    assert ws.last()["status"] == "pressed"
    assert c.actions == [("key_press", {"key": "e"})]


def test_ws_emergency_stop_without_game_mode_is_honest(monkeypatch):
    """Игровой режим выключен: не врём, что остановили контроллер."""
    monkeypatch.setattr(main, "game", None)
    ws = FakeWS()
    asyncio.run(main._handle_command(ws, {"type": "game", "action": "emergency_stop"}))
    reply = ws.last()
    assert reply["status"] == "stopped"
    assert reply["controller"] is False
    assert reply["tts_cancelled"] is False


def test_ws_emergency_stop_reports_controller_failure(monkeypatch):
    class Broken(MockController):
        def emergency_stop(self):
            raise RuntimeError("backend умер")

    monkeypatch.setattr(main, "game", Broken())
    ws = FakeWS()
    asyncio.run(main._handle_command(ws, {"type": "game", "action": "emergency_stop"}))
    reply = ws.last()
    assert reply["status"] == "error" and "backend" in reply["error"]


def test_ws_emergency_stop_cancels_speech(monkeypatch):
    monkeypatch.setattr(main, "game", MockController())
    started = []

    def slow_synth(chunk):
        started.append(chunk)
        import time
        time.sleep(0.05)
        return b"wav"

    monkeypatch.setattr(main, "tts_engine", types.SimpleNamespace(synthesize=slow_synth))
    session = main.SessionState("s")
    ws = FakeWS()
    text = " ".join(f"Предложение номер {i}." for i in range(60))

    async def scenario():
        main._start_tts(session, ws, text)
        await asyncio.sleep(0.01)
        await main._handle_command(
            ws, {"type": "game", "action": "emergency_stop"}, session)
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    reply = ws.last()
    assert reply["tts_cancelled"] is True and reply["controller"] is True
    assert session.tts_task is None
    assert len(ws.frames) < len(main._split_for_tts(text))
