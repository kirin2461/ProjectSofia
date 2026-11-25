"""Автономный цикл: разбор JSON, бюджет шагов, барьер, повторы."""

import threading

import pytest

from modules import autopilot as ap


class FakeBridge:
    def __init__(self, results=None):
        self.executed = []
        self.url = "https://example.com/"
        self.results = results or {}
        self.stopped = False

    def start(self, url="about:blank"):
        return None

    def current_url(self):
        return self.url

    def title(self):
        return "Пример"

    def read_text(self, max_chars=3000):
        return "Текст страницы с кнопкой Войти"

    def tabs(self):
        return [{"index": 0, "title": "Пример", "url": self.url}]

    def execute(self, action):
        self.executed.append(action)
        kind = action.get("action")
        if kind in self.results:
            return self.results[kind]
        return {"ok": True, "detail": f"сделано {kind}"}


def llm_from(actions):
    """Ответы модели по очереди; дальше — done."""
    queue = list(actions)

    def ask(system, prompt):
        if not queue:
            return '{"action":"done","summary":"всё"}'
        item = queue.pop(0)
        return item if isinstance(item, str) else __import__("json").dumps(item, ensure_ascii=False)

    return ask


def run(actions, **kwargs):
    bridge = FakeBridge(kwargs.pop("results", None))
    events = []
    pilot = ap.Autopilot(bridge=bridge, emitter=events.append,
                         llm=llm_from(actions), **kwargs)
    return pilot.run("найти информацию"), bridge, events, pilot


def test_extract_json_plain():
    assert ap._extract_json('{"action":"click","text":"Войти"}') == {
        "action": "click", "text": "Войти"}


def test_extract_json_from_markdown():
    raw = 'Вот шаг:\n```json\n{"action":"navigate","url":"https://ya.ru"}\n```'
    assert ap._extract_json(raw)["url"] == "https://ya.ru"


def test_extract_json_with_text_around():
    raw = 'Думаю. {"action":"done","summary":"готово"} — вот так.'
    assert ap._extract_json(raw)["action"] == "done"


def test_extract_json_none_on_garbage():
    assert ap._extract_json("извините, не понял") is None
    assert ap._extract_json("") is None


def test_extract_json_handles_braces_in_strings():
    raw = '{"action":"type","field":"x","text":"{сложный} текст"}'
    assert ap._extract_json(raw)["text"] == "{сложный} текст"


def test_loop_executes_and_finishes():
    result, bridge, events, _ = run([
        {"action": "click", "text": "Войти"},
        {"action": "done", "summary": "вошла"},
    ])
    assert result == {"ok": True, "summary": "вошла", "steps": 1}
    assert [a["action"] for a in bridge.executed] == ["click"]
    assert any(e["type"] == "finish" for e in events)


def test_done_on_first_step_has_no_actions():
    result, bridge, _, _ = run([{"action": "done", "summary": "уже открыто"}])
    assert result["ok"] is True and bridge.executed == []


def test_step_budget_stops():
    result, bridge, _, _ = run(
        [{"action": "scroll", "dy": 100 + i} for i in range(10)],
        max_steps=3,
    )
    assert result["ok"] is False
    assert "бюджет" in result["summary"]
    assert len(bridge.executed) == 3


def test_repeat_detection_stops():
    result, bridge, _, _ = run([
        {"action": "click", "text": "Далее"},
        {"action": "click", "text": "Далее"},
        {"action": "click", "text": "Далее"},
    ])
    assert result["ok"] is False
    assert "застряла" in result["summary"]
    assert len(bridge.executed) == 2


def test_non_json_reply_is_retried():
    result, bridge, _, _ = run([
        "не понял задачу",
        {"action": "press", "key": "Enter"},
    ])
    assert result["ok"] is True
    assert [a["action"] for a in bridge.executed] == ["press"]


def test_ask_action_pauses_for_human():
    result, bridge, events, _ = run([{"action": "ask", "summary": "нужен вход"}])
    assert result["ok"] is False
    assert result["summary"] == "нужен вход"
    assert any(e["type"] == "ask" for e in events)
    assert bridge.executed == []


def test_money_action_asks_confirmation_and_denies(monkeypatch):
    monkeypatch.setattr(ap.settings, "AUTOPILOT_BARRIER", True)
    bridge = FakeBridge()
    events = []
    pilot = ap.Autopilot(
        bridge=bridge, emitter=events.append,
        llm=llm_from([{"action": "click", "text": "Купить"}]), max_steps=3,
    )
    pilot.confirm_timeout = 0.2  # не ждём человека в тесте

    def decline(event):
        events.append(event)
        if event["type"] == "confirm":
            threading.Thread(target=pilot.confirm, args=(False,), daemon=True).start()

    pilot.emit = decline
    result = pilot.run("купить книгу")

    assert result["ok"] is False
    assert "не подтверждено" in result["summary"]
    assert bridge.executed == []
    assert any(e["type"] == "declined" for e in events)


def test_money_action_runs_when_confirmed(monkeypatch):
    monkeypatch.setattr(ap.settings, "AUTOPILOT_BARRIER", True)
    bridge = FakeBridge()
    pilot = ap.Autopilot(
        bridge=bridge,
        llm=llm_from([{"action": "click", "text": "Купить"},
                      {"action": "done", "summary": "куплено"}]), max_steps=3,
    )

    def approve(event):
        if event["type"] == "confirm":
            threading.Thread(target=pilot.confirm, args=(True,), daemon=True).start()

    pilot.emit = approve
    result = pilot.run("купить книгу")
    assert result["ok"] is True
    assert bridge.executed[0]["text"] == "Купить"


def test_barrier_disabled_by_setting(monkeypatch):
    monkeypatch.setattr(ap.settings, "AUTOPILOT_BARRIER", False)
    result, bridge, events, _ = run([
        {"action": "click", "text": "Купить"},
        {"action": "done", "summary": "ок"},
    ])
    assert result["ok"] is True
    assert not any(e["type"] == "confirm" for e in events)


def test_stop_from_event_ends_loop():
    bridge = FakeBridge()
    pilot = ap.Autopilot(bridge=bridge, llm=llm_from([{"action": "scroll"}]))
    pilot.emit = lambda e: pilot.stop() if e["type"] == "start" else None
    result = pilot.run("что-то")
    assert result["ok"] is False
    assert "остановлено" in result["summary"]
    assert bridge.executed == []


def test_browser_error_is_reported(monkeypatch):
    bridge = FakeBridge()

    def boom(*args, **kwargs):
        from modules.browser_agent import BrowserError
        raise BrowserError("Chrome не найден")

    bridge.start = boom
    pilot = ap.Autopilot(bridge=bridge, llm=llm_from([]))
    result = pilot.run("открыть сайт")
    assert result["ok"] is False
    assert "Chrome не найден" in result["summary"]


def test_empty_goal_does_nothing():
    bridge = FakeBridge()
    pilot = ap.Autopilot(bridge=bridge, llm=llm_from([]))
    assert pilot.run("   ")["ok"] is False
    assert bridge.executed == []
