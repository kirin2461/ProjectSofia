"""Браузерный слой: барьер на опасные действия и разбор команд CDP."""

import json

import pytest

from modules import browser_agent as ba


class FakeWS:
    """Заглушка websocket: отдаёт заготовленные ответы по id."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, timeout=None):
        return json.dumps(self.replies.pop(0))

    def close(self):
        self.closed = True


def conn_with(replies):
    c = ba.CDPConnection("ws://fake")
    ws = FakeWS(replies)
    c._ws = ws  # минуя реальное соединение
    return c, ws


def test_cdp_skips_events_and_returns_result():
    c, ws = conn_with([
        {"method": "Page.loadEventFired", "params": {}},
        {"id": 1, "result": {"value": 42}},
    ])
    assert c.send("Runtime.evaluate", {"expression": "1"}) == {"value": 42}
    assert ws.sent[0]["method"] == "Runtime.evaluate"
    assert ws.sent[0]["id"] == 1


def test_cdp_passes_session_id():
    c, ws = conn_with([{"id": 1, "result": {}}])
    c.send("Input.insertText", {"text": "привет"}, session="S1")
    assert ws.sent[0]["sessionId"] == "S1"


def test_cdp_raises_on_error_reply():
    c, _ = conn_with([{"id": 1, "error": {"message": "No target"}}])
    with pytest.raises(ba.BrowserError, match="No target"):
        c.send("Page.navigate", {"url": "x"})


def test_endpoint_and_profile_paths():
    assert ba.ChromeBridge(port=9333).endpoint.endswith(":9333")
    assert ba.profile_dir("sofia").name == "sofia"
    user = ba.profile_dir("user")
    assert user.name == "User Data"


def test_needs_confirmation_for_payment_url():
    assert ba.needs_confirmation({"action": "navigate", "url": "https://shop/checkout"})
    assert ba.needs_confirmation({"action": "navigate", "url": "https://bank.example/x"})


def test_needs_confirmation_for_buy_button():
    assert ba.needs_confirmation({"action": "click", "text": "Купить сейчас"})
    assert ba.needs_confirmation({"action": "click", "text": "Buy now"})


def test_needs_confirmation_for_sending_message():
    assert ba.needs_confirmation({"action": "click", "text": "Отправить"})
    assert ba.needs_confirmation({"action": "press", "key": "Send"})


def test_confirmation_not_needed_for_reading():
    assert ba.needs_confirmation({"action": "click", "text": "Читать далее"}, "https://wiki") == ""
    assert ba.needs_confirmation({"action": "navigate", "url": "https://ru.wikipedia.org"}) == ""
    assert ba.needs_confirmation({"action": "scroll", "dy": 900}) == ""
    assert ba.needs_confirmation({"action": "type", "field": "поиск", "text": "энтропия"}) == ""


def test_url_wins_over_action_text():
    assert ba.needs_confirmation({"action": "click", "text": "Продолжить"},
                                 "https://shop.example/checkout")


def test_execute_unknown_action():
    bridge = ba.ChromeBridge(port=1)
    res = bridge.execute({"action": "teleport"})
    assert res["ok"] is False and "неизвестное" in res["detail"]


def test_execute_done_is_noop():
    bridge = ba.ChromeBridge(port=1)
    res = bridge.execute({"action": "done", "summary": "готово"})
    assert res == {"ok": True, "detail": "готово"}


def test_execute_reports_browser_error(monkeypatch):
    bridge = ba.ChromeBridge(port=1)

    def boom(*args, **kwargs):
        raise ba.BrowserError("не найдено: Войти")

    monkeypatch.setattr(bridge, "click", boom)
    res = bridge.execute({"action": "click", "text": "Войти"})
    assert res["ok"] is False
    assert "не найдено" in res["detail"]


def test_find_js_escapes_quotes(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    seen = {}

    def fake_eval(expression, session=None):
        seen["js"] = expression
        return None

    monkeypatch.setattr(bridge, "_eval", fake_eval)
    bridge.find('кнопка "кавычки"', "")
    assert '\\"кавычки\\"' in seen["js"]
    assert '"text"' in seen["js"]


def test_navigate_adds_scheme(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    sent = []

    monkeypatch.setattr(bridge, "ensure_page", lambda: "S")
    monkeypatch.setattr(bridge, "_cmd", lambda method, params=None, session=None:
                        sent.append((method, params)) or {})
    monkeypatch.setattr(bridge, "_wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "current_url", lambda: "https://example.com/")
    bridge.navigate("example.com")
    assert sent == [("Page.navigate", {"url": "https://example.com"})]


# ── Жизненный цикл Chrome: подключение к уже запущенному ────
class FakeCDP:
    """Подделка CDPConnection: запоминает команды, ничего не открывает."""

    used: list = []

    def __init__(self, ws_url, timeout=20.0):
        self.ws_url = ws_url
        self.methods = []
        FakeCDP.used.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send(self, method, params=None, session=None):
        self.methods.append(method)
        return {}

    def close(self):
        self.closed = True


def test_start_attaches_when_port_already_listening(monkeypatch):
    """Прошлый Chrome держит порт — не падаем, а подключаемся к нему."""
    bridge = ba.ChromeBridge(port=9333, profile="sofia")
    connected = []
    monkeypatch.setattr(ba.ChromeBridge, "running", lambda self: True)
    monkeypatch.setattr(ba.ChromeBridge, "_connect",
                        lambda self: connected.append(True))

    bridge.start("about:blank")

    assert connected == [True]
    assert bridge._our_process is False      # чужой процесс не наш
    assert bridge._proc is None


def test_start_without_chrome_raises(monkeypatch):
    bridge = ba.ChromeBridge(port=9333, chrome=None)
    bridge.chrome = None
    monkeypatch.setattr(ba.ChromeBridge, "running", lambda self: False)

    with pytest.raises(ba.BrowserError, match="Chrome не найден"):
        bridge.start()


def test_stop_attached_sofia_closes_browser_via_cdp(monkeypatch):
    FakeCDP.used = []
    bridge = ba.ChromeBridge(port=9333, profile="sofia")
    monkeypatch.setattr(ba, "CDPConnection", FakeCDP)
    monkeypatch.setattr(ba, "_http_json",
                        lambda url, timeout=3.0: {"webSocketDebuggerUrl": "ws://x"})
    monkeypatch.setattr(ba.ChromeBridge, "running", lambda self: False)

    bridge.stop_chrome()

    assert [c.ws_url for c in FakeCDP.used] == ["ws://x"]
    assert FakeCDP.used[0].methods == ["Browser.close"]


def test_stop_attached_user_profile_never_closes_their_chrome(monkeypatch):
    """Ваш браузер при профиле user не закрывается — только отцепляемся."""
    bridge = ba.ChromeBridge(port=9333, profile="user")
    bridge._conn = FakeCDP("ws://ours")
    FakeCDP.used = []                       # считаем только новые соединения
    monkeypatch.setattr(ba, "CDPConnection", FakeCDP)
    monkeypatch.setattr(ba.ChromeBridge, "running", lambda self: True)

    bridge.stop_chrome()

    assert FakeCDP.used == []                # никаких Browser.close
    assert bridge._conn is None              # но соединение закрыто


def test_stop_own_process_kills_only_ours(monkeypatch):
    bridge = ba.ChromeBridge(port=9333, profile="user")
    killed = []

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            killed.append(True)

    bridge._proc = Proc()
    bridge._our_process = True
    bridge.stop_chrome()

    assert killed == [True]
    assert bridge._proc is None


# ── Зависший Chrome своего профиля ──────────────────────────
class FakeProc:
    def __init__(self, cmdline, name="chrome.exe"):
        self.info = {"name": name, "cmdline": cmdline}
        self.killed = False

    def kill(self):
        self.killed = True


def _profile_arg(path):
    return f"--user-data-dir={path}"


def test_stale_profile_processes_matches_only_our_profile(monkeypatch, tmp_path):
    ours = FakeProc(["chrome.exe", _profile_arg(tmp_path / "sofia"), "--type=renderer"])
    theirs = FakeProc(["chrome.exe", _profile_arg(tmp_path / "User Data")])
    other_app = FakeProc(["python.exe", _profile_arg(tmp_path / "sofia")], name="python.exe")
    plain = FakeProc(["chrome.exe"])
    import psutil
    monkeypatch.setattr(psutil, "process_iter",
                        lambda attrs=None: [ours, theirs, other_app, plain])

    found = ba._profile_processes(tmp_path / "sofia")

    assert found == [ours]


def test_kill_stale_never_touches_user_profile(monkeypatch, tmp_path):
    """Профиль user — ваш браузер: его процессы не убиваем никогда."""
    calls = []
    monkeypatch.setattr(ba, "_profile_processes", lambda path: calls.append(path) or [])
    bridge = ba.ChromeBridge(port=9333, profile="user")
    assert bridge._kill_stale() == 0
    assert calls == []


def test_kill_stale_kills_leftovers_of_sofia_profile(monkeypatch, tmp_path):
    left = [FakeProc(["chrome.exe"]), FakeProc(["chrome.exe"])]
    monkeypatch.setattr(ba, "_profile_processes", lambda path: left)
    monkeypatch.setattr(ba.time, "sleep", lambda s: None)
    bridge = ba.ChromeBridge(port=9333, profile="sofia")
    assert bridge._kill_stale() == 2
    assert all(p.killed for p in left)


class QuitPopen:
    """Chrome, который сразу вышел: запуск ушёл зависшему экземпляру."""

    launched = 0

    def __init__(self, args, **kw):
        QuitPopen.launched += 1
        self.args = args

    def poll(self):
        return 0


def test_start_retries_once_after_killing_stale_chrome(monkeypatch):
    QuitPopen.launched = 0
    bridge = ba.ChromeBridge(port=9333, profile="sofia", chrome="chrome.exe")
    state = {"up": False}
    killed = []

    def kill_stale(self):
        killed.append(True)
        state["up"] = True          # после уборки новый Chrome отдаёт порт
        return 1

    monkeypatch.setattr(ba.subprocess, "Popen", QuitPopen)
    monkeypatch.setattr(ba.ChromeBridge, "running", lambda self: state["up"] and
                        QuitPopen.launched >= 2)
    monkeypatch.setattr(ba.ChromeBridge, "_kill_stale", kill_stale)
    monkeypatch.setattr(ba.ChromeBridge, "_connect", lambda self: None)
    monkeypatch.setattr(ba.time, "sleep", lambda s: None)

    bridge.start()

    assert killed == [True]
    assert QuitPopen.launched == 2


def test_start_gives_up_with_clear_error_when_nothing_to_clean(monkeypatch):
    QuitPopen.launched = 0
    bridge = ba.ChromeBridge(port=9333, profile="sofia", chrome="chrome.exe")
    monkeypatch.setattr(ba.subprocess, "Popen", QuitPopen)
    monkeypatch.setattr(ba.ChromeBridge, "running", lambda self: False)
    monkeypatch.setattr(ba.ChromeBridge, "_kill_stale", lambda self: 0)
    monkeypatch.setattr(ba.time, "sleep", lambda s: None)

    with pytest.raises(ba.BrowserError, match="порт отладки"):
        bridge.start()
    assert QuitPopen.launched == 1


def test_stop_own_process_waits_for_exit(monkeypatch):
    """Иначе следующий запуск попадёт в ещё живой Chrome и выйдет."""
    bridge = ba.ChromeBridge(port=9333, profile="sofia")
    waited = []

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            waited.append(timeout)
            return 0

    bridge._proc = Proc()
    bridge._our_process = True
    monkeypatch.setattr(ba.ChromeBridge, "_kill_stale", lambda self: 0)
    bridge.stop_chrome()

    assert waited and waited[0] > 0


# ── Клавиши и вкладки поиска ────────────────────────────────
def test_press_enter_sends_text_parameter(monkeypatch):
    """Без поля text Enter — «немая» клавиша: форма в фоне не отправится."""
    bridge = ba.ChromeBridge(port=1)
    sent = []
    monkeypatch.setattr(bridge, "ensure_page", lambda: "S")
    monkeypatch.setattr(bridge, "_cmd", lambda method, params=None, session=None:
                        sent.append((method, params)) or {})

    bridge.press("Enter")

    assert [m for m, _ in sent] == ["Input.dispatchKeyEvent"] * 2
    down, up = sent[0][1], sent[1][1]
    assert down["type"] == "keyDown" and down["text"] == "\r"
    assert down["code"] == "Enter" and down["windowsVirtualKeyCode"] == 13
    assert "text" not in up


def test_press_regular_key_has_no_text(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    sent = []
    monkeypatch.setattr(bridge, "ensure_page", lambda: "S")
    monkeypatch.setattr(bridge, "_cmd", lambda method, params=None, session=None:
                        sent.append((method, params)) or {})

    bridge.press("Escape")

    assert "text" not in sent[0][1]


def test_has_target_reports_missing_tab(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    monkeypatch.setattr(bridge, "_cmd", lambda method, params=None, session=None: {
        "targetInfos": [{"targetId": "a", "type": "page"}]
    })
    assert bridge.has_target("a") is True
    assert bridge.has_target("b") is False


def test_has_target_false_when_chrome_gone(monkeypatch):
    bridge = ba.ChromeBridge(port=1)

    def boom(*args, **kwargs):
        raise ba.BrowserError("нет связи")

    monkeypatch.setattr(bridge, "_cmd", boom)
    assert bridge.has_target("a") is False


def test_cookies_becomes_dict(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    seen = {}

    def fake_cmd(method, params=None, session=None):
        seen["method"] = method
        return {"cookies": [{"name": "SID", "value": "1"}, {"value": "skip"}]}

    monkeypatch.setattr(bridge, "ensure_page", lambda: "S")
    monkeypatch.setattr(bridge, "_cmd", fake_cmd)
    assert bridge.cookies() == {"SID": "1"}
    assert seen["method"] == "Network.getCookies"


def test_focus_emulation_is_enabled_for_session(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    seen = {}
    monkeypatch.setattr(bridge, "ensure_page", lambda: "S")
    monkeypatch.setattr(bridge, "_cmd",
                        lambda method, params=None, session=None:
                        seen.update(method=method, params=params, session=session) or {})
    bridge.focus_emulation(True)
    assert seen["method"] == "Emulation.setFocusEmulationEnabled"
    assert seen["params"] == {"enabled": True}
    assert seen["session"] == "S"


def test_owned_marks_whether_we_launched_chrome():
    bridge = ba.ChromeBridge(port=1)
    assert bridge.owned is False
    bridge._our_process = True
    assert bridge.owned is True


def test_profile_follows_app_data_dir(monkeypatch, tmp_path):
    """Профиль живёт в data приложения, а не внутри _internal сборки:
    иначе вход в Google теряется при пересборке и падает в Program Files."""
    monkeypatch.setenv("SOFIA_HOME", str(tmp_path))
    assert ba.profile_dir("sofia") == tmp_path / "data" / "browser" / "sofia"


def test_screens_dir_follows_app_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SOFIA_HOME", str(tmp_path))
    assert ba.screens_dir() == tmp_path / "data" / "screens"


# ── Пустое окно about:blank ─────────────────────────────────
def _pages_cmd(pages, sent=None):
    def cmd(method, params=None, session=None):
        if sent is not None:
            sent.append((method, params))
        if method == "Target.getTargets":
            return {"targetInfos": [dict(type="page", **p) for p in pages]}
        if method == "Browser.getWindowForTarget":
            return {"windowId": 7}
        return {}
    return cmd


def test_minimize_windows_puts_every_window_down(monkeypatch):
    """Свёрнутое окно не видно — иначе человек смотрит в пустой about:blank."""
    bridge = ba.ChromeBridge(port=1)
    sent = []
    monkeypatch.setattr(bridge, "_cmd",
                        _pages_cmd([{"targetId": "T1", "url": "about:blank"}], sent))

    assert bridge.minimize_windows() == 1
    assert ("Browser.setWindowBounds",
            {"windowId": 7, "bounds": {"windowState": "minimized"}}) in sent


def test_minimize_windows_does_nothing_in_headless(monkeypatch):
    bridge = ba.ChromeBridge(port=1, headless=True)
    monkeypatch.setattr(bridge, "_cmd", _pages_cmd([{"targetId": "T1"}]))

    assert bridge.minimize_windows() == 0


def test_start_minimizes_the_window_it_launched(monkeypatch):
    bridge = ba.ChromeBridge(port=9333, profile="sofia", chrome="chrome.exe",
                             minimized=True)
    calls = []
    monkeypatch.setattr(bridge, "running", lambda: False)
    monkeypatch.setattr(bridge, "_launch", lambda url: calls.append(("launch", url)))
    monkeypatch.setattr(bridge, "_connect", lambda: None)
    monkeypatch.setattr(bridge, "minimize_windows",
                        lambda: calls.append(("min",)) or 1)

    bridge.start("https://www.google.com/")

    assert calls == [("launch", "https://www.google.com/"), ("min",)]


def test_start_keeps_window_visible_when_not_minimized(monkeypatch):
    bridge = ba.ChromeBridge(port=9333, profile="sofia", chrome="chrome.exe")
    calls = []
    monkeypatch.setattr(bridge, "running", lambda: False)
    monkeypatch.setattr(bridge, "_launch", lambda url: calls.append("launch"))
    monkeypatch.setattr(bridge, "_connect", lambda: None)
    monkeypatch.setattr(bridge, "minimize_windows", lambda: calls.append("min"))

    bridge.start()

    assert calls == ["launch"]


def test_own_page_navigates_blank_tab_instead_of_opening_second_window(monkeypatch):
    """Второе окно оставляло на экране бесполезную пустую вкладку."""
    bridge = ba.ChromeBridge(port=1)
    sent = []
    attached = []
    monkeypatch.setattr(bridge, "_pages",
                        lambda: [{"targetId": "T1", "url": "about:blank"}])
    monkeypatch.setattr(bridge, "_attach",
                        lambda t: attached.append(t) or setattr(bridge, "_session", "S"))
    monkeypatch.setattr(bridge, "open_window",
                        lambda url, minimized=True: pytest.fail("второе окно не нужно"))
    monkeypatch.setattr(bridge, "_wait_ready", lambda timeout=25.0: None)
    monkeypatch.setattr(bridge, "minimize_windows", lambda: 0)

    def cmd(method, params=None, session=None):
        sent.append((method, params))
        return {"windowId": 7} if method == "Browser.getWindowForTarget" else {}

    monkeypatch.setattr(bridge, "_cmd", cmd)

    assert bridge.own_page("https://www.google.com/") == "T1"
    assert attached == ["T1"]
    assert ("Page.navigate", {"url": "https://www.google.com/"}) in sent
    assert ("Browser.setWindowBounds",
            {"windowId": 7, "bounds": {"windowState": "minimized"}}) in sent


def test_own_page_opens_new_window_when_the_tab_shows_another_site(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    opened = []
    monkeypatch.setattr(bridge, "_pages",
                        lambda: [{"targetId": "T9", "url": "https://example.com/x"}])
    monkeypatch.setattr(bridge, "open_window",
                        lambda url, minimized=True: opened.append(url) or "T2")

    assert bridge.own_page("https://www.google.com/") == "T2"
    assert opened == ["https://www.google.com/"]


def test_own_page_does_not_hide_window_when_not_asked(monkeypatch):
    bridge = ba.ChromeBridge(port=1)
    sent = []
    monkeypatch.setattr(bridge, "_pages",
                        lambda: [{"targetId": "T1", "url": "about:blank"}])
    monkeypatch.setattr(bridge, "_attach", lambda t: setattr(bridge, "_session", "S"))
    monkeypatch.setattr(bridge, "_wait_ready", lambda timeout=25.0: None)
    monkeypatch.setattr(bridge, "_cmd",
                        _pages_cmd([{"targetId": "T1"}], sent))

    bridge.own_page("https://www.google.com/", minimized=False)

    assert not [m for m, _ in sent if m == "Browser.setWindowBounds"]


class LiveConn:
    alive = True


class DeadConn:
    alive = False


def test_start_keeps_live_connection_and_its_tab(monkeypatch):
    """Повторный start() рвал сессию вкладки — и поиск открывал новое окно."""
    bridge = ba.ChromeBridge(port=9333, profile="sofia")
    bridge._conn = LiveConn()
    bridge._target, bridge._session = "T1", "S"
    monkeypatch.setattr(bridge, "running", lambda: True)
    monkeypatch.setattr(bridge, "_connect",
                        lambda: pytest.fail("нельзя переподключаться на ходу"))

    bridge.start()

    assert (bridge._target, bridge._session) == ("T1", "S")


def test_start_reconnects_after_the_socket_died(monkeypatch):
    bridge = ba.ChromeBridge(port=9333, profile="sofia")
    bridge._conn = DeadConn()
    called = []
    monkeypatch.setattr(bridge, "running", lambda: True)
    monkeypatch.setattr(bridge, "_connect", lambda: called.append(True))

    bridge.start()

    assert called == [True]
