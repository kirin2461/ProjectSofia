"""Экран и курсор: уступка человеку, клики, юникод-ввод (через поддельный user32)."""

import ctypes
import time
from pathlib import Path

import pytest

from modules import screen_control as sc


class FakeUser32:
    def __init__(self, idle_ms=1000, pos=(0, 0)):
        self.idle_ms = idle_ms
        self.pos = list(pos)
        self.moves = []
        self.events = []
        self.inputs = []

    def GetLastInputInfo(self, ref):
        info = ref._obj
        info.dwTime = sc.kernel32.GetTickCount() - self.idle_ms
        return 1

    def GetSystemMetrics(self, index):
        return {76: -1920, 77: 0, 78: 4480, 79: 1440}[index]

    def SetCursorPos(self, x, y):
        self.pos = [x, y]
        self.moves.append((x, y))
        return 1

    def GetCursorPos(self, ref):
        ref._obj.x, ref._obj.y = self.pos
        return 1

    def mouse_event(self, flags, dx, dy, data, extra):
        self.events.append(flags)

    def SendInput(self, count, ref, size):
        self.inputs.append(ref._obj.ki)
        return 1


class FakeKernel32:
    def __init__(self):
        self.ticks = 5_000_000

    def GetTickCount(self):
        return self.ticks


@pytest.fixture
def fake(monkeypatch):
    u, k = FakeUser32(), FakeKernel32()
    monkeypatch.setattr(sc, "user32", u)
    monkeypatch.setattr(sc, "kernel32", k)
    return u, k


def test_idle_seconds_from_last_input(fake):
    u, k = fake
    u.idle_ms = 2500
    assert sc.idle_seconds() == pytest.approx(2.5)


def test_idle_survives_tick_overflow(fake):
    u, k = fake
    k.ticks = 1000
    u.idle_ms = -(2 ** 32 - 3000)   # ввод был до переполнения счётчика
    assert sc.idle_seconds() == pytest.approx(3.0)


def test_user_is_active_threshold(fake):
    u, _ = fake
    u.idle_ms = 500
    assert sc.user_is_active(1.5) is True
    u.idle_ms = 4000
    assert sc.user_is_active(1.5) is False


def test_wait_for_user_idle_gives_up(fake, monkeypatch):
    u, _ = fake
    u.idle_ms = 0
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    assert sc.wait_for_user_idle(threshold=1.0, timeout=0.2) is False


def test_wait_for_user_idle_returns_when_user_stops(fake, monkeypatch):
    u, _ = fake
    u.idle_ms = 5000
    assert sc.wait_for_user_idle(threshold=1.0, timeout=2.0) is True


def test_virtual_screen_covers_all_monitors(fake):
    assert sc.virtual_screen() == (-1920, 0, 4480, 1440)


def test_click_moves_cursor_and_clicks_left(fake, monkeypatch):
    u, _ = fake
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    sc.click(300, 400)
    assert u.moves == [(300, 400)]
    assert u.events == [sc.MOUSEEVENTF_LEFTDOWN, sc.MOUSEEVENTF_LEFTUP]


def test_click_right_button(fake, monkeypatch):
    u, _ = fake
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    sc.click(10, 20, button="right")
    assert u.events == [sc.MOUSEEVENTF_RIGHTDOWN, sc.MOUSEEVENTF_RIGHTUP]


def test_position(fake):
    assert sc.position() == (0, 0)


def test_type_text_sends_unicode_with_keyup(fake, monkeypatch):
    u, _ = fake
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    sc.type_text("яOk")
    sent = [(chr(k.wScan), k.dwFlags) for k in u.inputs]
    assert sent == [("я", sc.KEYEVENTF_UNICODE), ("я", sc.KEYEVENTF_UNICODE | sc.KEYEVENTF_KEYUP),
                    ("O", sc.KEYEVENTF_UNICODE), ("O", sc.KEYEVENTF_UNICODE | sc.KEYEVENTF_KEYUP),
                    ("k", sc.KEYEVENTF_UNICODE), ("k", sc.KEYEVENTF_UNICODE | sc.KEYEVENTF_KEYUP)]


def test_type_text_newline_as_carriage_return(fake, monkeypatch):
    u, _ = fake
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    sc.type_text("\n")
    assert [chr(k.wScan) for k in u.inputs] == ["\r", "\r"]


def test_screenshot_creates_png(tmp_path, monkeypatch):
    from PIL import Image

    calls = {}

    def fake_grab(all_screens=False):
        calls["all"] = all_screens
        return Image.new("RGB", (8, 6), "white")

    monkeypatch.setattr("PIL.ImageGrab.grab", fake_grab)
    path = sc.screenshot(str(tmp_path / "shot.png"))
    assert calls["all"] is True
    assert (tmp_path / "shot.png").stat().st_size > 0
    assert Image.open(path).size == (8, 6)


def test_screenshot_default_path_lives_in_data_dir(tmp_path, monkeypatch):
    """Без пути снимок уходит в data/screens, а не рядом с исходниками.

    В exe исходники лежат внутри архива, писать туда нельзя.
    """
    from PIL import Image

    monkeypatch.setattr("PIL.ImageGrab.grab",
                        lambda all_screens=False: Image.new("RGB", (4, 3), "black"))
    monkeypatch.setattr("modules.browser_agent.screens_dir", lambda: tmp_path / "screens")
    path = sc.screenshot()
    assert Path(path) == tmp_path / "screens" / "screen.png"
    assert Path(path).stat().st_size > 0


def test_input_struct_size_matches_windows():
    # SendInput падает, если размер структуры не тот
    assert ctypes.sizeof(sc._INPUT) == 40
