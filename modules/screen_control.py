"""Экран и курсор Windows: скриншоты, движение мыши, ввод, уступка человеку.

Отдельный слой от браузера: браузером София управляет через CDP и курсор
не трогает. Этот модуль нужен только для действий в обычных программах —
и здесь главное правило: если человек за компьютером, София уступает.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def idle_seconds() -> float:
    """Сколько секунд человек не трогал мышь и клавиатуру."""
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    now = kernel32.GetTickCount()
    # GetTickCount переполняется раз в 49.7 суток — учитываем переход
    delta = (now - info.dwTime) & 0xFFFFFFFF
    return max(0.0, delta / 1000.0)


def virtual_screen() -> Tuple[int, int, int, int]:
    """Общий прямоугольник всех мониторов: (x, y, ширина, высота)."""
    x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return x, y, w, h


def screenshot(path: Optional[str] = None) -> str:
    """Снимок всего экрана (все мониторы). Возвращает путь к PNG."""
    from PIL import ImageGrab

    img = ImageGrab.grab(all_screens=True)
    if not path:
        from modules.browser_agent import screens_dir
        path = str(screens_dir() / "screen.png")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target, "PNG")
    return str(target)


def move_to(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))


def position() -> Tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _mouse(flags: int) -> None:
    user32.mouse_event(flags, 0, 0, 0, 0)


def click(x: int, y: int, button: str = "left") -> None:
    """Плавно подвести курсор и кликнуть (в координатах рабочего стола)."""
    move_to(x, y)
    time.sleep(0.15)
    if button == "right":
        _mouse(MOUSEEVENTF_RIGHTDOWN)
        _mouse(MOUSEEVENTF_RIGHTUP)
    else:
        _mouse(MOUSEEVENTF_LEFTDOWN)
        _mouse(MOUSEEVENTF_LEFTUP)


def _send_unicode(ch: str, up: bool = False) -> None:
    inp = _INPUT(type=1)  # INPUT_KEYBOARD
    inp.ki = _KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0),
                         0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def type_text(text: str, delay: float = 0.01) -> None:
    """Ввод юникод-текста (работает и для русского)."""
    for ch in text:
        if ch == "\n":
            # без пары down/up клавиша остаётся «зажатой»
            _send_unicode("\r")
            _send_unicode("\r", up=True)
            continue
        _send_unicode(ch)
        _send_unicode(ch, up=True)
        time.sleep(delay)


def user_is_active(threshold: float = 1.5) -> bool:
    """Трогал ли человек мышь/клавиатуру только что."""
    return idle_seconds() < threshold


def wait_for_user_idle(threshold: float = 1.5, timeout: float = 30.0) -> bool:
    """Подождать, пока человек отойдёт от управления. False — не дождались."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not user_is_active(threshold):
            return True
        time.sleep(0.5)
    return False
