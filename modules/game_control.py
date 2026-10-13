"""
Game Control — руки Софии.

Управление мышью и клавиатурой для игр и приложений.
- Рефлексивные действия: прицеливание, уклонение (без LLM, < 100 мс)
- Когнитивные действия: выполнение плана от мозга

Зависимости:
- pydirectinput (Windows, для игр)
- pyautogui (кроссплатформенный fallback)

Примечание: на VDS (нет GUI, нет игр) этот модуль работает в mock-режиме.
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionPlan:
    action: str  # move_to, click_at, press_key, combo, wait
    target: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    key: Optional[str] = None
    keys: Optional[List[str]] = None
    duration: float = 0.1
    delay: float = 0.0


class BaseController:
    """Базовый класс управления."""

    def move_mouse(self, x: int, y: int, duration: float = 0.1):
        raise NotImplementedError

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left"):
        raise NotImplementedError

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.2):
        raise NotImplementedError

    def scroll(self, amount: int, direction: str = "vertical"):
        raise NotImplementedError

    def key_press(self, key: str):
        raise NotImplementedError

    def key_combo(self, keys: List[str]):
        raise NotImplementedError

    def type_text(self, text: str, interval: float = 0.01):
        raise NotImplementedError


class MockController(BaseController):
    """Mock-контроллер для VDS. Печатает действия в лог вместо реальных."""

    def _log(self, action: str, **kwargs):
        logger.info(f"[MockController] {action}: {kwargs}")

    def move_mouse(self, x: int, y: int, duration: float = 0.1):
        self._log("move_mouse", x=x, y=y, duration=duration)

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left"):
        self._log("click", x=x, y=y, button=button)

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.2):
        self._log("drag", start=start, end=end, duration=duration)

    def scroll(self, amount: int, direction: str = "vertical"):
        self._log("scroll", amount=amount, direction=direction)

    def key_press(self, key: str):
        self._log("key_press", key=key)

    def key_combo(self, keys: List[str]):
        self._log("key_combo", keys=keys)

    def type_text(self, text: str, interval: float = 0.01):
        self._log("type_text", text=text[:50], interval=interval)


class GameController(BaseController):
    """Реальный контроллер через pydirectinput / pyautogui."""

    def __init__(self, use_directinput: bool = True):
        self.use_directinput = use_directinput
        self._backend = None
        self._screen_size = (1920, 1080)
        self._load_backend()

    def _load_backend(self):
        if self.use_directinput:
            try:
                import pydirectinput
                self._backend = pydirectinput
                # Отключаем fail-safe для игр (осторожно!)
                pydirectinput.FAILSAFE = False
                logger.info("[GameController] Используется pydirectinput")
                return
            except ImportError:
                logger.warning("pydirectinput не найден, fallback на pyautogui")

        try:
            import pyautogui
            self._backend = pyautogui
            pyautogui.FAILSAFE = True
            logger.info("[GameController] Используется pyautogui")
        except ImportError:
            raise RuntimeError("Не установлен pydirectinput или pyautogui")

    def _humanize(self, value: float, variance: float = 0.1) -> float:
        """Добавляет случайную вариацию для имитации человека."""
        return value * (1 + random.uniform(-variance, variance))

    def _add_jitter(self, x: int, y: int, max_jitter: int = 2) -> Tuple[int, int]:
        return (
            max(0, min(self._screen_size[0], x + random.randint(-max_jitter, max_jitter))),
            max(0, min(self._screen_size[1], y + random.randint(-max_jitter, max_jitter)))
        )

    def move_mouse(self, x: int, y: int, duration: float = 0.1):
        x, y = self._add_jitter(x, y)
        duration = self._humanize(duration)
        if hasattr(self._backend, 'moveTo'):
            self._backend.moveTo(x, y, duration=duration)
        else:
            self._backend.moveTo(x, y, duration=duration)

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left"):
        if x is not None and y is not None:
            self.move_mouse(x, y, duration=0.05)
        if hasattr(self._backend, 'click'):
            self._backend.click(button=button)

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.2):
        self.move_mouse(start[0], start[1], duration=0.05)
        if hasattr(self._backend, 'mouseDown'):
            self._backend.mouseDown()
        self.move_mouse(end[0], end[1], duration=duration)
        if hasattr(self._backend, 'mouseUp'):
            self._backend.mouseUp()

    def scroll(self, amount: int, direction: str = "vertical"):
        if hasattr(self._backend, 'scroll'):
            self._backend.scroll(amount)

    def key_press(self, key: str):
        key = key.lower()
        if hasattr(self._backend, 'press'):
            self._backend.press(key)
        elif hasattr(self._backend, 'keyDown'):
            self._backend.keyDown(key)
            self._backend.keyUp(key)

    def key_combo(self, keys: List[str]):
        keys = [k.lower() for k in keys]
        if hasattr(self._backend, 'keyDown'):
            for key in keys:
                self._backend.keyDown(key)
            for key in reversed(keys):
                self._backend.keyUp(key)

    def type_text(self, text: str, interval: float = 0.01):
        if hasattr(self._backend, 'typewrite'):
            self._backend.typewrite(text, interval=interval)


class ReactiveActions:
    """Рефлексивные действия: прицеливание, уклонение. Без LLM."""

    def __init__(self, controller: BaseController):
        self.controller = controller
        self._target: Optional[Tuple[int, int]] = None
        self._running = False

    def track_target(self, bbox: Tuple[int, int, int, int], smooth: float = 0.3):
        """Мышь следует за центром bbox (объекта)."""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        if self._target is None:
            self._target = (center_x, center_y)
        else:
            # Экспоненциальное сглаживание
            self._target = (
                int(self._target[0] + smooth * (center_x - self._target[0])),
                int(self._target[1] + smooth * (center_y - self._target[1]))
            )

        self.controller.move_mouse(self._target[0], self._target[1], duration=0.02)

    def dodge(self, direction: str):
        """Быстрое движение в сторону + прыжок."""
        direction_keys = {
            "forward": "w", "backward": "s",
            "left": "a", "right": "d"
        }
        key = direction_keys.get(direction, "w")
        self.controller.key_press(key)
        self.controller.key_press("space")

    def react_to_flash(self, threshold: int = 200):
        """Реакция на вспышку (заглушка — нужен захват экрана)."""
        logger.info("[ReactiveActions] Реакция на вспышку: уклонение")
        self.dodge("backward")

    def emergency_stop(self):
        """Мгновенная остановка всех действий."""
        logger.warning("[ReactiveActions] ЭКСТРЕННАЯ ОСТАНОВКА")
        self._target = None


class CognitiveActions:
    """Когнитивные действия: выполнение плана от LLM."""

    def __init__(self, controller: BaseController):
        self.controller = controller
        self._action_log: List[dict] = []

    def parse_plan(self, plan_text: str) -> List[ActionPlan]:
        """Парсит текстовый план от LLM в структурированные действия."""
        plans = []
        # Простой парсер — можно расширить
        lines = plan_text.strip().split("\n")
        for line in lines:
            line = line.strip().lower()
            if line.startswith("move"):
                # "move to 100,200" или "move mouse to 100 200"
                coords = self._extract_coords(line)
                if coords:
                    plans.append(ActionPlan(action="move_to", coordinates=coords))
            elif line.startswith("click"):
                coords = self._extract_coords(line)
                plans.append(ActionPlan(action="click_at", coordinates=coords))
            elif line.startswith("press"):
                key = line.replace("press", "").strip()
                plans.append(ActionPlan(action="press_key", key=key))
            elif line.startswith("wait"):
                duration = self._extract_number(line) or 1.0
                plans.append(ActionPlan(action="wait", duration=duration))
        return plans

    def _extract_coords(self, text: str) -> Optional[Tuple[int, int]]:
        """Извлекает координаты из строки."""
        import re
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            return (int(numbers[0]), int(numbers[1]))
        return None

    def _extract_number(self, text: str) -> Optional[float]:
        import re
        match = re.search(r'\d+\.?\d*', text)
        return float(match.group()) if match else None

    def execute_plan(self, plan: ActionPlan):
        """Выполняет одно действие."""
        logger.info(f"[CognitiveActions] Выполняю: {plan}")

        if plan.action == "move_to":
            if plan.coordinates:
                self.controller.move_mouse(*plan.coordinates, duration=plan.duration)
        elif plan.action == "click_at":
            if plan.coordinates:
                self.controller.click(*plan.coordinates)
            else:
                self.controller.click()
        elif plan.action == "press_key":
            if plan.key:
                self.controller.key_press(plan.key)
        elif plan.action == "combo":
            if plan.keys:
                self.controller.key_combo(plan.keys)
        elif plan.action == "wait":
            time.sleep(plan.duration)

        # Логирование
        self._action_log.append({
            "timestamp": time.time(),
            "action": plan.action,
            "target": plan.target,
            "coordinates": plan.coordinates
        })

    def execute_plans(self, plans: List[ActionPlan]):
        """Выполняет последовательность действий."""
        for plan in plans:
            self.execute_plan(plan)
            if plan.delay > 0:
                time.sleep(plan.delay)

    def get_log(self) -> List[dict]:
        return self._action_log.copy()


class ActionSafety:
    """Безопасность действий."""

    def __init__(self, screen_size: Tuple[int, int] = (1920, 1080)):
        self.screen_size = screen_size
        self.max_actions_per_second = 10
        self._last_action_time = 0.0
        self._action_count = 0
        self._window_start = 0.0

    def validate(self, plan: ActionPlan) -> bool:
        """Проверяет, не выходит ли действие за пределы экрана."""
        if plan.coordinates:
            x, y = plan.coordinates
            if x < 0 or x > self.screen_size[0] or y < 0 or y > self.screen_size[1]:
                logger.warning(f"[ActionSafety] Координаты вне экрана: {plan.coordinates}")
                return False
        return True

    def rate_limit(self) -> bool:
        """Не более N действий в секунду."""
        now = time.time()
        if now - self._window_start > 1.0:
            self._window_start = now
            self._action_count = 0
        self._action_count += 1
        if self._action_count > self.max_actions_per_second:
            logger.warning("[ActionSafety] Превышен rate limit")
            return False
        return True

    def emergency_stop(self):
        """Мгновенная остановка."""
        logger.warning("[ActionSafety] 🛑 ЭКСТРЕННАЯ ОСТАНОВКА")
        # В реальности — остановить все потоки


# Глобальная переменная
game_controller: Optional[GameController] = None


def init_game_controller(use_mock: bool = False, **kwargs) -> BaseController:
    """Инициализация глобального контроллера."""
    global game_controller
    if use_mock:
        game_controller = MockController()
    else:
        game_controller = GameController(**kwargs)
    return game_controller
