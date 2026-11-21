"""Phase-6-04: Игровой режим — Vision + Game + Stream (ОПЦИОНАЛЬНО).

Связывает зрение, управление и стриминг в единый цикл. Загружается только
если включены флаги VISION_ENABLED, GAME_MODE_ENABLED, STREAM_ENABLED.

Цикл работает в отдельном потоке, чтобы не блокировать FastAPI.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GameProfile:
    name: str = "unknown"
    window_title: str = ""
    hp_region: Optional[list] = None
    key_bindings: Dict[str, str] = field(default_factory=dict)
    reactive_rules: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "GameProfile":
        hp = data.get("hp_region")
        if isinstance(hp, dict):
            hp = [hp.get("x", 0), hp.get("y", 0),
                  hp.get("width", 0), hp.get("height", 0)]
        return cls(
            name=data.get("name", "unknown"),
            window_title=data.get("window_title", ""),
            hp_region=hp,
            key_bindings=data.get("key_bindings", {}),
            reactive_rules=data.get("reactive_rules", []),
            raw=data,
        )

    @classmethod
    def load(cls, path: str) -> "GameProfile":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class GameMode:
    """Единый игровой цикл: восприятие → реакция → стратегия → действие → стрим."""

    def __init__(self, vision, controller, streamer, brain=None, emotion=None):
        self.vision = vision
        self.controller = controller
        self.streamer = streamer
        self.brain = brain          # функция think(context) -> plan (опц.)
        self.emotion = emotion
        self.running = False
        self.profile: Optional[GameProfile] = None
        self._thread: Optional[threading.Thread] = None
        self._last_think = 0.0
        self._think_interval = 0.5  # сек
        self._fps_target = 60

    def start(self, game_profile: dict):
        self.profile = (
            GameProfile.from_dict(game_profile)
            if isinstance(game_profile, dict) else game_profile
        )
        self.running = True
        if self.streamer:
            self.streamer.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("GameMode запущен: %s", self.profile.name)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.streamer and self.streamer.is_active():
            self.streamer.stop()
        logger.info("GameMode остановлен")

    def emergency_stop(self):
        logger.warning("GameMode: ЭКСТРЕННАЯ ОСТАНОВКА")
        self.running = False

    def _loop(self):
        from modules.game_control import ReactiveActions
        reactive = ReactiveActions(self.controller) if self.controller else None
        prev_hp = 100

        while self.running:
            frame_start = time.time()
            try:
                frame = self.vision.capture.get_frame() if self.vision else None
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Рефлексивный слой (каждый кадр)
                detections = self.vision.reactive.detect(frame)
                enemies = [d for d in detections if "person" in d.label.lower()
                           or "enemy" in d.label.lower()]
                if enemies and reactive:
                    reactive.track_target(enemies[0].bbox)

                # Когнитивный слой (каждые think_interval)
                now = time.time()
                if now - self._last_think >= self._think_interval:
                    self._last_think = now
                    self._cognitive_step(frame, detections)

                # Стриминг (каждый кадр)
                if self.streamer and self.streamer.is_active():
                    det_dicts = [
                        {"bbox": d.bbox, "label": d.label, "confidence": d.confidence}
                        for d in detections
                    ]
                    thoughts = self.vision.get_context() if self.vision else ""
                    emotion_name = (
                        self.emotion.dominant_emotion() if self.emotion else "нейтрально"
                    )
                    self.streamer.send_frame(
                        frame, thoughts=thoughts, emotion=emotion_name,
                        detections=det_dicts,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("GameMode loop error: %s", e)
                time.sleep(0.1)

            # Поддержка целевого FPS
            elapsed = time.time() - frame_start
            sleep = max(0, 1.0 / self._fps_target - elapsed)
            time.sleep(sleep)

    def _cognitive_step(self, frame, detections):
        """Стратегическое решение (заглушка без LLM в прототипе)."""
        if self.brain is None:
            return
        try:
            context = self.vision.get_context() if self.vision else ""
            plan = self.brain(context)
            if plan and self.controller:
                from modules.game_control import CognitiveActions
                ca = CognitiveActions(self.controller)
                for action in ca.parse_plan(str(plan)):
                    ca.execute_plan(action)
        except Exception as e:  # noqa: BLE001
            logger.debug("cognitive_step: %s", e)


def load_profile(profiles_dir: str, name: str) -> dict:
    """Загрузить профиль игры из data/game_profiles/<name>.json."""
    path = Path(profiles_dir) / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
