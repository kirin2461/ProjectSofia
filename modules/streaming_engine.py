"""
Streaming Engine — трансляция Софии.

Захват экрана + оверлей + стриминг на платформы.
- Запись в файл (MP4)
- Трансляция через OBS WebSocket
- RTMP (Twitch/YouTube) через ffmpeg

Зависимости:
- opencv-python (cv2)
- obs-websocket-py (для OBS)
- ffmpeg (системный)

Примечание: на VDS (нет GUI, нет ffmpeg) этот модуль работает в mock-режиме.
"""

import io
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    fps: float
    bitrate: float
    dropped_frames: int
    latency_ms: float


class Overlay:
    """Оверлей: мысли, эмоции, bounding boxes."""

    def __init__(self, font_scale: float = 0.6):
        self.font_scale = font_scale
        self.font_color = (0, 255, 0)  # Зелёный Sci-Fi
        self.panel_color = (0, 0, 0, 180)  # Полупрозрачный чёрный
        self.bbox_color = (0, 255, 255)  # Cyan

    def render_frame(
        self,
        frame: np.ndarray,
        thoughts: str = "",
        emotion: str = "нейтрально",
        stats: Optional[dict] = None,
        detections: Optional[List[dict]] = None
    ) -> np.ndarray:
        """Рендерит оверлей поверх кадра."""
        import cv2

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Полупрозрачная панель снизу
        panel_h = 120
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Мысли Софии
        thought_text = thoughts[:200] if thoughts else "Молчу..."
        cv2.putText(frame, f"🧠 {thought_text}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, 1)

        # Эмоция
        emotion_text = f"😊 {emotion}"
        cv2.putText(frame, emotion_text, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 200, 0), 1)

        # Статистика
        if stats:
            stats_text = f"FPS: {stats.get('fps', 0):.1f} | Latency: {stats.get('latency', 0):.0f}ms"
            cv2.putText(frame, stats_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Bounding boxes
        if detections:
            for det in detections:
                x, y, bw, bh = det.get("bbox", (0, 0, 0, 0))
                label = det.get("label", "?")
                conf = det.get("confidence", 0)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), self.bbox_color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.bbox_color, 1)

        return frame


class MockStreamer:
    """Mock-стример для VDS. Сохраняет кадры как PNG."""

    def __init__(self, output_dir: str = "data/streams/mock"):
        self.output_dir = output_dir
        self._active = False
        self._frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def start_stream(self, **kwargs):
        self._active = True
        self._frame_count = 0
        logger.info(f"[MockStreamer] Записываю кадры в {self.output_dir}")

    def stop_stream(self):
        self._active = False
        logger.info(f"[MockStreamer] Остановлен. Сохранено {self._frame_count} кадров")

    def send_frame(self, frame: np.ndarray):
        if not self._active:
            return
        import cv2
        path = os.path.join(self.output_dir, f"frame_{self._frame_count:06d}.png")
        cv2.imwrite(path, frame)
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            logger.info(f"[MockStreamer] Сохранено {self._frame_count} кадров")

    def is_active(self) -> bool:
        return self._active


class FileRecorder:
    """Запись в MP4 через cv2.VideoWriter."""

    def __init__(self, output_dir: str = "data/streams"):
        self.output_dir = output_dir
        self._writer: Optional[cv2.VideoWriter] = None
        self._active = False
        os.makedirs(output_dir, exist_ok=True)

    def start_stream(self, width: int = 1920, height: int = 1080, fps: int = 30):
        import cv2
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.output_dir, f"stream_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        self._active = True
        logger.info(f"[FileRecorder] Запись: {path}")

    def stop_stream(self):
        self._active = False
        if self._writer:
            self._writer.release()
            self._writer = None
        logger.info("[FileRecorder] Запись завершена")

    def send_frame(self, frame: np.ndarray):
        if self._active and self._writer:
            # Убедимся, что размер совпадает
            if frame.shape[1] != 1920 or frame.shape[0] != 1080:
                import cv2
                frame = cv2.resize(frame, (1920, 1080))
            self._writer.write(frame)

    def is_active(self) -> bool:
        return self._active


class RTMPStreamer:
    """Трансляция через ffmpeg (Twitch/YouTube)."""

    def __init__(self, rtmp_url: str):
        self.rtmp_url = rtmp_url
        self._process: Optional[subprocess.Popen] = None
        self._active = False

    def start_stream(self, width: int = 1920, height: int = 1080, fps: int = 30, bitrate: str = "4500k"):
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",  # stdin
            "-c:v", "libx264",
            "-preset", "fast",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-f", "flv",
            self.rtmp_url
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self._active = True
        logger.info(f"[RTMPStreamer] Трансляция на {self.rtmp_url}")

    def stop_stream(self):
        self._active = False
        if self._process:
            self._process.stdin.close()
            self._process.wait(timeout=5)
            self._process = None
        logger.info("[RTMPStreamer] Трансляция остановлена")

    def send_frame(self, frame: np.ndarray):
        if not self._active or not self._process or not self._process.stdin:
            return
        try:
            self._process.stdin.write(frame.tobytes())
            self._process.stdin.flush()
        except BrokenPipeError:
            logger.error("[RTMPStreamer] Pipe broken — стрим упал")
            self.stop_stream()

    def is_active(self) -> bool:
        return self._active and self._process is not None and self._process.poll() is None


class OBSController:
    """Управление OBS через WebSocket."""

    def __init__(self, host: str = "localhost", port: int = 4455, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self._client = None

    def connect(self):
        try:
            import obswebsocket
            from obswebsocket import obsws, requests
            self._client = obsws(self.host, self.port, self.password)
            self._client.connect()
            logger.info("[OBSController] Подключен к OBS")
        except ImportError:
            logger.error("obs-websocket-py не установлен. pip install obs-websocket-py")
        except Exception as e:
            logger.error(f"[OBSController] Ошибка подключения: {e}")

    def start_recording(self):
        if self._client:
            try:
                self._client.call(requests.StartRecording())
                logger.info("[OBSController] Запись начата")
            except Exception as e:
                logger.error(f"[OBSController] Ошибка: {e}")

    def stop_recording(self):
        if self._client:
            try:
                self._client.call(requests.StopRecording())
                logger.info("[OBSController] Запись остановлена")
            except Exception as e:
                logger.error(f"[OBSController] Ошибка: {e}")

    def switch_scene(self, scene_name: str):
        if self._client:
            try:
                self._client.call(requests.SetCurrentProgramScene(sceneName=scene_name))
                logger.info(f"[OBSController] Сцена: {scene_name}")
            except Exception as e:
                logger.error(f"[OBSController] Ошибка: {e}")


class Streamer:
    """Главный класс: выбирает платформу и управляет потоком."""

    def __init__(
        self,
        platform: str = "file",  # file, rtmp, obs, mock
        output_dir: str = "data/streams",
        rtmp_url: Optional[str] = None,
        use_mock: bool = False
    ):
        self.platform = platform
        self.use_mock = use_mock
        self.overlay = Overlay()
        self._backend = None
        self._active = False

        if use_mock:
            self._backend = MockStreamer(output_dir)
        elif platform == "file":
            self._backend = FileRecorder(output_dir)
        elif platform == "rtmp":
            if not rtmp_url:
                raise ValueError("Для RTMP нужен rtmp_url")
            self._backend = RTMPStreamer(rtmp_url)
        elif platform == "obs":
            self._backend = OBSController()
        else:
            raise ValueError(f"Неизвестная платформа: {platform}")

    def start(self, **kwargs):
        if self._backend:
            if isinstance(self._backend, OBSController):
                self._backend.connect()
                self._backend.start_recording()
            else:
                self._backend.start_stream(**kwargs)
            self._active = True

    def stop(self):
        if self._backend:
            if isinstance(self._backend, OBSController):
                self._backend.stop_recording()
            else:
                self._backend.stop_stream()
            self._active = False

    def send_frame(self, frame: np.ndarray, **overlay_kwargs):
        """Рендерит оверлей и отправляет кадр."""
        framed = self.overlay.render_frame(frame, **overlay_kwargs)
        if self._backend and hasattr(self._backend, 'send_frame'):
            self._backend.send_frame(framed)

    def is_active(self) -> bool:
        if self._backend and hasattr(self._backend, 'is_active'):
            return self._backend.is_active()
        return self._active


# Глобальная переменная
streamer: Optional[Streamer] = None


def init_streamer(use_mock: bool = False, **kwargs) -> Streamer:
    """Инициализация глобального стримера."""
    global streamer
    streamer = Streamer(use_mock=use_mock, **kwargs)
    return streamer
