"""
Vision Engine — глаза Софии.

Двухслойная система компьютерного зрения:
- Рефлексивный слой (ReactiveVision): YOLOv8-nano + детекция движения. < 50 мс.
- Когнитивный слой (CognitiveVision): Qwen3-VL-2B. 1-2 раза/сек.

Зависимости:
- dxcam (Windows, GPU-захват) или mss (кроссплатформенный)
- ultralytics (YOLO)
- ollama / openai (VLM)

Примечание: на VDS (CPU-only, нет GUI) этот модуль работает в mock-режиме.
"""

import asyncio
import base64
import io
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    bbox: tuple  # (x, y, w, h)
    confidence: float


@dataclass
class VisionDescription:
    text: str
    ui_elements: List[dict]
    timestamp: float


class MockScreenCapture:
    """Заглушка для VDS (нет экрана). Генерирует случайные кадры."""

    def __init__(self, fps: int = 60, region: Optional[tuple] = None):
        self.fps = fps
        self.region = region or (0, 0, 1920, 1080)
        self._running = False
        self._frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[MockScreenCapture] Запущен")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("[MockScreenCapture] Остановлен")

    def _loop(self):
        while self._running:
            # Меняем кадр слегка, чтобы имитировать движение
            self._frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            time.sleep(1.0 / self.fps)

    def get_frame(self) -> np.ndarray:
        return self._frame.copy()


class ScreenCapture:
    """Реальный захват экрана. Использует dxcam на Windows, mss на Linux."""

    def __init__(self, fps: int = 60, region: Optional[tuple] = None):
        self.fps = fps
        self.region = region
        self._capture = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        try:
            import dxcam
            self._capture = dxcam.create(region=self.region)
            logger.info("[ScreenCapture] Используется dxcam (GPU)")
        except ImportError:
            try:
                import mss
                self._capture = mss.mss()
                logger.info("[ScreenCapture] Используется mss (CPU)")
            except ImportError:
                raise RuntimeError("Не установлен dxcam или mss. Установите: pip install dxcam mss")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._capture:
            if hasattr(self._capture, 'stop'):
                self._capture.stop()
        logger.info("[ScreenCapture] Остановлен")

    def _loop(self):
        while self._running:
            try:
                if hasattr(self._capture, 'grab'):
                    # dxcam
                    frame = self._capture.grab()
                elif hasattr(self._capture, 'grab'):
                    # mss
                    monitor = self.region or self._capture.monitors[1]
                    sct_img = self._capture.grab(monitor)
                    import numpy as np
                    frame = np.array(sct_img)
                else:
                    continue

                if frame is not None:
                    with self._lock:
                        self._frame = frame
            except Exception as e:
                logger.error(f"[ScreenCapture] Ошибка захвата: {e}")
            time.sleep(1.0 / self.fps)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


class ReactiveVision:
    """Рефлексивный слой: YOLOv8-nano + детекция движения. < 50 мс."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or "yolov8n.pt"
        self._prev_frame = None
        self._device = "cpu"  # fallback

    def load(self):
        try:
            from ultralytics import YOLO
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.model_path)
            self.model.to(self._device)
            logger.info(f"[ReactiveVision] YOLO загружен на {self._device}")
        except ImportError:
            logger.warning("[ReactiveVision] ultralytics не установлен. Установите: pip install ultralytics")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model(frame, verbose=False, device=self._device)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(Detection(
                label=self.model.names[int(box.cls)],
                bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                confidence=float(box.conf)
            ))
        return detections

    def detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Определение движения через разницу кадров."""
        if prev_frame is None or curr_frame is None:
            return np.zeros_like(curr_frame)
        gray_prev = np.mean(prev_frame, axis=2).astype(np.uint8)
        gray_curr = np.mean(curr_frame, axis=2).astype(np.uint8)
        diff = cv2.absdiff(gray_prev, gray_curr)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return thresh


class CognitiveVision:
    """Когнитивный слой: VLM (Qwen3-VL-2B). 1-2 раза/сек."""

    def __init__(self, model_name: str = "qwen3-vl:2b"):
        self.model_name = model_name
        self.client = None

    async def load(self, ollama_host: str = "http://localhost:11434"):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(base_url=f"{ollama_host}/v1", api_key="ollama")
            logger.info(f"[CognitiveVision] Клиент создан для {self.model_name}")
        except ImportError:
            logger.error("[CognitiveVision] openai не установлен")

    async def describe(self, frame: np.ndarray, prompt: str = "Что ты видишь?") -> str:
        if self.client is None:
            return "[VLM не загружен]"

        # Кодируем кадр в base64
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[CognitiveVision] Ошибка VLM: {e}")
            return f"[Ошибка VLM: {e}]"

    async def describe_ui(self, frame: np.ndarray) -> dict:
        """Специальный промпт для распознавания UI/игровых элементов."""
        prompt = (
            "Опиши игровой интерфейс на скриншоте. "
            "Найди: здоровье (HP), ману/энергию, уровень, кнопки, врагов, предметы. "
            "Ответь в формате JSON: {\"hp\": \"...\", \"enemies\": [...], \"items\": [...]}"
        )
        text = await self.describe(frame, prompt)
        # Здесь можно добавить парсинг JSON
        return {"raw": text, "parsed": None}


class VisionEngine:
    """Главный класс: объединяет захват, рефлексивный и когнитивный слои."""

    def __init__(
        self,
        fps: int = 60,
        region: Optional[tuple] = None,
        yolo_model: Optional[str] = None,
        vlm_model: str = "qwen3-vl:2b",
        use_mock: bool = False
    ):
        self.use_mock = use_mock
        self.capture = MockScreenCapture(fps, region) if use_mock else ScreenCapture(fps, region)
        self.reactive = ReactiveVision(yolo_model)
        self.cognitive = CognitiveVision(vlm_model)
        self._running = False
        self._vision_task: Optional[asyncio.Task] = None
        self._last_description = ""
        self._last_detections: List[Detection] = []

    def start(self):
        self.capture.start()
        self.reactive.load()
        self._running = True
        logger.info("[VisionEngine] Запущен")

    def stop(self):
        self._running = False
        if self._vision_task:
            self._vision_task.cancel()
        self.capture.stop()
        logger.info("[VisionEngine] Остановлен")

    async def run(self, ollama_host: str = "http://localhost:11434"):
        """Фоновая задача: когнитивное описание каждые 500 мс."""
        await self.cognitive.load(ollama_host)

        while self._running:
            try:
                frame = self.capture.get_frame()
                if frame is not None:
                    # Рефлексивный слой (можно вызывать чаще извне)
                    self._last_detections = self.reactive.detect(frame)

                    # Когнитивный слой (1 раз в 500 мс)
                    description = await self.cognitive.describe(frame)
                    self._last_description = description
                    logger.debug(f"[Vision] {description[:100]}...")

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VisionEngine] Ошибка: {e}")
                await asyncio.sleep(1.0)

    def get_context(self) -> str:
        """Форматированный контекст для монолога."""
        detections_str = "; ".join([f"{d.label} ({d.confidence:.2f})" for d in self._last_detections])
        return (
            f"[VISION] На экране видно:\n"
            f"- Описание: {self._last_description}\n"
            f"- Объекты: {detections_str or 'нет'}\n"
        )


# Глобальная переменная для использования в main.py
vision_engine: Optional[VisionEngine] = None


def init_vision_engine(use_mock: bool = False, **kwargs) -> VisionEngine:
    """Инициализация глобального движка зрения."""
    global vision_engine
    vision_engine = VisionEngine(use_mock=use_mock, **kwargs)
    return vision_engine
