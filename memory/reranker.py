"""Необязательный реранкер: cross-encoder поверх найденных кандидатов.

Векторный поиск (bi-encoder) быстро достаёт похожие абзацы, но плохо различает
«похоже по теме» и «отвечает на вопрос». Cross-encoder читает пару
«вопрос + абзац» целиком и потому точнее — за это платят скоростью, поэтому
он ставится после отбора кандидатов, а не вместо него.

Модели в комплекте нет: она весит сотни мегабайт и нужна не всем. Если каталог
не задан или в нём нет ONNX-модели, модуль молча выключен и поиск работает
как раньше — никаких «а вдруг упадёт».
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

WEIGHT = 0.6              # вклад реранкера в итоговую оценку
MAX_LENGTH = 320          # токенов на пару: длинные абзацы обрезаются
MODEL_FILE = "model_quantized.onnx"
ALT_MODEL_FILE = "model.onnx"


def _dir() -> Optional[Path]:
    raw = os.getenv("SOFIA_RERANK_DIR", "").strip()
    if not raw:
        try:
            from config import settings
            raw = str(getattr(settings, "LIBRARY_RERANK_DIR", "") or "").strip()
        except Exception:  # noqa: BLE001
            raw = ""
    return Path(raw) if raw else None


def model_path() -> Optional[Path]:
    """Путь к ONNX-файлу, если он есть на диске."""
    folder = _dir()
    if folder is None or not folder.is_dir():
        return None
    for name in (MODEL_FILE, ALT_MODEL_FILE):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def available() -> bool:
    """Есть ли модель. Зависимости проверяются при первом использовании."""
    return model_path() is not None


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class OnnxReranker:
    """Ленивая обёртка: модель грузится при первом запросе и один раз."""

    def __init__(self, folder: Optional[Path | str] = None):
        self.folder = Path(folder) if folder else None
        self._session = None
        self._tokenizer = None
        self._failed = False

    def _load(self) -> bool:
        if self._session is not None and self._tokenizer is not None:
            return True
        if self._failed:
            return False
        path = model_path() if self.folder is None else self._local_model()
        if path is None:
            self._failed = True
            return False
        try:
            import onnxruntime as ort          # тяжёлый импорт: только по делу
            from tokenizers import Tokenizer

            tokenizer_file = path.parent / "tokenizer.json"
            if not tokenizer_file.is_file():
                logger.info("реранкер: нет tokenizer.json рядом с %s", path.name)
                self._failed = True
                return False

            self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
            self._tokenizer.enable_truncation(max_length=MAX_LENGTH)
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            logger.info("реранкер подключён: %s", path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.info("реранкер выключен (%s)", e)
            self._failed = True
            return False

    def _local_model(self) -> Optional[Path]:
        if self.folder is None or not self.folder.is_dir():
            return None
        for name in (MODEL_FILE, ALT_MODEL_FILE):
            candidate = self.folder / name
            if candidate.is_file():
                return candidate
        return None

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Оценки 0..1 для каждой пары «вопрос — абзац»."""
        texts = [str(t or "") for t in texts]
        if not texts or not self._load():
            return []
        try:
            pairs = [(query or "", t) for t in texts]
            encodings = self._tokenizer.encode_batch(pairs)
            ids = [e.ids for e in encodings]
            mask = [e.attention_mask for e in encodings]
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
            }
            # Часть моделей ждёт тип токенов; передаём только известные входы
            names = {i.name for i in self._session.get_inputs()}
            inputs = {k: v for k, v in inputs.items() if k in names}
            outputs = self._session.run(None, inputs)
            logits = outputs[0]
            return [_sigmoid(float(row[0]) if hasattr(row, "__len__") else float(row))
                    for row in logits]
        except Exception as e:  # noqa: BLE001
            logger.info("реранкер не смог оценить: %s", e)
            self._failed = True
            return []


_DEFAULT: Optional[OnnxReranker] = None


def _reranker() -> OnnxReranker:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = OnnxReranker()
    return _DEFAULT


def rerank(hits: Sequence, query: str, weight: float = WEIGHT) -> List:
    """Переставляет находки по cross-encoder-оценке, сохраняя исходный порядок при сбое."""
    hits = list(hits or [])
    if len(hits) < 2 or not (query or "").strip():
        return hits
    scores = _reranker().score(query, [getattr(h, "text", "") for h in hits])
    if len(scores) != len(hits):
        return hits
    for hit, score in zip(hits, scores):
        hit.score = round(weight * score + (1.0 - weight) * min(1.0, hit.score), 4)
    return sorted(hits, key=lambda h: h.score, reverse=True)


def describe() -> str:
    """Строка для /state и лога."""
    if not available():
        return "выключен"
    return f"включён: {model_path()}"
