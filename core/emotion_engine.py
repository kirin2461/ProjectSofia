"""Phase-4-02: PAD движок эмоций.

Модель PAD (Pleasure, Arousal, Dominance) с 8 базовыми эмоциями,
экспоненциальным затуханием к нейтрали и влиянием на генерацию ответа.

Не требует GPU — эвристический анализ работает везде.
"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from config.settings import DECAY_LAMBDA, EMOTION_STATE_PATH

logger = logging.getLogger(__name__)

# 8 базовых эмоций → координаты PAD (-1..1)
EMOTIONS: Dict[str, Tuple[float, float, float]] = {
    "радость": (0.8, 0.6, 0.4),
    "грусть": (-0.8, -0.4, -0.4),
    "гнев": (-0.6, 0.8, 0.6),
    "страх": (-0.7, 0.7, -0.6),
    "доверие": (0.5, 0.2, -0.3),
    "отвращение": (-0.6, 0.3, 0.1),
    "удивление": (0.3, 0.8, 0.1),
    "ожидание": (0.2, 0.4, 0.5),
    "нейтрально": (0.0, 0.0, 0.0),
}

# Простой лексикон для эвристики (слово → эмоция)
_LEXICON: Dict[str, Tuple[str, float]] = {
    "спасибо": ("радость", 0.5),
    "здорово": ("радость", 0.6),
    "класс": ("радость", 0.6),
    "ура": ("радость", 0.8),
    "люблю": ("радость", 0.7),
    "грустно": ("грусть", 0.6),
    "плохо": ("грусть", 0.5),
    "жаль": ("грусть", 0.5),
    "устал": ("грусть", 0.4),
    "дурак": ("гнев", 0.7),
    "ненавижу": ("гнев", 0.8),
    "злюсь": ("гнев", 0.7),
    "бесит": ("гнев", 0.7),
    "идиот": ("гнев", 0.8),
    "боюсь": ("страх", 0.7),
    "страшно": ("страх", 0.7),
    "опасно": ("страх", 0.5),
    "фу": ("отвращение", 0.6),
    "мерзко": ("отвращение", 0.7),
    "ого": ("удивление", 0.6),
    "вот это да": ("удивление", 0.7),
    "неужели": ("удивление", 0.5),
    "жду": ("ожидание", 0.5),
    "скоро": ("ожидание", 0.4),
}

# Эффекты событий: event → (dP, dA, dD)
_EVENTS: Dict[str, Tuple[float, float, float]] = {
    "критик_одобрил": (0.1, 0.0, 0.1),
    "критик_нашёл_ошибку": (-0.1, 0.1, -0.1),
    "диалог_успешен": (0.05, 0.05, 0.05),
    "hp_low": (-0.3, 0.4, -0.2),
    "победа": (0.5, 0.4, 0.4),
    "поражение": (-0.5, 0.2, -0.4),
}


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _has_word(text: str, word: str) -> bool:
    """Совпадение целого слова (а не подстроки: «ура» внутри «дурак»)."""
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None


@dataclass
class EmotionEngine:
    P: float = 0.0
    A: float = 0.0
    D: float = 0.0
    confidence: float = 0.5
    base_state: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    decay_lambda: float = DECAY_LAMBDA
    _last_update: float = field(default_factory=time.time)

    # ── Обновление ─────────────────────────────────────────
    def update_from_text(self, text: str) -> Optional[str]:
        """Эвристический анализ эмоции по ключевым словам."""
        self.decay()
        low = text.lower()
        best: Optional[Tuple[str, float]] = None
        for word, (emotion, weight) in _LEXICON.items():
            if _has_word(low, word):
                if best is None or weight > best[1]:
                    best = (emotion, weight)
        if best is None:
            return None

        emotion, weight = best
        tp, ta, td = EMOTIONS[emotion]
        # Притягиваем состояние к эмоции: чем сильнее слово, тем ближе к её центроиду
        pull = min(0.9, 0.3 + weight * 0.5)
        self.P = _clamp(self.P + (tp - self.P) * pull)
        self.A = _clamp(self.A + (ta - self.A) * pull)
        self.D = _clamp(self.D + (td - self.D) * pull)
        self.confidence = min(1.0, 0.3 + weight * 0.6)
        self._last_update = time.time()
        return emotion

    def update_from_event(self, event: str, intensity: float = 1.0):
        """Прямое изменение состояния по событию."""
        self.decay()
        dp, da, dd = _EVENTS.get(event, (0.0, 0.0, 0.0))
        self.P = _clamp(self.P + dp * intensity)
        self.A = _clamp(self.A + da * intensity)
        self.D = _clamp(self.D + dd * intensity)
        self._last_update = time.time()

    def set_pad(self, p: float, a: float, d: float, confidence: float = 0.8):
        """Прямая установка PAD (например, из LLM-анализа)."""
        self.P, self.A, self.D = _clamp(p), _clamp(a), _clamp(d)
        self.confidence = max(0.0, min(1.0, confidence))
        self._last_update = time.time()

    # ── Затухание ──────────────────────────────────────────
    def decay(self, dt_seconds: Optional[float] = None):
        """Экспоненциальное возвращение к base_state."""
        now = time.time()
        if dt_seconds is None:
            dt_seconds = now - self._last_update
        if dt_seconds <= 0:
            return
        factor = 1 - math.exp(-self.decay_lambda * dt_seconds)
        bp, ba, bd = self.base_state
        self.P += (bp - self.P) * factor
        self.A += (ba - self.A) * factor
        self.D += (bd - self.D) * factor
        self._last_update = now

    # ── Производные ─────────────────────────────────────────
    def dominant_emotion(self) -> str:
        """Ближайшая по евклиду базовая эмоция."""
        best_name, best_dist = "нейтрально", float("inf")
        for name, (p, a, d) in EMOTIONS.items():
            dist = (p - self.P) ** 2 + (a - self.A) ** 2 + (d - self.D) ** 2
            if dist < best_dist:
                best_dist, best_name = dist, name
        return best_name

    def to_prompt_modifier(self) -> dict:
        """Влияние эмоции на генерацию: temperature + tone."""
        temperature = max(0.3, min(1.2, 0.7 + self.A * 0.2))
        if self.P < -0.5:
            tone = "ответь сочувственно и осторожно"
        elif self.A > 0.7:
            tone = "ответь энергично"
        elif self.D > 0.5:
            tone = "ответь уверенно"
        else:
            tone = "ответь спокойно и дружелюбно"
        return {"temperature": round(temperature, 2), "tone_instruction": tone}

    def to_dict(self) -> dict:
        self.decay()
        return {
            "P": round(self.P, 3),
            "A": round(self.A, 3),
            "D": round(self.D, 3),
            "dominant_emotion": self.dominant_emotion(),
            "confidence": round(self.confidence, 3),
        }

    # ── Персистентность ─────────────────────────────────────
    def save(self, path: str = EMOTION_STATE_PATH):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"P": self.P, "A": self.A, "D": self.D,
                     "confidence": self.confidence, "ts": self._last_update},
                    f, ensure_ascii=False,
                )
        except OSError as e:
            logger.warning("Не удалось сохранить эмоции: %s", e)

    def load(self, path: str = EMOTION_STATE_PATH):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            self.P, self.A, self.D = d["P"], d["A"], d["D"]
            self.confidence = d.get("confidence", 0.5)
            self._last_update = d.get("ts", time.time())
            self.decay()
        except (OSError, KeyError, json.JSONDecodeError):
            pass
