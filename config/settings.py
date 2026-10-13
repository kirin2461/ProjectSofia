"""Конфигурация Project Sofia.

Поддерживает три профиля:
- target: ноутбук с RTX 5050 (GPU, 16 GB RAM)
- dev: VDS без GPU (CPU-only, 8 GB RAM)
- local: переопределения через settings_local.py

Локальные переопределения — в settings_local.py (не в git).
"""

import os

PROFILE = os.getenv("SOFIA_PROFILE", "target")

# ═══════════════════════════════════════════════════════
# TARGET (ноутбук) — профиль по умолчанию, единственный с GPU
# ═══════════════════════════════════════════════════════
TARGET = {
    "OLLAMA_HOST": "http://localhost:11434",
    "MODEL_BRAIN": "qwen3:4b",
    "MODEL_EXTRACTOR": "qwen3:1.7b",      # быстрая текстовая модель для фактов
    "MODEL_EMBED": "bge-m3",
    "MODEL_VLM": "qwen3-vl:2b",           # мультимодальная модель (зрение + текст)
    "NUM_CTX": 32768,
    "NUM_PREDICT": 512,
    "TEMPERATURE": 0.7,
    "MEMORY_DB_PATH": "data/memory_db",
    "LIBRARY_PATH": "data/library",
    "VECTOR_DB_PATH": "data/vector_db",
    "VOICE_SAMPLES_DIR": "data/voice_samples",
    "XTTS_MODEL": "tts_models/multilingual/multi-dataset/xtts_v2",
    "CLIPS_RULES_DIR": "core/clips_rules",
    "SAFETY_MODE": "standard",
    "API_PORT": 8000,
    "API_HOST": "0.0.0.0",
    "DEBUG": False,
    "MAX_HISTORY": 10,
    "DECAY_LAMBDA": 0.0003,  # полураспад PAD ~40 мин
    "HAS_GPU": True,

    # ═══════════════════════════════════════════════════
    # ОПЦИОНАЛЬНЫЕ РАСШИРЕНИЯ (Phase 6)
    # ═══════════════════════════════════════════════════
    "VISION_ENABLED": False,       # Qwen3-VL-2B (~1.5 GB VRAM)
    "GAME_MODE_ENABLED": False,    # pydirectinput / pyautogui
    "STREAM_ENABLED": False,       # ffmpeg / OBS WebSocket
}

# ═══════════════════════════════════════════════════════
# DEV (VDS) — CPU-only, без GPU, 8 GB RAM
# ═══════════════════════════════════════════════════════
DEV = {
    **TARGET,
    "HAS_GPU": False,
    "DEBUG": True,
    "API_HOST": "0.0.0.0",
    # На VDS не запускаем Ollama — нет GPU
    # Эти настройки для совместимости кода
}

# ═══════════════════════════════════════════════════════
# Выбор профиля
# ═══════════════════════════════════════════════════════
PROFILES = {
    "target": TARGET,
    "dev": DEV,
}

_config = PROFILES.get(PROFILE, TARGET)

# Экспорт переменных
globals().update(_config)

# ═══════════════════════════════════════════════════════
# Локальные переопределения (settings_local.py)
# ═══════════════════════════════════════════════════════
try:
    from config.settings_local import *  # noqa: F401,F403
except ImportError:
    pass

# ═══════════════════════════════════════════════════════
# Производные значения
# ═══════════════════════════════════════════════════════
OLLAMA_API = f"{OLLAMA_HOST}/v1"

# ═══════════════════════════════════════════════════════
# УМНАЯ ОПТИМИЗАЦИЯ VRAM
# ═══════════════════════════════════════════════════════
# Если зрение ВКЛЮЧЕНО — Qwen3-VL-2B уже загружена в VRAM (~1.5 GB).
# Зачем держать ещё qwen3:1.7b (~1.2 GB) для экстракции фактов?
# Используем VLM как экстрактор — экономим 1.2 GB VRAM!
#
# Логика:
#   VISION_ENABLED = False → экстрактор = qwen3:1.7b (быстро, дёшево)
#   VISION_ENABLED = True  → экстрактор = qwen3-vl:2b (уже в памяти)
# ═══════════════════════════════════════════════════════

if VISION_ENABLED:
    MODEL_EXTRACTOR_ACTIVE = MODEL_VLM
    # Примечание: VLM на тексте чуть медленнее чисто текстовой 1.7b,
    # но экономия 1.2 GB VRAM стоит того.
    # Если скорость критична — переопределите в settings_local.py:
    #   MODEL_EXTRACTOR_ACTIVE = "qwen3:1.7b"
else:
    MODEL_EXTRACTOR_ACTIVE = MODEL_EXTRACTOR
