"""Конфигурация Project Sofia.

Поддерживает три профиля:
- default (target): RTX 5050 8GB / 16GB RAM
- dev: VDS с большими ресурсами
- local: переопределения через settings_local.py

Локальные переопределения — в settings_local.py (не в git).
"""

import os

PROFILE = os.getenv("SOFIA_PROFILE", "target")

# ═══════════════════════════════════════════════════════
# TARGET (ноутбук) — профиль по умолчанию
# ═══════════════════════════════════════════════════════
TARGET = {
    "OLLAMA_HOST": "http://localhost:11434",
    "MODEL_BRAIN": "qwen3:4b",
    "MODEL_EXTRACTOR": "qwen3:1.7b",
    "MODEL_EMBED": "bge-m3",
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
}

# ═══════════════════════════════════════════════════════
# DEV (VDS) — для разработки и тестирования
# ═══════════════════════════════════════════════════════
DEV = {
    **TARGET,
    "MODEL_BRAIN": "qwen3:4b",  # можно переопределить в settings_local
    "NUM_CTX": 32768,
    "DEBUG": True,
    "API_HOST": "0.0.0.0",
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
