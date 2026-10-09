"""Конфигурация Project Sofia.

Локальные переопределения — в settings_local.py (не в git).
"""

import os

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API = f"{OLLAMA_HOST}/v1"

# Модели
MODEL_BRAIN = "qwen3:4b"
MODEL_EXTRACTOR = "qwen3:1.7b"
MODEL_EMBED = "bge-m3"

# Контекст
NUM_CTX = 32768
NUM_PREDICT = 512
TEMPERATURE = 0.7

# Память
MEMORY_DB_PATH = "data/memory_db"
LIBRARY_PATH = "data/library"
VECTOR_DB_PATH = "data/vector_db"

# TTS
VOICE_SAMPLES_DIR = "data/voice_samples"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# CLIPS
CLIPS_RULES_DIR = "core/clips_rules"

# Безопасность
SAFETY_MODE = "standard"  # strict / standard / permissive

# Порты
API_PORT = 8000

# Локальные переопределения
try:
    from config.settings_local import *  # noqa: F401,F403
except ImportError:
    pass
