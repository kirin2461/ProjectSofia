"""Конфигурация Project Sofia.

Поддерживает три профиля:
- target: ноутбук с RTX 5050 (GPU, 16 GB RAM)
- dev: VDS без GPU (CPU-only, 8 GB RAM)
- local: переопределения через settings_local.py

Порядок приоритетов (позднее перекрывает раннее):
1. профиль (TARGET / DEV) — значения по умолчанию
2. переменные окружения / .env
3. settings_local.py — локальные переопределения (не в git)
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # .env в корне проекта, если установлен python-dotenv
except ImportError:
    pass


def _env(name: str, default, cast=None):
    """Прочитать переменную окружения с приведением типа."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if cast is not None:
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default
    return value


PROFILE = _env("SOFIA_PROFILE", "target")

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
    "XTTS_LANGUAGE": "ru",
    # auto: GPU, если свободно >= TTS_MIN_FREE_VRAM GB, иначе CPU
    "TTS_DEVICE": "auto",
    "TTS_MIN_FREE_VRAM": 2.0,
    "CLIPS_RULES_DIR": "core/clips_rules",
    "SAFETY_MODE": "standard",
    "SAFETY_LOG_PATH": "data/safety_log.jsonl",
    "ACTION_LOG_PATH": "data/action_log.jsonl",
    "EMOTION_STATE_PATH": "data/emotion_state.json",
    "XP_PATH": "data/xp.json",
    "API_PORT": 8000,
    "API_HOST": "0.0.0.0",
    "DEBUG": False,
    "MAX_HISTORY": 10,
    "DECAY_LAMBDA": 0.0003,  # полураспад PAD ~40 мин
    "HAS_GPU": True,

    # Память
    "EPISODES_PER_RECALL": 3,
    "LIBRARY_CHUNK_SIZE": 800,
    "LIBRARY_CHUNK_OVERLAP": 80,
    "EPISODE_SUMMARY_DAYS": 7,

    # ═══════════════════════════════════════════════════
    # ОПЦИОНАЛЬНЫЕ РАСШИРЕНИЯ (Phase 6)
    # ═══════════════════════════════════════════════════
    "VISION_ENABLED": False,       # Qwen3-VL-2B (~1.5 GB VRAM)
    "GAME_MODE_ENABLED": False,    # pydirectinput / pyautogui
    "STREAM_ENABLED": False,       # ffmpeg / OBS WebSocket
    "STREAM_PLATFORM": "file",     # file | rtmp | obs
    "STREAM_RTMP_URL": "",
    "STREAM_OUTPUT_DIR": "data/streams",
    "AUTO_STREAM": False,
    "VISION_MOCK": False,
    "GAME_MOCK": False,
    "STREAM_MOCK": False,
    "GAME_PROFILES_DIR": "data/game_profiles",
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
    "VISION_MOCK": True,
    "GAME_MOCK": True,
    "STREAM_MOCK": True,
}

# ═══════════════════════════════════════════════════════
# Выбор профиля
# ═══════════════════════════════════════════════════════
PROFILES = {
    "target": TARGET,
    "dev": DEV,
}

_config = dict(PROFILES.get(PROFILE, TARGET))

# ═══════════════════════════════════════════════════════
# Переменные окружения (.env) — перекрывают профиль
# ═══════════════════════════════════════════════════════
_ENV_MAP = {
    "OLLAMA_HOST": (str,),
    "LLM_PROVIDER": (str,),
    "LLM_BASE_URL": (str,),
    "LLM_API_KEY": (str,),
    "MODEL_BRAIN": (str,),
    "MODEL_EXTRACTOR": (str,),
    "MODEL_EMBED": (str,),
    "MODEL_VLM": (str,),
    "SAFETY_MODE": (str,),
    "API_PORT": (int,),
    "API_HOST": (str,),
    "VISION_ENABLED": (bool,),
    "GAME_MODE_ENABLED": (bool,),
    "STREAM_ENABLED": (bool,),
    "STREAM_PLATFORM": (str,),
    "STREAM_RTMP_URL": (str,),
    "TEMPERATURE": (float,),
    "MAX_HISTORY": (int,),
    "DEBUG": (bool,),
    "XTTS_MODEL_PATH": (str,),
    "TTS_DEVICE": (str,),
    "TTS_MIN_FREE_VRAM": (float,),
}
for _key, (_cast,) in _ENV_MAP.items():
    if os.getenv(_key) is not None:
        _config[_key] = _env(_key, _config.get(_key), _cast)

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
# Провайдер LLM (OpenAI-совместимый слой). Модель общается через /v1.
#   ollama — нативный Ollama (по умолчанию, локально, GPU)
#   jan    — Jan.ai Local API Server (OpenAI-совместимый, /v1)
#   openai — внешний OpenAI-совместимый сервер (LM Studio, vLLM и т.п.)
PROVIDER_DEFAULTS = {
    "ollama": ("http://localhost:11434", "ollama"),
    "llamacpp": ("http://127.0.0.1:8080", "none"),   # llama-server (GGUF)
    "jan": ("http://127.0.0.1:1337", "not-needed"),
    "openai": ("https://api.openai.com", ""),
}


def resolve_llm_endpoints(provider: str, base_url: str = None, api_key: str = None,
                          ollama_host: str = None):
    """Вернуть (base_url, api_key, api_url) для провайдера.

    base_url/api_key — явные переопределения (из .env). Для ollama базой по
    умолчанию служит OLLAMA_HOST, чтобы старое поведение сохранилось.
    """
    provider = (provider or "ollama").strip().lower()
    default_base, default_key = PROVIDER_DEFAULTS.get(
        provider, PROVIDER_DEFAULTS["ollama"]
    )
    if provider == "ollama" and ollama_host:
        default_base = ollama_host
    base = (base_url or default_base).rstrip("/")
    key = api_key if api_key is not None else default_key
    return base, key, f"{base}/v1"


LLM_PROVIDER = _env("LLM_PROVIDER", "ollama").strip().lower()
LLM_BASE_URL, LLM_API_KEY, LLM_API = resolve_llm_endpoints(
    LLM_PROVIDER,
    base_url=_env("LLM_BASE_URL", None),
    api_key=_env("LLM_API_KEY", None),
    ollama_host=OLLAMA_HOST,
)
# Обратная совместимость: старые модули импортируют OLLAMA_API
OLLAMA_API = LLM_API


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
