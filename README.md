# Project Sofia

**SOAR + LLM CASCADE + COMPUTER VISION** — автономный ИИ-ассистент, игрок и стример.

- **SOAR**: цели, импассы, чанкинг (обучение навыкам)
- **LLM Cascade**: qwen3:1.7b (экстрактор) → qwen3:4b (мозг + критик)
- **CLIPS**: детерминированная безопасность и жизненный цикл памяти
- **Chroma**: эпизодическая память + библиотека книг
- **XTTS v2**: клонирование голоса, полностью локально
- **Qwen3-VL-2B**: компьютерное зрение — "глаза" Софии
- **YOLOv8-nano**: рефлексивное зрение — детекция объектов в реальном времени
- **Game Control**: управление мышью и клавиатурой — "руки" Софии
- **Streaming Engine**: захват, оверлей, трансляция на Twitch/YouTube

## 🎯 Целевое устройство (Target)

**Где работает ассистент — единственная машина с GPU:**
- NVIDIA RTX 5050 8 ГБ VRAM / 16 ГБ RAM
- AMD Ryzen 7 260AI + 780M (iGPU ограничена в BIOS до 512 МБ)
- Windows 11

### VRAM-бюджет в режиме "Игра"
```
qwen3:4b (мозг)        ~5.7 GB  ← постоянно
Qwen3-VL-2B (глаза)    ~1.5 GB  ← постоянно
YOLOv8-nano (рефлексы) ~0.1 GB  ← постоянно
────────────────────────────────
Итого:                 ~7.3 GB  ← влезает!

XTTS v2 (~2 GB):       выгружается во время игры
                        (голос отключается или заменяется на beep)
```

## 🖥️ Среда разработки (Dev)

**Где пишется код — CPU-only:**
- VDS: 8 ядер CPU / **8 ГБ RAM** / 75 ГБ SSD
- **Нет GPU** — нельзя запускать LLM, TTS, Vision
- Linux (Ubuntu 22.04+ рекомендуется)

⚠️ **Важно:**
- На VDS **нет GPU** — Ollama будет работать на CPU (невероятно медленно)
- На VDS только **8 ГБ RAM** — меньше, чем на ноутбуке
- 75 ГБ — это диск (SSD), не оперативка

## ⚡ Быстрый старт

### На VDS (разработка — только код, без LLM/GPU)
```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Папки
mkdir -p data/voice_samples data/library data/memory_db data/vector_db data/streams data/game_profiles

# 3. Линтинг и проверка синтаксиса
python -m py_compile core/*.py memory/*.py modules/*.py main.py

# 4. Юнит-тесты (без LLM — mock'и)
# pytest tests/unit/
```

### На ноутбуке (target — полный запуск)
```bash
# 1. Ollama (Windows): https://ollama.com/download/windows
# 2. В PowerShell:
ollama pull qwen3:4b qwen3:1.7b bge-m3

# 3. Для зрения (опционально, если нужен VLM):
ollama pull qwen3-vl:2b

# 4. Зависимости
pip install -r requirements.txt

# 5. Проверка
python scripts/check_ollama.py
python scripts/benchmark_target.py

# 6. Запуск (базовый чат)
uvicorn main:app --host 0.0.0.0 --reload

# 7. Запуск игрового режима (в коде)
# WebSocket: {"type": "game", "action": "start", "profile": "minecraft"}
```

## 🎮 Игровой режим

София может играть в игры и стримить процесс:

1. **Видит экран** через `dxcam` (60 FPS) + Qwen3-VL-2B (описание сцены)
2. **Детектирует объекты** через YOLOv8-nano (враги, предметы)
3. **Прицеливается и уклоняется** через рефлексивный слой (< 100 мс)
4. **Принимает стратегические решения** через мозг (qwen3:4b)
5. **Управляет мышью и клавиатурой** через `pydirectinput`
6. **Стримит** с оверлеем мыслей и эмоций через ffmpeg/OBS

### Профили игр

Создавайте профили в `data/game_profiles/<игра>.json`:

```json
{
  "name": "Minecraft",
  "window_title": "Minecraft*",
  "hp_region": [20, 20, 100, 20],
  "key_bindings": {
    "move_forward": "W",
    "move_backward": "S",
    "move_left": "A",
    "move_right": "D",
    "attack": "LMB",
    "use": "RMB",
    "jump": "SPACE",
    "inventory": "E"
  },
  "reactive_rules": [
    {"trigger": "enemy_near", "action": "aim", "priority": 10},
    {"trigger": "hp_low", "action": "heal", "priority": 9}
  ]
}
```

## 📂 Документация
- [Архитектура](docs/ARCHITECTURE.md) — поток данных, компоненты, среды, VRAM-бюджет
- [Дорожная карта](docs/ROADMAP.md) — фазы и чеклисты
- [Деплой](docs/DEPLOYMENT.md) — как переносить с VDS на ноутбук
- [Промпт для агента](docs/AGENT_PROMPT.md) — инструкция AI-разработчику на VDS
- [Задачи для нейросети](docs/TASKS.md) — готовые промпты для AI-ассистентов

## 🗂️ Структура
- `core/` — когнитивный цикл, эмоции, навыки, CLIPS-правила
- `memory/` — эпизодическая память, индексатор библиотеки
- `modules/` — TTS, аватар, зрение, управление, стриминг
- `config/` — настройки (локальные в `settings_local.py`)
- `scripts/` — утилиты, бенчмарки, проверки

## 🎙️ Голос
Для клонирования положите 5–8 WAV-файлов (10–30 сек, 22+ кГц, моно, тихая комната) в `data/voice_samples/`. Не коммитьте их в git.

## 📺 Стриминг
Для трансляции на Twitch/YouTube:
1. Установите ffmpeg (добавьте в PATH)
2. Установите OBS Studio + obs-websocket
3. Получите Stream Key на платформе
4. Настройте в `config/settings_local.py`:
   ```python
   STREAM_PLATFORM = "rtmp"
   STREAM_RTMP_URL = "rtmp://live.twitch.tv/app/YOUR_KEY"
   ```

## 📜 Лицензия
- Код: GPL-3.0
- XTTS v2: Coqui Public Model License (некоммерческое использование)

---

**Как использовать этот репозиторий:** откройте [docs/TASKS.md](docs/TASKS.md), скопируйте задачу и вставьте в промпт вашему AI-ассистенту. Агент работает на VDS (CPU-only, 8 GB RAM), но код пишется под ноутбук (RTX 5050, 16 GB RAM).
