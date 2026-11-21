# Project Sofia

**SOAR + LLM CASCADE** — автономный ИИ-ассистент с когнитивной архитектурой.

**Основа (всегда включено):**
- **SOAR**: цели, импассы, чанкинг (обучение навыкам)
- **LLM Cascade**: qwen3:1.7b (экстрактор) → qwen3:4b (мозг + критик)
- **CLIPS**: детерминированная безопасность и жизненный цикл памяти
- **Chroma**: эпизодическая память + библиотека книг
- **XTTS v2**: клонирование голоса, полностью локально

**Опционально (расширения):**
- **Qwen3-VL-2B**: компьютерное зрение — "глаза" Софии. При включении **автоматически заменяет** qwen3:1.7b как экстрактор — экономия 1.2 GB VRAM!
- **YOLOv8-nano**: рефлексивное зрение — детекция объектов в реальном времени
- **Game Control**: управление мышью и клавиатурой — "руки" Софии
- **Streaming Engine**: захват, оверлей, трансляция на Twitch/YouTube

> Расширения отключаются в конфиге. Базовый ассистент работает без них.

## 🎯 Целевое устройство (Target)

**Где работает ассистент — единственная машина с GPU:**
- NVIDIA RTX 5050 8 ГБ VRAM / 16 ГБ RAM
- AMD Ryzen 7 260AI + 780M (iGPU ограничена в BIOS до 512 МБ)
- Windows 11

### VRAM-бюджет

```
БАЗОВЫЙ РЕЖИМ (ассистент):
qwen3:4b (мозг)        ~5.7 GB  ← постоянно
qwen3:1.7b (экстрактор)~1.2 GB  ← постоянно
────────────────────────────────
Итого база:            ~6.9 GB  ← запас 1+ GB

ИГРОВОЙ РЕЖИМ (опционально):
qwen3:4b (мозг)        ~5.7 GB  ← постоянно
qwen3-vl:2b (VLM)      ~1.5 GB  ← зрение + экстрактор (заменяет 1.7b!)
YOLOv8-nano (рефлексы) ~0.1 GB  ← если включено
────────────────────────────────
Итого с играми:        ~7.3 GB  ← влезает!

XTTS v2 (голос)        ~2.0 GB  ← по требованию, выгружается в игре
```

**Ключевая оптимизация:** при включении зрения VLM (qwen3-vl:2b) автоматически берёт на себя роль экстрактора фактов. Не нужно держать qwen3:1.7b отдельно — **экономия 1.2 GB VRAM**.

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
mkdir -p data/voice_samples data/library data/memory_db data/vector_db

# 3. Линтинг и проверка синтаксиса
python -m py_compile core/*.py memory/*.py modules/*.py main.py

# 4. Юнит-тесты (без LLM — mock'и)
# pytest tests/unit/
```

### На ноутбуке (target — полный запуск)

Вариант A — **llama.cpp / llama-server** (локальные GGUF, без Ollama):
```powershell
# 1. Скачать сборку llama.cpp (CUDA) один раз:
#    https://github.com/ggml-org/llama.cpp/releases
#    -> llama-bXXXXX-bin-win-cuda-12.4-x64.zip + cudart-llama-bin-win-cuda-12.4-x64.zip
#    распаковать оба в %LOCALAPPDATA%\llama.cpp\bXXXXX

# 2. Запустить сервер (модель и число слоёв — параметрами):
powershell -ExecutionPolicy Bypass -File scripts\start_llm_server.ps1 `
    -Model "C:\путь\к\model.gguf" -Alias my-model -GpuLayers 29

# 3. Прописать в .env:
#    LLM_PROVIDER=llamacpp
#    MODEL_BRAIN=my-model
#    MODEL_EXTRACTOR=my-model

# 4. Проверка и запуск
python scripts/check_ollama.py
uvicorn main:app --host 127.0.0.1 --reload
```

Вариант B — **Ollama** (Windows): https://ollama.com/download/windows
```bash
ollama pull qwen3:4b qwen3:1.7b bge-m3          # + qwen3-vl:2b для зрения
pip install -r requirements.txt
python scripts/check_ollama.py
uvicorn main:app --host 0.0.0.0 --reload
# в .env: LLM_PROVIDER=ollama
```

```bash
# Включить зрение (VLM станет экстрактором, 1.7b не нужен):
# В config/settings_local.py:
#   VISION_ENABLED = True
#   GAME_MODE_ENABLED = True
#   STREAM_ENABLED = True
```

## 🎮 Игровой режим (опционально)

Если включён в конфиге (`VISION_ENABLED = True`), София получает "глаза":

1. **Видит экран** через `dxcam` (60 FPS) + Qwen3-VL-2B (описание сцены)
2. **Детектирует объекты** через YOLOv8-nano (враги, предметы)
3. **Прицеливается и уклоняется** через рефлексивный слой (< 100 мс)
4. **Принимает стратегические решения** через мозг (qwen3:4b)
5. **Управляет мышью и клавиатурой** через `pydirectinput`
6. **Стримит** с оверлеем мыслей и эмоций через ffmpeg/OBS

При этом Qwen3-VL-2B **автоматически** обрабатывает экстракцию фактов из диалога — отдельная qwen3:1.7b не загружается.

Без включения в конфиге — модули не загружаются и не занимают VRAM.

### Профили игр

Создавайте профили в `data/game_profiles/<игра>.json` (см. `data/game_profiles/example.json`).

## 📂 Документация
- [Архитектура](docs/ARCHITECTURE.md) — поток данных, компоненты, среды, VRAM-бюджет
- [Дорожная карта](docs/ROADMAP.md) — фазы и чеклисты
- [Деплой](docs/DEPLOYMENT.md) — как переносить с VDS на ноутбук
- [Промпт для агента](docs/AGENT_PROMPT.md) — инструкция AI-разработчику на VDS
- [Задачи для нейросети](docs/TASKS.md) — готовые промпты для AI-ассистентов

## 🗂️ Структура
- `core/` — когнитивный цикл, эмоции, навыки, CLIPS-правила
- `memory/` — эпизодическая память, индексатор библиотеки
- `modules/` — TTS, аватар, зрение (опц.), управление (опц.), стриминг (опц.)
- `config/` — настройки (локальные в `settings_local.py`)
- `scripts/` — утилиты, бенчмарки

## 🎙️ Голос (XTTS v2)
Установка (опционально, Phase 1):
```powershell
# 1. torch под GPU RTX 50xx (Blackwell — только CUDA 12.8+):
pip install --index-url https://download.pytorch.org/whl/cu128 `
    "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128"
# 2. движок голоса:
pip install -r requirements-voice.txt
# 3. проверка (скачает модель ~1.9 GB при первом запуске):
python scripts/test_voice.py
```
Для клонирования голоса положите 5–8 WAV-файлов (10–30 сек, 22+ кГц, моно, тихая
комната) в `data/voice_samples/`. Без образцов используется встроенный диктор.
Не коммитьте образцы в git.

Движок сам выбирает устройство: GPU, если свободно ≥ `TTS_MIN_FREE_VRAM` GB
(по умолчанию 2 ГБ), иначе CPU. На 8-ГБ карте вместе с 9B-LLM свободной VRAM
не остаётся, поэтому XTTS работает на CPU: real-time factor ≈ 3.5 (≈3.5 с
синтеза на 1 с речи). Чтобы ответ не молчал десятки секунд, София озвучивает
его по предложениям — первый аудио-фрагмент уходит почти сразу, браузер
проигрывает их по очереди.

Переопределения (`.env` или `settings_local.py`): `TTS_DEVICE` (`auto`/`cpu`/
`cuda`), `TTS_MIN_FREE_VRAM` (порог в ГБ). XTTS v2 — некоммерческая лицензия
CPML; движок выставляет `COQUI_TOS_AGREED=1` автоматически.

## 📺 Стриминг (опционально)
Для трансляции на Twitch/YouTube:
1. Установите ffmpeg (добавьте в PATH)
2. Установите OBS Studio + obs-websocket
3. Получите Stream Key на платформе
4. Настройте в `config/settings_local.py`:
   ```python
   STREAM_ENABLED = True
   STREAM_PLATFORM = "rtmp"
   STREAM_RTMP_URL = "rtmp://live.twitch.tv/app/YOUR_KEY"
   ```

## 📜 Лицензия
- Код: GPL-3.0
- XTTS v2: Coqui Public Model License (некоммерческое использование)

---

**Как использовать этот репозиторий:** откройте [docs/TASKS.md](docs/TASKS.md), скопируйте задачу и вставьте в промпт вашему AI-ассистенту. Агент работает на VDS (CPU-only, 8 GB RAM), но код пишется под ноутбук (RTX 5050, 16 GB RAM).
