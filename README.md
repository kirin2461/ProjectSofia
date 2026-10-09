# Project Sofia

**SOAR + LLM CASCADE** — автономный ИИ-ассистент с когнитивной архитектурой.

- **SOAR**: цели, импассы, чанкинг (обучение навыкам)
- **LLM Cascade**: qwen3:1.7b (экстрактор) → qwen3:4b (мозг + критик)
- **CLIPS**: детерминированная безопасность и жизненный цикл памяти
- **Chroma**: эпизодическая память + библиотека книг
- **XTTS v2**: клонирование голоса, полностью локально

## 🎯 Целевое устройство (Target)

**Где работает ассистент — единственная машина с GPU:**
- NVIDIA RTX 5050 8 ГБ VRAM / 16 ГБ RAM
- AMD Ryzen 7 260AI + 780M (iGPU ограничена в BIOS до 512 МБ)
- Windows 11

## 🖥️ Среда разработки (Dev)

**Где пишется код — CPU-only:**
- VDS: 8 ядер CPU / **8 ГБ RAM** / 75 ГБ SSD
- **Нет GPU** — нельзя запускать LLM и TTS
- Linux (Ubuntu 22.04+ рекомендуется)

⚠️ **Важно:**
- На VDS **нет GPU** — Ollama будет работать на CPU (невероятно медленно)
- На VDS только **8 ГБ RAM** — меньше, чем на ноутбуке
- 75 ГБ — это диск (SSD), не оперативка

## ⚡ Быстрый старт

### На VDS (разработка — только код, без LLM)
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
```bash
# 1. Ollama (Windows): https://ollama.com/download/windows
# 2. В PowerShell:
ollama pull qwen3:4b qwen3:1.7b bge-m3

# 3. Зависимости
pip install -r requirements.txt

# 4. Проверка
python scripts/check_ollama.py
python scripts/benchmark_target.py

# 5. Запуск
uvicorn main:app --host 0.0.0.0 --reload
```

## 📂 Документация
- [Архитектура](docs/ARCHITECTURE.md) — поток данных, компоненты, среды
- [Дорожная карта](docs/ROADMAP.md) — фазы и чеклисты
- [Деплой](docs/DEPLOYMENT.md) — как переносить с VDS на ноутбук
- [Промпт для агента](docs/AGENT_PROMPT.md) — инструкция AI-разработчику на VDS
- [Задачи для нейросети](docs/TASKS.md) — готовые промпты для AI-ассистентов

## 🗂️ Структура
- `core/` — когнитивный цикл, эмоции, навыки, CLIPS-правила
- `memory/` — эпизодическая память, индексатор библиотеки
- `modules/` — TTS, аватар, интеграции
- `config/` — настройки (локальные в `settings_local.py`)
- `scripts/` — утилиты, бенчмарки, проверки

## 🎙️ Голос
Для клонирования положите 5–8 WAV-файлов (10–30 сек, 22+ кГц, моно, тихая комната) в `data/voice_samples/`. Не коммитьте их в git.

## 📜 Лицензия
- Код: GPL-3.0
- XTTS v2: Coqui Public Model License (некоммерческое использование)

---

**Как использовать этот репозиторий:** откройте [docs/TASKS.md](docs/TASKS.md), скопируйте задачу и вставьте в промпт вашему AI-ассистенту. Агент работает на VDS (CPU-only, 8 GB RAM), но код пишется под ноутбук (RTX 5050, 16 GB RAM).
