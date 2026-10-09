# Деплой: с VDS на ноутбук

## Философия

- **VDS** — фабрика кода. Здесь пишется и собирается. **CPU-only, 8 GB RAM.**
- **Ноутбук** — дом. Здесь работает ассистент. **RTX 5050, 16 GB RAM.**

## ⚠️ Критические ограничения VDS

### Нет GPU
- На VDS **нет видеокарты**
- Ollama будет работать на CPU — **невероятно медленно** (1-2 ток/с вместо 40+)
- **Не пытайтесь запускать LLM на VDS** — это бессмысленно
- XTTS (голос) на VDS не работает — требует CUDA

### 8 GB RAM
- Даже без GPU RAM ограничена
- Нельзя запускать полный стек

## Что делать на VDS

```bash
# ✅ Пишем код
nano/vim/cursor core/cognitive_cycle.py

# ✅ Проверяем синтаксис
python -m py_compile core/*.py memory/*.py modules/*.py main.py

# ✅ Юнит-тесты с mock (без LLM)
# pytest tests/unit/ -v

# ✅ Линтинг
# ruff check . || flake8 .

# ✅ Git
git add . && git commit -m "feat: ..." && git push
```

## Что НЕ делать на VDS

```bash
# ❌ Не запускай Ollama для тестов
ollama run qwen3:4b  # Будет 1 ток/с на CPU — бессмысленно

# ❌ Не запускай полный бэкенд
uvicorn main:app  # Потребует Ollama, не хватит RAM

# ❌ Не тестируй XTTS
python modules/tts_engine.py  # Нет CUDA, упадёт

# ❌ Не запускай benchmark_target.py
# Он предназначен только для ноутбука с GPU
```

## Подготовка VDS

```bash
# 1. Базовое окружение (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3-pip python3-venv git

# 2. Репозиторий
git clone https://github.com/kirin2461/ProjectSofia.git
cd ProjectSofia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Готово к разработке (без LLM)
```

**Не устанавливай Ollama на VDS** — она не нужна без GPU.

## Подготовка ноутбука (Target)

```powershell
# 1. Ollama для Windows
# Скачать с https://ollama.com/download/windows

# 2. В PowerShell:
ollama pull qwen3:4b qwen3:1.7b bge-m3

# 3. Python
pip install -r requirements.txt

# 4. 780M в BIOS
# Перезагрузка → BIOS → UMA Frame Buffer Size → 512 MB

# 5. Файл подкачки
# Минимум 16384 МБ на SSD
```

## Перенос кода

```bash
# На VDS
git add .
git commit -m "feat: описание"
git push origin main

# На ноутбуке
git pull origin main
```

### Что НЕ переносится

| Папка | Почему | Что делать |
|---|---|---|
| `venv/` | Зависит от ОС | Создать заново |
| `data/memory_db/` | Локальная база | Переносить опционально |
| `data/vector_db/` | Индексы книг | Переиндексировать на ноутбуке |
| `data/voice_samples/` | Голос | Скопировать вручную |
| `data/library/` | Книги | Скопировать вручную |
| `data/safety_log.jsonl` | Логи | Не переносить |

## Тестирование

### На VDS (только без LLM)

```bash
# Синтаксис
python -m py_compile main.py

# Импорты
python -c "from core.cognitive_cycle import extract_facts; print('OK')"

# Не запускай тесты, требующие Ollama!
```

### На ноутбуке (полное)

```bash
# Проверка моделей
python scripts/check_ollama.py

# Бенчмарк (VRAM, скорость, контекст)
python scripts/benchmark_target.py

# Настройка
python scripts/setup.py

# Запуск
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### На VDS: "CUDA not available" при тесте
Это нормально — на VDS нет GPU. Тестируйте на ноутбуке.

### На ноутбуке: Ollama не видит GPU
```powershell
# Проверить драйверы NVIDIA
nvidia-smi
# Должно быть: CUDA 12.x, драйвер 550+
```

### На ноутбуке: не хватает RAM
- Закрыть Chrome/Edge
- Уменьшить `num_ctx` в `settings.py`

## Чеклист деплоя

- [ ] Код на VDS написан и проверен (синтаксис)
- [ ] `git push` выполнен
- [ ] На ноутбуке `git pull` выполнен
- [ ] Ollama модели скачаны
- [ ] `voice_samples/` скопированы
- [ ] `setup.py` прошёл без ошибок
- [ ] `check_ollama.py` прошёл
- [ ] `benchmark_target.py` прошёл
- [ ] **Полное тестирование на ноутбуке пройдено**

---

**Правило:** VDS = пишем код. Ноутбук = запускаем всё. На VDS нет GPU — не тестируй LLM там.
