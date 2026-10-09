# Деплой: с VDS на ноутбук

## Философия

- **VDS** — фабрика. Здесь пишется код, прогоняются тесты, собирается релиз.
- **Ноутбук** — дом. Здесь работает ассистент, здесь ограничения.

Никакого Docker, Kubernetes и прочей инфраструктуры — обе машины управляются вручную через git.

## Подготовка VDS

```bash
# 1. Базовое окружение
sudo apt update && sudo apt install -y python3-pip python3-venv git

# 2. Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 3. Модели
ollama pull qwen3:4b qwen3:1.7b bge-m3

# 4. Репозиторий
git clone https://github.com/kirin2461/ProjectSofia.git
cd ProjectSofia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Проверка (необязательно, но удобно)
ollama serve &
python scripts/check_ollama.py
```

## Подготовка ноутбука (Target)

```powershell
# 1. Ollama для Windows
# Скачать с https://ollama.com/download/windows
# Установить, убедиться что служба запущена (Ollama в трее)

# 2. В PowerShell:
ollama pull qwen3:4b qwen3:1.7b bge-m3

# 3. Python (установлен через Microsoft Store или python.org)
pip install -r requirements.txt

# 4. 780M в BIOS
# Перезагрузка → BIOS → UMA Frame Buffer Size → 512 MB
# Сохранить, выйти

# 5. Файл подкачки
# Система → Дополнительные параметры → Быстродействие → Дополнительно → Виртуальная память
# Минимум 16384 МБ на SSD
```

## Перенос кода

### Вариант A: Git (рекомендуется)

```bash
# На VDS
git add .
git commit -m "feat: новая фича"
git push origin main

# На ноутбуке
git pull origin main
```

### Вариант B: Rsync (если git не настроен)

```bash
# На VDS
rsync -avz --exclude 'venv/' --exclude '__pycache__/' --exclude 'data/' \
  ./ user@ноутбук-ip:/path/to/ProjectSofia/
```

### Что НЕ переносится

| Папка | Почему | Что делать |
|---|---|---|
| `venv/` | Зависит от ОС | Создать заново: `python -m venv venv` |
| `data/memory_db/` | Локальная база | Переносить опционально (экспорт/импорт Chroma) |
| `data/vector_db/` | Индексы книг | Переиндексировать на ноутбуке |
| `data/voice_samples/` | Голос | Скопировать вручную (не в git) |
| `data/library/` | Книги | Скопировать вручную или синхронизировать через облако |
| `data/safety_log.jsonl` | Логи | Не переносить |

## Проверка после переноса

```bash
# На ноутбуке
python scripts/check_ollama.py      # Все модели на месте?
python scripts/benchmark_target.py  # VRAM и RAM в норме?
python scripts/setup.py             # Папки созданы?

# Тестовый запуск
uvicorn main:app --host 0.0.0.0 --port 8000
# Открыть http://localhost:8000 в браузере
```

## Бенчмарк Target

`scripts/benchmark_target.py` проверяет:

1. **VRAM**: загружает qwen3:4b + XTTS, проверяет что суммарно < 7.5 GB
2. **RAM**: проверяет доступную память (> 10 GB свободно)
3. **Скорость**: 10 тестовых запросов к 4b, среднее latency < 3 сек
4. **Контекст**: проверяет 32k контекст с qwen3:4b

Если бенчмарк падает — не деплоить, чинить на VDS.

## Обновление моделей

Если на VDS протестировали новую модель (например, qwen3:8b для dev):

1. Убедиться, что она НЕ попадает в target-конфиг по умолчанию
2. В `config/settings.py` оставить `MODEL_BRAIN = "qwen3:4b"`
3. Для dev-переопределения использовать `settings_local.py` или переменные окружения

```python
# config/settings_local.py (только на VDS)
MODEL_BRAIN = "qwen3:8b"
NUM_CTX = 65536
```

## Troubleshooting

### Ollama не видит GPU на ноутбуке
```powershell
# Проверить
ollama list
# Если модели грузятся в RAM (медленно) — проверить драйверы NVIDIA
# Должно быть: CUDA 12.x, драйвер 550+
```

### Chroma не запускается
```bash
# Обычно конфликт версий sqlite
pip install --upgrade chromadb
```

### XTTS не находит CUDA
```python
# В modules/tts_engine.py добавить fallback
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Не хватает RAM
- Закрыть Chrome/Edge
- Проверить `data/` — возможно, векторная база разрослась
- Уменьшить `num_ctx` в `settings.py`

## Чеклист деплоя

- [ ] Код на VDS протестирован
- [ ] `benchmark_target.py` проходит на VDS (эмуляция) или ноутбуке
- [ ] Git push сделан
- [ ] На ноутбуке `git pull` выполнен
- [ ] Ollama модели скачаны
- [ ] `voice_samples/` скопированы
- [ ] `setup.py` прошёл без ошибок
- [ ] Тестовый диалог прошёл успешно
- [ ] Голос работает (если Phase 1 готова)

---

**Правило:** если что-то работает на VDS, но не работает на ноутбуке — баг. Чинить на VDS, тестировать на ноутбуке.
