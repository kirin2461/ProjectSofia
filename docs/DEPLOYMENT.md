# Деплой: с VDS на ноутбук

## Философия

- **VDS** — фабрика кода. Здесь пишется и собирается. **Но только 8 GB RAM** — нельзя запускать весь стек.
- **Ноутбук** — дом. Здесь работает ассистент. **16 GB RAM** — здесь можно всё.

## ⚠️ Критическое предупреждение про VDS

VDS имеет **8 GB RAM** — меньше, чем ноутбук (16 GB).

Что это значит:
- Нельзя одновременно держать qwen3:4b в GPU + Chroma в RAM + XTTS + FastAPI
- На VDS тестируйте компоненты по отдельности
- Полная интеграция — только на ноутбуке

## Подготовка VDS

```bash
# 1. Базовое окружение (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3-pip python3-venv git

# 2. Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 3. Модели (по необходимости, не все сразу)
ollama pull qwen3:4b
# ollama pull qwen3:1.7b  # когда нужен экстрактор
# ollama pull bge-m3      # когда нужна память

# 4. Репозиторий
git clone https://github.com/kirin2461/ProjectSofia.git
cd ProjectSofia
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Проверка (одна модель за раз!)
python scripts/check_ollama.py
```

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

### Вариант A: Git (рекомендуется)

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
| `data/library/` | Книги | Скопировать вручную или синхронизировать |
| `data/safety_log.jsonl` | Логи | Не переносить |

## Тестирование на VDS (с 8 GB RAM)

```bash
# 1. Тест одного компонента за раз
# Например, только когнитивный цикл без памяти:
python -c "from core.cognitive_cycle import extract_facts; ..."

# 2. Проверка RAM во время работы
# В другом терминале:
watch -n 1 free -h

# 3. Если RAM заканчивается — убейте Ollama и перезапустите:
sudo systemctl restart ollama
```

## Тестирование на ноутбуке (полное)

```bash
# На ноутбуке
python scripts/check_ollama.py
python scripts/benchmark_target.py
python scripts/setup.py
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Бенчмарк Target

`scripts/benchmark_target.py` проверяет:

1. **VRAM**: загружает qwen3:4b, проверяет < 7.5 GB
2. **RAM**: проверяет доступную память (> 10 GB свободно на ноутбуке)
3. **Скорость**: 10 запросов к 4b, среднее latency < 3 сек
4. **Контекст**: 32k с qwen3:4b

## Troubleshooting

### На VDS: не хватает RAM (8 GB)
```bash
# Симптом: OOM killer, процессы падают
# Решения:
# 1. Запускайте только один сервис за раз
# 2. Остановите Ollama перед тестированием Python-кода:
sudo systemctl stop ollama
# 3. Используйте swap (если не настроен):
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

### На ноутбуке: не хватает RAM (16 GB)
- Закрыть Chrome/Edge
- Проверить `data/` — векторная база может разрастись
- Уменьшить `num_ctx` в `settings.py`

## Чеклист деплоя

- [ ] Код на VDS написан и протестирован по частям
- [ ] `git push` выполнен
- [ ] На ноутбуке `git pull` выполнен
- [ ] Ollama модели скачаны
- [ ] `voice_samples/` скопированы
- [ ] `setup.py` прошёл без ошибок
- [ ] **Полное тестирование на ноутбуке пройдено**

---

**Правило:** VDS = пишем код. Ноутбук = запускаем всё. На VDS 8 GB RAM — не пытайтесь запустить полный стек.
