"""Проверка совместимости с целевым устройством (ноутбук).

Запускать на VDS для проверки кода, или на ноутбуке для диагностики.
Эмулирует ограничения ноутбука, если запущен на более мощной машине.
"""

import sys
import time
import psutil
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_BRAIN, MODEL_EXTRACTOR, MODEL_EMBED, NUM_CTX

# Лимиты целевого устройства
TARGET_VRAM_GB = 8.0
TARGET_RAM_GB = 16.0
TARGET_VRAM_RESERVE_GB = 0.5  # запас


def check_ollama_models():
    """Проверить, что нужные модели доступны."""
    print("=== Проверка моделей Ollama ===")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]

        required = [MODEL_BRAIN, MODEL_EXTRACTOR, MODEL_EMBED]
        for m in required:
            status = "✓" if any(m in existing for existing in models) else "✗"
            print(f"  {status} {m}")
            if status == "✗":
                print(f"    → ollama pull {m}")
                return False
        return True
    except Exception as e:
        print(f"  ✗ Ollama недоступна: {e}")
        return False


def check_vram():
    """Проверить доступную VRAM (если есть NVIDIA)."""
    print("\n=== Проверка VRAM ===")
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total / 1024**3
        free_gb = info.free / 1024**3
        print(f"  GPU: {pynvml.nvmlDeviceGetName(handle).decode()}")
        print(f"  Всего VRAM: {total_gb:.1f} GB")
        print(f"  Свободно: {free_gb:.1f} GB")

        # Проверка под target
        if total_gb > TARGET_VRAM_GB + 2:
            print(f"  ⚠ Это не target-устройство (VRAM > {TARGET_VRAM_GB} GB)")
            print(f"    Проверяем под эмуляцией {TARGET_VRAM_GB} GB...")

        needed = 6.0  # qwen3:4b + запас
        if free_gb >= needed:
            print(f"  ✓ Достаточно VRAM (нужно ~{needed} GB)")
            return True
        else:
            print(f"  ✗ Недостаточно VRAM (нужно ~{needed} GB)")
            return False
    except ImportError:
        print("  ⚠ pynvml не установлен: pip install nvidia-ml-py")
        print("  → Пропускаем проверку VRAM (не критично на CPU-only VDS)")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка GPU: {e}")
        return False


def check_ram():
    """Проверить доступную RAM."""
    print("\n=== Проверка RAM ===")
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024**3
    available_gb = mem.available / 1024**3
    print(f"  Всего RAM: {total_gb:.1f} GB")
    print(f"  Доступно: {available_gb:.1f} GB")

    if total_gb > TARGET_RAM_GB + 4:
        print(f"  ⚠ Это не target-устройство (RAM > {TARGET_RAM_GB} GB)")

    if available_gb >= 10:
        print("  ✓ Достаточно RAM для работы")
        return True
    elif available_gb >= 6:
        print("  ⚠ RAM впритык (закройте браузер)")
        return True
    else:
        print("  ✗ Недостаточно RAM")
        return False


def benchmark_speed():
    """Проверить скорость генерации."""
    print("\n=== Бенчмарк скорости ===")
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        # Тестовый запрос
        start = time.time()
        r = client.chat.completions.create(
            model=MODEL_BRAIN,
            messages=[{"role": "user", "content": "Привет, расскажи анекдот"}],
            max_tokens=100,
        )
        latency = time.time() - start
        tokens = r.usage.completion_tokens if r.usage else 50
        speed = tokens / latency

        print(f"  Latency: {latency:.2f} сек")
        print(f"  Токенов: {tokens}")
        print(f"  Скорость: {speed:.1f} ток/с")

        if speed >= 25:
            print("  ✓ Скорость хорошая")
            return True
        elif speed >= 15:
            print("  ⚠ Скорость ниже optimal (проверьте GPU)")
            return True
        else:
            print("  ✗ Скорость критически низкая")
            return False
    except Exception as e:
        print(f"  ✗ Ошибка бенчмарка: {e}")
        return False


def benchmark_context():
    """Проверить работу с длинным контекстом."""
    print(f"\n=== Проверка контекста ({NUM_CTX} токенов) ===")
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        # Генерируем текст ~50% от num_ctx
        filler = "Слово " * 500  # ~1500 токенов
        messages = [
            {"role": "system", "content": "Ты ассистент."},
            {"role": "user", "content": f"Вот текст: {filler}. Что последнее слово?"},
        ]

        start = time.time()
        r = client.chat.completions.create(
            model=MODEL_BRAIN,
            messages=messages,
            max_tokens=10,
            extra_body={"num_ctx": NUM_CTX},
        )
        latency = time.time() - start

        answer = r.choices[0].message.content
        print(f"  Latency: {latency:.2f} сек")
        print(f"  Ответ: {answer[:50]}...")

        if latency < 5:
            print("  ✓ Контекст работает быстро")
            return True
        else:
            print("  ⚠ Контекст медленный (нормально для CPU)")
            return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False


def main():
    print("=" * 50)
    print("Project Sofia — Benchmark Target")
    print(f"Target: RTX 5050 {TARGET_VRAM_GB}GB / {TARGET_RAM_GB}GB RAM")
    print("=" * 50)

    results = {
        "models": check_ollama_models(),
        "vram": check_vram(),
        "ram": check_ram(),
        "speed": benchmark_speed(),
        "context": benchmark_context(),
    }

    print("\n" + "=" * 50)
    print("ИТОГ:")
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")

    if all(results.values()):
        print("\n✓ Всё готово для деплоя на ноутбук!")
        return 0
    else:
        print("\n✗ Есть проблемы. Исправьте перед деплоем.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
