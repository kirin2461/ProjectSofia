"""Phase-0-01: Проверка доступности Ollama и моделей.

Запускать на ноутбуке (с GPU). На VDS Ollama не установлена — только
проверка синтаксиса (`python -m py_compile scripts/check_ollama.py`).

Использует только requests и openai (из requirements.txt).
"""

import sys
import time
from pathlib import Path

# Корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from config.settings import (  # noqa: E402
    LLM_API,
    LLM_API_KEY,
    LLM_PROVIDER,
    OLLAMA_HOST,
    MODEL_BRAIN,
    MODEL_EXTRACTOR,
    MODEL_EMBED,
)


def _fetch_tags():
    """GET /api/tags — список установленных моделей Ollama."""
    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def _fetch_v1_models():
    """GET /v1/models — список моделей любого OpenAI-совместимого сервера."""
    resp = requests.get(f"{LLM_API}/models", timeout=10,
                        headers={"Authorization": f"Bearer {LLM_API_KEY}"})
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def _model_present(model: str, installed: list) -> bool:
    """Ollama хранит теги вида 'qwen3:4b'; сверяем по префиксу."""
    base = model.split(":")[0]
    return any(name == model or name.startswith(base + ":") for name in installed)


def _bench_llm(client, model: str):
    """Тестовый chat-вызов, возвращает (ok, latency_ms, note)."""
    start = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Скажи 'ок'."}],
            max_tokens=10,
        )
    except Exception as e:  # noqa: BLE001
        return False, 0.0, f"ошибка: {e}"
    latency_ms = (time.perf_counter() - start) * 1000
    tokens = r.usage.completion_tokens if r.usage else 10
    tps = tokens / (latency_ms / 1000) if latency_ms else 0
    return True, latency_ms, f"~{tps:.0f} ток/с"


def _bench_embed(client, model: str):
    """Тестовый embedding-вызов, возвращает (ok, latency_ms, note)."""
    start = time.perf_counter()
    try:
        r = client.embeddings.create(model=model, input="test")
    except Exception as e:  # noqa: BLE001
        return False, 0.0, f"ошибка: {e}"
    latency_ms = (time.perf_counter() - start) * 1000
    dim = len(r.data[0].embedding)
    return True, latency_ms, f"embed {dim}d"


def main() -> int:
    is_ollama = LLM_PROVIDER == "ollama"
    where = OLLAMA_HOST if is_ollama else LLM_API
    print(f"Провайдер: {LLM_PROVIDER} ({where})\n")

    # 1. Доступность сервера и список моделей
    try:
        installed = _fetch_tags() if is_ollama else _fetch_v1_models()
    except requests.exceptions.ConnectionError:
        hint = "`ollama serve`" if is_ollama else "scripts/start_llm_server.ps1"
        print(f"✗ Сервер не отвечает: {where}\n  Запустите: {hint}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"✗ Ошибка подключения: {e}")
        return 1

    from openai import OpenAI
    client = OpenAI(base_url=LLM_API, api_key=LLM_API_KEY or "none")

    checks = [
        (MODEL_BRAIN, "llm"),
        (MODEL_EXTRACTOR, "llm"),
    ]
    # Эмбеддинги есть только у Ollama (у llama-server нужен --embeddings)
    if is_ollama:
        checks.append((MODEL_EMBED, "embed"))

    rows = []
    all_ok = True
    for model, kind in checks:
        if not _model_present(model, installed):
            hint = f"ollama pull {model}" if is_ollama else "выберите другую --alias"
            rows.append((model, "НЕТ", "—", hint))
            all_ok = False
            continue
        if kind == "embed":
            ok, latency, note = _bench_embed(client, model)
        else:
            ok, latency, note = _bench_llm(client, model)
        status = "OK" if ok else "FAIL"
        all_ok = all_ok and ok
        latency_str = f"{latency:.0f}ms" if ok else "—"
        rows.append((model, status, latency_str, note))

    # Таблица
    print(f"{'Модель':<14} {'Статус':<8} {'Latency':<9} Примечание")
    for model, status, latency, note in rows:
        print(f"{model:<14} {status:<8} {latency:<9} {note}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
