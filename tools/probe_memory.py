"""Frozen-пробник памяти: Chroma + ONNX-эмбеддинги под нагрузкой.

В приложении `add_episode` идёт в потоке цикла, а `recall_episodes` — в
рабочем потоке, поэтому эмбеддер используется из двух потоков одновременно.
Проверяем, переживает ли это frozen-сборка.
"""

import sys
import threading
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def show_memory():
    from memory import memory_manager as mm

    print("backend:", mm.BACKEND)
    print("add(прогрев):", mm.add_episode("прогрев эмбеддера"))

    stop = threading.Event()
    done = {"add": 0, "query": 0}
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            try:
                mm.add_episode(f"Пользователь: вопрос {i}\nСофия: ответ {i}",
                               {"session_id": "probe", "timestamp": 1.0 * i,
                                "P": 0.1, "A": 0.2, "D": 0.3,
                                "emotion": "спокойствие", "tags": ["dialog"]})
                done["add"] += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"add: {e}")
                break

    def reader():
        i = 0
        while not stop.is_set():
            i += 1
            try:
                mm.recall_episodes(f"вопрос номер {i}")
                done["query"] += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"query: {e}")
                break

    threads = [threading.Thread(target=writer), threading.Thread(target=writer),
               threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()

    try:
        for i in range(20):
            mm.add_episode(f"главный поток {i}", {"session_id": "main"})
            mm.recall_episodes(f"главный {i}")
            print(f"  круг {i}: add={done['add']} query={done['query']}",
                  flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)

    print("итого:", done, "ошибок:", len(errors), errors[:3])


if __name__ == "__main__":
    show_memory()
    print("ГОТОВО")
