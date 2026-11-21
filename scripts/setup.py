"""Phase-0-02: Подготовка окружения — создание папок данных.

Можно полностью запускать на VDS (не требует GPU):
    python scripts/setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    VOICE_SAMPLES_DIR,
    LIBRARY_PATH,
    MEMORY_DB_PATH,
    VECTOR_DB_PATH,
    STREAM_OUTPUT_DIR,
    GAME_PROFILES_DIR,
)

ROOT = Path(__file__).resolve().parent.parent


def ensure_dir(rel_path: str) -> bool:
    """Создаёт папку, если её нет. Возвращает True, если создана сейчас."""
    path = (ROOT / rel_path).resolve()
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return not existed


def main() -> int:
    dirs = [
        VOICE_SAMPLES_DIR,
        LIBRARY_PATH,
        MEMORY_DB_PATH,
        VECTOR_DB_PATH,
        STREAM_OUTPUT_DIR,
        GAME_PROFILES_DIR,
    ]

    for rel in dirs:
        created = ensure_dir(rel)
        mark = "✓" if created else "•"
        suffix = "создана" if created else "уже есть"
        print(f"{mark} {rel} {suffix}")

    # Предупреждение, если нет голосовых образцов
    voice_dir = (ROOT / VOICE_SAMPLES_DIR).resolve()
    wavs = list(voice_dir.glob("*.wav"))
    if not wavs:
        print(
            f"\n⚠ {VOICE_SAMPLES_DIR} пустая — добавьте 5–8 WAV "
            "(10–30 сек, 22+ кГц, моно) для клонирования голоса."
        )
    else:
        print(f"\n✓ Найдено {len(wavs)} WAV-образцов голоса.")

    print("\nГотово. Проверьте модели: python scripts/check_ollama.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
