"""Проверка голосового стека: XTTS v2 синтез в файл.

Запуск:
    python scripts/test_voice.py
    python scripts/test_voice.py --text "Привет!" --out out.wav

Если в data/voice_samples есть *.wav — голос клонируется по ним,
иначе берётся встроенный диктор XTTS.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import VOICE_SAMPLES_DIR  # noqa: E402
from modules.tts_engine import TTSEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Привет! Я София. Голосовой стек работает.")
    ap.add_argument("--out", default="voice_test.wav")
    ap.add_argument("--speaker", default=None,
                    help="встроенный диктор XTTS (если нет образцов)")
    args = ap.parse_args()

    engine = TTSEngine()
    print(f"device: {engine.device}")
    samples = engine.speaker_samples()
    print(f"образцы голоса: {len(samples)} шт. в {VOICE_SAMPLES_DIR}")

    t0 = time.perf_counter()
    engine._load()  # noqa: SLF001 — нужен явный прогрев, чтобы мерить отдельно
    load_s = time.perf_counter() - t0
    if engine._tts is None:  # noqa: SLF001
        print(f"✗ модель не загрузилась: {engine._load_error}")  # noqa: SLF001
        return 1
    print(f"загрузка модели: {load_s:.1f}s")

    speakers = getattr(engine._tts, "speakers", None) or []  # noqa: SLF001
    if speakers and not samples:
        print(f"встроенных дикторов: {len(speakers)}; первый: {speakers[0]}")

    t0 = time.perf_counter()
    audio = engine.synthesize(args.text)
    synth_s = time.perf_counter() - t0
    if not audio:
        print("✗ синтез не удался")
        return 1

    Path(args.out).write_bytes(audio)
    seconds = max(0.0, (len(audio) - 44) / 2 / 24000)  # XTTS: 24 кГц, 16 бит
    print(f"✓ {len(audio) / 1024:.0f} KB, ~{seconds:.1f}s речи за {synth_s:.1f}s "
          f"→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
