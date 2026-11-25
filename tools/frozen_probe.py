"""Frozen-пробник: Soar и CLIPS в собранном виде, без torch/FastAPI."""

import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

here = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(here.parent))
print("frozen:", bool(getattr(sys, "frozen", False)), "base:", here)
print("MEIPASS:", getattr(sys, "_MEIPASS", None))


def show_soar():
    from core.soar_bridge import NativeSoar

    chunks = Path(tempfile.mkdtemp()) / "chunks.soar"
    soar = NativeSoar(chunks_path=str(chunks))
    print("soar:", soar.status())

    script = [
        ("привет", False),
        ("как это?", False),          # vague + question без контекста → tie
        ("спасибо", True),
        ("как это?", False),          # должен сработать выученный чанк
        ("Что такое энтропия?", True),
        ("расскажи про космос", True),
    ]
    for i, (text, ctx) in enumerate(script, 1):
        action = soar.step(text, has_context=ctx)
        print(f"  #{i} {text!r} ctx={ctx} → {action.operator} | last={soar.status()['last']!r}"
              f" | chunks={len(soar.learned_skills)}")
        soar.feedback(text, action, success=True)
        print(f"     после feedback: chunks={len(soar.learned_skills)}")

    print("файл чанков существует:", chunks.exists(),
          chunks.stat().st_size if chunks.exists() else 0)
    if chunks.exists():
        print("---- чанки ----")
        print(chunks.read_text(encoding="utf-8")[:600])
    soar.close()


def show_clips():
    from core.safety_engine import SafetyEngine

    engine = SafetyEngine()
    print("clips:", engine.clips.available, engine.clips.rule_count)
    for text in ["привет", "как это?", "Что такое энтропия?", "спасибо",
                 "мой CVV 123", "Покажи карту мира"]:
        r = engine.check_input(text)
        print(f"  {text!r} → {r.action} ({r.rule})")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("clips", "both"):
        show_clips()
    if which in ("soar", "both"):
        show_soar()
    print("ГОТОВО")
