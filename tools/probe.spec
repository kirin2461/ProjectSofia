# -*- mode: python ; coding: utf-8 -*-
"""Быстрая frozen-сборка пробника: только Soar + CLIPS (для отладки падений)."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "tools" / "frozen_probe.py")],
    pathex=[str(ROOT)],
    binaries=collect_dynamic_libs("soar_sml"),
    datas=[
        (str(ROOT / "core" / "clips_rules"), "core/clips_rules"),
        (str(ROOT / "core" / "soar_rules"), "core/soar_rules"),
        (str(ROOT / "config"), "config"),
    ],
    hiddenimports=collect_submodules("clips") + [
        "soar_sml", "soar_sml._Python_sml_ClientInterface",
        "core.soar_bridge", "core.safety_engine",
    ],
    excludes=["torch", "torchaudio", "transformers", "chromadb", "TTS",
              "PySide6", "fastapi", "uvicorn", "openai", "numpy"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="probe",
    debug=False, strip=False, upx=False, console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="probe")
