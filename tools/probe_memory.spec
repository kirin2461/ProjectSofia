# -*- mode: python ; coding: utf-8 -*-
"""Быстрая frozen-сборка пробника памяти: Chroma + ONNX (без torch/Qt)."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "tools" / "probe_memory.py")],
    pathex=[str(ROOT)],
    datas=[
        (str(ROOT / "config"), "config"),
        (str(ROOT / "memory"), "memory"),
    ],
    hiddenimports=collect_submodules("chromadb") + collect_submodules("tokenizers") + [
        "memory.memory_manager", "onnxruntime", "tqdm", "huggingface_hub",
        "overrides", "mmh3", "pybase64", "orjson",
    ],
    excludes=["torch", "torchaudio", "transformers", "TTS", "PySide6",
              "fastapi", "uvicorn", "openai", "soar_sml", "clips"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="probe_memory",
    debug=False, strip=False, upx=False, console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="probe_memory")
