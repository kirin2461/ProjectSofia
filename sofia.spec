# -*- mode: python ; coding: utf-8 -*-
"""Сборка автономной Софии (PyInstaller, onedir).

    pyinstaller sofia.spec --noconfirm

Даёт dist/Sofia/Sofia.exe: интерфейс Qt6, встроенный llama-server, XTTS,
векторная память. Всё, чего нет на диске у пользователя (GGUF-модель,
llama.cpp), лаунчер докачает при первом запуске.

Всё тяжёлое (CUDA-сборка llama.cpp, кэш XTTS) кладётся в build/ скриптом
build.ps1 и попадает внутрь пакета — тогда интернет не нужен вовсе.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH)
BUILD = ROOT / "build"

datas = [
    (str(ROOT / "static"), "static"),
    (str(ROOT / "core" / "clips_rules"), "core/clips_rules"),
    (str(ROOT / "core" / "soar_rules"), "core/soar_rules"),
    (str(ROOT / "data" / "game_profiles" / "example.json"), "data/game_profiles"),
]

# ── Эмбеддинги (ONNX) ──────────────────────────────────────────────────────
# Без них поиск по книгам вырождается в случайный: all-MiniLM из Chroma
# обучена на английском. Кэш загрузок HF внутрь не кладём.
embed_dir = ROOT / "assets" / "embed"
if embed_dir.is_dir():
    for f in sorted(embed_dir.rglob("*")):
        if f.is_file() and ".cache" not in f.parts:
            datas.append((str(f), str(f.parent.relative_to(ROOT))))
else:
    print("[spec] нет assets/embed — поиск по книгам будет работать хуже")

# ── Данные пакетов, которые XTTS читает при загрузке ───────────────────────
# TTS/.models.json — каталог моделей (без него «No such file .models.json»),
# ko_speech_tools — таблицы jamo/g2p, импортируются токенизатором XTTS.
datas += collect_data_files(
    "TTS", include_py_files=False,
    excludes=["**/demos/**", "**/server/**", "**/*.md", "**/*.ipynb"],
)
datas += collect_data_files("ko_speech_tools")

# ── Тяжёлые ресурсы, подготовленные build.ps1 ──────────────────────────────
llama_dir = BUILD / "llama"
if llama_dir.is_dir():
    datas.append((str(llama_dir), "bin/llama"))

hf_cache = BUILD / "hf-cache"
if hf_cache.is_dir():
    datas.append((str(hf_cache), "hf-cache"))

chroma_cache = BUILD / "chroma-cache"
if chroma_cache.is_dir():
    datas.append((str(chroma_cache), "chroma-cache"))

# Tesseract целиком (exe + DLL + tessdata rus/eng): запускается как
# отдельный процесс, поэтому DLL должны лежать рядом с exe.
tess_dir = BUILD / "tesseract"
if tess_dir.is_dir():
    datas.append((str(tess_dir), "bin/tesseract"))
else:
    print("[spec] нет build/tesseract — сканы PDF останутся без текста")

# ── Метаданные: код читает версии через importlib.metadata ─────────────────
for dist in (
    "torch", "torchaudio", "torchcodec", "transformers", "tokenizers",
    "huggingface_hub", "safetensors", "coqui-tts", "coqpit-config", "trainer",
    "chromadb", "chromadb-rust-bindings", "onnxruntime", "soundfile", "librosa",
    "numba", "numpy", "scipy", "pymupdf", "uvicorn", "fastapi", "starlette",
    "openai", "watchdog", "psutil", "python-dotenv",
):
    try:
        datas += copy_metadata(dist)
    except Exception as exc:  # noqa: BLE001 — пакета может не быть
        print(f"[spec] нет метаданных {dist}: {exc}")

# ── Скрытые импорты ────────────────────────────────────────────────────────
hiddenimports = [
    "main",
    "app.bootstrap", "app.envfile", "app.llm", "app.paths", "app.runtime",
    "app.server", "app.ui",
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto", "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "websockets", "wsproto", "h11", "dotenv",
    "soundfile", "audioread", "pydub",
    "fitz", "pymupdf",
    # Импортируются лениво (внутри функций) — перечисляем явно
    "core.jsonx", "core.query_rewrite", "core.context",
    "memory.book_search", "memory.citation_check", "memory.feedback",
    "memory.profile", "memory.reranker",
    "modules.calculator", "modules.skills", "modules.autopilot",
    "modules.web_search",
    "sympy", "sympy.parsing.sympy_parser",
]
hiddenimports += collect_submodules("TTS")
hiddenimports += collect_submodules("trainer")
hiddenimports += collect_submodules("coqpit")
hiddenimports += collect_submodules("chromadb")
hiddenimports += collect_submodules("chromadb_rust_bindings")

# ── Продукционные системы: CLIPS (clipspy) и Soar (soar-sml) ───────────────
hiddenimports += collect_submodules("clips")
hiddenimports += ["soar_sml", "soar_sml._Python_sml_ClientInterface"]
binaries = collect_dynamic_libs("soar_sml")   # Soar.dll рядом с .pyd

# ВНИМАНИЕ: unittest исключать нельзя — его импортирует
# torch.utils._config_module (torch.distributed). Без него первый импорт
# torch.distributed обрывается после _c10d_init, а повторный роняет процесс
# через abort() («GradBucket already defined», код 0xC0000409).
excludes = [
    "tkinter", "test", "pytest", "IPython", "notebook", "jupyter",
    "matplotlib", "PySide6.Qt3D", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtQuick3D", "PySide6.QtBluetooth", "PySide6.QtMultimediaWidgets",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtWebEngineCore.dummy",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Sofia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Sofia",
)
