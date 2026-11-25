"""Упаковка: тяжёлые ресурсы, без которых сборка «самодостаточна», на месте.

Проверки не требуют PyInstaller: смотрим, что код ищет ресурсы внутри
сборки и что spec их туда кладёт.
"""

from pathlib import Path

import pytest

from app import paths
from memory import embedder as emb

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "sofia.spec").read_text(encoding="utf-8")


def test_model_dir_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFIA_EMBED_DIR", str(tmp_path / "m"))
    assert emb._model_dir() == tmp_path / "m"


def test_model_dir_inside_bundle(tmp_path, monkeypatch):
    monkeypatch.delenv("SOFIA_EMBED_DIR", raising=False)
    monkeypatch.setattr(paths, "bundled_dir", lambda: tmp_path / "_internal")

    assert emb._model_dir() == (
        tmp_path / "_internal" / "assets" / "embed" / "multilingual-e5-small"
    )


def test_spec_bundles_embeddings():
    # model_quantized.onnx + tokenizer.json в assets/embed, кэш HF — нет
    assert 'ROOT / "assets" / "embed"' in SPEC
    assert '".cache" not in f.parts' in SPEC


def test_spec_bundles_tesseract():
    assert 'BUILD / "tesseract"' in SPEC
    assert '"bin/tesseract"' in SPEC


def test_build_script_prepares_both():
    ps = (ROOT / "build.ps1").read_text(encoding="utf-8")
    assert "multilingual-e5-small" in ps
    assert "rus.traineddata" in ps


def test_embedded_model_files_present_if_dir_exists():
    d = ROOT / "assets" / "embed" / "multilingual-e5-small"
    if not d.is_dir():
        pytest.skip("модель эмбеддингов не подготовлена")
    assert (d / "tokenizer.json").is_file()
    assert any((d / "onnx").glob("*.onnx")), "нет onnx-файлов модели"
