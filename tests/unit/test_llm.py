"""Выбор GGUF-модели, поиск llama-server, аргументы запуска."""

import struct
from pathlib import Path

from app.llm import (
    ModelInfo,
    build_command,
    find_llama_server,
    gguf_meta,
    pick_model,
    scan_models,
)


def _gguf_bytes(pairs) -> bytes:
    """Минимальный GGUF: header + KV-пары (типы: 8=str, 4=u32, 5=i32)."""
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    for key, (vtype, value) in pairs:
        raw = key.encode("utf-8")
        out += struct.pack("<Q", len(raw)) + raw + struct.pack("<I", vtype)
        if vtype == 8:
            vb = value.encode("utf-8")
            out += struct.pack("<Q", len(vb)) + vb
        elif vtype == 4:
            out += struct.pack("<I", value)
        elif vtype == 5:
            out += struct.pack("<i", value)
    return out


def _fake_model(root: Path, name: str, size_gb: float, meta=None, yml=None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    path = d / "model.gguf"
    path.write_bytes(_gguf_bytes(meta or []))
    with open(path, "r+b") as f:  # разреженный файл нужного размера
        f.truncate(int(size_gb * 1024 ** 3))
    if yml:
        (d / "model.yml").write_text(yml, encoding="utf-8")
    return path


def test_gguf_meta_reads_scalars(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf_bytes([
        ("general.architecture", (8, "qwen3")),
        ("qwen3.block_count", (4, 36)),
        ("qwen3.context_length", (4, 32768)),
    ]))
    meta = gguf_meta(path)
    assert meta["general.architecture"] == "qwen3"
    assert meta["qwen3.block_count"] == 36


def test_scan_skips_aux_and_tmp(tmp_path):
    good = _fake_model(tmp_path, "brain", 0.2, [("qwen3.block_count", (4, 36))],
                       "name: brain-9b\nembedding: false\n")
    (good.parent / "mmproj.gguf").write_bytes(b"x" * 1024)
    (good.parent / "model.gguf.tmp").write_bytes(b"x" * 1024)
    _fake_model(tmp_path, "embed", 0.2, yml="name: mini\nembedding: true\n")
    _fake_model(tmp_path, "tiny", 0.001)

    models = scan_models([tmp_path], min_size_gb=0.05)
    names = {m.name for m in models}
    assert "brain-9b" in names
    assert "mini" in names  # embeddings видны, но помечены
    assert all("mmproj" not in str(m.path) for m in models)
    assert len(models) == 2
    brain = next(m for m in models if m.name == "brain-9b")
    assert brain.layers == 36
    assert brain.embedding is False
    assert next(m for m in models if m.name == "mini").embedding is True


def test_pick_prefers_largest_that_fits_vram():
    models = [
        ModelInfo(Path("a"), "small", 4.0, 32, False),
        ModelInfo(Path("b"), "medium", 5.7, 36, False),
        ModelInfo(Path("c"), "huge", 9.8, 40, False),
    ]
    assert pick_model(models, vram_gb=8.0, ram_gb=16.0).name == "medium"


def test_pick_ignores_embeddings():
    models = [
        ModelInfo(Path("a"), "embed", 0.05, None, True),
        ModelInfo(Path("b"), "chat", 3.0, 32, False),
    ]
    assert pick_model(models, vram_gb=8.0, ram_gb=16.0).name == "chat"


def test_pick_falls_back_to_partial_offload():
    models = [
        ModelInfo(Path("a"), "huge", 9.8, 40, False),
        ModelInfo(Path("b"), "bigger", 14.0, 48, False),
    ]
    # 8 VRAM + 50% от 16 RAM = 16 GB → 14 помещается частично, 16 не влезает
    assert pick_model(models, vram_gb=8.0, ram_gb=16.0).name == "bigger"


def test_pick_smallest_when_nothing_fits():
    models = [ModelInfo(Path("a"), "a", 30.0, None, False),
              ModelInfo(Path("b"), "b", 20.0, None, False)]
    assert pick_model(models, vram_gb=8.0, ram_gb=8.0).name == "b"


def test_pick_empty():
    assert pick_model([], vram_gb=8.0, ram_gb=16.0) is None


def test_find_llama_server_prefers_newest_build(tmp_path):
    for build in ("b12000", "b12606", "b9000"):
        d = tmp_path / build
        d.mkdir(parents=True)
        (d / "llama-server.exe").write_bytes(b"exe")
    found = find_llama_server([tmp_path])
    assert found is not None and found.parent.name == "b12606"


def test_find_llama_server_none(tmp_path):
    assert find_llama_server([tmp_path]) is None


def test_build_command_has_required_flags(tmp_path):
    model = ModelInfo(tmp_path / "model.gguf", "my-model", 5.0, 36, False)
    cmd = build_command(Path("C:/llama/llama-server.exe"), model, port=8080, ctx=8192,
                        threads=8)
    assert cmd[0].endswith("llama-server.exe")
    joined = " ".join(cmd)
    for flag in ("--model", "--alias my-model", "--port 8080", "--ctx-size 8192",
                 "--n-gpu-layers", "--jinja", "--no-webui", "--parallel 1",
                 "--cache-type-k q8_0", "--threads 8"):
        assert flag in joined


def test_build_command_respects_explicit_ngl(tmp_path):
    model = ModelInfo(tmp_path / "model.gguf", "m", 5.0, 36, False)
    cmd = build_command(Path("llama-server.exe"), model, port=8080, ctx=4096, ngl=20)
    idx = cmd.index("--n-gpu-layers")
    assert cmd[idx + 1] == "20"
