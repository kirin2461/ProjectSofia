"""Слот контекста: одно число на llama-server и на бюджет истории.

llama-server поднимался с --ctx-size из pick_ctx() (8192 в реальном логе),
а config/settings.NUM_CTX=32768 не попадал никуда: num_ctx сервер
игнорирует. Бюджет истории считался от NUM_CTX, то есть от числа, которого
у сервера нет.
"""

import os
from pathlib import Path

import pytest

from app.llm import ModelInfo
from app.runtime import (
    DEFAULT_CTX,
    LARGE_CTX,
    Runtime,
    pick_ctx,
    resolve_ctx,
)
from config.settings import NUM_CTX


def _model(size_gb=4.0, layers=32):
    return ModelInfo(Path("m/model.gguf"), "brain", size_gb, layers, False)


@pytest.fixture
def no_env_ctx(monkeypatch):
    monkeypatch.delenv("NUM_CTX", raising=False)


def test_settings_and_runtime_agree_on_default(no_env_ctx):
    assert DEFAULT_CTX == NUM_CTX


def test_ctx_from_process_env_wins(no_env_ctx, monkeypatch, tmp_path):
    monkeypatch.setenv("NUM_CTX", "12288")
    assert resolve_ctx(tmp_path) == 12288


def test_ctx_from_env_file(no_env_ctx, tmp_path):
    (tmp_path / ".env").write_text("NUM_CTX=20480\n", encoding="utf-8")
    assert resolve_ctx(tmp_path) == 20480


def test_process_env_beats_env_file(no_env_ctx, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("NUM_CTX=20480\n", encoding="utf-8")
    monkeypatch.setenv("NUM_CTX", "12288")
    assert resolve_ctx(tmp_path) == 12288


def test_missing_ctx_means_hardware_guess(no_env_ctx, tmp_path):
    assert resolve_ctx(tmp_path) is None


def test_broken_ctx_falls_back(no_env_ctx, monkeypatch, tmp_path):
    monkeypatch.setenv("NUM_CTX", "много")
    assert resolve_ctx(tmp_path) is None


@pytest.mark.parametrize("value", ["0", "-4096", "12"])
def test_ctx_below_sane_minimum_ignored(no_env_ctx, monkeypatch, tmp_path, value):
    monkeypatch.setenv("NUM_CTX", value)
    assert resolve_ctx(tmp_path) is None


def test_pick_ctx_still_hardware_aware():
    assert pick_ctx(_model(4.0), vram_gb=8.0) == LARGE_CTX
    assert pick_ctx(_model(7.0), vram_gb=8.0) == DEFAULT_CTX
    assert pick_ctx(_model(1.0), vram_gb=0.0) == DEFAULT_CTX


# ── Публикация слота в конфиг ──────────────────────────────
def test_apply_env_publishes_slot(no_env_ctx, tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    rt.model = _model()
    rt.api_port = 8123
    rt.ctx = 16384
    rt._apply_env()
    assert os.environ["NUM_CTX"] == "16384"


def test_apply_env_publishes_default_when_nothing_chosen(no_env_ctx, tmp_path):
    rt = Runtime(tmp_path, lambda *_: None)
    rt.model = _model()
    rt.api_port = 8124
    rt._apply_env()
    assert os.environ["NUM_CTX"] == str(DEFAULT_CTX)
