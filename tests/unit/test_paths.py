"""Пути лаунчера: портативная установка, запись данных, каталоги."""

from pathlib import Path

from app import paths


def test_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.BASE_ENV, str(tmp_path))
    assert paths.resolve_base_dir() == tmp_path.resolve()


def test_default_uses_install_dir(monkeypatch):
    monkeypatch.delenv(paths.BASE_ENV, raising=False)
    assert paths.resolve_base_dir() == paths.install_dir()


def test_unwritable_install_falls_back_to_localappdata(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.BASE_ENV, raising=False)
    monkeypatch.setattr(paths, "install_dir", lambda: tmp_path / "Program Files")
    monkeypatch.setattr(paths, "is_writable", lambda _p: False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert paths.resolve_base_dir() == (tmp_path / "local" / paths.APP_NAME).resolve()


def test_ensure_dirs_creates_layout(tmp_path):
    paths.ensure_dirs(tmp_path)
    for rel in (
        "data",
        "data/library",
        "data/voice_samples",
        "data/game_profiles",
        "data/streams",
        "logs",
    ):
        assert (tmp_path / Path(rel)).is_dir()


def test_env_path(tmp_path):
    assert paths.env_path(tmp_path) == tmp_path / ".env"


def test_is_writable_on_real_dir(tmp_path):
    assert paths.is_writable(tmp_path) is True
