"""Первый запуск без действий пользователя: каталоги, .env, выбор модели."""

from pathlib import Path

from app.bootstrap import build_env_updates, prepare_home, pick_threads
from app.envfile import read_env
from app.llm import ModelInfo


def test_build_env_updates_llamacpp():
    updates = build_env_updates(
        provider="llamacpp",
        base_url="http://127.0.0.1:8080",
        model="qwythos-9b",
        api_port=8123,
    )
    assert updates["LLM_PROVIDER"] == "llamacpp"
    assert updates["LLM_BASE_URL"] == "http://127.0.0.1:8080"
    assert updates["MODEL_BRAIN"] == "qwythos-9b"
    assert updates["MODEL_EXTRACTOR"] == "qwythos-9b"
    assert updates["API_PORT"] == "8123"
    assert updates["API_HOST"] == "127.0.0.1"


def test_build_env_updates_ollama_keeps_extractor_default():
    updates = build_env_updates(provider="ollama", model="qwen3:4b", api_port=8000)
    assert updates["LLM_PROVIDER"] == "ollama"
    assert updates["OLLAMA_HOST"] == "http://localhost:11434"
    assert "MODEL_EXTRACTOR" not in updates


def test_build_env_updates_marks_selected_model():
    updates = build_env_updates(
        provider="llamacpp",
        base_url="http://127.0.0.1:8080",
        model="qwythos-9b",
        api_port=8123,
        model_path="C:/models/m/model.gguf",
    )
    assert updates["SOFIA_MODEL_PATH"] == "C:/models/m/model.gguf"
    assert updates["MODEL_BRAIN"] == "qwythos-9b"
    assert updates["MODEL_EXTRACTOR"] == "qwythos-9b"


def test_prepare_home_creates_env_and_dirs(tmp_path):
    home = prepare_home(tmp_path)
    assert home == tmp_path
    env = read_env(tmp_path / ".env")
    assert env["API_HOST"] == "127.0.0.1"
    assert env["SOFIA_PROFILE"] == "target"
    for rel in ("data/library", "data/voice_samples", "data/game_profiles", "logs"):
        assert (tmp_path / rel).is_dir()


def test_prepare_home_keeps_existing_env(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=ollama\nMODEL_BRAIN=qwen3:4b\n", encoding="utf-8")
    prepare_home(tmp_path)
    env = read_env(env_path)
    assert env["MODEL_BRAIN"] == "qwen3:4b"
    assert env["API_HOST"] == "127.0.0.1"


def test_prepare_home_seeds_game_profiles(tmp_path):
    prepare_home(tmp_path)
    profiles = list((tmp_path / "data" / "game_profiles").glob("*.json"))
    assert profiles, "должен появиться пример профиля игры"


def test_pick_threads_reasonable():
    assert 2 <= pick_threads(cores=16) <= 16
    assert pick_threads(cores=2) >= 1


def test_apply_model_updates_env(tmp_path):
    from app.bootstrap import apply_model_choice

    prepare_home(tmp_path)
    model = ModelInfo(Path("x/model.gguf"), "brain-x", 5.0, 36, False)
    apply_model_choice(tmp_path, model)
    env = read_env(tmp_path / ".env")
    assert env["SOFIA_MODEL_PATH"] == str(model.path)


def test_saved_model_used_when_present(tmp_path):
    from app.bootstrap import apply_model_choice, choose_model

    gguf = tmp_path / "m" / "model.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF" + b"\0" * 64)
    other = ModelInfo(tmp_path / "other.gguf", "other", 3.0, 30, False)
    saved = ModelInfo(gguf, "saved", 9.0, 33, False)
    prepare_home(tmp_path)
    apply_model_choice(tmp_path, saved)
    picked = choose_model(tmp_path, [other, saved], vram_gb=8.0, ram_gb=16.0)
    assert picked.path == gguf


def test_missing_saved_model_falls_back_to_auto(tmp_path):
    from app.bootstrap import apply_model_choice, choose_model

    prepare_home(tmp_path)
    gone = ModelInfo(tmp_path / "gone.gguf", "gone", 3.0, 30, False)
    apply_model_choice(tmp_path, gone)
    small = ModelInfo(tmp_path / "small.gguf", "small", 3.0, 30, False)
    assert choose_model(tmp_path, [small], vram_gb=8.0, ram_gb=16.0) == small


def test_bundled_xtts_cache_used_offline(tmp_path):
    from app.bootstrap import XTTS_REPO_DIR, configure_bundled_caches

    (tmp_path / "hf-cache" / "hub" / XTTS_REPO_DIR / "snapshots").mkdir(parents=True)
    env = {}
    used = configure_bundled_caches(tmp_path, env)
    assert "xtts" in used
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hf-cache" / "hub")
    assert env["HF_HUB_OFFLINE"] == "1"


def test_bundled_cache_respects_user_setting(tmp_path):
    from app.bootstrap import XTTS_REPO_DIR, configure_bundled_caches

    (tmp_path / "hf-cache" / "hub" / XTTS_REPO_DIR / "snapshots").mkdir(parents=True)
    env = {"HF_HUB_CACHE": "D:/mine"}
    assert configure_bundled_caches(tmp_path, env) == []
    assert env == {"HF_HUB_CACHE": "D:/mine"}


def test_no_bundle_changes_nothing(tmp_path):
    from app.bootstrap import configure_bundled_caches

    env = {}
    assert configure_bundled_caches(tmp_path, env) == []
    assert env == {}
