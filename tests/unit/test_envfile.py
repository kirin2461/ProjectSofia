"""Правка .env без потери комментариев и порядка строк."""

from app.envfile import parse_env, read_env, upsert_env, upsert_text

SAMPLE = """# Локальный llama-server
LLM_PROVIDER=llamacpp
MODEL_BRAIN=qwythos-9b
MODEL_EXTRACTOR=qwythos-9b

# Ollama для эмбеддингов
OLLAMA_HOST=http://localhost:11434

API_PORT=8000
"""


def test_parse_strips_quotes_and_skips_comments():
    assert parse_env('A=1\n# c\nB="two words"\nC=\nD=\'x\'\n') == {
        "A": "1",
        "B": "two words",
        "C": "",
        "D": "x",
    }


def test_parse_ignores_invalid_lines():
    assert parse_env("broken line\nA=1\n=2\n") == {"A": "1"}


def test_upsert_replaces_in_place_and_appends_new():
    out = upsert_text(SAMPLE, {"MODEL_BRAIN": "new-model", "API_PORT": "8123", "EXTRA": "1"})
    assert "MODEL_BRAIN=new-model" in out
    assert "API_PORT=8123" in out
    assert "EXTRA=1" in out
    # комментарии и порядок сохранены
    assert "# Ollama для эмбеддингов" in out
    assert out.index("LLM_PROVIDER") < out.index("MODEL_BRAIN") < out.index("OLLAMA_HOST")
    assert out.index("MODEL_BRAIN") < out.index("API_PORT")


def test_upsert_keeps_unrelated_values():
    out = parse_env(upsert_text(SAMPLE, {"API_PORT": "9000"}))
    assert out["LLM_PROVIDER"] == "llamacpp"
    assert out["OLLAMA_HOST"] == "http://localhost:11434"


def test_upsert_env_writes_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(SAMPLE, encoding="utf-8")
    merged = upsert_env(path, {"API_HOST": "127.0.0.1"})
    assert merged["API_HOST"] == "127.0.0.1"
    assert read_env(path)["API_HOST"] == "127.0.0.1"
    assert path.read_text(encoding="utf-8").startswith("# Локальный llama-server")


def test_read_env_missing_file(tmp_path):
    assert read_env(tmp_path / "nope.env") == {}


def test_upsert_text_adds_trailing_newline():
    assert upsert_text("A=1", {"B": "2"}).endswith("\n")
