"""Реранкер: включается только при наличии модели, при сбое — молча выключается."""

from dataclasses import dataclass

from memory import reranker


@dataclass
class Hit:
    text: str
    score: float = 0.5


def test_unavailable_without_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SOFIA_RERANK_DIR", str(tmp_path / "нет"))
    assert not reranker.available()
    assert reranker.model_path() is None
    assert reranker.describe() == "выключен"


def test_available_with_model_file(monkeypatch, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    monkeypatch.setenv("SOFIA_RERANK_DIR", str(tmp_path))
    assert reranker.available()
    assert "включён" in reranker.describe()


def test_prefers_quantized_model(monkeypatch, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "model_quantized.onnx").write_bytes(b"x")
    monkeypatch.setenv("SOFIA_RERANK_DIR", str(tmp_path))
    assert reranker.model_path().name == "model_quantized.onnx"


def test_model_without_tokenizer_disables_quietly(monkeypatch, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    monkeypatch.setenv("SOFIA_RERANK_DIR", str(tmp_path))
    r = reranker.OnnxReranker(tmp_path)
    assert r.score("вопрос", ["абзац"]) == []
    assert r._failed


def test_rerank_reorders_by_score(monkeypatch):
    hits = [Hit("раз", 0.9), Hit("два", 0.2)]
    monkeypatch.setattr(reranker._reranker(), "score",
                        lambda q, texts: [0.1, 0.95])
    out = reranker.rerank(hits, "вопрос")
    assert [h.text for h in out] == ["два", "раз"]
    assert out[0].score > out[1].score


def test_rerank_blends_with_original(monkeypatch):
    hits = [Hit("раз", 1.0), Hit("два", 1.0)]
    monkeypatch.setattr(reranker._reranker(), "score", lambda q, texts: [0.0, 0.0])
    out = reranker.rerank(hits, "вопрос", weight=0.6)
    assert abs(out[0].score - 0.4) < 0.01


def test_rerank_keeps_order_when_scores_missing(monkeypatch):
    hits = [Hit("раз", 0.9), Hit("два", 0.2)]
    monkeypatch.setattr(reranker._reranker(), "score", lambda q, texts: [])
    assert [h.text for h in reranker.rerank(hits, "вопрос")] == ["раз", "два"]


def test_rerank_single_hit_untouched(monkeypatch):
    def boom(q, texts):
        raise AssertionError("не должен вызываться")

    monkeypatch.setattr(reranker._reranker(), "score", boom)
    assert len(reranker.rerank([Hit("раз", 0.9)], "вопрос")) == 1


def test_rerank_empty_query_untouched():
    assert reranker.rerank([Hit("а"), Hit("б")], "") == [Hit("а", 0.5), Hit("б", 0.5)]


def test_rerank_empty_hits():
    assert reranker.rerank([], "вопрос") == []


def test_sigmoid_bounds():
    assert 0.0 < reranker._sigmoid(-50) < 0.01
    assert 0.99 < reranker._sigmoid(50) <= 1.0
    assert abs(reranker._sigmoid(0) - 0.5) < 1e-9
