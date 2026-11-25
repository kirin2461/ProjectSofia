"""Тесты SOAR-моста: настоящий Soar (SML) и резервная симуляция."""

import pytest

from core.soar_bridge import (
    CLARIFY_QUESTION,
    NativeSoar,
    SimulatedSoar,
    SOARBridge,
    classify,
)


# ── Признаки сообщения (их видят правила) ───────────────────
@pytest.mark.parametrize(
    "text,has_context,expected",
    [
        ("привет!", False, {"greeting": True, "question": False, "vague": False}),
        ("расскажи про космос", False, {"greeting": False, "question": True}),
        ("что такое энтропия?", False, {"greeting": False, "question": True}),
        ("мне нравится эта музыка", False, {"question": False, "vague": False}),
        ("как это?", False, {"question": True, "vague": True}),
        ("как это?", True, {"context": True}),
    ],
)
def test_classify(text, has_context, expected):
    features = classify(text, has_context)
    for key, value in expected.items():
        assert features[key] is value, feature_message(text, key, features)


def feature_message(text, key, features):
    return f"{text!r}: {key}={features[key]} в {features}"


# ── Настоящий Soar ──────────────────────────────────────────
@pytest.fixture
def native(tmp_path):
    pytest.importorskip("soar_sml")
    soar = NativeSoar(chunks_path=str(tmp_path / "chunks.soar"))
    yield soar
    soar.close()


def test_native_engine_is_real_soar(native):
    assert native.status()["engine"] == "soar"
    assert native.status()["active"] is True
    assert native._rules >= 10
    assert native.learned_skills == []


@pytest.mark.parametrize(
    "text,has_context,operator",
    [
        ("привет", False, "op_greet"),
        ("здравствуй, София!", False, "op_greet"),
        ("расскажи про чёрные дыры", False, "op_answer_question"),
        ("что такое энтропия?", False, "op_answer_question"),
        ("мне нравится эта книга", False, "op_answer_question"),
        ("как это?", False, "op_ask_clarification"),
        ("как это?", True, "op_answer_question"),
    ],
)
def test_native_decisions(native, text, has_context, operator):
    action = native.step(text, has_context=has_context)
    assert action.operator == operator
    if operator == "op_ask_clarification":
        assert action.payload["question"] == CLARIFY_QUESTION


def test_native_learns_chunk_at_tie_impasse(native):
    assert native.learned_skills == []
    native.step("как это?")
    assert len(native.learned_skills) == 1
    assert native.learned_skills[0].startswith("chunk")
    native.feedback("как это?", native.step("как это?"), success=True)
    assert native.chunks_path.is_file()
    text = native.chunks_path.read_text(encoding="utf-8")
    assert "ask-clarification" in text


def test_native_chunks_survive_restart(tmp_path):
    pytest.importorskip("soar_sml")
    path = str(tmp_path / "chunks.soar")

    first = NativeSoar(chunks_path=path)
    first.step("как это?")
    learned = first.learned_skills
    assert learned
    first.feedback("как это?", first.step("как это?"), success=True)
    first.close()

    second = NativeSoar(chunks_path=path)
    try:
        assert second.learned_skills == learned
        assert second.step("как это?").operator == "op_ask_clarification"
    finally:
        second.close()


def test_native_does_not_leak_working_memory(native):
    for i in range(50):
        native.step(f"расскажи факт номер {i}")
    link = native._agent.GetInputLink()
    messages = [
        link.GetChild(i)
        for i in range(link.GetNumberChildren())
        if link.GetChild(i).GetAttribute() == "message"
    ]
    assert len(messages) == 1
    assert native._agent.GetNumberCommands() <= 5


def test_native_failure_keeps_answering(native, monkeypatch):
    def boom(features):
        raise RuntimeError("Soar упал")

    monkeypatch.setattr(native, "_run", boom)
    assert native.step("привет").operator == "op_answer_question"


# ── Мост ────────────────────────────────────────────────────
def test_bridge_prefers_native_by_default(tmp_path, monkeypatch):
    pytest.importorskip("soar_sml")
    monkeypatch.setattr("core.soar_bridge.SOAR_CHUNKS_PATH", str(tmp_path / "c.soar"))
    bridge = SOARBridge()
    try:
        assert bridge.engine == "soar"
        assert bridge.status()["rules"] >= 10
    finally:
        bridge.close()


def test_bridge_falls_back_to_simulation():
    bridge = SOARBridge(prefer_native=False)
    assert bridge.engine == "simulated"
    assert bridge.status()["active"] is False


def test_bridge_needs_clarification_returns_question():
    bridge = SOARBridge(prefer_native=False)
    assert bridge.needs_clarification("как это?") == CLARIFY_QUESTION


# ── Симуляция (резерв) ──────────────────────────────────────
def test_greeting_operator():
    bridge = SOARBridge(prefer_native=False)
    assert bridge.step("привет").operator == "op_greet"


def test_ambiguous_goes_to_clarification():
    bridge = SOARBridge(prefer_native=False)
    action = bridge.step("как это?")
    assert action.operator == "op_ask_clarification"
    assert action.payload["question"]


def test_question_operator():
    bridge = SOARBridge(prefer_native=False)
    assert bridge.step("расскажи про космос подробнее").operator == "op_answer_question"


def test_propose_goal_id():
    bridge = SOARBridge(prefer_native=False)
    assert bridge.propose_goal("цель").startswith("goal_")


def test_chunking_after_three_successes():
    sim = SimulatedSoar()
    ctx = "расскажи про космос"
    action = sim.step(ctx)
    assert sim.learned_skills == []
    for _ in range(3):
        sim.feedback(ctx, action, success=True)
    assert len(sim.learned_skills) == 1
    # Выученный чанк возвращается без изменений
    assert sim.step(ctx).operator == action.operator


def test_no_chunk_on_failure():
    sim = SimulatedSoar()
    ctx = "расскажи про космос"
    action = sim.step(ctx)
    for _ in range(5):
        sim.feedback(ctx, action, success=False)
    assert sim.learned_skills == []
