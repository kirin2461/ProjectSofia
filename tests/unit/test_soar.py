"""Юнит-тесты SOAR-моста (Phase 5-01). Симуляция, без Soar SML."""

from core.soar_bridge import SOARBridge, SimulatedSoar


def test_greeting_operator():
    bridge = SOARBridge()
    action = bridge.step("привет")
    assert action.operator == "op_greet"


def test_ambiguous_goes_to_clarification():
    bridge = SOARBridge()
    action = bridge.step("как это?")
    assert action.operator == "op_ask_clarification"
    assert action.payload["question"]


def test_question_operator():
    bridge = SOARBridge()
    assert bridge.step("расскажи про космос подробнее").operator == "op_answer_question"


def test_propose_goal_id():
    bridge = SOARBridge()
    gid = bridge.propose_goal("цель")
    assert gid.startswith("goal_")


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


def test_needs_clarification_returns_question():
    bridge = SOARBridge()
    assert bridge.needs_clarification("как это?")
