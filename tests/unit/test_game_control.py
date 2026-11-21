"""Юнит-тесты управления игрой (Phase 6-02). Mock-контроллер, без GUI."""

import logging
import time

import pytest

from modules.game_control import (
    ActionPlan,
    ActionSafety,
    CognitiveActions,
    MockController,
    ReactiveActions,
)


@pytest.fixture
def mock():
    return MockController()


def test_parse_move_and_click():
    ca = CognitiveActions(MockController())
    plans = ca.parse_plan("move to 100,200\nclick 50 60")
    assert plans[0].action == "move_to"
    assert plans[0].coordinates == (100, 200)
    assert plans[1].action == "click_at"
    assert plans[1].coordinates == (50, 60)


def test_parse_press_and_wait():
    ca = CognitiveActions(MockController())
    plans = ca.parse_plan("press e\nwait 2.5")
    assert plans[0].action == "press_key"
    assert plans[0].key == "e"
    assert plans[1].action == "wait"
    assert plans[1].duration == 2.5


def test_parse_unknown_lines_ignored():
    ca = CognitiveActions(MockController())
    assert ca.parse_plan("думать о жизни\nmove 10 20")[0].action == "move_to"


def test_execute_plan_logs(mock, caplog):
    ca = CognitiveActions(mock)
    ca.execute_plan(ActionPlan(action="move_to", coordinates=(10, 20)))
    ca.execute_plan(ActionPlan(action="press_key", key="w"))
    log = ca.get_log()
    assert [e["action"] for e in log] == ["move_to", "press_key"]


def test_execute_plans_sequence(mock):
    ca = CognitiveActions(mock)
    ca.execute_plans([ActionPlan("press_key", key="a"),
                      ActionPlan("press_key", key="b")])
    assert len(ca.get_log()) == 2


def test_action_safety_rejects_out_of_screen():
    s = ActionSafety(screen_size=(1920, 1080))
    assert s.validate(ActionPlan("move_to", coordinates=(100, 100)))
    assert not s.validate(ActionPlan("move_to", coordinates=(-1, 100)))
    assert not s.validate(ActionPlan("move_to", coordinates=(2000, 100)))


def test_action_safety_rate_limit():
    s = ActionSafety()
    results = [s.rate_limit() for _ in range(10)]
    assert all(results)
    assert s.rate_limit() is False


def test_reactive_track_smoothing(mock):
    ra = ReactiveActions(mock)
    ra.track_target((0, 0, 100, 100))  # цель без предыдущей
    assert ra._target == (50, 50)
    ra.track_target((100, 100, 100, 100))  # плавно смещается
    assert 50 < ra._target[0] < 150


def test_reactive_dodge(mock):
    ra = ReactiveActions(mock)
    ra.dodge("left")  # не падает
    ra.emergency_stop()
    assert ra._target is None


def test_mock_controller_does_not_touch_system(mock, caplog):
    caplog.set_level(logging.INFO)
    mock.move_mouse(1, 2)
    mock.click()
    mock.type_text("hi")
    assert "MockController" in caplog.text
