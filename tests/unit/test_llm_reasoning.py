"""Окружение llama-server: размышления выключаются рабочим флагом.

llama-server на LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
отвечает варнингом «deprecated. Use --reasoning on/--reasoning off instead».
Переменная для --reasoning — LLAMA_ARG_REASONING (есть в llama-common.dll
рядом с LLAMA_ARG_REASONING_EFFORT/PRESERVE).
"""

import pytest

from app.llm import server_env


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_REASONING", raising=False)
    monkeypatch.delenv("LLAMA_ARG_CHAT_TEMPLATE_KWARGS", raising=False)


def test_reasoning_turned_off(clean):
    assert server_env()["LLAMA_ARG_REASONING"] == "off"


def test_deprecated_template_kwargs_not_set(clean):
    assert "LLAMA_ARG_CHAT_TEMPLATE_KWARGS" not in server_env()


def test_explicit_reasoning_setting_wins(clean, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_REASONING", "auto")
    assert server_env()["LLAMA_ARG_REASONING"] == "auto"


def test_fit_still_set(clean):
    assert server_env()["LLAMA_ARG_FIT"] == "on"


def test_env_is_a_copy(clean):
    import os

    env = server_env()
    env["SOFIA_TEST_MARK"] = "1"
    assert "SOFIA_TEST_MARK" not in os.environ
