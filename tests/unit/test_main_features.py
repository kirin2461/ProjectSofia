"""Связки main.py: расчёт в промпте, профиль, книги, статусы для /state."""

import asyncio

import pytest

import main
from memory.profile import make_profile


# ── калькулятор в ответе ────────────────────────────────────
def test_math_hint_gives_number():
    assert "8" in main._math_hint("сколько будет 2+2*3?")


def test_math_hint_skips_plain_text():
    assert main._math_hint("привет, как дела") == ""


def test_math_hint_survives_garbage():
    assert main._math_hint("2+") == "" or isinstance(main._math_hint("2+"), str)


# ── профиль ─────────────────────────────────────────────────
def test_learn_profile_adds_facts(tmp_path, monkeypatch):
    prof = make_profile(tmp_path / "p.json")
    monkeypatch.setattr(main, "profile", prof)
    main._learn_profile("- любит физику\n- зовут Алексей\n")
    assert len(prof) == 2


def test_learn_profile_ignores_noise(tmp_path, monkeypatch):
    prof = make_profile(tmp_path / "p.json")
    monkeypatch.setattr(main, "profile", prof)
    main._learn_profile("привет\nспасибо")
    assert len(prof) == 0


def test_learn_profile_without_profile_is_noop(monkeypatch):
    monkeypatch.setattr(main, "profile", None)
    main._learn_profile("- любит физику")     # не должно бросать


def test_learn_profile_skips_long_lines(tmp_path, monkeypatch):
    prof = make_profile(tmp_path / "p.json")
    monkeypatch.setattr(main, "profile", prof)
    main._learn_profile("ф" * 500)
    assert len(prof) == 0


# ── книги: несколько формулировок ───────────────────────────
def test_recall_books_passes_queries(monkeypatch):
    captured = {}

    def fake_search(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "search_books", fake_search)
    monkeypatch.setattr(main.settings, "LIBRARY_ENABLED", True)
    main._recall_books("Что такое энтропия?", prev_user="")
    assert captured["query"].startswith("Что такое")
    assert captured["queries"] and captured["queries"][0].startswith("энтропия")
    assert captured["top_k"] >= 1


def test_recall_books_respects_followup(monkeypatch):
    captured = {}

    def fake_search(query, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "search_books", fake_search)
    monkeypatch.setattr(main.settings, "LIBRARY_ENABLED", True)
    main._recall_books("а подробнее?", prev_user="Что такое энтропия?")
    assert any("энтропия" in q for q in captured["queries"])


def test_recall_books_off_returns_empty(monkeypatch):
    monkeypatch.setattr(main.settings, "LIBRARY_ENABLED", False)
    assert main._recall_books("что такое энтропия") == []


def test_recall_books_survives_search_error(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("нет индекса")

    monkeypatch.setattr(main, "search_books", boom)
    monkeypatch.setattr(main.settings, "LIBRARY_ENABLED", True)
    assert main._recall_books("вопрос") == []


# ── статусы для /state ──────────────────────────────────────
def test_reranker_status_is_string(monkeypatch):
    monkeypatch.setattr(main.settings, "LIBRARY_RERANK_DIR", "", raising=False)
    monkeypatch.delenv("SOFIA_RERANK_DIR", raising=False)
    assert isinstance(main._reranker_status(), str)


def test_skills_stats_shape():
    stats = main._skills_stats()
    assert "total" in stats and "by_domain" in stats


def test_state_route_is_registered():
    paths = {getattr(r, "path", "") for r in main.app.routes}
    assert "/api/feedback" in paths and "/state" in paths


def test_feedback_endpoint_accepts_rating(tmp_path, monkeypatch):
    """Оценка пишется в файл, а не теряется в недрах эндпоинта."""
    monkeypatch.setattr(main.feedback, "DEFAULT_PATH", tmp_path / "f.jsonl")
    entry = main.feedback.record(1, "вопрос", "ответ")
    assert entry["rating"] == 1
    assert main.feedback.stats()["total"] == 1


@pytest.mark.parametrize("value", ["-1", "0", "1"])
def test_feedback_rating_values(value):
    assert int(value) in (-1, 0, 1)


def test_feedback_meta_only_for_the_same_turn(monkeypatch):
    """Оценка старого ответа не должна получить стратегию последнего хода."""
    monkeypatch.setattr(main, "_LAST_TURN", {"question": "Сколько будет 17*23+4?",
                                             "strategy": "уточни", "sources": 2})
    old = main._feedback_meta({"question": "Что про энтропию?"})
    assert old == {"strategy": "", "sources": 0}
    same = main._feedback_meta({"question": "Сколько будет 17*23+4?"})
    assert same == {"strategy": "уточни", "sources": 2}
    bare = main._feedback_meta({})          # без вопроса — оцениваем последний ход
    assert bare == {"strategy": "уточни", "sources": 2}


# ── Поиск в интернете ──────────────────────────────────────
def test_get_web_uses_settings_engine(monkeypatch):
    from modules import web_search
    sentinel = object()
    monkeypatch.setattr(main, "_web", None)
    monkeypatch.setattr(web_search, "engine", lambda: sentinel)
    assert main._get_web() is sentinel
    assert main._get_web() is sentinel      # второй раз — тот же, без Chrome
    monkeypatch.setattr(main, "_web", None)


def test_search_web_returns_none_on_timeout(monkeypatch):
    import time as _time

    def slow(query):
        _time.sleep(0.5)
        return "поздно"

    monkeypatch.setattr(main, "_web_lookup", slow)
    monkeypatch.setattr(main.settings, "WEB_SEARCH_TIMEOUT", 0.05)
    assert asyncio.run(main._search_web("курс")) is None


def test_search_web_passes_answer_through(monkeypatch):
    class Answer:
        query = "курс"
        results = []
        answer = ""
        engine = "google"
        seconds = 0.1
        error = ""

        def to_dict(self):
            return {"query": self.query}

    monkeypatch.setattr(main, "_web_lookup", lambda q: Answer())
    assert asyncio.run(main._search_web("курс")).query == "курс"


def test_web_lookup_swallows_engine_errors(monkeypatch):
    class Boom:
        def search(self, query):
            raise RuntimeError("Chrome не найден")

    monkeypatch.setattr(main, "_get_web", lambda: Boom())
    assert main._web_lookup("курс") is None


def test_self_search_skipped_when_calculator_already_answered():
    """Модель любит написать «ПОИСК: Сколько будет 17*23+4» — гуглить
    арифметику незачем, калькулятор уже посчитал (в exe это стоило 22 с)."""
    assert main._self_search_query("Сколько будет 17*23+4", hint="17*23+4 = 395") == ""


def test_self_search_kept_for_real_questions():
    assert main._self_search_query("курс доллара к рублю", hint="") == "курс доллара к рублю"


def test_self_search_disabled_by_settings(monkeypatch):
    monkeypatch.setattr(main.settings, "WEB_SEARCH_ENABLED", False)
    assert main._self_search_query("курс доллара", hint="") == ""


def test_state_exposes_web_section():
    data = asyncio.run(main.state())
    payload = data.body.decode("utf-8")
    assert '"web"' in payload and "engine" in payload


def test_payload_text_accepts_common_keys():
    assert main._payload_text("просто текст") == "просто текст"
    assert main._payload_text('{"text": "привет"}') == "привет"
    assert main._payload_text('{"message": "привет"}') == "привет"
    assert main._payload_text('{"content": "привет"}') == "привет"


def test_payload_text_keeps_raw_on_unknown_shape():
    raw = '{"foo": 1}'
    assert main._payload_text(raw) == raw      # не текст — пусть решает дальше
    assert main._payload_text('{"text": ""}') == '{"text": ""}'
    assert main._payload_text("{битый json") == "{битый json"


# ── открытие браузера из меню ───────────────────────────────
def test_browser_open_page_is_not_about_blank():
    """Пустое поле адреса больше не приводит к пустому окну about:blank."""
    from modules.web_search import GOOGLE_HOME
    assert main._browser_open_url({}) == GOOGLE_HOME
    assert main._browser_open_url(None) == GOOGLE_HOME
    assert main._browser_open_url({"url": "   "}) == GOOGLE_HOME


def test_browser_open_keeps_typed_address():
    assert main._browser_open_url({"url": " cbr.ru "}) == "cbr.ru"


class FakeBrowser:
    def __init__(self, running):
        self._running = running
        self.calls = []

    def running(self):
        return self._running

    def start(self, url="about:blank"):
        self.calls.append(("start", url))

    def navigate(self, url, timeout=25.0):
        self.calls.append(("navigate", url))
        return url

    def current_url(self):
        return "https://www.cbr.ru/"

    def title(self):
        return "ЦБ"


def test_browser_open_launches_chrome_at_the_page(monkeypatch):
    """Раньше Chrome стартовал на about:blank и только потом переходил."""
    browser = FakeBrowser(running=False)
    monkeypatch.setattr(main, "_get_browser", lambda: browser)
    monkeypatch.setattr(main.settings, "BROWSER_ENABLED", True)

    asyncio.run(main.browser_open({"url": "cbr.ru"}))

    assert browser.calls == [("start", "cbr.ru")]


def test_browser_open_navigates_running_chrome(monkeypatch):
    browser = FakeBrowser(running=True)
    monkeypatch.setattr(main, "_get_browser", lambda: browser)
    monkeypatch.setattr(main.settings, "BROWSER_ENABLED", True)

    asyncio.run(main.browser_open({"url": "cbr.ru"}))

    assert browser.calls == [("navigate", "cbr.ru")]
