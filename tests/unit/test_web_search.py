"""Интернет: когда искать, разбор выдачи и поиск в своём окне Chrome."""

import time

import pytest

from modules import web_search as ws


# ═══════════════════════════════════════════════════════════
# Когда искать
# ═══════════════════════════════════════════════════════════
@pytest.mark.parametrize("text", [
    "погугли, сколько стоит билет на «Ласточку»",
    "София, поищи в интернете последнюю версию Python",
    "загугли курс биткоина",
    "google it: python 3.14 release date",
    "узнай в сети, во сколько закрывается поликлиника",
])
def test_explicit_request_searches(text):
    ok, why = ws.should_search(text)
    assert ok and why


@pytest.mark.parametrize("text", [
    "какая сегодня погода в Москве?",
    "какой сейчас курс доллара?",
    "кто выиграл чемпионат мира по футболу 2022?",
    "что нового в последней версии Chrome?",
])
def test_fresh_question_searches(text):
    assert ws.should_search(text)[0]


@pytest.mark.parametrize("text", [
    "расскажи, как устроена солнечная система",
    "объясни разницу между массой и весом",
    "придумай название для рассказа",
    "привет",
    "да",
])
def test_ordinary_text_does_not_search(text):
    ok, why = ws.should_search(text)
    assert not ok and why == ""


def test_fresh_word_without_question_does_not_search():
    assert not ws.should_search("вчера весь день шёл дождь, и я промок")[0]


def test_make_query_strips_polite_request():
    q = ws.make_query("София, пожалуйста, погугли в интернете: курс евро")
    assert "погугли" not in q.lower()
    assert "пожалуйста" not in q.lower()
    assert "курс евро" in q.lower()
    assert not q.startswith((",", ":", "—"))


def test_make_query_keeps_long_text_as_is():
    text = "объясни, почему небо голубое, и как это связано с рассеянием"
    assert ws.make_query(text) == text


def test_make_query_joins_short_followup_with_previous():
    q = ws.make_query("а сейчас?", "погода в Казани")
    assert "погода в Казани" in q
    assert "сейчас" in q


def test_make_query_empty_for_empty_text():
    assert ws.make_query("") == ""


def test_parse_search_request_gives_query_and_clean_monologue():
    mono = "Нужен свежий курс.\nПОИСК: курс доллара к рублю\nОтвет короткий."
    query, cleaned = ws.parse_search_request(mono)
    assert query == "курс доллара к рублю"
    assert "ПОИСК" not in cleaned
    assert "Нужен свежий курс." in cleaned
    assert "Ответ короткий." in cleaned


def test_parse_search_request_english_line():
    assert ws.parse_search_request("SEARCH: weather in Oslo")[0] == "weather in Oslo"


def test_parse_search_request_accepts_markdown_line():
    assert ws.parse_search_request("**ПОИСК:** «счёт матча»")[0] == "счёт матча"


@pytest.mark.parametrize("line", ["ПОИСК: нет", "ПОИСК: -", "SEARCH: none"])
def test_parse_search_request_no_means_nothing(line):
    query, cleaned = ws.parse_search_request(f"размышляю\n{line}")
    assert query == ""
    assert "ПОИСК" not in cleaned and "SEARCH" not in cleaned


def test_parse_search_request_without_line_keeps_text():
    mono = "просто монолог"
    assert ws.parse_search_request(mono) == ("", mono)


# ═══════════════════════════════════════════════════════════
# Заглушки сети
# ═══════════════════════════════════════════════════════════
class FakeResponse:
    def __init__(self, text="", status=200, headers=None):
        self.text = text
        self.status_code = status
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        import json
        return json.loads(self.text)

    def iter_content(self, n=65536):
        yield self.text.encode("utf-8")

    def close(self):
        self.closed = True


class FakeSession:
    """Подмена requests.Session: страницы по адресу, запись обращений."""

    def __init__(self, pages=None, status=200, headers=None):
        self.pages = dict(pages or {})
        self.status = status
        self.headers = headers
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.headers is not None:
            return FakeResponse("", self.status, self.headers)
        if url not in self.pages:
            return FakeResponse("", 404)
        return FakeResponse(self.pages[url], self.status)


BING_HTML = """
<html><body><ol>
<li class="b_algo">
  <h2><a href="https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9leGFtcGxlLm9yZy9h">Первый результат</a></h2>
  <div class="b_caption"><p>Короткое описание первого.</p></div>
</li>
<li class="b_algo">
  <h2><a href="https://example.com/second">Второй результат</a></h2>
  <p>Описание второго результата.</p>
</li>
<li class="b_algo"><h2><span>Без ссылки</span></h2></li>
</ol></body></html>
"""


PAGE_HTML = """
<html><body>
<nav><a href="/a">Меню</a><a href="/b">Ещё меню</a></nav>
<script>var junk = "не текст";</script>
<h1>Столица Австралии</h1>
<p>Канберра — столица Австралии, крупнейший город страны — Сидней.</p>
<p>Город был основан в 1913 году как компромисс между Мельбурном и Сиднеем.</p>
<footer><p>Подвал сайта, который нам не нужен совсем ни разу.</p></footer>
</body></html>
"""


# ═══════════════════════════════════════════════════════════
# Разбор выдачи и чтение страниц
# ═══════════════════════════════════════════════════════════
def test_decode_bing_url_restores_target():
    href = ("https://www.bing.com/ck/a?!&&p=1&u=a1aHR0cHM6Ly9leGFtcGxlLm9yZy9h"
            "&ntb=1")
    assert ws.decode_bing_url(href) == "https://example.org/a"
    assert ws.decode_bing_url("https://example.org/plain") == "https://example.org/plain"


def test_parse_bing_reads_titles_urls_snippets():
    results = ws.parse_bing(BING_HTML)
    assert [r.title for r in results] == ["Первый результат", "Второй результат"]
    assert results[0].url == "https://example.org/a"
    assert results[0].snippet.startswith("Короткое описание")
    assert results[1].site == "example.com"


def test_parse_bing_survives_garbage():
    assert ws.parse_bing("не html") == []


def test_extract_paragraphs_drops_nav_script_footer():
    paras = ws.extract_paragraphs(PAGE_HTML)
    text = " ".join(paras)
    assert "Канберра" in text
    assert "не текст" not in text
    assert "Подвал" not in text
    assert "Меню" not in text


def test_extract_paragraphs_survives_garbage():
    assert ws.extract_paragraphs("<<<") == []


def test_best_paragraphs_prefers_overlap_and_keeps_order():
    paras = ws.extract_paragraphs(PAGE_HTML)
    picked = ws.best_paragraphs(paras, "когда основана Канберра?")
    assert "1913" in picked
    assert picked.index("Канберра") < picked.index("1913")


def test_best_paragraphs_without_overlap_gives_beginning():
    picked = ws.best_paragraphs(["Первый абзац.", "Второй абзац."], "совсем другое")
    assert picked == "Первый абзац."


def test_best_paragraphs_empty():
    assert ws.best_paragraphs([], "что угодно") == ""


def test_format_results_names_site_and_skips_empty():
    a = ws.WebAnswer(query="q", results=[
        ws.WebResult(title="Австралия", url="https://ru.wikipedia.org/wiki/A",
                     text="Канберра — столица."),
        ws.WebResult(title="Пустое", url="https://x.org/"),
    ])
    text = ws.format_results(a)
    assert "[1] ru.wikipedia.org — Австралия" in text
    assert "Канберра" in text
    assert "Пустое" not in text


def test_format_results_includes_instant_answer():
    a = ws.WebAnswer(query="курс", answer="82,68 Российский рубль",
                     results=[ws.WebResult(title="t", url="https://cbr.ru/x",
                                           snippet="курс на сегодня")])
    text = ws.format_results(a)
    assert "82,68" in text
    assert "cbr.ru" in text


def test_format_results_none_is_empty():
    assert ws.format_results(None) == ""


def test_web_answer_sites_are_unique():
    a = ws.WebAnswer(query="q", results=[
        ws.WebResult(title="a", url="https://www.lenta.ru/1"),
        ws.WebResult(title="b", url="https://lenta.ru/2"),
        ws.WebResult(title="c", url="https://ria.ru/3"),
    ])
    assert a.sites() == ["lenta.ru", "ria.ru"]


# ═══════════════════════════════════════════════════════════
# WebSearch: обычная выдача без браузера
# ═══════════════════════════════════════════════════════════
def test_web_search_falls_back_to_wikipedia():
    wiki = ('{"query": {"pages": {"1": {"index": 1, "title": "Канберра",'
            ' "fullurl": "https://ru.wikipedia.org/wiki/Канберра",'
            ' "extract": "Канберра — столица Австралии."}}}}')
    session = FakeSession(pages={
        "https://ru.wikipedia.org/w/api.php": wiki,
    })
    s = ws.WebSearch(session=session, fetch_pages=0)

    def boom(query):
        raise RuntimeError("bing молчит")

    s.bing = boom
    answer = s.search("столица Австралии")
    assert answer.ok
    assert answer.results[0].engine == "wikipedia"
    assert "Канберра" in answer.results[0].text


def test_web_search_cache_avoids_second_request():
    session = FakeSession(pages={"https://www.bing.com/search": BING_HTML})
    s = ws.WebSearch(session=session, fetch_pages=0)
    first = s.search("что-то")
    second = s.search("ЧТО-ТО")
    assert first is second
    assert len(session.calls) == 1


def test_web_search_empty_on_failure():
    class Dead:
        def get(self, *a, **k):
            raise RuntimeError("нет сети")

    s = ws.WebSearch(session=Dead(), fetch_pages=0)
    answer = s.search("что-то")
    assert not answer.ok
    assert answer.error


def test_web_search_empty_query_is_not_sent():
    session = FakeSession()
    s = ws.WebSearch(session=session, fetch_pages=0)
    assert not s.search("   ").ok
    assert session.calls == []


def test_web_search_reads_pages_for_question():
    session = FakeSession(pages={
        "https://www.bing.com/search": (
            '<ol><li class="b_algo"><h2><a href="https://ex.org/au">Австралия</a>'
            "</h2><p>страница про Австралию</p></li></ol>"
        ),
        "https://ex.org/au": PAGE_HTML,
    })
    s = ws.WebSearch(session=session, fetch_pages=1)
    answer = s.search("столица Австралии")
    assert "Канберра" in answer.results[0].text
    assert answer.results[0].to_dict()["read"] is True


# ═══════════════════════════════════════════════════════════
# Google в своём окне Chrome
# ═══════════════════════════════════════════════════════════
def test_decode_google_href_from_ping():
    href = "https://www.google.com/goto?url=CAES"
    ping = ("/url?sa=t&source=web&rct=j&url=https://ru.wikipedia.org/wiki/A"
            "&ved=0")
    assert ws.decode_google_href(href, ping) == "https://ru.wikipedia.org/wiki/A"


def test_decode_google_href_from_query():
    href = "/url?q=https://example.org/page&sa=U"
    assert ws.decode_google_href(href) == "https://example.org/page"


def test_decode_google_href_goto_without_target_is_empty():
    assert ws.decode_google_href("https://www.google.com/goto?url=CAESabc") == ""
    assert ws.decode_google_href("") == ""


def test_clean_answer_keeps_answer_and_drops_service_lines():
    top = (
        "Курс доллара\n"
        "82,68 Российский рубль\n"
        "Результатов: примерно 12 300 000 (0,42 сек.)\n"
        "Все результаты\n"
        "Что-то ещё"
    )
    text = ws.clean_answer(top)
    assert "82,68" in text
    assert "Результатов" not in text
    assert "Что-то ещё" not in text


def test_clean_answer_caps_length():
    top = "\n".join(["Очень длинная строка ответа"] * 40)
    assert len(ws.clean_answer(top, limit=60)) <= 60


def test_clean_answer_empty():
    assert ws.clean_answer("") == ""


class FakeBridge:
    """Подмена ChromeBridge: запоминает команды, отдаёт заготовленную выдачу."""

    def __init__(self, top="Курс доллара\n82,68 Российский рубль",
                 results=None, url="https://www.google.com/?hl=ru",
                 ready=True, target="t-search", cookies=None, stuck=False,
                 page_text=""):
        self.top = top
        self.results = results if results is not None else []
        self.url = url
        self.ready = ready
        self.stuck = stuck
        self.page_text = page_text
        self.target = target
        self.cookies_value = cookies or {}
        self.opened = []
        self.reused = []
        self.navigated = []
        self.typed = []
        self.pressed = []
        self.focus = None
        self.closed = []
        self.nav_timeouts = []
        self.stopped = False
        self.our_process = True
        self._target = None
        self._session = None

    @property
    def session(self):
        return self._session

    @property
    def owned(self):
        return self.our_process

    def start(self):
        pass

    def own_page(self, url="about:blank", minimized=True):
        self.reused.append((url, minimized))
        self._target = self.target
        self._session = "S1"
        if not self.stuck:      # капча не отпускает: страница /sorry/ остаётся
            self.url = url
        return self.target

    def open_window(self, url="about:blank", minimized=True):
        self.opened.append((url, minimized))
        self._target = self.target
        self._session = "S1"
        if not self.stuck:      # капча не отпускает: страница /sorry/ остаётся
            self.url = url
        return self.target

    def has_target(self, target_id):
        return target_id == self._target

    def focus_emulation(self, enabled=True):
        self.focus = enabled

    def navigate(self, url, timeout=25.0):
        self.navigated.append(url)
        self.nav_timeouts.append(timeout)
        if not self.stuck:          # капча не отпускает: Google возвращает /sorry/
            self.url = url
        return url

    def current_url(self):
        return self.url

    def read_text(self, max_chars=6000):
        return self.page_text[:max_chars]

    def cookies(self):
        return dict(self.cookies_value)

    def type_text(self, text, field="", selector="", enter=False):
        self.typed.append(text)
        return text

    def press(self, key):
        self.pressed.append(key)

    def close_target(self, target_id):
        self.closed.append(target_id)
        if target_id == self._target:
            self._target = None
            self._session = None

    def stop_chrome(self):
        self.stopped = True

    def _eval(self, expression, session=None):
        if expression == ws._GOOGLE_MARK_JS:
            return True
        if expression == ws._GOOGLE_FOCUS_JS:
            return True
        if expression == ws._GOOGLE_READY_JS:
            return self.ready
        if expression == ws._GOOGLE_TOP_JS:
            return self.top
        if expression.startswith(ws._GOOGLE_RESULTS_JS[:40]):
            return self.results
        return None


GOOGLE_ITEM = {
    "title": "Курс доллара ЦБ РФ",
    "href": "https://www.google.com/goto?url=CAESxyz",
    "ping": "/url?sa=t&url=https://www.cbr.ru/currency_base/daily/&ved=1",
    "cite": "cbr.ru › currency_base › daily",
    "text": "Курс доллара ЦБ РФ 82,68 рубля на сегодня.",
}


def google_reader(pages=None):
    return ws.WebSearch(session=FakeSession(pages=pages or {}), fetch_pages=1)


def test_browser_search_types_query_and_reads_results():
    bridge = FakeBridge(results=[GOOGLE_ITEM])
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader())
    answer = s.search("курс доллара")
    assert answer.engine == "google"
    assert answer.answer.startswith("Курс доллара")
    assert answer.results[0].url == "https://www.cbr.ru/currency_base/daily/"
    assert answer.results[0].site == "cbr.ru"
    assert bridge.typed == ["курс доллара"]
    assert bridge.pressed == ["Enter"]
    assert bridge.reused and bridge.reused[0][1] is True   # свёрнутое окно
    assert bridge.opened == []          # пустой about:blank не остаётся
    assert bridge.focus is True


def test_browser_search_reuses_window_between_queries():
    bridge = FakeBridge(results=[GOOGLE_ITEM])
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader())
    s.search("первый вопрос")
    s.search("второй вопрос")
    assert len(bridge.reused) == 1
    assert bridge.opened == []
    assert bridge.typed == ["первый вопрос", "второй вопрос"]


def test_browser_search_caches_answer():
    bridge = FakeBridge(results=[GOOGLE_ITEM])
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader())
    first = s.search("курс доллара")
    second = s.search("Курс доллара")
    assert first is second
    assert bridge.typed == ["курс доллара"]


def test_browser_search_reads_one_page():
    page = PAGE_HTML
    bridge = FakeBridge(results=[GOOGLE_ITEM], top="")
    reader = google_reader({"https://www.cbr.ru/currency_base/daily/": page})
    answer = ws.BrowserSearch(bridge=bridge, reader=reader, fetch_pages=1).search(
        "Канберра столица"
    )
    assert "Канберра" in answer.results[0].text


def test_browser_search_resolves_goto_only_copy():
    item = dict(GOOGLE_ITEM, ping="")
    bridge = FakeBridge(results=[item])
    session = FakeSession(headers={"location": "https://www.cbr.ru/real/"})
    reader = ws.WebSearch(session=session, fetch_pages=0)
    answer = ws.BrowserSearch(bridge=bridge, reader=reader).search("курс")
    assert answer.results[0].url == "https://www.cbr.ru/real/"


def test_browser_search_keeps_result_without_url():
    item = {"title": "Курс", "href": "https://www.google.com/goto?url=CAES",
            "ping": "", "cite": "www.cbr.ru › daily", "text": "82,68 рубля"}
    bridge = FakeBridge(results=[item])
    answer = ws.BrowserSearch(bridge=bridge, reader=google_reader()).search("курс")
    assert answer.results[0].url == ""
    assert answer.results[0].site == "cbr.ru"


# Выдача по теме запроса «курс доллара»: нерелевантное отсекается фильтром
BING_RATE_HTML = """
<html><body><ol>
<li class="b_algo">
  <h2><a href="https://www.banki.ru/products/currency/usd/">Курс доллара на сегодня</a></h2>
  <p>Курс доллара ЦБ РФ.</p>
</li>
</ol></body></html>
"""


def test_browser_search_captcha_falls_back_to_reader():
    bridge = FakeBridge(url="https://www.google.com/sorry/index?continue=x",
                        ready=False, stuck=True)
    reader = google_reader({"https://www.bing.com/search": BING_RATE_HTML})
    answer = ws.BrowserSearch(bridge=bridge, reader=reader, fetch_pages=0).search(
        "курс доллара"
    )
    assert answer.results
    assert answer.results[0].engine == "bing"
    assert "капча" in answer.error


def test_browser_search_without_browser_falls_back():
    reader = google_reader({"https://www.bing.com/search": BING_RATE_HTML})
    s = ws.BrowserSearch(bridge_factory=_broken_bridge, reader=reader,
                         fetch_pages=0)
    answer = s.search("курс доллара")
    assert answer.results and answer.error


def _broken_bridge():
    raise RuntimeError("Chrome не найден")


def test_browser_search_no_ready_page_falls_back():
    bridge = FakeBridge(ready=False)
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader(
        {"https://www.bing.com/search": BING_RATE_HTML}), fetch_pages=0, timeout=0.2)
    answer = s.search("курс доллара")
    assert answer.results[0].engine == "bing"


def test_browser_search_empty_query_does_nothing():
    bridge = FakeBridge()
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader())
    assert not s.search("").ok
    assert bridge.opened == []


def test_browser_search_shutdown_closes_own_window():
    bridge = FakeBridge(results=[GOOGLE_ITEM])
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader())
    s.search("курс")
    s.shutdown()
    assert bridge.closed == ["t-search"]
    assert bridge.stopped is True


def test_browser_search_shutdown_keeps_foreign_chrome():
    bridge = FakeBridge(results=[GOOGLE_ITEM])
    bridge.our_process = False
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader())
    s.search("курс")
    s.shutdown()
    assert bridge.closed == ["t-search"]
    assert bridge.stopped is False


def test_browser_search_reads_blocked_page_in_chrome():
    """Сайт закрыт от простых запросов — читаем тем же окном браузера."""
    bridge = FakeBridge(results=[GOOGLE_ITEM],
                        page_text="Канберра — столица Австралии и крупнейший "
                                  "город страны — Сидней, а не Канберра.")
    answer = ws.BrowserSearch(bridge=bridge, reader=google_reader(),
                              fetch_pages=1).search("столица Австралии")
    assert "Канберра" in answer.results[0].text
    assert "https://www.cbr.ru/currency_base/daily/" in bridge.navigated


def test_clean_answer_stops_at_people_also_ask():
    top = "Вопросы по теме\nБыл ли Сидней столицей Австралии?\nКанберра?"
    assert ws.clean_answer(top) == ""


def test_clean_answer_drops_bare_tickers():
    top = "Конвертер валют\n1 Доллар США равно\n82,55 Российский рубль\nAINFT\nAave"
    text = ws.clean_answer(top)
    assert "AINFT" not in text and "Aave" not in text
    assert "82,55" in text


def test_clean_answer_drops_web_results_heading():
    assert ws.clean_answer("Web results\nЧто-то ещё") == ""


def test_result_snippet_from_block_drops_title_and_cite():
    item = {"title": "Курс", "href": "https://cbr.ru/x", "cite": "cbr.ru › x",
            "text": "Курс cbr.ru › x · Перевести эту страницу 82,68 рубля"}
    bridge = FakeBridge(results=[item])
    answer = ws.BrowserSearch(bridge=bridge, reader=google_reader()).search("курс")
    assert answer.results[0].snippet == "82,68 рубля"


def test_result_snippet_prefers_google_snippet():
    item = dict(GOOGLE_ITEM, snippet="Готовый сниппет", text="что-то другое")
    bridge = FakeBridge(results=[item])
    answer = ws.BrowserSearch(bridge=bridge, reader=google_reader()).search("курс")
    assert answer.results[0].snippet == "Готовый сниппет"


def test_clean_answer_drops_tickers_after_the_number():
    """В виджете курса под ответом идёт карусель крипты — это не ответ."""
    top = ("Конвертер валют\n1 Доллар США равно\n82,55 Российский рубль\n"
           "Aerodrome Finance\nArtificial Superintelligence Alliance\n"
           "BitTorrent [New]")
    text = ws.clean_answer(top)
    assert "82,55" in text
    for junk in ("Aerodrome", "Artificial Superintelligence", "BitTorrent"):
        assert junk not in text


def test_clean_answer_keeps_city_before_temperature():
    text = ws.clean_answer("Москва\n−3 °C\nОщущается как −6 °C")
    assert text.startswith("Москва") and "−3" in text


def _two_items():
    second = dict(GOOGLE_ITEM, title="Курс на завтра",
                  ping="/url?sa=t&url=https://quote.rbc.ru/dollar/&ved=2",
                  cite="quote.rbc.ru › dollar")
    return [dict(GOOGLE_ITEM), second]


def test_browser_search_reads_one_page_when_google_already_answered():
    """Есть готовый ответ — хватит одной страницы: быстрее, а факт уже есть."""
    bridge = FakeBridge(results=_two_items())
    answer = ws.BrowserSearch(bridge=bridge, reader=google_reader(),
                              fetch_pages=2).search("курс доллара")
    assert answer.answer
    assert len(bridge.navigated) == 1


def test_browser_search_reads_both_pages_without_instant_answer():
    bridge = FakeBridge(top="", results=_two_items())
    ws.BrowserSearch(bridge=bridge, reader=google_reader(),
                     fetch_pages=2).search("что такое канберра")
    assert len(bridge.navigated) == 2


def test_browser_read_has_its_own_short_timeout():
    """Страница-долгожитель не должна съедать весь бюджет поиска."""
    bridge = FakeBridge(top="", results=_two_items())
    s = ws.BrowserSearch(bridge=bridge, reader=google_reader(), fetch_pages=2,
                         timeout=20.0)
    s.search("что такое канберра")
    assert bridge.nav_timeouts
    assert max(bridge.nav_timeouts) <= ws.BROWSER_READ_TIMEOUT < 20.0


def test_self_search_drops_trailing_aside():
    """Модель любит дописать пояснение в скобках — в запрос оно не идёт."""
    q, _ = ws.parse_search_request(
        "надо глянуть.\nПОИСК: какое ваше имя? (имеется в виду собеседник)")
    assert q == "какое ваше имя?"


def test_self_search_rejects_overlong_line():
    long_line = "ПОИСК: " + "очень длинный поток слов " * 12
    q, text = ws.parse_search_request(long_line)
    assert q == ""
    assert "ПОИСК" not in text          # строку всё равно вырезали


def test_self_search_keeps_normal_query():
    q, text = ws.parse_search_request("ПОИСК: курс доллара к рублю сегодня")
    assert q == "курс доллара к рублю сегодня"
    assert text == ""


def test_bing_engine_is_interchangeable_with_google_one():
    """main на выключении зовёт _get_web().shutdown — у обоих движков он есть."""
    s = ws.WebSearch(session=FakeSession())
    assert callable(s.search) and callable(s.shutdown)
    s.shutdown()                      # без Chrome закрывать нечего
    s.shutdown()                      # повторно — тоже не падает


def test_engine_bing_has_shutdown(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "WEB_SEARCH_ENGINE", "bing")
    ws.reset_default()
    try:
        assert isinstance(ws.engine(), ws.WebSearch)
        ws.engine().shutdown()
    finally:
        ws.reset_default()


def test_dedupe_results_drops_same_url_keeps_order():
    """В exe banki.ru пришёл в источниках дважды одной и той же ссылкой."""
    a = ws.WebResult(title="Курс", url="https://www.banki.ru/products/currency/usd/")
    b = ws.WebResult(title="Курс ЦБ", url="https://www.banki.ru/products/currency/usd")
    c = ws.WebResult(title="Investing", url="", site_name="ru.investing.com")
    d = ws.WebResult(title="Investing 2", url="", site_name="ru.investing.com")
    e = ws.WebResult(title="БКС", url="https://bcs-express.ru/kurs")
    out = ws.dedupe_results([a, b, c, d, e])
    assert [r.title for r in out] == ["Курс", "Investing", "БКС"]


def test_dedupe_results_merges_text_into_kept_one():
    a = ws.WebResult(title="A", url="https://x.ru/p")
    b = ws.WebResult(title="B", url="https://x.ru/p#top", text="прочитанный текст")
    out = ws.dedupe_results([a, b])
    assert len(out) == 1 and out[0].text == "прочитанный текст"


# ── Нерелевантная выдача ────────────────────────────────────
def test_engine_query_drops_question_words():
    """Bing на «какой сейчас курс…» отдавал словари на слово «какой»."""
    assert ws.engine_query("какой сейчас курс доллара к рублю") == "сейчас курс доллара к рублю"
    assert ws.engine_query("Какая погода в Москве") == "погода в Москве"
    assert ws.engine_query("что такое энтропия") == "энтропия"
    assert ws.engine_query("кто такой Илон Маск") == "Илон Маск"
    assert ws.engine_query("сколько стоит биткоин") == "сколько стоит биткоин"


def test_engine_query_keeps_short_queries_intact():
    assert ws.engine_query("погода") == "погода"
    assert ws.engine_query("кто такой") == "кто такой"       # пусто не делаем


def test_relevant_drops_dictionary_pages_for_question_words():
    query = "какой сейчас курс доллара к рублю"
    good = ws.WebResult(title="Курс доллара к рублю на сегодня — Банки.ру",
                        url="https://www.banki.ru/products/currency/usd/",
                        snippet="Курс доллара ЦБ РФ на сегодня 82,52 руб.")
    wikt = ws.WebResult(title="какой — Викисловарь",
                        url="https://ru.wiktionary.org/wiki/какой",
                        snippet="Местоимение. Употребляется в вопросе о качестве.")
    gram = ws.WebResult(title="Поиск по слову «какой» — Грамота.ру",
                        url="https://gramota.ru/poisk?query=какой",
                        snippet="Толковый словарь: какой, местоимение.")
    assert ws.relevant([good, wikt, gram], query) == [good]


def test_relevant_keeps_everything_when_query_has_no_content_words():
    r = ws.WebResult(title="Что-то", url="https://x.ru", snippet="текст")
    assert ws.relevant([r], "какой и что") == [r]


def test_bing_search_filters_irrelevant_and_reports_it():
    """Все результаты мимо — честно пусто с причиной, а не словари в промпт."""
    html = """<html><body><ol>
    <li class="b_algo"><h2><a href="https://ru.wiktionary.org/wiki/x">какой — Викисловарь</a></h2>
    <p>Местоимение.</p></li></ol></body></html>"""
    session = FakeSession(pages={})

    def get(url, **kw):
        session.calls.append((url, kw))
        if "bing.com" in url:
            return FakeResponse(html, 200)
        return FakeResponse("", 404)

    session.get = get
    s = ws.WebSearch(session=session, fetch_pages=0)
    s.wikipedia = lambda q, lang="ru": []
    ans = s.search("какой сейчас курс доллара к рублю")
    assert ans.results == []
    assert "нерелевант" in ans.error


def test_format_results_tells_model_not_to_invent_when_empty():
    ans = ws.WebAnswer(query="курс доллара", results=[], error="bing: выдача нерелевантна")
    text = ws.format_results(ans)
    assert "ничего" in text.lower() and "не придумывай" in text.lower()
