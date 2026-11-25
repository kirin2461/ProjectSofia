"""Поиск в интернете: сама ищет, сама читает, сама решает.

Быстрый путь без браузера: страница выдачи Bing (HTML) → 1–2 сайта из
выдачи → абзацы, в которых есть слова вопроса. Обычно 1–3 секунды. Если
Bing не ответил — запасной источник, Википедия.

Когда искать, решают два механизма (см. main.py):
  * `should_search` — мгновенная эвристика: прямая просьба («погугли») или
    вопрос о свежем (курс, погода, новости, «сегодня»);
  * сама модель — в монологе строкой `ПОИСК: запрос`, если понимает, что
    точного ответа не знает (`parse_search_request`).

Всё остальное София отвечает из своей головы и книг: поиск на каждую
реплику сделал бы ответы медленными, а её — «ленивой».

Глубокая работа с сайтами (войти, нажать, заполнить) — это автопилот в
Chrome (`modules/autopilot.py`), здесь только чтение.
"""

from __future__ import annotations

import base64
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from modules.browser_agent import release_target, reserve_target

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"}

SEARCH_TIMEOUT = 6.0
PAGE_TIMEOUT = 6.0
MAX_PAGE_BYTES = 1_500_000
CACHE_TTL = 600.0
CACHE_SIZE = 64
PAGE_CHARS = 1200          # из одной страницы в промпт
TOTAL_CHARS = 2600         # весь блок «из интернета»
MAX_SELF_QUERY = 120       # длиннее — это размышление, а не запрос

# Сайты, которые не читаем: видео, соцсети за логином, файлы
_SKIP_HOSTS = ("youtube.com", "youtu.be", "vk.com", "instagram.com",
               "facebook.com", "tiktok.com", "x.com", "twitter.com",
               "t.me", "ok.ru", "rutube.ru")
_SKIP_EXT = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar",
             ".mp4", ".mp3", ".jpg", ".png")


# ── Когда искать ────────────────────────────────────────────
_EXPLICIT = re.compile(
    r"(погугли|загугли|поищи|найди|посмотри|глянь|проверь|узнай)"
    r"[^.?!]{0,40}?(в интернете|в сети|онлайн|в гугле|в яндексе|в бинге)"
    r"|(погугли|загугли|гугл(ани|ни))"
    r"|\b(search the web|google it|look it up online|search online)\b",
    re.I,
)
_FRESH = re.compile(
    r"\b(сегодня|сейчас|на данный момент|в данный момент|текущ\w*|актуальн\w*|"
    r"последн\w* (новост|верси|данн)\w*|новост\w*|курс\w* (доллар|евро|юан|рубл|"
    r"валют|биткоин|bitcoin|акци)\w*|погод\w*|прогноз\w*|цен[аыуе]\b|стоимост\w*|"
    r"сколько стоит|вышел|вышла|вышло|выйдет|релиз\w*|в этом году|"
    r"в прошлом месяце|на этой неделе|счёт матча|счет матча|кто выиграл|"
    r"20[2-3]\d|"
    r"today|right now|latest|news|current|price|weather|released?)\b",
    re.I,
)
_QUESTION = re.compile(
    r"\?|\b(какой|какая|какое|какие|сколько|что|кто|когда|где|почему|зачем|"
    r"как|есть ли|ли|what|who|when|where|which|how|is there)\b",
    re.I,
)
_STRIP_EXPLICIT = re.compile(
    r"(пожалуйста|плиз|софия|соня)[,\s]*"
    r"|(погугли|загугли|гуглани|гуглни|поищи|найди|посмотри|глянь|проверь|узнай)"
    r"(\s+(мне|нам|пожалуйста))?(\s+(в интернете|в сети|онлайн|в гугле|в яндексе))?"
    r"|(в интернете|в сети|онлайн|в гугле|в яндексе)",
    re.I,
)
_REQUEST_LINE = re.compile(r"^\s*[*_>\-\s]*(ПОИСК|SEARCH)\s*[:：]\s*(.+?)\s*$",
                           re.I | re.M)


def should_search(text: str) -> Tuple[bool, str]:
    """Нужен ли интернет для этой реплики. Возвращает (да/нет, причина)."""
    t = " ".join((text or "").split())
    if len(t) < 6:
        return False, ""
    if _EXPLICIT.search(t):
        return True, "просьба поискать"
    if _FRESH.search(t) and _QUESTION.search(t):
        return True, "вопрос о свежем"
    return False, ""


def make_query(text: str, prev_user: str = "") -> str:
    """Реплика → поисковый запрос: без «погугли», вежливостей и знаков."""
    q = _STRIP_EXPLICIT.sub(" ", text or "")
    q = re.sub(r"^[\s,:;—–-]+|[\s?!.,:;—–-]+$", "", " ".join(q.split()))
    q = re.sub(r"^(про|о|об|что|мне)\s+", "", q, flags=re.I) if len(q.split()) > 2 else q
    if len(q.split()) <= 2 and prev_user:
        # «а сейчас?» / «погугли это» — смысл в предыдущем вопросе
        base = make_query(prev_user)
        q = f"{base} {q}".strip() if q.lower() not in ("это", "его", "её", "") else base
    return q[:200]


def parse_search_request(monologue: str) -> Tuple[str, str]:
    """Строка `ПОИСК: …` из монолога → (запрос, монолог без этой строки)."""
    text = monologue or ""
    m = _REQUEST_LINE.search(text)
    if not m:
        return "", text
    query = m.group(2).strip().strip("*_ \t«»\"'`").strip()
    # Модель любит дописать пояснение в скобках: в запросе это лишний шум
    query = re.sub(r"\s*\([^)]*\)\s*", " ", query)
    query = re.sub(r"\s*[—–-]\s*(имеется|это)\b.*$", "", query, flags=re.I)
    query = " ".join(query.split()).strip(" .,:;—–-")
    if query.lower() in ("нет", "не нужно", "не нужен", "none", "no", "-", "—"):
        query = ""
    if len(query) > MAX_SELF_QUERY:
        query = ""      # не запрос, а размышление: так искать нельзя
    cleaned = (text[:m.start()] + text[m.end():]).strip()
    return query[:200], cleaned


# ── Результаты ──────────────────────────────────────────────
@dataclass
class WebResult:
    title: str
    url: str
    snippet: str = ""
    text: str = ""                 # абзацы со страницы (если читали)
    engine: str = "bing"
    site_name: str = ""            # домен, если ссылку раскрыть не удалось

    @property
    def site(self) -> str:
        if self.site_name:
            return self.site_name
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "site": self.site,
                "snippet": self.snippet[:300], "read": bool(self.text)}


@dataclass
class WebAnswer:
    query: str
    results: List[WebResult] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    answer: str = ""               # мгновенный ответ Google: курс, погода, счёт
    engine: str = "bing"

    @property
    def ok(self) -> bool:
        return bool(self.results)

    def sites(self) -> List[str]:
        out: List[str] = []
        for r in self.results:
            if r.site and r.site not in out:
                out.append(r.site)
        return out

    def to_dict(self) -> dict:
        return {"query": self.query, "seconds": round(self.seconds, 2),
                "results": [r.to_dict() for r in self.results],
                "answer": self.answer, "engine": self.engine,
                "error": self.error}


# ── Разбор выдачи ───────────────────────────────────────────
def _text(node) -> str:
    return " ".join(node.text_content().split()) if node is not None else ""


def decode_bing_url(href: str) -> str:
    """Ссылка-переходник Bing (`/ck/a?...&u=a1<base64>`) → настоящий адрес."""
    if not href or "bing.com/ck/" not in href:
        return href or ""
    try:
        u = parse_qs(urlparse(href).query).get("u", [""])[0]
        if u.startswith("a1"):
            raw = u[2:]
            raw += "=" * (-len(raw) % 4)
            return base64.urlsafe_b64decode(raw).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        pass
    return href


def parse_bing(html: str, limit: int = 6) -> List[WebResult]:
    from lxml import html as lhtml

    try:
        doc = lhtml.fromstring(html)
    except Exception:  # noqa: BLE001
        return []
    out: List[WebResult] = []
    for li in doc.xpath('//li[contains(concat(" ", normalize-space(@class), " "), " b_algo ")]'):
        link = li.xpath(".//h2//a[@href]")
        if not link:
            continue
        url = decode_bing_url(link[0].get("href", ""))
        if not url.startswith("http"):
            continue
        title = _text(link[0])
        snippet = ""
        for xp in ('.//div[contains(@class,"b_caption")]//p', ".//p",
                   './/div[contains(@class,"b_snippet")]'):
            node = li.xpath(xp)
            if node:
                snippet = _text(node[0])
                break
        snippet = re.sub(r"^\d{1,2} \w+\.? (назад|\d{4} г\.)\s*·\s*", "", snippet)
        out.append(WebResult(title=title, url=url, snippet=snippet))
        if len(out) >= limit:
            break
    return out


# Вопросительные слова в начале: поисковик без браузера ищет по ним
# словари («какой — Викисловарь»), а смысл запроса — в остальных словах.
_LEAD_QUESTION = re.compile(
    r"^(что\s+так(ое|ие)|кто\s+так(ой|ая|ие)|как(ой|ая|ое|ие|ов|ова)|"
    r"what\s+is|who\s+is)\s+",
    re.I,
)
# Слова «когда» — не о чём: по ним одним страницу релевантной не считаем
_TIME_WORDS = {"сейчас", "сегодн", "текущ", "актуальн", "нынешн", "now", "today", "current"}


def engine_query(query: str) -> str:
    """Запрос для выдачи без браузера: без вопросительного начала."""
    q = " ".join((query or "").split())
    short = _LEAD_QUESTION.sub("", q, count=1).strip()
    return short if short else q


def _content_tokens(text: str) -> set:
    from memory.book_search import tokenize
    return {t for t in tokenize(text) if t not in _TIME_WORDS}


def relevant(results: List[WebResult], query: str) -> List[WebResult]:
    """Оставить результаты, где есть хоть одно содержательное слово запроса."""
    need = _content_tokens(query)
    if not need:
        return list(results)
    out = []
    for r in results:
        have = _content_tokens(f"{r.title} {r.snippet} {r.text[:600]}")
        if need & have:
            out.append(r)
    return out


def _skip(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    return any(host == h or host.endswith("." + h) for h in _SKIP_HOSTS) or \
        path.endswith(_SKIP_EXT)


def _result_key(r: WebResult) -> str:
    """Одна страница — один ключ: без www, якоря и хвостового слэша."""
    if not r.url:
        return "site:" + r.site
    u = urlparse(r.url)
    host = u.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{u.path.rstrip('/')}?{u.query}"


def dedupe_results(results: List[WebResult]) -> List[WebResult]:
    """Убрать повторы одной страницы; прочитанный текст не терять."""
    kept: Dict[str, WebResult] = {}
    for r in results:
        key = _result_key(r)
        first = kept.get(key)
        if first is None:
            kept[key] = r
        elif r.text and not first.text:
            first.text = r.text
    return list(kept.values())


# ── Чтение страницы ─────────────────────────────────────────
_DROP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside",
              "form", "svg", "iframe", "button", "select", "template")


def extract_paragraphs(html: str) -> List[str]:
    """Осмысленные абзацы страницы: без меню, скриптов и подвалов."""
    from lxml import html as lhtml

    try:
        doc = lhtml.fromstring(html)
    except Exception:  # noqa: BLE001
        return []
    for bad in doc.xpath("|".join(f"//{t}" for t in _DROP_TAGS)):
        bad.drop_tree()
    paras: List[str] = []
    seen = set()
    for node in doc.xpath("//p|//li|//h1|//h2|//h3|//td|//blockquote|//dd|//pre"):
        text = " ".join(node.text_content().split())
        if len(text) < 40 or text in seen:
            continue
        # Списки ссылок (меню) — много коротких ссылок на мало текста
        links = node.xpath(".//a")
        if links and sum(len(_text(a)) for a in links) > 0.6 * len(text):
            continue
        seen.add(text)
        paras.append(text)
    return paras


def best_paragraphs(paras: List[str], query: str, max_chars: int = PAGE_CHARS) -> str:
    """Абзацы с наибольшим числом слов запроса, в исходном порядке."""
    from memory.book_search import tokenize

    terms = set(tokenize(query))
    if not paras:
        return ""
    scored = []
    for i, p in enumerate(paras):
        toks = tokenize(p)
        overlap = len(terms & set(toks)) if terms else 0
        # цифры важны для «курса» и «счёта»
        digits = 0.3 if re.search(r"\d", p) else 0.0
        scored.append((overlap + digits, i, p))
    scored.sort(key=lambda s: (-s[0], s[1]))
    picked, used = [], 0
    for score, i, p in scored:
        if terms and score < 1:
            break
        if used + len(p) > max_chars and picked:
            continue
        picked.append((i, p[:max_chars]))
        used += min(len(p), max_chars)
        if used >= max_chars:
            break
    if not picked:                       # слова не совпали — хотя бы начало
        picked = [(0, paras[0][:max_chars])]
    return "\n".join(p for _, p in sorted(picked))


class WebSearch:
    """Поиск + чтение страниц с кешем на 10 минут."""

    def __init__(self, session=None, fetch_pages: int = 2, max_results: int = 5):
        self._session = session
        self.fetch_pages = fetch_pages
        self.max_results = max_results
        self._cache: Dict[str, Tuple[float, WebAnswer]] = {}
        self._lock = threading.Lock()

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(HEADERS)
        return self._session

    # ── источники ───────────────────────────────────────────
    def bing(self, query: str) -> List[WebResult]:
        r = self.session.get(
            "https://www.bing.com/search",
            params={"q": query, "setlang": "ru"},
            timeout=SEARCH_TIMEOUT,
        )
        r.raise_for_status()
        return parse_bing(r.text, limit=self.max_results + 3)

    def wikipedia(self, query: str, lang: str = "ru") -> List[WebResult]:
        api = f"https://{lang}.wikipedia.org/w/api.php"
        r = self.session.get(api, params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrlimit": 3, "prop": "extracts|info",
            "exintro": 1, "explaintext": 1, "exchars": 1200, "inprop": "url",
        }, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        pages = (r.json().get("query") or {}).get("pages") or {}
        out = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            extract = " ".join((page.get("extract") or "").split())
            out.append(WebResult(title=page.get("title", ""),
                                 url=page.get("fullurl", ""),
                                 snippet=extract[:300], text=extract,
                                 engine="wikipedia"))
        return out

    def fetch(self, url: str) -> str:
        r = self.session.get(url, timeout=PAGE_TIMEOUT, stream=True)
        try:
            ctype = r.headers.get("content-type", "")
            if r.status_code >= 400 or "html" not in ctype.lower():
                return ""
            raw = b""
            for chunk in r.iter_content(65536):
                raw += chunk
                if len(raw) > MAX_PAGE_BYTES:
                    break
        finally:
            r.close()
        enc = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else None
        try:
            return raw.decode(enc or r.apparent_encoding or "utf-8", "replace")
        except LookupError:
            return raw.decode("utf-8", "replace")

    # ── главный вызов ───────────────────────────────────────
    def search(self, query: str, read_pages: Optional[int] = None) -> WebAnswer:
        query = " ".join((query or "").split())[:200]
        if not query:
            return WebAnswer(query="", error="пустой запрос")
        key = query.lower()
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < CACHE_TTL:
                return hit[1]

        t0 = time.time()
        answer = WebAnswer(query=query)
        plain = engine_query(query)
        try:
            found = [r for r in self.bing(plain) if not _skip(r.url)]
            results = relevant(found, query)
            if found and not results:
                answer.error = "bing: выдача нерелевантна"
        except Exception as e:  # noqa: BLE001
            logger.info("Bing недоступен: %s", e)
            answer.error = f"bing: {e}"
            results = []
        if not results:
            try:
                results = relevant(self.wikipedia(plain), query)
            except Exception as e:  # noqa: BLE001
                logger.info("Википедия недоступна: %s", e)
                answer.error = (answer.error + f"; wiki: {e}").strip("; ")
        results = dedupe_results(results)[:self.max_results]

        n = self.fetch_pages if read_pages is None else read_pages
        to_read = [r for r in results if not r.text][:max(0, n)]
        if to_read:
            with ThreadPoolExecutor(max_workers=len(to_read)) as pool:
                pages = list(pool.map(self._safe_fetch, [r.url for r in to_read]))
            for res, html in zip(to_read, pages):
                if html:
                    res.text = best_paragraphs(extract_paragraphs(html), query)

        answer.results = results
        answer.seconds = time.time() - t0
        if results:
            with self._lock:
                if len(self._cache) >= CACHE_SIZE:
                    oldest = min(self._cache, key=lambda k: self._cache[k][0])
                    self._cache.pop(oldest, None)
                self._cache[key] = (time.time(), answer)
        logger.info("Интернет: «%s» → %d результатов, %.1f с",
                    query, len(results), answer.seconds)
        return answer

    def shutdown(self) -> None:
        """Освободить сессию и кеш. Без Chrome закрывать тут нечего."""
        with self._lock:
            session, self._session = self._session, None
            self._cache.clear()
        if session is not None:
            try:
                session.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("Сессия поиска не закрылась: %s", e)

    def _safe_fetch(self, url: str) -> str:
        try:
            return self.fetch(url)
        except Exception as e:  # noqa: BLE001
            logger.debug("страница %s не прочиталась: %s", url, e)
            return ""


def format_results(answer: Optional[WebAnswer], max_chars: int = TOTAL_CHARS) -> str:
    """Блок для промпта: мгновенный ответ, сайт, заголовок, текст."""
    if answer is None:
        return ""
    if not answer.answer and not answer.results:
        # Иначе модель «вспоминает» цифры сама и выдаёт их за найденные
        return (f"Поиск в интернете по запросу «{answer.query}» ничего полезного "
                "не нашёл. Скажи об этом прямо и не придумывай цифры и факты.")
    parts, used = [], 0
    if answer.answer:
        head = f"Готовый ответ на странице поиска: {answer.answer}"
        parts.append(head)
        used += len(head)
    for i, r in enumerate(answer.results, 1):
        body = (r.text or r.snippet).strip()
        if not body:
            continue
        if used + len(body) > max_chars and parts:
            body = r.snippet.strip()
            if not body or used + len(body) > max_chars:
                continue
        used += len(body)
        parts.append(f"[{i}] {r.site} — {r.title}\n{body}")
    return "\n\n".join(parts)


# ── Google в своём окне Chrome ──────────────────────────────
# Зачем так: обычная выдача без браузера то работает, то нет, а Google
# в настоящем Chrome отвечает за 1–1,5 секунды. Окно свёрнуто, курсор
# пользователя не трогается: всё делается через DevTools Protocol.
GOOGLE_HOME = "https://www.google.com/?hl=ru"
GOOGLE_TIMEOUT = 8.0
BROWSER_READ_TIMEOUT = 6.0
ANSWER_CHARS = 320

_GOOGLE_MARK_JS = "window.__s = 1"
_GOOGLE_FOCUS_JS = (
    "(() => { const e = document.querySelector("
    "'textarea[name=q],input[name=q]');"
    " if (!e) return false; e.focus(); e.select(); return true; })()"
)
_GOOGLE_READY_JS = (
    "window.__s === undefined && !!document.querySelector('#rso, #center_col')"
)
_GOOGLE_TOP_JS = (
    "(() => { const c = document.querySelector('#center_col');"
    " return c ? c.innerText.slice(0, 900) : ''; })()"
)
_GOOGLE_RESULTS_JS = r"""
(() => {
  const seen = new Set(), out = [];
  for (const h of document.querySelectorAll('#rso a h3, #center_col a h3')) {
    const a = h.closest('a');
    if (!a || seen.has(a.href)) continue;
    seen.add(a.href);
    let box = a;
    while (box.parentElement &&
           box.parentElement.querySelectorAll('h3').length === 1) {
      box = box.parentElement;
    }
    const pick = (sel) => {
      const n = box.querySelector(sel);
      return n ? (n.innerText || '').replace(/\s+/g, ' ').trim() : '';
    };
    const c = box.querySelector('cite');
    out.push({title: (h.innerText || '').trim(), href: a.href,
              ping: a.getAttribute('ping') || '',
              cite: c ? c.innerText.replace(/\s+/g, ' ').trim() : '',
              snippet: pick('.VwiC3b') || pick('[data-sncf]'),
              text: (box.innerText || '').replace(/\s+/g, ' ').trim()
                    .slice(0, 400)});
    if (out.length >= __LIMIT__) break;
  }
  return out;
})()
"""

# Строки над выдачей, после которых начинаются уже обычные результаты
_STOP_LINE = re.compile(
    r"^(все результаты|результаты поиска|показать все|ещё результаты|"
    r"web results|картинки|видео|новости|карты|покупки|книги|"
    r"вопросы по теме|популярные запросы|люди также спрашивают|"
    r"похожие запросы|связанные запросы|показать ещё)\b",
    re.I,
)
_JUNK_LINE = re.compile(
    r"(результатов: примерно|\(function\(\)|перейти к|"
    r"перевести эту страницу|отказ от обязательств|"
    r"^https?://|^[\w.-]+\.(ru|com|org|net|io|by)$|"
    r"^[\w\s.-]{0,20}биржа|^криптовалют)",
    re.I,
)


def clean_answer(text: str, limit: int = ANSWER_CHARS) -> str:
    """Верх блока выдачи без служебных строк: там обычно готовый ответ.

    Ответ узнаётся по цифрам (курс, температура, счёт). До первой такой
    строки оставляем название («Москва», «Конвертер валют»), после неё
    строки без цифр — уже карусель тикеров и ссылки, их выбрасываем.
    """
    lines: List[str] = []
    lead = 0
    for raw in (text or "").splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if _STOP_LINE.match(line):
            break
        if _JUNK_LINE.search(line) or len(line) > 400:
            continue
        if re.search(r"\d", line):
            lead = 2
        elif lead < 2:
            lead += 1
        else:
            continue
        lines.append(line)
        if len(lines) >= 6:
            break
    out = "\n".join(lines).strip()
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].strip() or out[:limit]
    return out


def decode_google_href(href: str, ping: str = "") -> str:
    """Настоящий адрес из ссылки Google: сначала `ping`, потом `?q=`."""
    for raw in (ping, href):
        if not raw:
            continue
        try:
            qs = parse_qs(urlparse(raw).query)
        except ValueError:
            continue
        for key in ("url", "q", "u"):
            value = qs.get(key, [""])[0]
            if value.startswith("http") and "google." not in urlparse(value).netloc:
                return value
    return ""


def _host_of(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _host_from_cite(cite: str) -> str:
    """`cbr.ru › currency_base` → `cbr.ru` (когда ссылку раскрыть не вышло)."""
    first = (cite or "").split("›")[0].strip()
    if first.startswith("http"):
        first = urlparse(first).netloc
    first = first.split("/")[0].strip()
    if not re.match(r"^[\w.-]+\.[a-z]{2,}$", first, re.I):
        return ""
    return first[4:] if first.startswith("www.") else first


def _default_bridge():
    from config import settings
    from modules.browser_agent import ChromeBridge, PROFILE_SOFIA

    return ChromeBridge(port=int(getattr(settings, "BROWSER_PORT", 9222)),
                        profile=PROFILE_SOFIA, minimized=True)


class BrowserSearch:
    """Google через свой Chrome: набрала запрос, прочитала выдачу, закрыла окно.

    Окно живёт между запросами (открыть Chrome дороже, чем один поиск),
    поэтому его цель помечается `reserve_target`: автопилот в неё не залезет.
    """

    def __init__(self, bridge=None, bridge_factory=None, reader=None,
                 fetch_pages: int = 2, max_results: int = 5,
                 timeout: float = GOOGLE_TIMEOUT):
        self._bridge = bridge
        self._factory = bridge_factory
        self.reader = reader if reader is not None else WebSearch()
        self.fetch_pages = fetch_pages
        self.max_results = max_results
        self.timeout = timeout
        self._target: Optional[str] = None
        self._cookie_jar: Dict[str, str] = {}
        self._cache: Dict[str, Tuple[float, WebAnswer]] = {}
        self._lock = threading.Lock()
        self.err = ""

    # ── окно ────────────────────────────────────────────────
    def _bus(self):
        if self._bridge is None:
            self._bridge = (self._factory or _default_bridge)()
        return self._bridge

    def _open(self, bus) -> None:
        if self._target and bus.has_target(self._target):
            return
        # Окно после запуска Chrome уже есть: второй сделал бы пустую вкладку
        self._target = bus.own_page(GOOGLE_HOME, minimized=True)
        reserve_target(self._target)
        bus.focus_emulation(True)

    def _wait_results(self, bus, session) -> None:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                if bus._eval(_GOOGLE_READY_JS, session):
                    time.sleep(0.25)   # блок ответа дорисовывается позже выдачи
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.1)
        raise RuntimeError(f"Google не показал выдачу за {self.timeout:.0f} с")

    # ── ссылки ──────────────────────────────────────────────
    def _resolve(self, href: str) -> str:
        """Переходник `/goto?…` → адрес: обычно виден в 302 от Google."""
        url = href if href.startswith("http") else "https://www.google.com" + href
        try:
            r = self.reader.session.get(
                url, allow_redirects=False, timeout=SEARCH_TIMEOUT,
                cookies=self._cookie_jar or None,
            )
            loc = r.headers.get("location") or ""
            if loc.startswith("http") and "google." not in urlparse(loc).netloc:
                return loc
        except Exception as e:  # noqa: BLE001
            logger.debug("переходник Google не раскрылся: %s", e)
        return ""

    def _to_result(self, item: dict) -> Optional[WebResult]:
        href = str(item.get("href") or "")
        url = decode_google_href(href, str(item.get("ping") or ""))
        if not url and href.startswith("http") and "google." not in urlparse(href).netloc:
            url = href
        if not url and ("/goto" in href or "/url?" in href):
            url = self._resolve(href)
        site = _host_of(url) or _host_from_cite(str(item.get("cite") or ""))
        if not url and not site:
            return None
        title = str(item.get("title") or site).strip()
        snippet = str(item.get("snippet") or "").strip()
        if not snippet:
            # Блок целиком: заголовок, домен и «Перевести эту страницу» — лишние
            snippet = str(item.get("text") or "").strip()
            if title and snippet.startswith(title):
                snippet = snippet[len(title):].strip()
            cite = str(item.get("cite") or "").strip()
            if cite and snippet.startswith(cite):
                snippet = snippet[len(cite):].strip()
            snippet = re.sub(r"^(·\s*)?Перевести эту страницу\s*", "", snippet)
            snippet = snippet.lstrip(" ·|—-")
        return WebResult(title=title, url=url, snippet=snippet[:300],
                         engine="google", site_name=site)

    # ── главный вызов ───────────────────────────────────────
    def _ask(self, query: str, read_pages: Optional[int] = None) -> WebAnswer:
        t0 = time.time()
        bus = self._bus()
        bus.start()
        self._open(bus)
        session = getattr(bus, "session", None)

        url = ""
        try:
            url = bus.current_url() or ""
        except Exception:  # noqa: BLE001
            pass
        if "google." not in url or "/sorry/" in url:
            bus.navigate(GOOGLE_HOME, timeout=self.timeout)
        if not bus._eval(_GOOGLE_FOCUS_JS, session):
            bus.navigate(GOOGLE_HOME, timeout=self.timeout)
            if not bus._eval(_GOOGLE_FOCUS_JS, session):
                raise RuntimeError("Google не показал поле поиска")

        bus._eval(_GOOGLE_MARK_JS, session)      # чтобы отличить новую страницу
        bus.type_text(query)
        bus.press("Enter")
        try:
            self._wait_results(bus, session)
        except Exception:
            if "/sorry/" in (bus.current_url() or ""):
                raise RuntimeError("Google просит проверку (капча)")
            raise

        url = bus.current_url() or ""
        if "/sorry/" in url:
            raise RuntimeError("Google просит проверку (капча)")

        top = clean_answer(bus._eval(_GOOGLE_TOP_JS, session) or "")
        try:
            self._cookie_jar = bus.cookies()
        except Exception:  # noqa: BLE001
            self._cookie_jar = {}

        raw = bus._eval(_GOOGLE_RESULTS_JS.replace("__LIMIT__", str(self.max_results + 2)),
                        session) or []
        results: List[WebResult] = []
        for item in raw:
            res = self._to_result(item)
            if res is not None:
                results.append(res)
        results = dedupe_results([r for r in results if not _skip(r.url or "")])
        results = results[:self.max_results]

        n = self.fetch_pages if read_pages is None else read_pages
        to_read = [r for r in results if r.url and not r.text][:max(0, n)]
        if to_read:
            with ThreadPoolExecutor(max_workers=len(to_read)) as pool:
                pages = list(pool.map(lambda r: self._read(r.url, query), to_read))
            for res, text in zip(to_read, pages):
                res.text = text
            # Сайты, закрытые от простых запросов (антибот, JS), читаем тем же
            # окном: для страницы это обычный браузер, а не скрипт. Если Google
            # уже дал готовый ответ, второй страницы достаточно одной.
            rest = [r for r in to_read if not r.text]
            if top:
                rest = rest[:1]
            for res in rest:
                res.text = self._read_in_browser(bus, res.url, query)

        return WebAnswer(query=query, results=results, answer=top,
                         seconds=time.time() - t0, engine="google")

    def _read_in_browser(self, bus, url: str, query: str) -> str:
        try:
            bus.navigate(url, timeout=min(self.timeout, BROWSER_READ_TIMEOUT))
            text = bus.read_text(max_chars=12000) or ""
        except Exception as e:  # noqa: BLE001
            logger.debug("страница %s не открылась в Chrome: %s", url, e)
            return ""
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return best_paragraphs([ln for ln in lines if len(ln) >= 40], query)

    def _read(self, url: str, query: str) -> str:
        try:
            return best_paragraphs(extract_paragraphs(self.reader.fetch(url)), query)
        except Exception as e:  # noqa: BLE001
            logger.debug("страница %s не прочиталась: %s", url, e)
            return ""

    def search(self, query: str, read_pages: Optional[int] = None) -> WebAnswer:
        query = " ".join((query or "").split())[:200]
        if not query:
            return WebAnswer(query="", error="пустой запрос")
        key = query.lower()
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < CACHE_TTL:
                return hit[1]

            t0 = time.time()
            self.err = ""
            answer: Optional[WebAnswer] = None
            try:
                answer = self._ask(query, read_pages)
            except Exception as e:  # noqa: BLE001
                self.err = str(e)
                logger.info("Google через браузер не вышел: %s", e)

            if answer is None or not answer.ok:
                # Запасной путь без браузера: Bing, потом Википедия.
                why = self.err or (answer.error if answer is not None else "")
                fallback = self.reader.search(query, read_pages)
                if answer is not None:
                    fallback.answer = answer.answer
                fallback.error = why or fallback.error
                fallback.seconds = time.time() - t0
                answer = fallback
            if answer.ok:
                if len(self._cache) >= CACHE_SIZE:
                    oldest = min(self._cache, key=lambda k: self._cache[k][0])
                    self._cache.pop(oldest, None)
                self._cache[key] = (time.time(), answer)
            logger.info("Интернет (%s): «%s» → %d результатов, %.1f с",
                        answer.engine, query, len(answer.results), answer.seconds)
            return answer

    def shutdown(self) -> None:
        """Закрыть окно поиска. Чужой Chrome (не наш процесс) не гасим."""
        with self._lock:
            bus = self._bridge
            target, self._target = self._target, None
            if bus is None:
                return
            if target:
                release_target(target)
                try:
                    bus.close_target(target)
                except Exception:  # noqa: BLE001
                    pass
            if getattr(bus, "owned", False):
                try:
                    bus.stop_chrome()
                except Exception as e:  # noqa: BLE001
                    logger.debug("Chrome поиска не закрылся: %s", e)
            self._bridge = None


_DEFAULT: Optional[WebSearch] = None
_DEFAULT_SEARCH = None


def engine():
    """Поисковик по настройкам: Google в своём Chrome или обычная выдача."""
    global _DEFAULT_SEARCH
    if _DEFAULT_SEARCH is None:
        try:
            from config import settings
            kind = str(getattr(settings, "WEB_SEARCH_ENGINE", "google") or "").lower()
            pages = int(getattr(settings, "WEB_FETCH_PAGES", 2))
            limit = int(getattr(settings, "WEB_MAX_RESULTS", 5))
        except Exception:  # noqa: BLE001
            kind, pages, limit = "bing", 2, 5
        if kind == "google":
            _DEFAULT_SEARCH = BrowserSearch(
                reader=WebSearch(fetch_pages=pages, max_results=limit),
                fetch_pages=pages, max_results=limit,
            )
        else:
            _DEFAULT_SEARCH = WebSearch(fetch_pages=pages, max_results=limit)
    return _DEFAULT_SEARCH


def reset_default() -> None:
    global _DEFAULT, _DEFAULT_SEARCH
    _DEFAULT = None
    _DEFAULT_SEARCH = None


def default() -> WebSearch:
    return engine()
