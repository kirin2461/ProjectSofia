"""Управление настоящим Chrome через DevTools Protocol (CDP).

Зачем CDP, а не клики по картинке: 4B-модель не умеет надёжно находить
кнопку по пикселям, но отлично выбирает действие из списка и называет
текст кнопки. CDP даёт доступ к настоящему DOM и тексту страницы, поэтому
«нажми Войти» работает и на сложных сайтах, и в headless-режиме.

Соединение синхронное (websockets.sync) — модуль вызывается из рабочего
потока, чтобы не блокировать event loop сервера.

Профили:
  sofia — отдельная папка data/browser/sofia (входы сохраняются, ваш
          основной Chrome не трогается);
  user  — ваш профиль Chrome. Требует закрытого браузера: Chrome не
          отдаёт порт отладки уже запущенному процессу.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9222
PROFILE_SOFIA = "sofia"
PROFILE_USER = "user"

_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

# Действия, которые могут стоить денег или говорить от имени пользователя.
# Выполняются только после подтверждения (см. Autopilot).
_MONEY_WORDS = (
    "оплат", "купи", "купить", "заказ", "оформить заказ", "платёж", "платеж",
    "перевести", "перевод", "списать", "подписк", "тариф", "баланс",
    "pay", "buy", "checkout", "purchase", "subscribe", "billing", "donate",
)
_SEND_WORDS = (
    "отправить", "отправь", "послать", "опубликовать", "разместить",
    "send", "publish", "post", "submit", "tweet", "reply",
)
_MONEY_URLS = (
    "checkout", "payment", "pay.", "billing", "wallet", "bank", "sber",
    "tinkoff", "alfabank", "paypal", "stripe.com/pay", "oplata",
)


class BrowserError(RuntimeError):
    """Браузер недоступен или команда не выполнилась."""


# Вкладки, занятые фоновой работой (поиск в интернете): автопилот и панель
# их не видят и не перехватывают, иначе два потока водили бы одну вкладку.
_RESERVED: set = set()


def reserve_target(target_id: str) -> None:
    if target_id:
        _RESERVED.add(target_id)


def release_target(target_id: str) -> None:
    _RESERVED.discard(target_id)


# ── Профили и запуск ────────────────────────────────────────
def find_chrome() -> Optional[str]:
    env = os.environ.get("SOFIA_CHROME")
    if env and Path(env).exists():
        return env
    for cand in _CHROME_CANDIDATES:
        if Path(cand).exists():
            return cand
    return shutil.which("chrome") or shutil.which("msedge")


def data_dir() -> Path:
    """Куда приложению можно писать: папка сборки или %LOCALAPPDATA%/Sofia."""
    try:
        from app.paths import resolve_base_dir
        return resolve_base_dir()
    except Exception:  # noqa: BLE001 — модуль работает и без лаунчера
        return Path(__file__).resolve().parent.parent


def profile_dir(name: str = PROFILE_SOFIA) -> Path:
    """Папка профиля браузера."""
    if name == PROFILE_USER:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Google" / "Chrome" / "User Data"
    return data_dir() / "data" / "browser" / "sofia"


def screens_dir() -> Path:
    return data_dir() / "data" / "screens"


def _same_path(a: str, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(a.strip('"'))) == \
            os.path.normcase(os.path.abspath(str(b)))
    except (TypeError, ValueError):
        return False


def _profile_processes(profile_path: Path) -> list:
    """Процессы chrome.exe, запущенные именно с этой папкой профиля."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name not in ("chrome.exe", "chrome"):
                continue
            for arg in proc.info.get("cmdline") or []:
                if arg.startswith("--user-data-dir=") and \
                        _same_path(arg.split("=", 1)[1], profile_path):
                    found.append(proc)
                    break
        except Exception:  # noqa: BLE001 — процесс исчез, доступа нет
            continue
    return found


def _http_json(url: str, timeout: float = 3.0) -> dict:
    import requests

    return requests.get(url, timeout=timeout).json()


class CDPConnection:
    """Одно websocket-соединение с браузером: команды по id, события мимо."""

    def __init__(self, ws_url: str, timeout: float = 20.0):
        self._url = ws_url
        self._timeout = timeout
        self._stack = contextlib.ExitStack()
        self._ws = None
        self._id = 0

    def _open(self):
        if self._ws is not None:
            return
        from websockets.sync.client import connect

        self._ws = self._stack.enter_context(
            connect(self._url, max_size=32 * 1024 * 1024, open_timeout=self._timeout)
        )

    def close(self):
        self._stack.close()
        self._ws = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def alive(self) -> bool:
        return self._ws is not None

    def send(self, method: str, params: Optional[dict] = None,
             session: Optional[str] = None) -> dict:
        self._open()
        self._id += 1
        msg_id = self._id
        payload: Dict[str, Any] = {"id": msg_id, "method": method,
                                   "params": params or {}}
        if session:
            payload["sessionId"] = session
        try:
            self._ws.send(json.dumps(payload))
            deadline = time.time() + self._timeout
            while True:
                if time.time() > deadline:
                    raise BrowserError(f"CDP: нет ответа на {method}")
                raw = self._ws.recv(timeout=max(1.0, deadline - time.time()))
                reply = json.loads(raw)
                if reply.get("id") != msg_id:
                    continue  # события (Page.loadEventFired и прочие) не наши
                if "error" in reply:
                    err = reply["error"]
                    raise BrowserError(f"CDP {method}: {err.get('message')}")
                return reply.get("result", {})
        except BrowserError:
            raise
        except Exception as e:  # noqa: BLE001
            self.close()
            raise BrowserError(f"связь с браузером потеряна: {e}") from e


class ChromeBridge:
    """Высокоуровневые действия на странице: переход, клик, ввод, чтение."""

    def __init__(self, port: int = DEFAULT_PORT, profile: str = PROFILE_SOFIA,
                 chrome: Optional[str] = None, headless: bool = False,
                 minimized: bool = False):
        self.port = port
        self.profile = profile
        self.chrome = chrome or find_chrome()
        self.headless = headless
        self.minimized = minimized
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[CDPConnection] = None
        self._session: Optional[str] = None
        self._target: Optional[str] = None
        self._our_process = False
        self._page_ready = False

    # ── жизненный цикл ──────────────────────────────────────
    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        try:
            _http_json(f"{self.endpoint}/json/version", timeout=1.0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _kill_stale(self) -> int:
        """Добить Chrome своего профиля, оставшийся без порта отладки.

        Такой процесс держит папку профиля: новый запуск передаёт ему
        управление и сразу выходит. Профиль user — ваш браузер, его не трогаем.
        """
        if self.profile == PROFILE_USER:
            return 0
        procs = _profile_processes(profile_dir(self.profile))
        for proc in procs:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        if procs:
            logger.info("Chrome: закрыт зависший экземпляр профиля %s (%d проц.)",
                        self.profile, len(procs))
            time.sleep(1.0)            # Windows отпускает блокировку профиля
        return len(procs)

    def start(self, url: str = "about:blank") -> None:
        """Подключиться к уже запущенному Chrome или запустить свой."""
        if self._conn is not None and self._conn.alive and self.running():
            return                     # уже подключены: сессию вкладки не рвём
        if not self.running():
            if not self.chrome:
                raise BrowserError("Chrome не найден — установите Google Chrome")
            try:
                self._launch(url)
            except BrowserError as e:
                if "порт отладки" not in str(e) or not self._kill_stale():
                    raise
                self._launch(url)      # вторая попытка — уже на чистом профиле
        self._connect()
        if self.minimized:
            self.minimize_windows()    # незачем держать на экране пустую вкладку

    def _launch(self, url: str) -> None:
        """Запустить Chrome и дождаться порта отладки."""
        args = [
            self.chrome,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={profile_dir(self.profile)}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-popup-blocking",
        ]
        if self.profile == PROFILE_USER:
            args.append("--profile-directory=Default")
        if self.headless:
            args.append("--headless=new")
        args.append(url)
        logger.info("Chrome: запуск (%s)", self.profile)
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._our_process = True
        for _ in range(60):
            if self.running():
                break
            if self._proc.poll() is not None:
                # Chrome мог передать запуск уже работающему экземпляру
                # с этой папкой профиля и выйти сам.
                if self.running():
                    break
                raise BrowserError(
                    "Chrome не отдал порт отладки. Если открыт ваш основной "
                    "браузер, закройте его или выберите профиль Софии."
                )
            time.sleep(0.25)
        else:
            raise BrowserError("Chrome не запустился за 15 секунд")

    def _connect(self) -> None:
        ver = _http_json(f"{self.endpoint}/json/version")
        ws_url = ver.get("webSocketDebuggerUrl")
        if not ws_url:
            raise BrowserError("Chrome не отдал адрес отладки")
        self._conn = CDPConnection(ws_url)
        self._session = None
        self._target = None

    @property
    def session(self) -> Optional[str]:
        """Сессия текущей вкладки (None, пока страница не прицеплена)."""
        return self._session

    @property
    def owned(self) -> bool:
        """Chrome запущен нами, а не найден уже работающим."""
        return bool(self._our_process)

    def close(self, kill: bool = False) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._session = None
        if kill and self._our_process and self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            # Дочерние процессы Chrome живут дольше главного и держат профиль
            self._kill_stale()
        self._proc = None

    def stop_chrome(self) -> None:
        """Закрыть браузер, запущенный нами (ваш Chrome не трогаем)."""
        if self._our_process:
            self.close(kill=True)
            return
        # Прицепились к уже работавшему Chrome. Через CDP его можно закрыть,
        # но только для отдельного профиля Софии: основной браузер
        # пользователя закрывать нельзя.
        if self.profile == PROFILE_SOFIA:
            try:
                ver = _http_json(f"{self.endpoint}/json/version")
                ws_url = ver.get("webSocketDebuggerUrl")
                if ws_url:
                    with CDPConnection(ws_url, timeout=5.0) as conn:
                        conn.send("Browser.close")
            except Exception as e:  # noqa: BLE001
                logger.debug("Chrome не закрылся через CDP: %s", e)
            for _ in range(20):          # Chrome гаснет не мгновенно
                if not self.running():
                    break
                time.sleep(0.15)
        self.close()

    # ── сессия страницы ─────────────────────────────────────
    def _cmd(self, method: str, params: Optional[dict] = None,
             session: Optional[str] = None) -> dict:
        if self._conn is None:
            self.start()
        return self._conn.send(method, params, session)

    def _pages(self) -> List[dict]:
        targets = self._cmd("Target.getTargets").get("targetInfos", [])
        return [t for t in targets
                if t.get("type") == "page" and t.get("targetId") not in _RESERVED]

    def ensure_page(self) -> str:
        """Сессия активной вкладки (создаёт вкладку, если её нет)."""
        if self._session:
            return self._session
        pages = self._pages()
        if not pages:
            created = self._cmd("Target.createTarget", {"url": "about:blank"})
            target = created["targetId"]
        else:
            target = pages[0]["targetId"]
        self._attach(target)
        return self._session

    def _attach(self, target_id: str) -> None:
        res = self._cmd("Target.attachToTarget",
                        {"targetId": target_id, "flatten": True})
        self._session = res["sessionId"]
        self._target = target_id
        self._cmd("Page.enable", {}, self._session)
        self._cmd("Runtime.enable", {}, self._session)

    def _minimize_window(self, target_id: str) -> bool:
        if self.headless:
            return False
        try:
            win = self._cmd("Browser.getWindowForTarget",
                            {"targetId": target_id})["windowId"]
            self._cmd("Browser.setWindowBounds",
                      {"windowId": win, "bounds": {"windowState": "minimized"}})
            return True
        except (BrowserError, KeyError):
            return False

    def minimize_windows(self) -> int:
        """Свернуть все её окна, чтобы они не заслоняли работу человека."""
        if self.headless:
            return 0
        done = 0
        for page in self._pages():
            if self._minimize_window(page["targetId"]):
                done += 1
        return done

    def own_page(self, url: str = "about:blank", minimized: bool = True) -> str:
        """Вкладка для её работы: пустую переиспользуем, чужую не трогаем.

        Chrome после запуска показывает about:blank. Если рядом открыть
        второе окно, пустое так и останется висеть на экране.
        """
        page = next((p for p in self._pages()
                     if (p.get("url") or "") in ("", "about:blank", url)), None)
        if page is None:
            return self.open_window(url, minimized=minimized)
        target = page["targetId"]
        self._attach(target)
        self._cmd("Page.navigate", {"url": url}, self._session)
        self._wait_ready()
        if minimized:
            self._minimize_window(target)
        return target

    def open_window(self, url: str = "about:blank", minimized: bool = True) -> str:
        """Отдельное окно для фоновой работы; вкладка становится текущей.

        Окно сворачивается сразу: человек продолжает работать в своём
        браузере, а София читает страницы у себя.
        """
        created = self._cmd("Target.createTarget",
                            {"url": url, "newWindow": True, "background": True})
        target = created["targetId"]
        if minimized:
            self._minimize_window(target)
        self._attach(target)
        return target

    def close_target(self, target_id: str) -> None:
        with contextlib.suppress(BrowserError):
            self._cmd("Target.closeTarget", {"targetId": target_id})
        if target_id == self._target:
            self._session = None
            self._target = None

    def has_target(self, target_id: str) -> bool:
        """Жива ли вкладка (после закрытия окна или падения Chrome — нет)."""
        try:
            infos = self._cmd("Target.getTargets").get("targetInfos", [])
        except BrowserError:
            return False
        return any(t.get("targetId") == target_id for t in infos)

    def focus_emulation(self, enabled: bool = True) -> None:
        """Клавиши доходят до свёрнутого окна.

        Без этого Enter в фоновом окне не отправляет форму: Chrome считает
        окно неактивным и глушит клавиатуру.
        """
        self._cmd("Emulation.setFocusEmulationEnabled", {"enabled": bool(enabled)},
                  self.ensure_page())

    def cookies(self) -> Dict[str, str]:
        """Cookies текущей страницы: ими раскрывается переходник Google."""
        items = self._cmd("Network.getCookies", {},
                          self.ensure_page()).get("cookies", [])
        return {c["name"]: c["value"] for c in items if c.get("name")}

    def show(self) -> None:
        """Развернуть окно текущей вкладки (если оно было свёрнуто)."""
        if self.headless:
            return
        self.ensure_page()
        with contextlib.suppress(BrowserError, KeyError):
            win = self._cmd("Browser.getWindowForTarget",
                            {"targetId": self._target})["windowId"]
            self._cmd("Browser.setWindowBounds",
                      {"windowId": win, "bounds": {"windowState": "normal"}})
            self._cmd("Target.activateTarget", {"targetId": self._target})

    def _eval(self, expression: str, session: Optional[str] = None) -> Any:
        sid = session or self.ensure_page()
        res = self._cmd(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True,
             "awaitPromise": True},
            sid,
        )
        if res.get("exceptionDetails"):
            desc = res["exceptionDetails"].get("text") or "ошибка JS"
            raise BrowserError(f"страница: {desc}")
        return (res.get("result") or {}).get("value")

    def _wait_ready(self, timeout: float = 20.0) -> None:
        sid = self.ensure_page()
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = self._eval("document.readyState", sid)
            except BrowserError:
                time.sleep(0.2)
                continue
            if state == "complete":
                time.sleep(0.3)  # дать догрузиться динамике
                return
            time.sleep(0.2)

    # ── действия ────────────────────────────────────────────
    def navigate(self, url: str, timeout: float = 25.0) -> str:
        if not re.match(r"^[a-z]+://", url, re.I):
            url = "https://" + url
        sid = self.ensure_page()
        self._cmd("Page.navigate", {"url": url}, sid)
        self._wait_ready(timeout)
        return self.current_url()

    def current_url(self) -> str:
        return self._eval("location.href") or "about:blank"

    def title(self) -> str:
        return self._eval("document.title") or ""

    def read_text(self, max_chars: int = 6000) -> str:
        """Видимый текст страницы (то, что реально видит человек)."""
        text = self._eval(
            "(() => { const b = document.body; "
            "return b ? (b.innerText || b.textContent || '') : ''; })()"
        ) or ""
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n… (обрезано)"
        return text

    def tabs(self) -> List[dict]:
        return [{"index": i, "title": t.get("title", ""), "url": t.get("url", "")}
                for i, t in enumerate(self._pages())]

    def open_tab(self, url: str) -> int:
        if not re.match(r"^[a-z]+://", url, re.I):
            url = "https://" + url
        self._cmd("Target.createTarget", {"url": url})
        self._session = None
        return len(self._pages()) - 1

    def switch_tab(self, index: int) -> dict:
        pages = self._pages()
        if not 0 <= index < len(pages):
            raise BrowserError(f"нет вкладки {index}")
        self._cmd("Target.activateTarget", {"targetId": pages[index]["targetId"]})
        self._attach(pages[index]["targetId"])
        self._wait_ready(10)
        return {"title": self.title(), "url": self.current_url()}

    def close_tab(self, index: int) -> None:
        pages = self._pages()
        if not 0 <= index < len(pages):
            raise BrowserError(f"нет вкладки {index}")
        self._cmd("Target.closeTarget", {"targetId": pages[index]["targetId"]})
        if index == 0:
            self._session = None

    _FIND_JS = """
    (q => {
      const norm = s => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
      let el = null;
      if (q.selector) el = document.querySelector(q.selector);
      if (!el && q.text) {
        const needle = norm(q.text);
        const nodes = [...document.querySelectorAll(
          'a,button,[role=button],[role=link],input[type=submit],' +
          'input[type=button],label,summary,[onclick]')];
        el = nodes.find(n => norm(n.innerText || n.value) === needle)
          || nodes.find(n => norm(n.innerText || n.value).includes(needle));
      }
      if (!el) return null;
      el.scrollIntoView({block: 'center', inline: 'center'});
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      return {x: r.left + r.width / 2, y: r.top + r.height / 2,
              tag: el.tagName.toLowerCase(),
              text: norm(el.innerText || el.value).slice(0, 120)};
    })(%s)
    """

    def find(self, text: str = "", selector: str = "") -> Optional[dict]:
        payload = json.dumps({"text": text or "", "selector": selector or ""},
                             ensure_ascii=False)
        return self._eval(self._FIND_JS % payload)

    def _click_at(self, x: float, y: float) -> None:
        sid = self.ensure_page()
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            params = {"type": kind, "x": x, "y": y, "button": "left",
                      "clickCount": 1}
            self._cmd("Input.dispatchMouseEvent", params, sid)

    def click(self, text: str = "", selector: str = "") -> str:
        """Нажать настоящей мышью в центр найденного элемента."""
        el = self.find(text, selector)
        if not el:
            raise BrowserError(f"не найдено: {text or selector}")
        self._click_at(el["x"], el["y"])
        return el["text"]

    _FIND_FIELD_JS = """
    (q => {
      const norm = s => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
      let el = q.selector ? document.querySelector(q.selector) : null;
      if (!el && q.field) {
        const needle = norm(q.field);
        const nodes = [...document.querySelectorAll(
          'input,textarea,[contenteditable=true]')];
        el = nodes.find(n => norm([n.placeholder, n.name, n.id,
          n.getAttribute('aria-label'), n.getAttribute('type')]
          .filter(Boolean).join(' ')).includes(needle));
      }
      if (!el) return null;
      el.scrollIntoView({block: 'center', inline: 'center'});
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) return null;
      el.focus();
      return {x: r.left + r.width / 2, y: r.top + r.height / 2};
    })(%s)
    """

    def find_field(self, field: str = "", selector: str = "") -> Optional[dict]:
        payload = json.dumps({"field": field or "", "selector": selector or ""},
                             ensure_ascii=False)
        return self._eval(self._FIND_FIELD_JS % payload)

    def type_text(self, text: str, field: str = "", selector: str = "",
                  enter: bool = False) -> str:
        """Ввести текст в поле (по подсказке/подписи/селектору)."""
        if field or selector:
            el = self.find_field(field, selector)
            if not el:
                raise BrowserError(f"поле не найдено: {field or selector}")
            self._click_at(el["x"], el["y"])
        sid = self.ensure_page()
        self._cmd("Input.insertText", {"text": text}, sid)
        if enter:
            self.press("Enter")
        return text

    _KEYS = {
        "enter": ("Enter", 13), "tab": ("Tab", 9), "esc": ("Escape", 27),
        "escape": ("Escape", 27), "backspace": ("Backspace", 8),
        "delete": ("Delete", 46), "down": ("ArrowDown", 40),
        "up": ("ArrowUp", 38), "page_down": ("PageDown", 34),
        "page_up": ("PageUp", 33),
    }

    def press(self, key: str) -> None:
        name = key.strip().lower()
        code, vk = self._KEYS.get(name, (key, 0))
        sid = self.ensure_page()
        # Enter/Tab без поля text — «немая» клавиша: форма не отправится.
        text = {"Enter": "\r", "Tab": "\t"}.get(code, "")
        for kind in ("keyDown", "keyUp"):
            params = {"type": kind, "key": code, "code": code,
                      "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
            if kind == "keyDown" and text:
                params["text"] = text
                params["unmodifiedText"] = text
            self._cmd("Input.dispatchKeyEvent", params, sid)

    def scroll(self, dy: int = 800) -> None:
        sid = self.ensure_page()
        self._cmd("Input.dispatchMouseEvent",
                  {"type": "mouseWheel", "x": 10, "y": 10,
                   "deltaX": 0, "deltaY": int(dy)}, sid)

    def set_viewport(self, width: int, height: int) -> None:
        """Размер «окна» страницы: для headless-снимков и проверки вёрстки."""
        sid = self.ensure_page()
        self._cmd("Emulation.setDeviceMetricsOverride",
                  {"width": int(width), "height": int(height),
                   "deviceScaleFactor": 1, "mobile": False}, sid)

    def screenshot(self, path: Optional[str] = None) -> str:
        sid = self.ensure_page()
        data = self._cmd("Page.captureScreenshot", {"format": "png"}, sid)["data"]
        raw = base64.b64decode(data)
        target = Path(path) if path else (screens_dir() / "browser.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return str(target)

    # ── диспетчер действий ──────────────────────────────────
    def execute(self, action: dict) -> dict:
        """Выполнить действие из JSON модели. Возвращает {ok, detail}."""
        kind = (action.get("action") or "").strip().lower()
        try:
            if kind == "navigate":
                url = self.navigate(str(action.get("url") or ""))
                return {"ok": True, "detail": f"открыто {url}"}
            if kind == "click":
                hit = self.click(str(action.get("text") or ""),
                                 str(action.get("selector") or ""))
                return {"ok": True, "detail": f"нажато «{hit}»"}
            if kind == "type":
                self.type_text(str(action.get("text") or ""),
                               str(action.get("field") or ""),
                               str(action.get("selector") or ""),
                               bool(action.get("enter")))
                return {"ok": True, "detail": "текст введён"}
            if kind == "press":
                self.press(str(action.get("key") or "Enter"))
                return {"ok": True, "detail": "клавиша нажата"}
            if kind == "scroll":
                self.scroll(int(action.get("dy") or 800))
                return {"ok": True, "detail": "прокручено"}
            if kind == "read":
                return {"ok": True, "detail": "страница прочитана"}
            if kind == "open_tab":
                idx = self.open_tab(str(action.get("url") or ""))
                return {"ok": True, "detail": f"новая вкладка {idx}"}
            if kind == "switch_tab":
                info = self.switch_tab(int(action.get("index") or 0))
                return {"ok": True, "detail": f"вкладка: {info['title']}"}
            if kind == "close_tab":
                self.close_tab(int(action.get("index") or 0))
                return {"ok": True, "detail": "вкладка закрыта"}
            if kind == "back":
                self._eval("history.back()")
                self._wait_ready(10)
                return {"ok": True, "detail": "назад"}
            if kind in ("done", "ask"):
                return {"ok": True, "detail": str(action.get("summary") or "")}
            return {"ok": False, "detail": f"неизвестное действие: {kind}"}
        except BrowserError as e:
            return {"ok": False, "detail": str(e)}


# ── Барьер: что нельзя делать без подтверждения ─────────────
def needs_confirmation(action: dict, url: str = "") -> str:
    """Причина подтверждения или пустая строка.

    Деньги и отправка от имени пользователя — единственные категории:
    всё остальное в автономном режиме делается без вопросов.
    """
    kind = (action.get("action") or "").lower()
    text = " ".join(str(action.get(k) or "")
                    for k in ("text", "field", "summary", "key", "url"))
    blob = f"{text} {url}".lower()
    # при вводе текста опасен не сам текст, а куда его вводят и где мы находимся
    money_blob = (f"{action.get('field', '')} {url}".lower()
                  if kind == "type" else blob)
    if any(w in money_blob for w in _MONEY_WORDS):
        return "действие похоже на платёж или покупку"
    if kind == "click" or kind == "press":
        if any(w in text.lower() for w in _SEND_WORDS):
            return "действие похоже на отправку от вашего имени"
    if kind == "navigate" and any(w in str(action.get("url", "")).lower()
                                 for w in _MONEY_URLS):
        return "переход на страницу оплаты"
    return ""
