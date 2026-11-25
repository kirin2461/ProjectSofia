"""Автономный режим: цель → действие → результат → следующее действие.

Работает в рабочем потоке (браузер синхронный), наружу отдаёт события
через callback. Модель выбирает один шаг из короткого списка — это
надёжнее, чем просить 4B-модель «управлять браузером» целиком.

Границы: деньги и отправка от имени пользователя требуют подтверждения
(modules.browser_agent.needs_confirmation), шаги ограничены бюджетом,
любое повторяющееся действие дважды подряд останавливает цикл.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

from config import settings
from core.jsonx import first_json
from modules.browser_agent import BrowserError, ChromeBridge, needs_confirmation

logger = logging.getLogger(__name__)

PAGE_CHARS = 3000
HISTORY_STEPS = 6

SYSTEM = (
    "Ты — София. Ты управляешь настоящим браузером по шагам. "
    "Ответ — ровно один JSON-объект, без markdown, без пояснений.\n"
    "Действия:\n"
    '{"action":"navigate","url":"https://..."} — открыть адрес\n'
    '{"action":"click","text":"текст кнопки или ссылки"} — нажать\n'
    '{"action":"type","field":"подсказка в поле","text":"что ввести",'
    '"enter":true} — ввести\n'
    '{"action":"press","key":"Enter"} — нажать клавишу\n'
    '{"action":"scroll","dy":900} — прокрутить (минус — вверх)\n'
    '{"action":"open_tab","url":"..."} — новая вкладка\n'
    '{"action":"switch_tab","index":0} / {"action":"close_tab","index":0}\n'
    '{"action":"back"} — назад\n'
    '{"action":"read"} — если текста на странице мало\n'
    '{"action":"done","summary":"что сделано"} — цель достигнута\n'
    '{"action":"ask","summary":"вопрос"} — без человека не обойтись\n'
    "Правила: один шаг за раз. Если нужной кнопки нет — сначала прочитай "
    "страницу и поищи другой текст. Не придумывай адреса: переходи только "
    "по ссылкам со страницы или по известному адресу. Перед done проверь, "
    "что получилось."
)


def _extract_json(text: str) -> Optional[dict]:
    """Достать первый JSON-объект из ответа модели (разбор — в core/jsonx)."""
    obj = first_json(text or "")
    return obj if isinstance(obj, dict) and obj.get("action") else None


def _same_action(a: dict, b: dict) -> bool:
    keys = ("action", "text", "url", "index", "key", "dy")
    return all(str(a.get(k)) == str(b.get(k)) for k in keys)


class Autopilot:
    """Один автономный прогон в браузере."""

    def __init__(
        self,
        bridge: Optional[ChromeBridge] = None,
        emitter: Optional[Callable[[dict], None]] = None,
        max_steps: int = 12,
        confirm_timeout: float = 180.0,
        llm: Optional[Callable[[str, str], str]] = None,
    ):
        self.bridge = bridge or ChromeBridge(
            port=settings.BROWSER_PORT,
            profile=settings.BROWSER_PROFILE,
        )
        self.emit = emitter or (lambda event: None)
        self.max_steps = max(1, int(max_steps))
        self.confirm_timeout = confirm_timeout
        self.llm = llm or _ask_llm

        self._stop = threading.Event()
        self._confirm = threading.Event()
        self._approved = False
        self._lock = threading.Lock()
        self._done_steps: List[dict] = []
        self._skill_hint = ""
        self.state = {
            "running": False,
            "goal": "",
            "steps": 0,
            "url": "",
            "title": "",
            "last": "",
            "blocked": "",
            "summary": "",
        }

    # ── управление снаружи ──────────────────────────────────
    def stop(self) -> None:
        self._stop.set()
        self._confirm.set()  # разбудить ожидание подтверждения

    def confirm(self, allow: bool) -> None:
        with self._lock:
            self._approved = bool(allow)
        self._confirm.set()

    def _ask_confirmation(self, reason: str, action: dict) -> bool:
        with self._lock:
            self._approved = False
            self._confirm.clear()
        self.emit({"type": "confirm", "reason": reason, "action": action})
        logger.info("Автономность: ждём подтверждения (%s)", reason)
        got = self._confirm.wait(self.confirm_timeout)
        with self._lock:
            approved = got and self._approved
        if not approved:
            self.emit({"type": "declined", "reason": reason})
        return approved

    def _log_step(self, action: dict, result: dict) -> str:
        kind = action.get("action")
        target = action.get("text") or action.get("url") or action.get("field") or ""
        mark = "✓" if result.get("ok") else "✗"
        line = f"{mark} {kind} {target}".strip()
        self.state["steps"] += 1
        self.state["last"] = line
        self.emit({"type": "step", "step": self.state["steps"],
                   "action": action, "result": result, "line": line,
                   "url": self.state["url"], "title": self.state["title"]})
        return line

    # ── основной цикл ───────────────────────────────────────
    def run(self, goal: str) -> dict:
        goal = (goal or "").strip()
        if not goal:
            return {"ok": False, "summary": "цель не задана", "steps": 0}
        self._stop.clear()
        self._done_steps = []
        self._skill_hint = ""
        self.state.update({"running": True, "goal": goal, "steps": 0,
                           "last": "", "blocked": "", "summary": ""})
        history: List[str] = []
        prev: Optional[dict] = None
        repeats = 0

        try:
            self.bridge.start()
            self._refresh_state()
            self._skill_hint = self._load_skill_hint(goal)
            self.emit({"type": "start", "goal": goal, "url": self.state["url"]})

            for _ in range(self.max_steps):
                if self._stop.is_set():
                    return self._finish(False, "остановлено пользователем")
                if self.state["blocked"]:
                    return self._finish(False, self.state["blocked"])

                page = self._page_context()
                prompt = self._build_prompt(goal, page, history, self._skill_hint)
                raw = self.llm(SYSTEM, prompt)
                action = _extract_json(raw)
                if not action:
                    history.append("✗ модель вернула не JSON, повторяю")
                    logger.warning("Автономность: не разобрал ответ: %s", raw[:200])
                    continue

                if prev and _same_action(prev, action):
                    repeats += 1
                    if repeats >= 2:
                        return self._finish(
                            False, f"застряла на действии «{action.get('action')}»"
                        )
                else:
                    repeats = 0
                prev = action

                kind = (action.get("action") or "").lower()
                if kind == "done":
                    return self._finish(True, str(action.get("summary") or goal))
                if kind == "ask":
                    self.emit({"type": "ask", "question": str(action.get("summary") or "")})
                    return self._finish(False, str(action.get("summary") or "нужен человек"))

                reason = (needs_confirmation(action, self.state["url"])
                          if settings.AUTOPILOT_BARRIER else "")
                if reason and not self._ask_confirmation(reason, action):
                    return self._finish(False, f"не подтверждено: {reason}")

                result = self.bridge.execute(action)
                line = self._log_step(action, result)
                if result.get("ok"):
                    self._done_steps.append(action)
                history.append(line)
                history[:] = history[-HISTORY_STEPS:]
                self._refresh_state()
                if not result.get("ok") and "не найдено" in str(result.get("detail")):
                    history.append("↳ элемента нет на странице, нужен другой текст")

            return self._finish(False, f"бюджет шагов исчерпан ({self.max_steps})")
        except BrowserError as e:
            logger.warning("Автономность: браузер — %s", e)
            return self._finish(False, f"браузер: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception("Автономность упала")
            return self._finish(False, f"ошибка: {e}")

    def _finish(self, ok: bool, summary: str) -> dict:
        self.state.update({"running": False, "summary": summary})
        self.emit({"type": "finish", "ok": ok, "summary": summary,
                   "steps": self.state["steps"]})
        logger.info("Автономность: %s — %s", "готово" if ok else "стоп", summary)
        if ok:
            self._save_skill()
        return {"ok": ok, "summary": summary, "steps": self.state["steps"]}

    # ── навыки (удачные прогоны → подсказка на будущее) ─────
    def _load_skill_hint(self, goal: str) -> str:
        try:
            from modules import skills
            text = skills.hint(goal, self.state.get("url", ""))
        except Exception as e:  # noqa: BLE001
            logger.debug("навыки недоступны: %s", e)
            return ""
        if text:
            logger.info("Автономность: есть похожий навык, подсказываю путь")
            self.emit({"type": "skill", "hint": text})
        return text

    def _save_skill(self) -> None:
        try:
            from modules import skills
            skills.save_trace(
                self.state.get("goal", ""),
                self._done_steps,
                ok=True,
                url=self.state.get("url", ""),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("навык не сохранён: %s", e)

    # ── состояние страницы ──────────────────────────────────
    def _refresh_state(self) -> None:
        try:
            self.state["url"] = self.bridge.current_url()
            self.state["title"] = self.bridge.title()
        except BrowserError:
            pass

    def _page_context(self) -> str:
        try:
            text = self.bridge.read_text(PAGE_CHARS)
            tabs = self.bridge.tabs()
        except BrowserError as e:
            return f"(страница недоступна: {e})"
        tab_list = " | ".join(
            f"[{t['index']}] {t['title'][:40] or t['url'][:40]}" for t in tabs[:6]
        )
        return (
            f"Адрес: {self.state['url']}\n"
            f"Заголовок: {self.state['title']}\n"
            f"Вкладки: {tab_list}\n"
            f"Текст страницы:\n{text}"
        )

    def _build_prompt(self, goal: str, page: str, history: List[str],
                      hint: str = "") -> str:
        steps = "\n".join(history) if history else "(пока ничего)"
        known = f"\nПрошлый удачный путь: {hint}\n" if hint else ""
        return (
            f"Цель: {goal}\n"
            f"{known}\n"
            f"{page}\n\n"
            f"Что уже сделано:\n{steps}\n\n"
            "Какой один шаг дальше? Ответ — JSON."
        )


def _ask_llm(system: str, prompt: str) -> str:
    """Один синхронный запрос к мозгу (в рабочем потоке)."""
    from openai import OpenAI

    client = OpenAI(base_url=settings.LLM_API,
                    api_key=settings.LLM_API_KEY or "none")
    r = client.chat.completions.create(
        model=getattr(settings, "MODEL_STEP_ACTIVE", settings.MODEL_BRAIN),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        extra_body={"num_ctx": settings.NUM_CTX, "num_predict": 200,
                    "think": False, "cache_prompt": True},
    )
    return r.choices[0].message.content or ""
