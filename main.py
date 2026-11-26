"""Point d'entrée: FastAPI + WebSocket. Оркестрация когнитивного цикла.

Базовый режим (всегда): экстрактор → монолог → ответ (стрим) → критик,
память, безопасность (CLIPS/regex), эмоции (PAD), голос (XTTS, опц.).

Опциональные расширения (Phase 6) загружаются только по флагам конфига:
VISION_ENABLED, GAME_MODE_ENABLED, STREAM_ENABLED.
"""

import asyncio
import collections
import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from config import settings
from core.cognitive_cycle import (
    SessionState,
    extract_facts,
    monologue,
    generate_response,
    critic,
)
from core.context import maybe_compact, estimate_tokens
from core.emotion_engine import EmotionEngine
from core.query_rewrite import build_queries, describe as describe_queries
from core.safety_engine import SafetyEngine, BLOCK, WARN
from core.soar_bridge import SOARBridge
from memory import feedback, memory_manager
from memory.book_search import search_books, format_sources
from memory.citation_check import verify as verify_citations
from memory.citation_check import describe as describe_citations
from modules import web_search as web

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sofia")

app = FastAPI(title="Project Sofia")


def _trace(stage: str, **extra):
    """Отметка в журнале (SOFIA_LOG_LEVEL=DEBUG).

    Нужна, когда падение в нативном коде (Soar, CLIPS, torch) убивает процесс
    без трейсбэка: последняя отметка показывает, на каком шаге это случилось.
    """
    logger.debug("шаг %s: %s", stage,
                 ", ".join(f"{k}={v}" for k, v in extra.items()) or "ok")

# ── Глобальное состояние ────────────────────────────────────
emotion = EmotionEngine()
safety = SafetyEngine()
soar = None                # type: ignore  # создаётся в startup(): Soar живёт в потоке uvicorn
tts_engine = None          # type: ignore
vision = None              # type: ignore
game = None                # type: ignore
stream = None              # type: ignore
_library_observer = None
_xp = {"total": 0}
# Профиль пользователя: устойчивые факты (не эпизоды) для промпта
profile = None              # type: ignore  # создаётся в startup()
# Статистика последнего хода: попадание в KV-кеш и размер сводки (для /state)
_LAST_CACHE: dict = {}
# Что происходило в последнем ходе: расчёт, цитаты, формулировки поиска
_LAST_TURN: dict = {}

# Браузер и автономный режим (ленивые: Chrome стартует по команде, не на входе)
_browser = None            # type: ignore
_browser_lock = threading.Lock()
_autopilot = None          # type: ignore
_autopilot_thread = None   # type: ignore
_autopilot_log: collections.deque = collections.deque(maxlen=60)

# Поиск в интернете: свой Chrome с профилем Софии, отдельное свёрнутое окно
_web = None                # type: ignore
_web_lock = threading.Lock()

HTML_PATH = Path(__file__).parent / "static" / "index.html"


# ── XP (простейший прогресс) ────────────────────────────────
def _load_xp():
    try:
        with open(settings.XP_PATH, encoding="utf-8") as f:
            _xp.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass


def _add_xp(amount: int = 1):
    _xp["total"] = _xp.get("total", 0) + amount
    try:
        Path(settings.XP_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(settings.XP_PATH, "w", encoding="utf-8") as f:
            json.dump(_xp, f)
    except OSError:
        pass


# ── Жизненный цикл ──────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global tts_engine, vision, game, stream, _library_observer, soar, profile

    emotion.load()
    _load_xp()

    try:
        from memory.profile import make_profile
        profile = make_profile(settings.PROFILE_PATH)
        logger.info("Профиль: фактов %d", len(profile))
    except Exception as e:  # noqa: BLE001
        logger.warning("Профиль недоступен: %s", e)
        profile = None

    # Soar привязывается к потоку, в котором создан kernel: только здесь,
    # в потоке uvicorn, а не при импорте main из UI-потока.
    try:
        soar = SOARBridge() if settings.SOAR_ENABLED else None
        if soar is not None:
            logger.info("SOAR: движок %s", soar.engine)
    except Exception as e:  # noqa: BLE001
        logger.error("SOAR не запустился: %s", e)
        soar = None

    # Голос (ленивая загрузка самой модели — при первом синтезе)
    try:
        if not settings.TTS_ENABLED:
            raise RuntimeError("отключено в настройках (TTS_ENABLED=false)")
        from modules.tts_engine import TTSEngine
        tts_engine = TTSEngine()
        logger.info("TTSEngine готов (device=%s)", tts_engine.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("TTS недоступен: %s", e)

    # Библиотека: watcher + дочитывание папки в фоне (книги могут быть большими)
    try:
        from memory.library_ingest import start_watcher, scan_library

        _library_observer = start_watcher(settings.LIBRARY_PATH)
        if settings.LIBRARY_ENABLED:
            def _index_library():
                try:
                    memory_manager.migrate_legacy()
                    scan_library(settings.LIBRARY_PATH)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Индексация библиотеки: %s", e)

            threading.Thread(target=_index_library, daemon=True, name="library-scan").start()
    except Exception as e:  # noqa: BLE001
        logger.warning("Library watcher не запущен: %s", e)

    # ── Опциональные расширения (Phase 6) ──────────────────
    if settings.VISION_ENABLED:
        try:
            from modules.vision_engine import init_vision_engine
            vision = init_vision_engine(
                use_mock=settings.VISION_MOCK, vlm_model=settings.MODEL_VLM
            )
            vision.start()
            asyncio.create_task(vision.run(settings.OLLAMA_HOST))
            logger.info("Vision Engine запущен")
        except Exception as e:  # noqa: BLE001
            logger.error("Vision Engine не запущен: %s", e)

    if settings.GAME_MODE_ENABLED:
        try:
            from modules.game_control import init_game_controller
            game = init_game_controller(use_mock=settings.GAME_MOCK)
            logger.info("Game Control готов")
        except Exception as e:  # noqa: BLE001
            logger.error("Game Control не запущен: %s", e)

    if settings.STREAM_ENABLED:
        try:
            from modules.streaming_engine import init_streamer
            stream = init_streamer(
                use_mock=settings.STREAM_MOCK,
                platform=settings.STREAM_PLATFORM,
                output_dir=settings.STREAM_OUTPUT_DIR,
                rtmp_url=settings.STREAM_RTMP_URL or None,
            )
            logger.info("Streaming Engine готов (platform=%s)", settings.STREAM_PLATFORM)
        except Exception as e:  # noqa: BLE001
            logger.error("Streaming Engine не запущен: %s", e)


@app.on_event("shutdown")
async def shutdown():
    emotion.save()
    if soar is not None:
        soar.close()
    if _autopilot is not None:
        _autopilot.stop()
    if _web is not None:
        await asyncio.to_thread(_web.shutdown)
    if _browser is not None:
        _browser.stop_chrome()
    if _library_observer:
        _library_observer.stop()
    if vision:
        vision.stop()
    if stream and stream.is_active():
        stream.stop()


# ── HTTP endpoints ──────────────────────────────────────────
@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Sofia</h1><p>static/index.html не найден</p>")


@app.get("/state")
async def state():
    data = emotion.to_dict()
    data["xp"] = _xp.get("total", 0)
    data["safety_mode"] = safety.mode
    data["vision_enabled"] = settings.VISION_ENABLED
    data["game_enabled"] = settings.GAME_MODE_ENABLED
    data["stream_enabled"] = settings.STREAM_ENABLED
    data["model"] = settings.MODEL_BRAIN
    data["provider"] = settings.LLM_PROVIDER
    data["tts"] = tts_engine is not None
    data["clips"] = {
        "active": safety.clips.available,
        "rules": safety.clips.rule_count,
    }
    data["soar"] = soar.status() if soar is not None else {"engine": "off", "active": False}
    data["books"] = {
        "active": settings.LIBRARY_ENABLED,
        "chunks": memory_manager.count_books(),
        "expand": bool(getattr(settings, "LIBRARY_PARENT_EXPAND", True)),
        "reranker": _reranker_status(),
    }
    data["context"] = {
        "summary_tokens": estimate_tokens(_LAST_CACHE.get("summary", "")),
        "last_cache": dict(_LAST_CACHE),
    }
    data["turn"] = dict(_LAST_TURN)
    data["feedback"] = feedback.stats()
    data["browser"] = {
        "enabled": settings.BROWSER_ENABLED,
        "running": _browser.running() if _browser is not None else False,
        "profile": settings.BROWSER_PROFILE,
        "autonomous": bool(_autopilot and _autopilot.state.get("running")),
        "skills": _skills_stats(),
    }
    data["web"] = {
        "enabled": settings.WEB_SEARCH_ENABLED,
        "engine": settings.WEB_SEARCH_ENGINE,
        "last": dict(_LAST_TURN.get("web") or {}),
    }
    return JSONResponse(data)


def _reranker_status() -> str:
    """Состояние реранкера без загрузки модели: только наличие файлов."""
    try:
        from memory import reranker
        return reranker.describe()
    except Exception:  # noqa: BLE001
        return "недоступен"


def _skills_stats() -> dict:
    """Сколько выученных навыков автопилота (без чтения при отключённом браузере)."""
    try:
        from modules import skills
        return skills.stats()
    except Exception:  # noqa: BLE001
        return {"total": 0, "by_domain": {}}


# ── Обратная связь по ответам ───────────────────────────────
def _feedback_meta(payload: dict) -> dict:
    """Стратегия и число источников для оценки.

    Берём их из последнего хода, только если оценивают именно его: иначе
    👍 старому ответу припишется стратегия совсем другого вопроса.
    """
    question = str(payload.get("question") or "").strip()
    same = bool(question) and question[:400] == str(_LAST_TURN.get("question") or "").strip()
    last = _LAST_TURN if same else {}
    try:
        sources = int(payload.get("sources") or last.get("sources") or 0)
    except (TypeError, ValueError):
        sources = 0
    return {"strategy": str(payload.get("strategy") or last.get("strategy") or ""),
            "sources": sources}


@app.post("/api/feedback")
async def api_feedback(payload: dict | None = None):
    """👍/👎 на последний ответ: единственный сигнал качества от человека."""
    payload = payload or {}
    try:
        rating = int(payload.get("rating", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="rating должен быть числом")
    rating = 1 if rating > 0 else (-1 if rating < 0 else 0)
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    # Без текста не угадываем: подставленный «последний ход» приписал бы
    # оценку чужому ответу
    if not question or not answer:
        raise HTTPException(status_code=400, detail="нужны question и answer оцениваемого ответа")
    meta = _feedback_meta(payload)
    entry = feedback.record(
        rating,
        question,
        answer,
        strategy=meta["strategy"],
        sources=meta["sources"],
        model=settings.MODEL_BRAIN,
        note=str(payload.get("note") or ""),
    )
    st = feedback.stats()
    _trace("feedback", summary=feedback.describe(st))
    return JSONResponse({"ok": True, "stats": st, "entry": entry})


# ── Браузер: Chrome через DevTools Protocol ─────────────────
def _get_browser():
    global _browser
    with _browser_lock:
        if _browser is None:
            from modules.browser_agent import ChromeBridge
            _browser = ChromeBridge(
                port=settings.BROWSER_PORT, profile=settings.BROWSER_PROFILE
            )
        return _browser


# ── Интернет: она ищет сама ─────────────────────────────────
def _get_web():
    global _web
    with _web_lock:
        if _web is None:
            from modules import web_search
            _web = web_search.engine()
        return _web


def _web_lookup(query: str):
    """Синхронный поиск: вызывается в рабочем потоке, не в event loop."""
    if not query:
        return None
    try:
        return _get_web().search(query)
    except Exception as e:  # noqa: BLE001
        logger.warning("поиск в интернете не удался: %s", e)
        return None


async def _search_web(query: str):
    """Поиск с потолком ожидания: лучше ответ без интернета, чем пауза.

    Поиск живёт в своём потоке: по таймауту он доработает и результат
    попадёт в кеш, но этот ход уже ответит без него.
    """
    budget = float(getattr(settings, "WEB_SEARCH_TIMEOUT", 12.0))
    try:
        return await asyncio.wait_for(asyncio.to_thread(_web_lookup, query),
                                      timeout=budget)
    except asyncio.TimeoutError:
        _trace("web timeout", query=query, budget=budget)
        return None


def _autopilot_event(event: dict) -> None:
    """События автономного режима — в журнал для панели и в лог."""
    item = dict(event)
    item["at"] = time.strftime("%H:%M:%S")
    _autopilot_log.append(item)
    if event.get("type") == "step":
        logger.info("АВТОНОМНО: %s", event.get("line"))


@app.get("/api/browser/state")
async def browser_state():
    b = _get_browser()
    info = {
        "enabled": settings.BROWSER_ENABLED,
        "running": b.running(),
        "profile": settings.BROWSER_PROFILE,
        "port": settings.BROWSER_PORT,
        "autonomous": bool(_autopilot and _autopilot.state.get("running")),
    }
    if info["running"]:
        try:
            info["url"] = await asyncio.to_thread(b.current_url)
            info["title"] = await asyncio.to_thread(b.title)
        except Exception as e:  # noqa: BLE001
            info["url"], info["title"] = "", f"(ошибка: {e})"
    return JSONResponse(info)


def _browser_open_url(payload: dict | None) -> str:
    """Куда открывать браузер: введённый адрес, а если пусто — Google."""
    url = ((payload or {}).get("url") or "").strip()
    if url:
        return url
    from modules.web_search import GOOGLE_HOME
    return GOOGLE_HOME


@app.post("/api/browser/open")
async def browser_open(payload: dict | None = None):
    if not settings.BROWSER_ENABLED:
        raise HTTPException(403, "браузер выключен в настройках")
    url = _browser_open_url(payload)
    b = _get_browser()

    def _go():
        if b.running():
            b.navigate(url)        # Chrome уже открыт — просто переходим
        else:
            b.start(url)           # иначе первый экран сразу нужный, без about:blank

    try:
        await asyncio.to_thread(_go)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
    return await browser_state()


@app.post("/api/browser/close")
async def browser_close():
    b = _get_browser()

    def _stop():
        b.stop_chrome()

    await asyncio.to_thread(_stop)
    return await browser_state()


@app.get("/api/browser/screenshot")
async def browser_screenshot():
    b = _get_browser()
    try:
        path = await asyncio.to_thread(b.screenshot)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
    return FileResponse(path, media_type="image/png")


# ── Автономный режим ───────────────────────────────────────
@app.get("/api/autopilot/state")
async def autopilot_state():
    st = dict(_autopilot.state) if _autopilot is not None else {}
    return JSONResponse({"state": st, "log": list(_autopilot_log)})


@app.post("/api/autopilot")
async def autopilot_start(payload: dict | None = None):
    """Запустить автономный прогон: она сама решает, что делать в браузере."""
    global _autopilot, _autopilot_thread
    if not settings.BROWSER_ENABLED:
        raise HTTPException(403, "браузер выключен в настройках")
    data = payload or {}
    goal = str(data.get("goal") or "").strip()
    if not goal:
        raise HTTPException(400, "нужна цель: что сделать в браузере")
    if _autopilot is not None and _autopilot.state.get("running"):
        raise HTTPException(409, "автономный режим уже выполняется")

    from modules.autopilot import Autopilot

    _autopilot_log.clear()
    _autopilot = Autopilot(
        bridge=_get_browser(),
        emitter=_autopilot_event,
        max_steps=int(data.get("max_steps") or settings.AUTOPILOT_MAX_STEPS),
    )
    url = str(data.get("url") or "").strip()
    goal_text = f"{goal}\n\nНачать с адреса: {url}" if url else goal
    _autopilot_thread = threading.Thread(
        target=_autopilot.run, args=(goal_text,), daemon=True, name="autopilot"
    )
    _autopilot_thread.start()
    return JSONResponse({"started": True, "goal": goal})


@app.post("/api/autopilot/stop")
async def autopilot_stop():
    if _autopilot is not None:
        _autopilot.stop()
    return JSONResponse({"stopped": True})


@app.post("/api/autopilot/confirm")
async def autopilot_confirm(payload: dict | None = None):
    """Ответ на запрос подтверждения (деньги/отправка)."""
    if _autopilot is None:
        raise HTTPException(404, "автономный режим не запущен")
    allow = bool((payload or {}).get("allow"))
    _autopilot.confirm(allow)
    return JSONResponse({"allow": allow})


@app.get("/health")
async def health():
    return {"status": "ok", "profile": settings.PROFILE}


# ── WebSocket чат ───────────────────────────────────────────
@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    await ws.accept()
    session = SessionState(session_id=str(uuid.uuid4()))
    logger.info("Новая сессия %s", session.session_id)

    try:
        while True:
            raw = await ws.receive_text()
            await _handle_message(ws, session, raw)
    except WebSocketDisconnect:
        logger.info("Сессия %s закрыта", session.session_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("Ошибка в сессии: %s", e)
    finally:
        _cancel_tts(session)


def _payload_text(raw: str) -> str:
    """Текст реплики из сообщения: обычный текст или JSON от клиента.

    Клиенты присылают по-разному — `text`, `message`, `content`. Неизвестный
    ключ не должен превращать разговор в переписку с JSON-строкой.
    """
    if not raw.startswith("{"):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict):
        return raw
    for key in ("text", "message", "content", "msg"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return raw


async def _handle_message(ws: WebSocket, session: SessionState, raw: str):
    # Поддержка JSON-команд (игра/стрим) и обычного текста
    user_msg = raw
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
            if payload.get("type") in ("game", "stream", "action"):
                await _handle_command(ws, payload, session)
                return
        except json.JSONDecodeError:
            pass
    user_msg = _payload_text(raw)

    # 1. Безопасность входа
    in_check = safety.check_input(user_msg)
    _trace("safety in", action=in_check.action, rule=in_check.rule)
    if in_check.action == BLOCK:
        await ws.send_text(json.dumps({
            "type": "error",
            "error": in_check.reason,
            "reason": in_check.rule,
        }))
        return

    # 0. Сжатие контекста: старая часть диалога превращается в сводку.
    #    До добавления новой реплики, иначе она попадёт в сводку сама.
    comp = await maybe_compact(session)
    if comp.get("compacted"):
        _trace("context", tokens=comp.get("tokens"), saved=comp.get("saved"))

    prev_user = next(
        (str(m.get("content") or "") for m in reversed(session.history_messages())
         if m.get("role") == "user"),
        "",
    )
    session.add_user(user_msg)
    emotion.update_from_text(user_msg)

    # Стратегия ответа: настоящий Soar выбирает оператор на текущее сообщение
    soar_action = None
    strategy = ""
    if soar is not None:
        soar_action = soar.step(user_msg, has_context=len(session.history_messages()) > 1)
        strategy = _soar_strategy(soar_action)
    _trace("soar", operator=getattr(soar_action, "operator", None))

    # 2. Экстрактор (фоном) + retrieval памяти и книг (параллельно)
    facts_task = asyncio.create_task(extract_facts(user_msg))
    recall_task = asyncio.create_task(
        asyncio.to_thread(memory_manager.recall_episodes, user_msg)
    )
    books_task = asyncio.create_task(
        asyncio.to_thread(_recall_books, user_msg, prev_user)
    )
    hint_task = asyncio.create_task(asyncio.to_thread(_math_hint, user_msg))

    # Интернет — только когда он правда нужен: прямая просьба или вопрос
    # о свежем (курс, погода, новости). Остальное — из головы и книг.
    search_now, why = (web.should_search(user_msg)
                       if settings.WEB_SEARCH_ENABLED else (False, ""))
    web_task = None
    web_query = web.make_query(user_msg, prev_user) if search_now else ""
    if web_query:
        _trace("web start", query=web_query, why=why)
        await ws.send_text(json.dumps({"type": "status",
                                       "text": "смотрю в интернете…"}))
        web_task = asyncio.create_task(_search_web(web_query))

    vision_ctx = vision.get_context() if vision else ""

    # Монолог ждёт факты, но с ограничением по времени (fallback)
    try:
        facts = await asyncio.wait_for(asyncio.shield(facts_task), timeout=4.0)
    except asyncio.TimeoutError:
        facts = ""  # не успел — работаем без фактов
    try:
        memories = await recall_task
    except Exception:  # noqa: BLE001
        memories = None

    books_ctx = ""
    try:
        hits = await asyncio.wait_for(asyncio.shield(books_task), timeout=8.0)
        books_ctx = format_sources(hits)
    except Exception as e:  # noqa: BLE001
        logger.warning("книги: поиск не удался (%s)", e)

    _trace("recall", facts=bool(facts), memories=bool(memories),
           books=len(books_ctx.splitlines()) if books_ctx else 0)

    session.last_facts = facts

    hint = ""
    try:
        hint = await asyncio.wait_for(asyncio.shield(hint_task), timeout=3.0)
    except Exception:  # noqa: BLE001
        hint = ""
    queries = _LAST_TURN.pop("queries", [])
    _LAST_TURN.clear()
    _LAST_TURN.update({"hint": hint, "queries": queries, "strategy": strategy})
    _trace("math", hint=bool(hint), queries=describe_queries(queries))

    # Свежие данные из интернета (поиск шёл параллельно с книгами и памятью)
    web_ctx, web_answer = "", None
    if web_task is not None:
        web_answer = await web_task
        if web_answer is not None:
            web_ctx = web.format_results(web_answer)
            _LAST_TURN["web"] = web_answer.to_dict()
            _trace("web", query=web_answer.query, engine=web_answer.engine,
                   results=len(web_answer.results),
                   answer=bool(web_answer.answer),
                   seconds=f"{web_answer.seconds:.1f}")

    # 3. Монолог (внутренний, не отправляем пользователю)
    history = session.history_messages()[:-1]  # без текущего user-msg
    try:
        mono = await asyncio.wait_for(
            monologue(
                facts, memories, user_msg, history, vision_ctx, strategy,
                summary=session.summary, sources=books_ctx, web=web_ctx,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning("monologue timeout — отвечаем без монолога")
        mono = ""
    except Exception as e:  # noqa: BLE001
        logger.error("monologue failed: %s", e)
        mono = ""
    session.last_monologue = mono
    logger.info("MONOLOG: %s", (mono or "")[:200])

    # Она сама может попросить поиск: `ПОИСК: запрос` в монологе.
    request, mono = web.parse_search_request(mono)
    request = _self_search_query(request, hint)
    if request and not web_ctx:
        _trace("web self", query=request)
        await ws.send_text(json.dumps({"type": "status",
                                       "text": "ищу в интернете…"}))
        web_answer = await _search_web(request)
        if web_answer is not None:
            web_ctx = web.format_results(web_answer)
            _LAST_TURN["web"] = web_answer.to_dict()
            _trace("web", query=web_answer.query, engine=web_answer.engine,
                   results=len(web_answer.results),
                   answer=bool(web_answer.answer),
                   seconds=f"{web_answer.seconds:.1f}")

    # Поправка по прошлым оценкам — в монолог, а не в системный блок: префикс
    # промпта должен оставаться неизменным ради KV-кеша.
    try:
        advice = await asyncio.to_thread(feedback.advice)
    except Exception:  # noqa: BLE001
        advice = ""
    if advice:
        mono = f"{mono}\n(поправка по прошлым оценкам: {advice})"
        _LAST_TURN["advice"] = advice

    # Что мы уже знаем о пользователе — тоже в монолог: факты о нём уместны
    # в любом ответе, а системный префикс остаётся неизменным.
    about = ""
    if profile is not None:
        try:
            about = await asyncio.to_thread(profile.as_prompt, 6, user_msg)
        except Exception:  # noqa: BLE001
            about = ""
    if about:
        mono = f"{mono}\n({about})"
        _LAST_TURN["profile_facts"] = len(profile)

    # 4. Ответ со стримингом (эмоция влияет на temperature/tone)
    mod = emotion.to_prompt_modifier()
    await ws.send_text(json.dumps({
        "type": "meta",
        "emotion": emotion.dominant_emotion(),
        "pad": emotion.to_dict(),
    }))

    full_response = []
    cache_stats: dict = {}
    try:
        async for token in generate_response(
            mono, user_msg, history,
            temperature=mod["temperature"],
            tone_instruction=mod["tone_instruction"],
            summary=session.summary,
            sources=books_ctx,
            stats=cache_stats,
            hint=hint,
            web=web_ctx,
        ):
            full_response.append(token)
            await ws.send_text(token)
    except Exception as e:  # noqa: BLE001
        logger.error("generate_response failed: %s", e)
        await ws.send_text("[Ошибка генерации]")

    if cache_stats:
        session.last_cache = cache_stats
        _LAST_CACHE.clear()
        _LAST_CACHE.update(cache_stats)
        _LAST_CACHE["summary"] = session.summary
        total = cache_stats.get("cache_n", 0) + cache_stats.get("prompt_n", 0)
        hit = (100.0 * cache_stats.get("cache_n", 0) / total) if total else 0.0
        _trace("cache", hit=f"{hit:.0f}%", cached=cache_stats.get("cache_n", 0),
               prompt=cache_stats.get("prompt_n", 0))

    response_text = "".join(full_response)
    _trace("reply", chars=len(response_text))

    # 5. Проверка цитат: если модель «процитировала» книгу, а такой фразы
    #    в найденных фрагментах нет — это видно в журнале и в /state.
    if books_ctx:
        try:
            report = verify_citations(response_text, books_ctx)
            _LAST_TURN["citations"] = report.to_dict()
            _trace("citation", summary=describe_citations(report))
        except Exception as e:  # noqa: BLE001
            logger.debug("проверка цитат не удалась: %s", e)

    # 6. Безопасность выхода
    out_check = safety.check_output(response_text)
    if out_check.action == BLOCK:
        response_text = "[ответ заблокирован фильтром безопасности]"
        await ws.send_text(json.dumps({
            "type": "replace", "text": response_text,
        }))

    # Ссылки-источники: кликабельны в интерфейсе (открываются в вашем браузере)
    if web_answer is not None and web_answer.results:
        await ws.send_text(json.dumps({
            "type": "sources",
            "web": [r.to_dict() for r in web_answer.results],
        }))

    await ws.send_text("[END]")
    session.add_assistant(response_text)
    # Для кнопок 👍/👎: оценка относится к последнему ходу этой сессии
    session.last_question = user_msg
    session.last_answer = response_text
    _LAST_TURN["question"] = user_msg[:400]
    _LAST_TURN["answer"] = response_text[:400]
    _LAST_TURN["sources"] = len(books_ctx)

    # 7. Голос (опционально)
    if tts_engine is not None:
        _start_tts(session, ws, response_text)
    _trace("turn done", tts=tts_engine is not None)

    # 8. Критик + запись в память + XP (фоном)
    asyncio.create_task(
        _post_turn(session, user_msg, response_text, facts, soar_action)
    )


def _recall_books(query: str, prev_user: str = ""):
    """Фрагменты книг для текущего вопроса (пусто, если библиотека отключена)."""
    if not settings.LIBRARY_ENABLED:
        return []
    try:
        # Несколько формулировок: бытовой вопрос и термины учебника редко
        # совпадают дословно, а лишний эмбеддинг стоит миллисекунды.
        queries = build_queries(query, prev_user=prev_user)
        _LAST_TURN["queries"] = queries
        return search_books(
            query,
            top_k=settings.LIBRARY_TOP_K,
            pool=settings.LIBRARY_POOL,
            min_score=settings.LIBRARY_MIN_SCORE,
            queries=queries[1:],
            expand=getattr(settings, "LIBRARY_PARENT_EXPAND", True),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("книги недоступны: %s", e)
        return []


def _self_search_query(request: str, hint: str) -> str:
    """Её собственный `ПОИСК:` — или пусто, если искать незачем."""
    if not request or not settings.WEB_SEARCH_ENABLED:
        return ""
    if hint:
        # Калькулятор уже посчитал: арифметику в интернете не ищем
        logger.info("ПОИСК «%s» пропущен: ответ дал калькулятор", request)
        return ""
    return request


def _math_hint(text: str) -> str:
    """Готовая арифметика для промпта (sympy импортируется только при цифрах)."""
    if not any(ch.isdigit() for ch in text or ""):
        return ""
    try:
        from modules.calculator import safe_hint
        return safe_hint(text)
    except Exception as e:  # noqa: BLE001
        logger.debug("калькулятор недоступен: %s", e)
        return ""


def _soar_strategy(action) -> str:
    """Решение Soar → подсказка монологу (не жёсткое правило)."""
    return {
        "op_greet": (
            "Пользователь здоровается: ответь коротким тёплым приветствием "
            "и предложи, чем помочь."
        ),
        "op_ask_clarification": (
            "Сообщение непонятно без контекста: задай один короткий "
            "уточняющий вопрос вместо догадок."
        ),
    }.get(action.operator, "")


async def _send_audio(ws: WebSocket, text: str):
    """Озвучка по предложениям: первое аудио уходит почти сразу.

    На CPU XTTS синтезирует ~в 3.5 раза медленнее реального времени, поэтому
    один большой фрагмент = долгая тишина. Браузер проигрывает фреймы по
    очереди (см. enqueueAudio в static/index.html).
    """
    try:
        for chunk in _split_for_tts(text):
            _trace("tts start", chars=len(chunk))
            audio = await asyncio.to_thread(tts_engine.synthesize, chunk)
            _trace("tts done", bytes=len(audio or b""))
            if audio:
                # Отмена посреди отправки не должна оборвать фрейм:
                # начатый send_bytes дописывается, дальше не идём
                await asyncio.shield(ws.send_bytes(audio))
    except asyncio.CancelledError:
        _trace("tts cancelled")
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("Отправка аудио не удалась: %s", e)


def _start_tts(session: SessionState, ws: WebSocket, text: str):
    """Озвучка ответа; предыдущая этой же сессии отменяется."""
    _cancel_tts(session)
    session.tts_task = asyncio.create_task(_send_audio(ws, text))
    return session.tts_task


def _cancel_tts(session) -> bool:
    """True — если озвучка ещё шла и была отменена."""
    task = getattr(session, "tts_task", None)
    if session is not None:
        session.tts_task = None
    if task is None or task.done():
        return False
    task.cancel()
    return True


def _split_for_tts(text: str, max_len: int = 220) -> list:
    """Разбить ответ на короткие фрагменты по границам предложений.

    Фрагменты — непрерывные куски текста, поэтому склейка возвращает
    исходник без потерь. Нормализуются только пробелы: любая
    последовательность пробельных символов становится одним пробелом,
    пробелы по краям отбрасываются. Инвариант:

        "".join(_split_for_tts(t)) == " ".join((t or "").split())

    Значит, пробел на месте разреза остаётся в конце предыдущего фрагмента
    (синтезатор обрезает его сам), а не выбрасывается. Каждый фрагмент
    непуст после strip() и не длиннее max_len.
    """
    import re

    text = " ".join((text or "").split())
    if not text:
        return []
    max_len = max(1, int(max_len))
    sentence = re.compile(r"[.!?…] ")

    chunks, start, total = [], 0, len(text)
    while start < total:
        if total - start <= max_len:
            chunks.append(text[start:])
            break

        # окно — самое большее max_len символов, начиная с текущей позиции
        window_end = start + max_len
        window = text[start:window_end]

        cut = None
        for m in sentence.finditer(window):
            cut = start + m.end()        # точку с пробелом оставляем слева
        if cut is None:
            space = text.rfind(" ", start, window_end)
            cut = space + 1 if space > start else window_end
        if cut <= start:                 # max_len == 1 и первым идёт пробел
            cut = min(start + 1, total)

        chunk = text[start:cut]
        if not chunk.strip() and cut < total:
            cut += 1
            chunk = text[start:cut]
        chunks.append(chunk)
        start = cut

    return chunks


def _learn_profile(facts: str) -> None:
    """Факты из экстрактора → профиль. Мусор отсеивает сам профиль."""
    if profile is None or not facts:
        return
    added = 0
    for raw in str(facts).splitlines():
        line = raw.strip().lstrip("-*•–—").strip()
        if not line or len(line) > 200:
            continue
        try:
            if profile.add(line, confidence=0.55, source="extractor"):
                added += 1
        except Exception as e:  # noqa: BLE001
            logger.debug("профиль: %s", e)
            break
    if added:
        _trace("profile", added=added, total=len(profile))


async def _post_turn(session, user_msg, response_text, facts, soar_action=None):
    try:
        critique = await critic(response_text, facts)
        logger.info("CRITIC: %s", (critique or "")[:200])
        _learn_profile(facts)
        approved = critique.strip().upper().startswith("OK")
        if approved:
            emotion.update_from_event("критик_одобрил")
        elif critique.strip():
            emotion.update_from_event("критик_нашёл_ошибку")

        if soar is not None and soar_action is not None:
            # Успех = критик одобрил: Soar сохраняет выученные чанки
            soar.feedback(user_msg, soar_action, success=approved)
        _trace("soar feedback", approved=approved)

        pad = emotion.to_dict()
        memory_manager.add_episode(
            f"Пользователь: {user_msg}\nСофия: {response_text}",
            {
                "session_id": session.session_id,
                "timestamp": time.time(),
                "P": pad["P"], "A": pad["A"], "D": pad["D"],
                "emotion": pad["dominant_emotion"],
                "tags": ["dialog"],
                "critique": (critique or "")[:500],
            },
        )
        _trace("episode saved")
        _add_xp(1)
        emotion.save()
        _trace("post turn done")
    except Exception as e:  # noqa: BLE001
        logger.warning("post_turn failed: %s", e)


# ── Команды игрового режима / стрима (Phase 6) ──────────────
async def _emergency_stop(session=None) -> dict:
    """Остановить всё, что София делает руками и голосом. Честный отчёт."""
    report = {"type": "game", "status": "stopped",
              "controller": False, "tts_cancelled": _cancel_tts(session)}
    if game is not None:
        try:
            game.emergency_stop()
            report["controller"] = True
        except Exception as e:  # noqa: BLE001
            logger.error("emergency_stop контроллера не сработал: %s", e)
            report["status"] = "error"
            report["error"] = str(e)
    logger.warning("EMERGENCY STOP: %s", report)
    return report


async def _handle_command(ws: WebSocket, payload: dict, session=None):
    ctype = payload.get("type")
    action = payload.get("action")

    # Стоп должен работать всегда: даже без игрового режима он глушит голос
    if ctype in ("game", "action") and action == "emergency_stop":
        await ws.send_text(json.dumps(await _emergency_stop(session)))
        return

    if ctype == "stream" and stream is not None:
        if action == "start":
            stream.start()
            await ws.send_text(json.dumps({"type": "stream", "status": "started"}))
        elif action == "stop":
            stream.stop()
            await ws.send_text(json.dumps({"type": "stream", "status": "stopped"}))
        else:
            await ws.send_text(json.dumps({
                "type": "stream", "status": "unknown_action"}))
        return

    if ctype in ("game", "action") and game is not None:
        if action == "resume":
            game.resume()
            await ws.send_text(json.dumps({"type": "game", "status": "resumed"}))
        elif action in ("click", "press_key") and getattr(game, "stopped", False):
            await ws.send_text(json.dumps({
                "type": "game", "status": "blocked",
                "error": "Экстренная остановка: сначала команда resume."}))
        elif action == "click":
            game.click(payload.get("x"), payload.get("y"))
            await ws.send_text(json.dumps({"type": "game", "status": "clicked"}))
        elif action == "press_key":
            game.key_press(payload.get("key", ""))
            await ws.send_text(json.dumps({"type": "game", "status": "pressed"}))
        else:
            await ws.send_text(json.dumps({"type": "game", "status": "unknown_action"}))
        return

    await ws.send_text(json.dumps({
        "type": "error",
        "error": "Расширение отключено в конфиге или неизвестная команда.",
    }))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
