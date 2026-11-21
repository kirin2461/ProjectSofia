"""Point d'entrée: FastAPI + WebSocket. Оркестрация когнитивного цикла.

Базовый режим (всегда): экстрактор → монолог → ответ (стрим) → критик,
память, безопасность (CLIPS/regex), эмоции (PAD), голос (XTTS, опц.).

Опциональные расширения (Phase 6) загружаются только по флагам конфига:
VISION_ENABLED, GAME_MODE_ENABLED, STREAM_ENABLED.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from config import settings
from core.cognitive_cycle import (
    SessionState,
    extract_facts,
    monologue,
    generate_response,
    critic,
)
from core.emotion_engine import EmotionEngine
from core.safety_engine import SafetyEngine, BLOCK, WARN
from memory import memory_manager

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sofia")

app = FastAPI(title="Project Sofia")

# ── Глобальное состояние ────────────────────────────────────
emotion = EmotionEngine()
safety = SafetyEngine()
tts_engine = None          # type: ignore
vision = None              # type: ignore
game = None                # type: ignore
stream = None              # type: ignore
_library_observer = None
_xp = {"total": 0}

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
    global tts_engine, vision, game, stream, _library_observer

    emotion.load()
    _load_xp()

    # Голос (ленивая загрузка самой модели — при первом синтезе)
    try:
        from modules.tts_engine import TTSEngine
        tts_engine = TTSEngine()
        logger.info("TTSEngine готов (device=%s)", tts_engine.device)
    except Exception as e:  # noqa: BLE001
        logger.warning("TTS недоступен: %s", e)

    # Библиотека: watcher
    try:
        from memory.library_ingest import start_watcher
        _library_observer = start_watcher(settings.LIBRARY_PATH)
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
    return JSONResponse(data)


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


async def _handle_message(ws: WebSocket, session: SessionState, raw: str):
    # Поддержка JSON-команд (игра/стрим) и обычного текста
    user_msg = raw
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
            if payload.get("type") in ("game", "stream", "action"):
                await _handle_command(ws, payload)
                return
            user_msg = payload.get("text", raw)
        except json.JSONDecodeError:
            pass

    # 1. Безопасность входа
    in_check = safety.check_input(user_msg)
    if in_check.action == BLOCK:
        await ws.send_text(json.dumps({
            "type": "error",
            "error": in_check.reason,
            "reason": in_check.rule,
        }))
        return

    session.add_user(user_msg)
    emotion.update_from_text(user_msg)

    # 2. Экстрактор (фоном) + retrieval памяти (параллельно)
    facts_task = asyncio.create_task(extract_facts(user_msg))
    recall_task = asyncio.create_task(
        asyncio.to_thread(memory_manager.recall_episodes, user_msg)
    )

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

    session.last_facts = facts

    # 3. Монолог (внутренний, не отправляем пользователю)
    history = session.history_messages()[:-1]  # без текущего user-msg
    try:
        mono = await asyncio.wait_for(
            monologue(facts, memories, user_msg, history, vision_ctx),
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

    # 4. Ответ со стримингом (эмоция влияет на temperature/tone)
    mod = emotion.to_prompt_modifier()
    await ws.send_text(json.dumps({
        "type": "meta",
        "emotion": emotion.dominant_emotion(),
        "pad": emotion.to_dict(),
    }))

    full_response = []
    try:
        async for token in generate_response(
            mono, user_msg, history,
            temperature=mod["temperature"],
            tone_instruction=mod["tone_instruction"],
        ):
            full_response.append(token)
            await ws.send_text(token)
    except Exception as e:  # noqa: BLE001
        logger.error("generate_response failed: %s", e)
        await ws.send_text("[Ошибка генерации]")

    response_text = "".join(full_response)

    # 5. Безопасность выхода
    out_check = safety.check_output(response_text)
    if out_check.action == BLOCK:
        response_text = "[ответ заблокирован фильтром безопасности]"
        await ws.send_text(json.dumps({
            "type": "replace", "text": response_text,
        }))

    await ws.send_text("[END]")
    session.add_assistant(response_text)

    # 6. Голос (опционально)
    if tts_engine is not None:
        asyncio.create_task(_send_audio(ws, response_text))

    # 7. Критик + запись в память + XP (фоном)
    asyncio.create_task(
        _post_turn(session, user_msg, response_text, facts)
    )


async def _send_audio(ws: WebSocket, text: str):
    """Озвучка по предложениям: первое аудио уходит почти сразу.

    На CPU XTTS синтезирует ~в 3.5 раза медленнее реального времени, поэтому
    один большой фрагмент = долгая тишина. Браузер проигрывает фреймы по
    очереди (см. enqueueAudio в static/index.html).
    """
    try:
        for chunk in _split_for_tts(text):
            audio = await asyncio.to_thread(tts_engine.synthesize, chunk)
            if audio:
                await ws.send_bytes(audio)
    except Exception as e:  # noqa: BLE001
        logger.warning("Отправка аудио не удалась: %s", e)


def _split_for_tts(text: str, max_len: int = 220) -> list:
    """Разбить ответ на короткие фрагменты по границам предложений."""
    import re

    text = " ".join((text or "").split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    chunks, buf = [], ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_len:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                chunks.append(buf)
            # слишком длинное «предложение» режем по запятым/пробелам
            while len(p) > max_len:
                cut = p.rfind(" ", 0, max_len)
                cut = cut if cut > 0 else max_len
                chunks.append(p[:cut].strip())
                p = p[cut:].strip()
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


async def _post_turn(session, user_msg, response_text, facts):
    try:
        critique = await critic(response_text, facts)
        logger.info("CRITIC: %s", (critique or "")[:200])
        if critique.strip().upper().startswith("OK"):
            emotion.update_from_event("критик_одобрил")
        elif critique.strip():
            emotion.update_from_event("критик_нашёл_ошибку")

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
        _add_xp(1)
        emotion.save()
    except Exception as e:  # noqa: BLE001
        logger.warning("post_turn failed: %s", e)


# ── Команды игрового режима / стрима (Phase 6) ──────────────
async def _handle_command(ws: WebSocket, payload: dict):
    ctype = payload.get("type")
    action = payload.get("action")

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
        if action == "emergency_stop":
            logger.warning("EMERGENCY STOP")
            await ws.send_text(json.dumps({"type": "game", "status": "stopped"}))
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
