"""Phase-5-01: мост к Soar — настоящая продукционная система Soar 9.6 (SML).

Если пакет soar-sml установлен, работает полноценный агент: правила из
core/soar_rules/sofia.soar решают, как отвечать на сообщение (приветствие,
ответ на вопрос, обычная реплика, уточнение). На неоднозначном сообщении
агент уходит в tie-импасс, правило разрешает его в пользу уточнения, а
встроенный чанкинг компилирует решение в новое правило — это и есть
обучение. Выученные чанки сохраняются в data/soar_chunks.soar и
подгружаются при следующем запуске.

Без soar-sml остаётся SimulatedSoar — лёгкая имитация того же контракта,
чтобы остальной код не падал.
"""

import logging
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import SOAR_CHUNKS_PATH, SOAR_RULES_DIR

logger = logging.getLogger(__name__)


@dataclass
class SoarAction:
    operator: str
    payload: dict = field(default_factory=dict)


# «расскажи про космос» без знака вопроса — всё равно запрос, а не реплика
_ASK_WORDS = frozenset(
    """расскажи расскажите объясни объясните покажи покажите скажи скажите
    найди найдите переведи переведите нарисуй напиши дай дайте помоги подскажи
    почему зачем как что где когда кто сколько какой какая какие чем""".split()
)
_GREETING = re.compile(
    r"^\s*(?:привет\w*|здравствуй\w*|хай|хей|ку|hello|hi|"
    r"добрый\s+(?:день|вечер)|доброе\s+утро|доброго\s+(?:дня|вечера))\b",
    re.I,
)
_VAGUE = ("это", "то самое", "как обычно", "сам знаешь", "как там", "ну")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

CLARIFY_QUESTION = "Уточни, пожалуйста, что именно ты имеешь в виду?"


def classify(text: str, has_context: bool = False) -> Dict[str, bool]:
    """Признаки сообщения — их видят правила Soar (и симуляция)."""
    low = (text or "").lower().strip()
    words = _WORD.findall(low)
    greeting = bool(_GREETING.match(low))
    question = "?" in low or (bool(words) and words[0] in _ASK_WORDS)
    vague = (len(words) <= 2 and "?" in low) or (
        len(words) <= 5 and any(marker in low for marker in _VAGUE)
    )
    return {
        "greeting": greeting,
        "question": question,
        "vague": vague,
        "context": bool(has_context),
    }


def _resolve_rules_file(rules_dir: str) -> Path:
    """Файл правил: настройка, сборка PyInstaller или папка рядом с модулем."""
    candidates = [Path(rules_dir) / "sofia.soar"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / rules_dir / "sofia.soar")
    candidates.append(Path(__file__).resolve().parent / "soar_rules" / "sofia.soar")
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return Path(rules_dir) / "sofia.soar"


class SimulatedSoar:
    """Резерв без SML: те же операторы, обучение — по счётчику успехов."""

    def __init__(self):
        self._goal_stack: List[str] = []
        self._chunks: Dict[str, SoarAction] = {}  # выученные правила
        self._success_counts: Dict[str, int] = {}

    def propose_goal(self, context: str) -> str:
        goal_id = f"goal_{len(self._goal_stack)}"
        self._goal_stack.append(context)
        return goal_id

    def _is_ambiguous(self, context: str) -> bool:
        low = context.lower()
        # Короткий или вопрос без конкретики → неоднозначно
        if len(low.split()) <= 2 and "?" in low:
            return True
        vague = ("это", "то самое", "как обычно", "сам знаешь")
        return any(v in low for v in vague)

    def step(self, context: str, has_context: bool = False) -> SoarAction:
        """Один такт: выбрать оператор (или уйти в импасс → уточнение)."""
        # Выученный чанк?
        key = self._chunk_key(context)
        if key in self._chunks:
            return self._chunks[key]

        low = context.lower()
        if re.search(r"\b(привет|здравствуй|хай|добрый)\b", low):
            return SoarAction("op_greet")
        if self._is_ambiguous(context):
            # Импасс → подцель "уточнить"
            self.propose_goal(f"clarify:{context}")
            return SoarAction(
                "op_ask_clarification",
                {"question": CLARIFY_QUESTION},
            )
        return SoarAction("op_answer_question")

    def feedback(self, context: str, action: SoarAction, success: bool):
        """Результат действия; при успехах — чанкинг (компиляция правила)."""
        if not success:
            return
        key = self._chunk_key(context)
        self._success_counts[key] = self._success_counts.get(key, 0) + 1
        # После 3 успехов одного типа — выучить правило
        if self._success_counts[key] >= 3 and key not in self._chunks:
            self._chunks[key] = action
            logger.info("SOAR chunk выучен: %s → %s", key, action.operator)

    @staticmethod
    def _chunk_key(context: str) -> str:
        return re.sub(r"\s+", " ", context.lower().strip())[:64]

    @property
    def learned_skills(self) -> List[str]:
        return list(self._chunks.keys())

    def status(self) -> dict:
        return {
            "engine": "simulated",
            "active": False,
            "rules": 0,
            "chunks": len(self._chunks),
            "learned": list(self._chunks.keys()),
            "last": "",
        }

    def close(self):
        pass


class NativeSoar:
    """Настоящий Soar: агент, правила, импасс, чанкинг."""

    OP_MAP = {
        "greet": "op_greet",
        "answer-question": "op_answer_question",
        "respond": "op_answer_question",
        "ask-clarification": "op_ask_clarification",
    }
    # Output-link растёт на одну команду за сообщение; раз в N реплик
    # перезапускаем агента, сохранив выученные чанки.
    RESTART_AFTER = 250

    def __init__(self, rules_dir: str = None, chunks_path: str = None):
        import soar_sml

        self._sml = soar_sml
        self.rules_path = _resolve_rules_file(rules_dir or SOAR_RULES_DIR)
        if not self.rules_path.is_file():
            raise FileNotFoundError(f"правила Soar не найдены: {self.rules_path}")
        self.chunks_path = Path(chunks_path or SOAR_CHUNKS_PATH)
        self._lock = threading.RLock()
        self._kernel = soar_sml.Kernel.CreateKernelInCurrentThread(True, 0)
        if self._kernel.HadError():
            raise RuntimeError(f"Soar kernel: {self._kernel.GetLastErrorDescription()}")
        self._agent = None
        self._rules = 0
        self._chunks: List[str] = []
        self._steps = 0
        self._msg_id = 0
        self._last = ""
        self._broken = False
        self._start_agent()

    # ── агент ───────────────────────────────────────────────
    def _start_agent(self):
        self._agent = self._kernel.CreateAgent("sofia")
        if self._agent is None:
            raise RuntimeError("Soar не создал агента")
        self._agent.ExecuteCommandLine("echo off")
        self._agent.LoadProductions(str(self.rules_path))
        if not self._agent.GetLastCommandLineResult():
            raise RuntimeError(
                f"Soar не загрузил правила: {self._agent.GetLastErrorDescription()}"
            )
        base = set(self._production_names())
        if self.chunks_path.is_file():
            self._agent.LoadProductions(str(self.chunks_path))
            if not self._agent.GetLastCommandLineResult():
                logger.warning("Soar: чанки не загрузились: %s",
                               self._agent.GetLastErrorDescription())
        self._rules = len(base)
        self._chunks = self._chunk_names()
        self._steps = 0
        logger.info("Soar готов: %d правил, %d чанков", self._rules, len(self._chunks))

    def _production_names(self) -> List[str]:
        out = self._agent.ExecuteCommandLine("print --all")
        return [n for n in out.split() if not n.startswith("Error")]

    def _chunk_names(self) -> List[str]:
        """Soar называет выученные правила chunk*..."""
        return [n for n in self._production_names() if n.startswith("chunk")]

    def _refresh_chunks(self):
        try:
            names = self._chunk_names()
        except Exception:  # noqa: BLE001
            return
        if names != self._chunks:
            logger.info("Soar: выучено чанков — %d", len(names))
            self._chunks = names

    def _restart(self):
        self._save_chunks()
        try:
            self._kernel.DestroyAgent(self._agent)
        except Exception as e:  # noqa: BLE001
            logger.warning("Soar: агент не уничтожен (%s) — создаю нового", e)
        self._start_agent()

    # ── такт ────────────────────────────────────────────────
    def step(self, context: str, has_context: bool = False) -> SoarAction:
        features = classify(context, has_context)
        if self._broken:
            return SoarAction("op_answer_question")
        with self._lock:
            if self._steps >= self.RESTART_AFTER:
                try:
                    self._restart()
                except Exception:  # noqa: BLE001
                    logger.exception("Soar: плановый перезапуск не удался")
                    self._broken = True
                    return SoarAction("op_answer_question")
            try:
                command = self._run(features)
            except Exception:  # noqa: BLE001
                logger.exception("Soar: сбой такта — перезапуск агента")
                try:
                    self._restart()
                except Exception:  # noqa: BLE001
                    logger.error("Soar: агент не перезапустился — дальше без Soar")
                    self._broken = True
                    return SoarAction("op_answer_question")
                return SoarAction("op_answer_question")
            self._steps += 1
            self._last = command or ""
            self._refresh_chunks()
        operator = self.OP_MAP.get(command, "op_answer_question")
        payload = {}
        if operator == "op_ask_clarification":
            payload["question"] = CLARIFY_QUESTION
        return SoarAction(operator, payload)

    def _run(self, features: Dict[str, bool]) -> Optional[str]:
        agent = self._agent
        self._msg_id += 1
        msg_id = self._msg_id

        link = agent.GetInputLink()
        if link is None:  # kernel живёт в другом потоке или уже остановлен
            raise RuntimeError("Soar: input-link недоступен")
        # вчерашнее сообщение убираем: правила смотрят только на текущее
        for i in range(link.GetNumberChildren() - 1, -1, -1):
            child = link.GetChild(i)
            if child.GetAttribute() == "message":
                child.DestroyWME()

        message = agent.CreateIdWME(link, "message")
        agent.CreateIntWME(message, "id", msg_id)
        for name in ("greeting", "question", "vague", "context"):
            agent.CreateStringWME(message, name, "yes" if features[name] else "no")
        agent.Commit()
        agent.RunSelfTilOutput()

        for i in range(agent.GetNumberCommands()):
            command = agent.GetCommand(i)
            if command.GetParameterValue("msg-id") == str(msg_id):
                return command.GetParameterValue("name")
        logger.warning("Soar: решение без команды (features=%s)", features)
        return None

    # ── обучение ────────────────────────────────────────────
    def feedback(self, context: str, action: SoarAction, success: bool):
        """Обратная связь; чанкинг Soar делает сам — здесь только сохранение."""
        if not success:
            return
        with self._lock:
            self._save_chunks()

    def _save_chunks(self):
        with self._lock:
            try:
                chunks = self._chunk_names()
            except Exception:  # noqa: BLE001
                logger.debug("Soar: не удалось перечислить правила")
                return
            if not chunks:
                self._chunks = []
                return
            sources = []
            for name in chunks:
                text = self._agent.ExecuteCommandLine(f"print {name}").strip()
                if text and not text.startswith("No production"):
                    sources.append(text)
            if not sources:
                return
            self._chunks = chunks
            try:
                self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.chunks_path.with_suffix(".tmp")
                tmp.write_text("\n\n".join(sources) + "\n", encoding="utf-8")
                tmp.replace(self.chunks_path)
                logger.info("Soar: сохранено чанков — %d", len(sources))
            except OSError as e:
                logger.warning("Soar: чанки не сохранены (%s)", e)

    def propose_goal(self, context: str) -> str:
        """Цель ставит Soar; снаружи — только метка для логов."""
        return f"soar:{_WORD.findall(context.lower())[:3]}"

    @property
    def learned_skills(self) -> List[str]:
        return list(self._chunks)

    def status(self) -> dict:
        return {
            "engine": "soar",
            "active": not self._broken,
            "rules": self._rules,
            "chunks": len(self._chunks),
            "learned": list(self._chunks),
            "last": self._last,
        }

    def close(self):
        with self._lock:
            if self._broken:
                return
            try:
                self._save_chunks()
            finally:
                self._broken = True  # после остановки kernel такты не выполняются
                try:
                    self._kernel.Shutdown()
                except Exception:  # noqa: BLE001
                    logger.debug("Soar: kernel уже остановлен")


class SOARBridge:
    """Единый интерфейс. Настоящий Soar, если доступен, иначе симуляция."""

    def __init__(self, prefer_native: bool = True):
        self._impl = None
        self._native = False
        if prefer_native:
            try:
                self._impl = NativeSoar()
                self._native = True
                logger.info("SOAR: настоящий Soar (%d правил)", self._impl._rules)
            except Exception as e:  # noqa: BLE001
                logger.warning("SOAR: настоящий Soar недоступен (%s) — симуляция", e)
        if self._impl is None:
            self._impl = SimulatedSoar()

    @property
    def native(self) -> bool:
        return self._native

    @property
    def engine(self) -> str:
        return "soar" if self._native else "simulated"

    def propose_goal(self, context: str) -> str:
        return self._impl.propose_goal(context)

    def step(self, context: str, has_context: bool = False) -> SoarAction:
        return self._impl.step(context, has_context)

    def feedback(self, context: str, action: SoarAction, success: bool):
        self._impl.feedback(context, action, success)

    def needs_clarification(self, context: str, has_context: bool = False) -> Optional[str]:
        """Если Soar решил уточнить — вернуть текст вопроса."""
        action = self._impl.step(context, has_context)
        if action.operator == "op_ask_clarification":
            return action.payload.get("question")
        return None

    def status(self) -> dict:
        return self._impl.status()

    def close(self):
        self._impl.close()

    @property
    def learned_skills(self) -> List[str]:
        return self._impl.learned_skills
