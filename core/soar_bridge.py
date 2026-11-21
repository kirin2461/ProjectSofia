"""Phase-5-01: SOAR bridge (исследовательская).

Если установлен PySoar/SML — используется он. Иначе — лёгкая Python-симуляция
операторов и импассов (subgoaling), чтобы остальной код не падал.

Операторы: op_greet, op_answer_question, op_ask_clarification, op_learn.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SoarAction:
    operator: str
    payload: dict = field(default_factory=dict)


class SimulatedSoar:
    """Минимальная симуляция SOAR: предложение операторов + импассы."""

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
        vague = ("это", "그", "то самое", "как обычно", "сам знаешь")
        return any(v in low for v in vague)

    def step(self, context: str) -> SoarAction:
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
                {"question": "Уточни, пожалуйста, что именно ты имеешь в виду?"},
            )
        if "?" in context:
            return SoarAction("op_answer_question")
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


class SOARBridge:
    """Единый интерфейс. Пытается PySoar, иначе симуляция."""

    def __init__(self):
        self._impl = None
        self._native = False
        try:
            import Python_sml_ClientInterface  # noqa: F401
            # Полноценная инициализация Soar SML здесь (вне scope прототипа)
            logger.info("Soar SML доступен, но используется симуляция прототипа")
        except ImportError:
            logger.info("Soar SML не найден — симуляция")
        self._impl = SimulatedSoar()

    def propose_goal(self, context: str) -> str:
        return self._impl.propose_goal(context)

    def step(self, context: str) -> SoarAction:
        return self._impl.step(context)

    def feedback(self, context: str, action: SoarAction, success: bool):
        self._impl.feedback(context, action, success)

    def needs_clarification(self, context: str) -> Optional[str]:
        """Если монолог неуверен — SOAR может предложить уточнение."""
        action = self._impl.step(context)
        if action.operator == "op_ask_clarification":
            return action.payload.get("question")
        return None

    @property
    def learned_skills(self) -> List[str]:
        return self._impl.learned_skills
