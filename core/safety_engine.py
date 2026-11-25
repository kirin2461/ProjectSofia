"""Phase-4-01: Движок безопасности на CLIPS с обязательным regex-fallback.

Текст разбивается на токены и уходит в продукционную систему CLIPS
(core/clips_rules/safety.clp) — правила видят последовательность слов, а не
подстроку, поэтому «карта мира» больше не путается с банковской картой.

Regex-набор ниже — резерв на случай, если clipspy нет или правила не
загрузились; если CLIPS работает, берётся более строгий из двух вердиктов.
Оба набора проверяются на одном корпусе (tests/unit/test_safety_clips.py)
и должны давать одинаковый вердикт.

Режимы:
- strict:      warn трактуется как block
- standard:    как описано (block/warn/allow)
- permissive:  только hard block
"""

import hashlib
import json
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import SAFETY_MODE, SAFETY_LOG_PATH, CLIPS_RULES_DIR

logger = logging.getLogger(__name__)

ALLOW = "allow"
WARN = "warn"
BLOCK = "block"

_SEVERITY = {ALLOW: 0, WARN: 1, BLOCK: 2}


@dataclass
class SafetyResult:
    action: str
    reason: str = ""
    confidence: float = 0.0
    rule: str = ""


@dataclass
class Token:
    """Слово или числовой кластер для фактов CLIPS."""

    pos: int
    kind: str  # word | number
    text: str
    dot: str = "no"
    digits: int = 0


# Числовой кластер: «123», «4111 1111 1111 1111», «+7 999 123 45 67».
# Разделители — только пробел, дефис и скобки (не больше двух подряд).
_CLUSTER = re.compile(r"\d(?:[ \t()\-]{0,2}\d)*")
# Слово: буквы/цифры, внутри допускаются дефис и апостроф («пин-код»).
_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)*")


def tokenize(text: str) -> List[Token]:
    """Разбить сообщение на токены в порядке появления.

    Числовые кластеры собираются первыми, чтобы «4111 1111 1111 1111»
    осталось одним токеном: по нему считается число цифр.
    """
    found: List[Tuple[int, Token]] = []
    masked = list(text)
    for m in _CLUSTER.finditer(text):
        raw = re.sub(r"\s+", " ", m.group(0)).strip()
        digits = sum(ch.isdigit() for ch in raw)
        dot = "yes" if m.end() < len(text) and text[m.end()] == "." else "no"
        found.append((m.start(), Token(0, "number", raw, dot, digits)))
        for i in range(m.start(), m.end()):
            masked[i] = " "

    for m in _WORD.finditer("".join(masked)):
        word = m.group(0).lower().replace("ё", "е")
        dot = "yes" if m.end() < len(text) and text[m.end()] == "." else "no"
        found.append((m.start(), Token(0, "word", word, dot)))

    found.sort(key=lambda item: item[0])
    tokens = [token for _, token in found]
    for i, token in enumerate(tokens):
        token.pos = i
    return tokens


def _resolve_rules_dir(rules_dir: str) -> Path:
    """Папка правил: настройка, рядом с модулем или внутри сборки PyInstaller."""
    candidates = [Path(rules_dir)]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / rules_dir)
    candidates.append(Path(__file__).resolve().parent / "clips_rules")
    for path in candidates:
        try:
            if path.is_dir():
                return path
        except OSError:
            continue
    return Path(rules_dir)


@dataclass
class _Pattern:
    name: str
    action: str
    reason: str
    confidence: float
    regex: re.Pattern


# ── Regex-правила (fallback и база) ─────────────────────────
_RULES: List[_Pattern] = [
    # BLOCK: финансовые данные
    _Pattern(
        "financial_cvv", BLOCK,
        "Я не могу обрабатывать платёжные данные (CVV/CVC).",
        0.95,
        re.compile(r"\b(?:cvv|cvc|код\s+безопасности)\b\s*[:\-]?\s*\d{3,4}", re.I),
    ),
    _Pattern(
        "financial_card", BLOCK,
        "Я не могу обрабатывать номера банковских карт.",
        0.9,
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    ),
    _Pattern(
        "financial_secret", BLOCK,
        "Я не могу обрабатывать пароли и платёжные данные.",
        0.8,
        re.compile(r"\b(?:мой|моя|мои)\s+(?:пароль|пин-?код|карт[аы]?)\b", re.I),
    ),
    # BLOCK: угрозы насилия (нужен адресат: «я тебя убью», «убью тебя»)
    _Pattern(
        "violence_threat", BLOCK,
        "Я не могу помогать с угрозами насилия.",
        0.85,
        re.compile(
            r"(?:(?:тебя|вас|его|её|ее|их)\s+(?:убью|убей|зарежу|взорву|прикончу|изобью)|"
            r"(?:убью|убей|зарежу|взорву|прикончу|изобью)\s+(?:тебя|вас|его|её|ее|их|себя)|"
            r"(?<!\w)насилие(?!\w)|kill\s+you|i\s+will\s+kill)",
            re.I,
        ),
    ),
    # BLOCK: запретные темы (изготовление оружия/взрывчатки, наркотики)
    _Pattern(
        "forbidden_topic", BLOCK,
        "Я не могу обсуждать изготовление оружия, взрывчатки или наркотиков.",
        0.8,
        re.compile(
            r"(?:как\s+(?:сделать|собрать|изготовить)\s+(?:бомбу|взрывчат|оружие|наркотик)|"
            r"рецепт\s+(?:метамфетамина|взрывчат)|"
            r"how\s+to\s+make\s+(?:a\s+bomb|explosive))",
            re.I,
        ),
    ),
    # WARN: персональные данные
    _Pattern(
        "personal_address", WARN,
        "Похоже на персональные данные (адрес). Проверьте согласие на обработку.",
        0.7,
        re.compile(
            r"\b(?:ул\.|улица|проспект|переулок|д\.\s*\d+|кв\.\s*\d+|"
            r"адрес\s+(?:проживания|регистрации))\b",
            re.I,
        ),
    ),
    _Pattern(
        "personal_phone", WARN,
        "Похоже на номер телефона — персональные данные.",
        0.7,
        re.compile(r"\+?\b\d[\d\s\-()]{8,}\d\b"),
    ),
    _Pattern(
        "personal_id", WARN,
        "Похоже на документ, удостоверяющий личность (персональные данные).",
        0.75,
        re.compile(r"\b(?:паспорт|снилс|инн)\b\s*(?:серия|номер|[:№])?\s*\d", re.I),
    ),
    # WARN: медицинские и юридические советы
    _Pattern(
        "medical_advice", WARN,
        "Это медицинская тема. Я не врач — нужна консультация специалиста.",
        0.6,
        re.compile(
            r"(?<!\w)(?:дозировк\w*|доз[аы]\s+лекарств\w*|пропиши(?:те)?|"
            r"диагноз\w*|лечить\s+меня|какие\s+таблетки)(?!\w)",
            re.I,
        ),
    ),
    _Pattern(
        "legal_advice", WARN,
        "Это юридический вопрос. Я не юрист — нужна консультация специалиста.",
        0.6,
        re.compile(
            r"(?<!\w)(?:юридическ\w*|консультаци[яю]\s+юрист\w*|"
            r"подать\s+в\s+суд|как\s+обойти\s+закон)(?!\w)",
            re.I,
        ),
    ),
]


class ClipsBackend:
    """Продукционная система CLIPS (clipspy 1.x). Без clipspy — отключена."""

    MAX_TOKENS = 400  # длинные ответы режем: правила локальные, хвоста хватает

    def __init__(self, rules_dir: str):
        self.env = None
        self.available = False
        self.rule_count = 0
        self._lock = threading.Lock()  # Environment не потокобезопасен
        try:
            import clips
        except ImportError:
            logger.warning("clipspy не установлен — используется regex-fallback")
            return

        rules_path = _resolve_rules_dir(rules_dir)
        files = sorted(rules_path.glob("*.clp")) if rules_path.is_dir() else []
        if not files:
            logger.warning("Правила CLIPS не найдены: %s", rules_path)
            return

        try:
            env = clips.Environment()
            for clp in files:
                env.load(str(clp))
            env.reset()
            self._token_template = env.find_template("token")
            self.env = env
            self.rule_count = len(list(env.rules()))
            self.available = self.rule_count > 0
            logger.info("CLIPS загружен: %d правил из %d файлов",
                        self.rule_count, len(files))
        except Exception as e:  # noqa: BLE001
            logger.error("CLIPS не завёлся (%s) — regex-fallback", e)

    def check(self, text: str) -> Optional[SafetyResult]:
        """Вердикт CLIPS; None — правила молчат (или CLIPS недоступен)."""
        if not self.available or self.env is None:
            return None
        tokens = tokenize(text)
        if len(tokens) > self.MAX_TOKENS:
            tokens = tokens[: self.MAX_TOKENS]
            for i, t in enumerate(tokens):
                t.pos = i
        try:
            with self._lock:
                self.env.reset()
                for t in tokens:
                    self._token_template.assert_fact(
                        pos=t.pos, kind=_symbol(t.kind), text=t.text,
                        dot=_symbol(t.dot), digits=t.digits,
                    )
                self.env.run()
                verdicts = [
                    (str(f["action"]), str(f["rule"]), str(f["reason"]),
                     float(f["confidence"]))
                    for f in self.env.facts()
                    if f.template.name == "verdict"
                ]
                self.env.reset()
        except Exception as e:  # noqa: BLE001
            logger.warning("CLIPS check failed (%s) — regex", e)
            return None

        if not verdicts:
            return None
        action, rule, reason, confidence = max(
            verdicts, key=lambda v: (_SEVERITY.get(v[0], 0), v[3])
        )
        return SafetyResult(action, reason, confidence, "clips:" + rule)


def _symbol(value: str):
    import clips

    return clips.Symbol(value)


def _stricter(a: SafetyResult, b: SafetyResult) -> SafetyResult:
    """Строгий из двух вердиктов; при равенстве — первый (CLIPS)."""
    if _SEVERITY.get(a.action, 0) >= _SEVERITY.get(b.action, 0):
        return a
    return b


class SafetyEngine:
    """Предфильтр входа и постфильтр выхода."""

    def __init__(
        self,
        rules_dir: str = CLIPS_RULES_DIR,
        mode: Optional[str] = None,
        log_path: str = SAFETY_LOG_PATH,
    ):
        self.mode = (mode or SAFETY_MODE).lower()
        self.log_path = log_path
        self.patterns = _RULES
        self.clips = ClipsBackend(rules_dir)

    # ── Публичный API ───────────────────────────────────────
    def check_input(self, text: str) -> SafetyResult:
        return self._check(text, direction="input")

    def check_output(self, text: str) -> SafetyResult:
        return self._check(text, direction="output")

    def _check(self, text: str, direction: str) -> SafetyResult:
        if not text or not text.strip():
            return SafetyResult(ALLOW)

        result = self._regex_check(text)
        clips_result = self.clips.check(text)
        if clips_result is not None:
            # Более строгий вердикт; при равенстве — вердикт продукционной системы
            result = _stricter(clips_result, result)

        result = self._apply_mode(result)
        if result.action != ALLOW:
            self._log(direction, result, text)
        return result

    # ── Внутреннее ──────────────────────────────────────────
    def _regex_check(self, text: str) -> SafetyResult:
        # Порядок важен: block проверяем раньше warn
        matches = [p for p in self.patterns if p.regex.search(text)]
        if not matches:
            return SafetyResult(ALLOW)

        blocks = [m for m in matches if m.action == BLOCK]
        warns = [m for m in matches if m.action == WARN]
        chosen = max(blocks or warns, key=lambda m: m.confidence)
        return SafetyResult(chosen.action, chosen.reason, chosen.confidence, chosen.name)

    def _apply_mode(self, result: SafetyResult) -> SafetyResult:
        if self.mode == "permissive" and result.action == WARN:
            return SafetyResult(ALLOW, result.reason, result.confidence, result.rule)
        if self.mode == "strict" and result.action == WARN:
            return SafetyResult(
                BLOCK, result.reason, result.confidence, result.rule + ":strict"
            )
        return result

    def _log(self, direction: str, result: SafetyResult, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        entry = {
            "timestamp": time.time(),
            "direction": direction,
            "mode": self.mode,
            "action": result.action,
            "reason": result.reason,
            "rule": result.rule,
            "text_hash": digest,
        }
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("Не удалось записать safety-лог: %s", e)
