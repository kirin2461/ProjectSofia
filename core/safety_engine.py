"""Phase-4-01: Движок безопасности на CLIPS с обязательным regex-fallback.

Текст разбивается на токены и уходит в продукционную систему CLIPS
(core/clips_rules/safety.clp) — правила видят последовательность слов, а не
подстроку, поэтому «карта мира» больше не путается с банковской картой.

Regex-набор ниже — резерв на случай, если clipspy нет или правила не
загрузились; если CLIPS работает, берётся более строгий из двух вердиктов.
Карты, телефоны и угрозы в резерве проверяются по тем же токенам, что видит
CLIPS (looks_like_card, looks_like_phone, is_violence_threat), чтобы вердикты
не расходились.
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
from typing import Callable, List, Optional, Tuple

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
    luhn: str = "no"  # yes — 13–19 цифр и сходится контрольная сумма Луна


def luhn_valid(digits: str) -> bool:
    """Проверка Луна (контрольная цифра банковских карт)."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


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
        only_digits = "".join(ch for ch in raw if ch.isdigit())
        digits = len(only_digits)
        luhn = "yes" if 13 <= digits <= 19 and luhn_valid(only_digits) else "no"
        dot = "yes" if m.end() < len(text) and text[m.end()] == "." else "no"
        found.append((m.start(), Token(0, "number", raw, dot, digits, luhn)))
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
    """Правило fallback: регулярка по тексту или проверка по токенам."""

    name: str
    action: str
    reason: str
    confidence: float
    regex: Optional[re.Pattern] = None
    tokens: Optional[Callable[[List[Token]], bool]] = None


# ── Словари токенных правил (те же списки — в core/clips_rules/safety.clp) ──
# Номер карты: 13–19 цифр и (сумма Луна или слово про карту в ±5 токенах)
CARD_CONTEXT_WINDOW = 5
_CARD_WORDS = frozenset({
    "карта", "карты", "карте", "карту", "картой",
    "карточка", "карточки", "карточке", "карточку", "карточкой",
    "card", "cvv", "cvc", "cvv2", "cvc2", "visa", "mastercard", "maestro",
    "expiry", "expires",
})

# «убью …» — угроза, если следом не «время», «таракана» и т.п.
_THREAT_VERBS = frozenset({"убью", "убей", "зарежу", "взорву", "прикончу", "изобью"})
_THREAT_EXCEPTIONS = frozenset({
    "время", "времени", "вечер", "час", "таракана", "комара", "муху",
    "процесс", "задачу",
})
# Угрозы и призывы, которые блокируются только при адресате рядом (±2 токена)
_TARGETED_VERBS = frozenset({
    "застрелю", "пристрелю", "задушу", "покалечу", "искалечу",
    "убейте", "зарежьте", "избейте", "прикончите", "перебейте", "застрелите",
})
# Инструкция/намерение: «как убить человека», «хочу избить соседа»
_INTENT_WORDS = frozenset({
    "как", "хочу", "хочется", "собираюсь", "планирую", "помоги", "помогите",
    "способ", "способы", "how", "want", "wanna",
})
_HARM_VERBS = frozenset({
    "убить", "зарезать", "покалечить", "искалечить", "изувечить", "избить",
    "застрелить", "пристрелить", "задушить", "отравить", "прикончить",
    "kill", "murder", "stab", "shoot", "poison",
})
_PERSON_TARGETS = frozenset({
    "человека", "человек", "людей", "ребенка", "детей", "жену", "мужа",
    "соседа", "соседку", "соседей", "мать", "маму", "отца", "папу", "брата",
    "сестру", "друга", "подругу", "девушку", "парня", "женщину", "мужчину",
    "учителя", "начальника", "одноклассника", "одноклассницу", "коллегу",
    "полицейского", "кого-то", "кого-нибудь", "кого-либо",
    "тебя", "вас", "его", "ее", "их", "всех", "него", "нее", "них",
    "person", "someone", "somebody", "people", "human", "humans", "man",
    "woman", "child", "children", "kids", "wife", "husband", "neighbor",
    "neighbour", "you", "him", "her", "them",
})
_KILL_TARGETS_EN = frozenset({"you", "him", "her", "them"})


def _word_at(tokens: List[Token], pos: int) -> Optional[str]:
    if 0 <= pos < len(tokens) and tokens[pos].kind == "word":
        return tokens[pos].text
    return None


def looks_like_card(tokens: List[Token]) -> bool:
    """13–19 цифр и (контрольная сумма Луна или контекст карты рядом)."""
    for t in tokens:
        if t.kind != "number" or not 13 <= t.digits <= 19:
            continue
        if t.luhn == "yes":
            return True
        lo, hi = t.pos - CARD_CONTEXT_WINDOW, t.pos + CARD_CONTEXT_WINDOW
        for c in tokens[max(0, lo): hi + 1]:
            if c.kind != "word":
                continue
            if c.text in _CARD_WORDS:
                return True
            if c.text == "действия" and _word_at(tokens, c.pos - 1) == "срок":
                return True
    return False


def looks_like_phone(tokens: List[Token]) -> bool:
    """10–12 цифр в одном кластере — как (token (digits ...)) в personal-phone."""
    return any(t.kind == "number" and 10 <= t.digits <= 12 for t in tokens)


def _target_near(tokens: List[Token], pos: int, after: int = -1) -> bool:
    for o in range(pos - 2, pos + 3):
        if o != pos and o > after and _word_at(tokens, o) in _PERSON_TARGETS:
            return True
    return False


def is_violence_threat(tokens: List[Token]) -> bool:
    """Угроза или инструкция причинить вред человеку (не тема «насилие»)."""
    for t in tokens:
        if t.kind != "word":
            continue
        w = t.text
        if w in _THREAT_VERBS and _word_at(tokens, t.pos + 1) not in _THREAT_EXCEPTIONS:
            return True
        if w in _TARGETED_VERBS and _target_near(tokens, t.pos):
            return True
        if w == "kill" and _word_at(tokens, t.pos + 1) in _KILL_TARGETS_EN:
            return True
        if w == "kill" and (
            _word_at(tokens, t.pos - 1) == "i'll"
            or (_word_at(tokens, t.pos - 1) == "will" and _word_at(tokens, t.pos - 2) == "i")
        ):
            return True
        if w in _INTENT_WORDS:
            for v in range(t.pos + 1, t.pos + 4):
                if _word_at(tokens, v) in _HARM_VERBS and _target_near(tokens, v, after=t.pos):
                    return True
    return False


# ── Fallback-правила (regex и токены) ───────────────────────
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
        tokens=looks_like_card,
    ),
    _Pattern(
        "financial_secret", BLOCK,
        "Я не могу обрабатывать пароли и платёжные данные.",
        0.8,
        re.compile(r"\b(?:мой|моя|мои)\s+(?:пароль|пин-?код|карт[аы]?)\b", re.I),
    ),
    # BLOCK: угрозы и инструкции насилия; само слово «насилие» — тема, не угроза
    _Pattern(
        "violence_threat", BLOCK,
        "Я не могу помогать с угрозами насилия.",
        0.85,
        tokens=is_violence_threat,
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
        tokens=looks_like_phone,
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
                        dot=_symbol(t.dot), digits=t.digits, luhn=_symbol(t.luhn),
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
        tokens: Optional[List[Token]] = None
        matches = []
        for p in self.patterns:
            if p.regex is not None and p.regex.search(text):
                matches.append(p)
            elif p.tokens is not None:
                if tokens is None:
                    tokens = tokenize(text)
                if p.tokens(tokens):
                    matches.append(p)
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
