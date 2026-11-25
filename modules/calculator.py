"""Безопасный калькулятор: 4B-модель врёт в арифметике, SymPy — нет.

Зачем отдельный слой: «сколько будет 2+2*3» и «sin(30°)» — это задачи, где
языковая модель угадывает, а вычисление даёт точный ответ за микросекунды.
Число попадает в промпт как проверенный факт, и модель его не пересчитывает.

Безопасность: выражение принимается только из белого списка символов и
функций; никакого eval, никаких имён с подчёркиванием.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from sympy import Abs, E, acos, asin, atan, cos, exp, log, pi, sin, sqrt, tan
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.core.numbers import Integer, Rational
from sympy.core.expr import Expr

logger = logging.getLogger(__name__)

MAX_LEN = 200

# Идентификаторы, которые разрешено писать в выражении
_LOCAL = {
    "pi": pi,
    "E": E,
    "e": E,
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "exp": exp,
    "log": log,
    "ln": log,
    "lg": lambda x: log(x, 10),
    "log10": lambda x: log(x, 10),
    "abs": Abs,
    "Abs": Abs,
    "deg": lambda x: x * pi / 180,
}

_IDENT_RE = re.compile(r"[A-Za-z_]+")
# Начало кандидата: цифра, скобка или известная функция
_CANDIDATE_RE = re.compile(
    r"(?:\d|[(]|(?:sqrt|sin|cos|tan|asin|acos|atan|log|ln|lg|exp|abs|deg)\s*\()"
    r"[0-9A-Za-z_+\-*/^().,\s°×÷]*"
)
_QUESTION_WORDS = re.compile(
    r"^\s*(?:а\s+)?(?:сколько\s+(?:будет|это)?|чему\s+(?:равно|равняется)|"
    r"посчитай(?:те)?|вычисли(?:те)?|найди(?:те)?)\s*",
    re.IGNORECASE,
)


@dataclass
class CalcResult:
    """Результат вычисления: либо значение, либо понятная причина отказа."""

    expr: str
    ok: bool
    value: str = ""
    error: str = ""

    def as_hint(self) -> str:
        """Строка для промпта: модель получает готовый ответ, а не задачу."""
        if not self.ok:
            return ""
        return f"Проверенный расчёт: {self.expr} = {self.value}"

    def to_dict(self) -> dict:
        return {"expr": self.expr, "ok": self.ok,
                "value": self.value, "error": self.error}


def _prepare(expr: str) -> str:
    """Приводит «человеческую» запись к символьной."""
    e = (expr or "").strip()
    e = e.replace("×", "*").replace("÷", "/")
    # Десятичная запятая: «1,5» → «1.5». Запятая между цифрами не может быть
    # разделителем аргументов, поэтому правило безопасно.
    e = re.sub(r"(?<=\d),(?=\d)", ".", e)
    e = e.replace("^", "**")
    # градусы: 30° → 30*pi/180
    e = re.sub(r"(\d+(?:\.\d+)?)\s*°", r"(\1*pi/180)", e)
    return e.strip()


def _bad_identifiers(expr: str) -> List[str]:
    """Имена, которых нет в белом списке (включая любые с «_»)."""
    return [name for name in _IDENT_RE.findall(expr) if name not in _LOCAL]


def _format(value) -> str:
    """Число → строка без хвостовых нулей; дроби — до 6 значащих цифр."""
    if isinstance(value, Integer):
        return str(int(value))
    if isinstance(value, Rational):
        return f"{float(value):.6g}"
    return f"{float(value):.6g}"


def solve(expr: str) -> CalcResult:
    """Вычислить выражение. Ошибки — в поле error, исключений наружу нет."""
    original = (expr or "").strip()
    if not original:
        return CalcResult(original, False, error="пустое выражение")
    if len(original) > MAX_LEN:
        return CalcResult(original, False, error="слишком длинное выражение")

    prepared = _prepare(original)
    bad = _bad_identifiers(prepared)
    if bad:
        return CalcResult(original, False,
                          error=f"недопустимые имена: {', '.join(sorted(set(bad)))}")

    try:
        # global_dict не ограничиваем: стандартные трансформации SymPy строят
        # Integer/Mul/Pow и без них не работают. От исполнения кода защищает
        # белый список имён выше: ни одно имя вне _LOCAL не доходит до разбора.
        parsed: Expr = parse_expr(
            prepared,
            local_dict=_LOCAL,
            transformations=standard_transformations,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("калькулятор: разбор не удался (%s): %s", original, e)
        return CalcResult(original, False, error="не удалось разобрать выражение")

    if getattr(parsed, "free_symbols", None):
        names = ", ".join(sorted(str(s) for s in parsed.free_symbols))
        return CalcResult(original, False, error=f"неизвестные величины: {names}")

    try:
        numeric = parsed.evalf()
    except Exception as e:  # noqa: BLE001
        return CalcResult(original, False, error=f"вычисление не удалось: {e}")

    if not getattr(numeric, "is_number", False):
        return CalcResult(original, False, error="результат не является числом")
    if numeric.is_finite is False:
        return CalcResult(original, False,
                          error="деление на ноль или бесконечный результат")
    if numeric.is_finite is None:
        return CalcResult(original, False, error="неопределённый результат")

    # Точная дробь красивее для простых случаев: 1/3 → 1/3, но 0.333333 для 2/6?
    try:
        value = _format(parsed if parsed.is_number else numeric)
    except (TypeError, ValueError):
        value = _format(numeric)
    if value in ("zoo", "nan", "oo", "-oo"):
        return CalcResult(original, False, error="результат не является числом")
    return CalcResult(original, True, value=value)


def needs_math(text: str) -> bool:
    """Есть ли в сообщении арифметика, которую стоит посчитать заранее."""
    t = text or ""
    if not re.search(r"\d", t):
        return False
    if re.search(r"\d\s*[+\-*/^×÷]\s*[-(\d]", t):
        return True
    if re.search(r"(?:sqrt|sin|cos|tan|log|ln|exp|abs|deg)\s*\(", t):
        return True
    if re.search(r"\d\s*°", t):
        return True
    return bool(_QUESTION_WORDS.search(t))


def _candidates(text: str) -> List[str]:
    """Куски текста, похожие на выражение: от длинных к коротким."""
    found = []
    for m in _CANDIDATE_RE.finditer(text or ""):
        chunk = m.group(0)
        if len(chunk) >= 3:
            found.append(chunk.strip())
    found.sort(key=len, reverse=True)
    return found


def extract_expression(text: str) -> Optional[str]:
    """Достаёт из фразы первое выражение, которое реально вычисляется.

    «сколько будет 2+2*3?» → «2+2*3»; «мне 30 лет» → None.
    Хвост обрезается по одному символу: регэксп не знает, где кончается
    арифметика и начинается обычный текст.
    """
    stripped = _QUESTION_WORDS.sub("", (text or "").strip())
    for chunk in _candidates(stripped):
        variant = chunk
        for _ in range(6):
            variant = variant.rstrip(" .,;:")
            if not variant:
                break
            if not re.search(r"\d", variant):
                break
            if not re.search(r"[+\-*/^×÷]|sqrt\(|sin\(|cos\(|tan\(|log\(|ln\(", variant):
                break
            result = solve(variant)
            if result.ok:
                return variant
            # отрезаем последний «мусорный» символ и пробуем снова
            variant = variant[:-1]
    return None


def hint_for(text: str) -> str:
    """Строка-подсказка для промпта или пустая, если считать нечего."""
    if not needs_math(text):
        return ""
    expr = extract_expression(text)
    if not expr:
        return ""
    result = solve(expr)
    return result.as_hint()


def safe_hint(text: str) -> str:
    """hint_for, который не может сорвать ответ.

    Считать — вспомогательный шаг: если sympy подавился странной записью,
    ответ всё равно должен уйти пользователю, просто без подсказки.
    """
    if not re.search(r"\d", text or ""):
        return ""
    try:
        return hint_for(text)
    except Exception:  # noqa: BLE001
        return ""
