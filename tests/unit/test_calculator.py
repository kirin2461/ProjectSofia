"""Калькулятор на SymPy: точный счёт вместо угадывания и запрет на исполнение кода."""

import math

import pytest

from modules import calculator as calc


def test_arithmetic_and_precedence():
    assert calc.solve("2+2*3").value == "8"
    assert calc.solve("(2+2)*3").value == "12"


def test_power_symbols():
    assert calc.solve("2^10").value == "1024"
    assert calc.solve("10**3").value == "1000"


def test_comma_as_decimal_separator():
    assert calc.solve("1,5*2").value == "3"


def test_functions():
    assert calc.solve("sqrt(16)").value == "4"
    assert calc.solve("sin(pi/2)").value == "1"
    assert calc.solve("lg(1000)").value == "3"


def test_degrees_are_converted():
    assert calc.solve("sin(30°)").value.startswith("0.5")
    assert calc.solve("cos(60°)").value.startswith("0.5")


def test_integer_result_has_no_trailing_zeros():
    assert calc.solve("7/7").value == "1"


def test_empty_expression_is_error():
    result = calc.solve("")
    assert not result.ok and "пустое" in result.error


def test_too_long_expression_rejected():
    result = calc.solve("1+" * 200 + "1")
    assert not result.ok and "длинное" in result.error


def test_division_by_zero_is_error_not_exception():
    result = calc.solve("1/0")
    assert not result.ok
    assert result.value == ""


def test_unknown_value_is_error():
    result = calc.solve("x+2")
    assert not result.ok and "x" in result.error


@pytest.mark.parametrize("evil", [
    "__import__('os').system('dir')",
    "(). __class__",
    "open('/etc/passwd')",
    "eval('2+2')",
    "exec('x=1')",
    "lambda: 1",
])
def test_code_injection_is_rejected(evil):
    result = calc.solve(evil)
    assert not result.ok
    assert result.error


def test_no_filesystem_side_effect(monkeypatch, tmp_path):
    marker = tmp_path / "pwned"
    result = calc.solve(f"open(r'{marker}', 'w')")
    assert not result.ok
    assert not marker.exists()


def test_hint_contains_result():
    result = calc.solve("2+2")
    assert result.as_hint() == "Проверенный расчёт: 2+2 = 4"


def test_failed_result_has_no_hint():
    assert calc.solve("x+1").as_hint() == ""


# ── когда вообще считать ────────────────────────────────────
@pytest.mark.parametrize("text", [
    "сколько будет 2+2*3?",
    "посчитай 15*4",
    "чему равно sin(pi/4)",
    "sqrt(2) это сколько",
    "угол 45° в радианах",
])
def test_needs_math_true(text):
    assert calc.needs_math(text)


@pytest.mark.parametrize("text", [
    "привет, как дела?",
    "мне 30 лет",
    "расскажи про энтропию",
    "в 2024 году вышла книга",
])
def test_needs_math_false(text):
    assert not calc.needs_math(text)


def test_extract_expression_from_sentence():
    assert calc.extract_expression("сколько будет 2+2*3?") == "2+2*3"


def test_extract_expression_ignores_plain_number():
    assert calc.extract_expression("мне 30 лет") is None


def test_hint_for_sentence():
    hint = calc.hint_for("сколько будет 12/4 + 1")
    assert hint.startswith("Проверенный расчёт:") and hint.endswith("4")


def test_hint_empty_for_chat():
    assert calc.hint_for("как дела?") == ""


def test_result_is_json_friendly():
    d = calc.solve("2+2").to_dict()
    assert d["ok"] is True and d["value"] == "4" and isinstance(d["expr"], str)


def test_value_rounded_to_six_significant_digits():
    """Контракт формата: не-целые — 6 значащих цифр, без хвостовых нулей."""
    assert calc.solve("2**0.5").value == "1.41421"
    assert calc.solve("pi").value == "3.14159"
    assert calc.solve("1/3").value == "0.333333"


def test_numeric_value_matches_python_for_simple_cases():
    assert float(calc.solve("3*7").value) == 21.0
    assert float(calc.solve("2**0.5").value) == pytest.approx(math.sqrt(2), rel=1e-5)
