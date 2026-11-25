"""Общий JSON-парсер: то, что модель вставляет в ответ вокруг JSON, не ломает разбор."""

from core import jsonx


def test_plain_object():
    assert jsonx.first_json('{"a": 1}') == {"a": 1}


def test_fenced_object():
    assert jsonx.first_json('```json\n{"вопрос": "а"}\n```') == {"вопрос": "а"}


def test_fence_without_language():
    assert jsonx.first_json("```\n{\"a\": 2}\n```") == {"a": 2}


def test_text_around_object():
    raw = 'Готово.\n{"action": "click", "text": "Да"}\nВсё, готово.'
    assert jsonx.first_json(raw)["action"] == "click"


def test_nested_objects():
    raw = '{"action": "type", "params": {"x": {"y": 1}}}'
    assert jsonx.first_json(raw)["params"]["x"]["y"] == 1


def test_braces_inside_strings_do_not_break_parsing():
    raw = '{"text": "формула {a} и {b}", "ok": true}'
    assert jsonx.first_json(raw)["text"] == "формула {a} и {b}"


def test_escaped_quotes():
    raw = '{"text": "он сказал \\"да\\""}'
    assert jsonx.first_json(raw)["text"] == 'он сказал "да"'


def test_garbage_returns_none():
    assert jsonx.first_json("никакого json тут нет") is None
    assert jsonx.first_json("") is None
    assert jsonx.first_json("{сломанный") is None


def test_truncated_object_returns_none():
    assert jsonx.first_json('{"a": 1') is None


def test_second_object_used_when_first_broken():
    raw = '{"a": } мусор {"b": 2}'
    assert jsonx.first_json(raw) == {"b": 2}


def test_json_list_strings():
    assert jsonx.json_list(["энтропия", "  тепло  ", ""]) == ["энтропия", "тепло"]


def test_json_list_scalar_and_wrong_types():
    assert jsonx.json_list("одна строка") == ["одна строка"]
    assert jsonx.json_list(42) == []
    assert jsonx.json_list([1, ["x"], "окно"]) == ["окно"]


def test_json_list_dedupes_and_limits():
    assert jsonx.json_list(["а б", "а б", "в г", "д е"], limit=2) == ["а б", "в г"]
