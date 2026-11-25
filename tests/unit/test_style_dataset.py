"""Датасет для стилевой LoRA: разбор ответов модели, дедупликация, JSONL."""

import json

from scripts import build_style_dataset as bs


def test_extract_json_from_fence():
    assert bs._extract_json('```json\n{"вопрос":"а","ответ":"б"}\n```') == {
        "вопрос": "а", "ответ": "б"
    }


def test_extract_json_without_json_extension():
    assert bs._extract_json('```\n{"a":1}\n```') == {"a": 1}


def test_extract_json_tolerates_text_around():
    raw = 'Готово.\n{"вопрос":"Что такое энтропия?","ответ":"Мера беспорядка."}\nВсё.'
    assert bs._extract_json(raw)["вопрос"] == "Что такое энтропия?"


def test_extract_json_none_for_garbage():
    assert bs._extract_json("никакого json тут нет") is None
    assert bs._extract_json("") is None


def test_to_pair_accepts_english_keys():
    raw = json.dumps({"question": "Вопрос?", "answer": "Ответ " * 30}, ensure_ascii=False)
    pair = bs.to_pair(raw)
    assert pair and pair["question"] == "Вопрос?"


def test_to_pair_normalizes_spaces():
    raw = json.dumps({"вопрос": "Что\u00a0такое  энтропия?",
                      "ответ": "Энтропия —\u2009мера беспорядка. " * 8}, ensure_ascii=False)
    pair = bs.to_pair(raw)
    assert "\u00a0" not in pair["question"] and "\u2009" not in pair["answer"]
    assert "  " not in pair["question"]


def test_to_pair_rejects_short_answer():
    raw = json.dumps({"вопрос": "Да?", "ответ": "Да."}, ensure_ascii=False)
    assert bs.to_pair(raw) is None


def test_to_pair_rejects_long_answer_and_question():
    long_a = json.dumps({"вопрос": "В?", "ответ": "я" * (bs.MAX_ANSWER + 1)},
                        ensure_ascii=False)
    assert bs.to_pair(long_a) is None
    long_q = json.dumps({"вопрос": "почему " * 60, "ответ": "я" * 200},
                        ensure_ascii=False)
    assert bs.to_pair(long_q) is None


def test_to_pair_rejects_empty_fields():
    assert bs.to_pair(json.dumps({"вопрос": "", "ответ": "я" * 200},
                                 ensure_ascii=False)) is None
    assert bs.to_pair(json.dumps({"вопрос": "В?", "ответ": ""})) is None


def test_build_skips_failures_and_duplicates():
    good = json.dumps({"вопрос": "Что такое энтропия?",
                       "ответ": "Энтропия — мера беспорядка системы. " * 4},
                      ensure_ascii=False)

    def chat(prompt):
        if "второй" in prompt:
            return "не JSON"
        return good          # один и тот же ответ на разные фрагменты → дубль

    pairs = bs.build(chat, ["первый фрагмент", "второй фрагмент", "третий фрагмент"])
    assert len(pairs) == 1


def test_build_keeps_distinct_pairs():
    answers = iter([
        json.dumps({"вопрос": "Что такое энтропия?",
                    "ответ": "Энтропия — мера беспорядка. " * 4}, ensure_ascii=False),
        json.dumps({"вопрос": "Что такое импульс?",
                    "ответ": "Импульс — произведение массы на скорость. " * 3},
                   ensure_ascii=False),
    ])
    pairs = bs.build(lambda prompt: next(answers), ["a", "b"])
    assert [p["question"] for p in pairs] == ["Что такое энтропия?", "Что такое импульс?"]


def test_to_jsonl_is_jsonl():
    text = bs.to_jsonl([{"question": "В?", "answer": "О."},
                        {"question": "В2?", "answer": "О2."}])
    lines = text.splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["messages"][0] == {"role": "user", "content": "В?"}
    assert first["messages"][1] == {"role": "assistant", "content": "О."}


def test_clean_collapses_whitespace_and_keeps_newlines_out():
    assert bs.clean("а\u00a0 б\n\nв") == "а б в"


def test_make_prompt_keeps_json_example_braces():
    prompt = bs.make_prompt("фрагмент")
    assert '{"вопрос"' in prompt            # шаблон не испорчен форматированием
    assert prompt.endswith("фрагмент")


def test_make_prompt_truncates_long_chunk():
    prompt = bs.make_prompt("я" * 5000)
    assert len(prompt) < 5000
    assert prompt.endswith("я" * 2500)


class _CollWithLimit:
    def __init__(self, docs):
        self.docs = docs
        self.kwargs = {}

    def get(self, limit=None):
        self.kwargs["limit"] = limit
        return {"documents": self.docs}


class _CollLikeFallback:
    """Как запасное хранилище: get() без limit — иначе TypeError."""

    def __init__(self, docs):
        self.docs = docs

    def get(self, where=None):
        return {"documents": self.docs}


def test_read_chunks_sorts_longest_first_and_applies_limit():
    coll = _CollWithLimit(["короткий", "я" * 800, "я" * 1200])
    chunks = bs.read_chunks(2, collection=coll)
    assert [len(c) for c in chunks] == [1200, 800]
    assert coll.kwargs["limit"] == 600


def test_read_chunks_ignores_short_texts():
    chunks = bs.read_chunks(10, collection=_CollWithLimit(["абв", "я" * 500]))
    assert chunks == ["я" * 500]


def test_read_chunks_works_without_limit_support():
    """Fallback-коллекция в памяти не принимает limit — не должно падать."""
    chunks = bs.read_chunks(3, collection=_CollLikeFallback(["я" * 500] * 2))
    assert len(chunks) == 2


def test_read_chunks_handles_empty_collection():
    assert bs.read_chunks(5, collection=_CollWithLimit([])) == []


def test_script_runs_as_file_not_only_as_module():
    """`python scripts/build_style_dataset.py` должен находить пакет memory."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(bs.__file__).resolve().parent.parent
    res = subprocess.run([sys.executable, "scripts/build_style_dataset.py", "--help"],
                         cwd=root, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "--limit" in res.stdout
