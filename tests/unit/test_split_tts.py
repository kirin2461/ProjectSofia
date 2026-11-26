"""Нарезка ответа для озвучки: ни один символ не должен потеряться.

Договорённость (см. docstring main._split_for_tts): пробелы нормализуются —
любая последовательность пробельных символов становится одним пробелом,
края обрезаются. После этого "".join(фрагменты) == нормализованный текст.
"""

import random

import pytest

import main

LIMIT = 220


def _norm(text: str) -> str:
    return " ".join(text.split())


def _check(text: str, limit: int = LIMIT) -> list:
    chunks = main._split_for_tts(text, max_len=limit)
    assert "".join(chunks) == _norm(text)
    assert sum(len(c) for c in chunks) == len(_norm(text))
    assert all(0 < len(c) <= limit for c in chunks)
    assert all(c.strip() for c in chunks)
    return chunks


def test_empty_string():
    assert main._split_for_tts("") == []
    assert main._split_for_tts(None) == []


def test_whitespace_only():
    assert main._split_for_tts("   \n\t  \r\n ") == []


def test_short_text_is_one_chunk():
    assert main._split_for_tts("Привет. Как дела?") == ["Привет. Как дела?"]


def test_whitespace_is_normalized():
    assert main._split_for_tts("  а \n\n  б\tв  ") == ["а б в"]


def test_text_without_spaces_longer_than_limit():
    assert main._split_for_tts("б" * 221) == ["б" * 220, "б"]
    chunks = _check("x" * 1000)
    assert len(chunks) == 5


def test_single_word_longer_than_limit():
    word = "а" * 500
    chunks = _check(f"Начало фразы. {word} конец.")
    assert max(len(c) for c in chunks) == LIMIT


def test_prefers_sentence_boundaries():
    text = "А" * 100 + ". " + "Б" * 100 + ". " + "В" * 100 + "."
    chunks = _check(text)
    assert chunks[0].rstrip().endswith(".")
    assert chunks[1].startswith("В")


def test_cut_at_space_keeps_the_space():
    """Раньше p[:cut].strip() + p[cut:].strip() съедали разделитель."""
    text = " ".join(["слово"] * 80)          # ≈ 480 символов, без точек
    chunks = _check(text)
    assert len(chunks) >= 3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_long_text_keeps_every_character(seed):
    rnd = random.Random(seed)
    words = ["энтропия", "мера", "беспорядка", "в", "системе", "и", "это",
             "не", "просто", "x" * 300, "функция", "состояния"]
    parts = []
    while sum(len(p) for p in parts) < 100_000:
        sentence = " ".join(rnd.choice(words) for _ in range(rnd.randint(1, 40)))
        parts.append(sentence + rnd.choice([".", "!", "?", "…", ",", ""]))
        parts.append(rnd.choice([" ", "  ", "\n", " \n\t "]))
    text = "".join(parts)
    _check(text)
