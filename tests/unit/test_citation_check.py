"""Проверка цитат: дословная опора на книгу vs правдоподобный пересказ."""

from memory import citation_check as cc

SOURCE = (
    "[1] (Сивухин, с. 112)\n"
    "Энтропия замкнутой системы не убывает при любых процессах, "
    "происходящих в ней. Второе начало термодинамики вводит функцию "
    "состояния S, изменение которой в обратимом процессе равно dQ/T."
)


def test_quote_present_verbatim_is_verified():
    answer = 'В книге сказано: «Энтропия замкнутой системы не убывает».'
    report = cc.verify(answer, SOURCE)
    assert report.checked == 1 and report.ok
    assert report.verified


def test_quote_missing_is_reported():
    answer = 'Здесь: «Энтропия возрастает только в открытых системах при нагреве».'
    report = cc.verify(answer, SOURCE)
    assert report.checked == 1
    assert not report.ok and len(report.missing) == 1


def test_capital_letters_and_yo_do_not_matter():
    answer = "«ЭНТРОПИЯ ЗАМКНУТОЙ СИСТЕМЫ НЕ УБЫВАЕТ»"
    assert cc.verify(answer, SOURCE).ok


def test_typographic_dash_and_newlines_are_normalized():
    source = "Энтропия — функция\nсостояния, она не убывает\nв замкнутой системе."
    answer = "«Энтропия – функция состояния, она не убывает»"
    assert cc.verify(answer, source).ok


def test_short_quotes_are_ignored():
    """Короткое в кавычках — термин, а не цитата."""
    answer = 'Термин «энтропия» тут ключевой.'
    report = cc.verify(answer, SOURCE)
    assert report.checked == 0 and report.ok


def test_straight_quotes_are_supported():
    answer = '"Второе начало термодинамики вводит функцию состояния"'
    assert cc.verify(answer, SOURCE).ok


def test_no_quotes_no_report():
    report = cc.verify("Просто ответ без цитат.", SOURCE)
    assert report.checked == 0 and report.sources_empty is False


def test_empty_sources_mark_report():
    answer = "«Энтропия замкнутой системы не убывает»"
    report = cc.verify(answer, "")
    assert report.sources_empty is True
    assert report.missing


def test_partial_quote_from_two_paragraphs_is_verified():
    answer = ("«Энтропия замкнутой системы не убывает при любых процессах, "
              "происходящих в ней»")
    assert cc.verify(answer, SOURCE).ok


def test_reordered_words_are_not_verified():
    """Смысл тот же, но цитатой это назвать нельзя."""
    answer = "«Не убывает энтропия системы замкнутой при процессах любых»"
    assert not cc.verify(answer, SOURCE).ok


def test_formula_in_quotes_does_not_crash():
    report = cc.verify("«dS = dQ/T для обратимого процесса»", SOURCE)
    assert report.checked == 1
    assert isinstance(report.to_dict(), dict)


def test_report_dict_shape():
    answer = "«Энтропия замкнутой системы не убывает» и «выдуманная фраза про тепло»"
    d = cc.verify(answer, SOURCE).to_dict()
    assert set(d) == {"checked", "verified", "missing", "sources_empty", "ok"}
    assert d["checked"] == 2 and d["verified"] == 1 and len(d["missing"]) == 1


def test_describe_is_quiet_without_quotes():
    assert cc.describe(None) == ""
    assert cc.describe(cc.CitationReport()) == ""


def test_describe_mentions_missing_count():
    report = cc.verify("«выдуманная фраза про тепло и работу»", SOURCE)
    assert "не подтверждено" in cc.describe(report)


class _Hit:
    def __init__(self, text):
        self.text = text

    def citation(self):
        return "Книга, с. 1"


def test_sources_from_hits_matches_prompt_format():
    block = cc.sources_from_hits([_Hit("абзац текста")])
    assert block.startswith("[1] (Книга, с. 1)")
    assert "абзац текста" in block


def test_sources_from_hits_empty():
    assert cc.sources_from_hits([]) == ""
