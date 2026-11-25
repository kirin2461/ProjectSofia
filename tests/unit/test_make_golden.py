"""Черновик golden set: выбор фрагментов, разбор ответа модели, формат JSONL."""

import json

from eval import metrics
from scripts import make_golden as mg

TEXT = "Энтропия замкнутой системы не убывает. " * 12


class FakeCollection:
    def __init__(self, docs, metas):
        self.docs, self.metas = docs, metas

    def get(self, limit=None, include=None):
        return {"documents": self.docs, "metadatas": self.metas}


def test_read_records_skips_short_and_archived():
    col = FakeCollection(
        [TEXT, "коротко", TEXT],
        [{"title": "Физика", "page_start": 3, "page_end": 4},
         {"title": "Физика", "page_start": 5},
         {"title": "Старое", "page_start": 1, "archived": True}],
    )
    recs = mg.read_records(col)
    assert len(recs) == 1
    assert recs[0]["book"] == "Физика" and recs[0]["page"] == 3 and recs[0]["page_end"] == 4


def test_read_records_falls_back_to_source():
    col = FakeCollection([TEXT], [{"source": "kurs.pdf", "page_start": 2}])
    assert mg.read_records(col)[0]["book"] == "kurs.pdf"


def test_pick_spreads_over_book():
    recs = [{"book": "А", "page": p, "page_end": p, "text": TEXT} for p in range(1, 101)]
    chosen = mg.pick(recs, per_book=4)
    pages = sorted(r["page"] for r in chosen)
    assert len(pages) == 4
    assert pages[0] <= 25 and pages[-1] > 75


def test_pick_takes_all_when_few():
    recs = [{"book": "А", "page": 1, "page_end": 1, "text": TEXT},
            {"book": "Б", "page": 2, "page_end": 2, "text": TEXT}]
    assert len(mg.pick(recs, per_book=5)) == 2


def test_pick_is_deterministic():
    recs = [{"book": "А", "page": p, "page_end": p, "text": TEXT} for p in range(50)]
    assert mg.pick(recs, 5) == mg.pick(recs, 5)


def test_to_question_parses_json_block():
    raw = 'Вот:\n```json\n{"вопрос": "Почему энтропия не убывает?"}\n```'
    assert mg.to_question(raw) == "Почему энтропия не убывает?"


def test_to_question_accepts_english_key():
    assert mg.to_question('{"question": "What is entropy here?"}') == "What is entropy here?"


def test_to_question_rejects_garbage():
    assert mg.to_question("нет json") is None
    assert mg.to_question('{"вопрос": "а?"}') is None
    assert mg.to_question('{"вопрос": "' + "о" * 300 + '"}') is None
    assert mg.to_question('{"вопрос": "О чём этот фрагмент рассказывает?"}') is None


def test_build_dedupes_and_records_pages():
    recs = [{"book": "Курс общей физики том 1 Сивухин", "page": 10, "page_end": 11, "text": TEXT},
            {"book": "Курс общей физики том 1 Сивухин", "page": 40, "page_end": 40, "text": TEXT}]
    cases = mg.build(lambda p: '{"вопрос": "Может ли энтропия уменьшаться?"}', recs)
    assert len(cases) == 1
    assert cases[0]["pages"] == [10, 11]
    assert cases[0]["book"] == "Курс общей физики том"


def test_build_survives_model_errors():
    def boom(prompt):
        raise RuntimeError("сервер упал")

    assert mg.build(boom, [{"book": "А", "page": 1, "page_end": 1, "text": TEXT}]) == []


def test_output_is_valid_golden():
    cases = mg.build(lambda p: '{"вопрос": "Что такое энтропия системы?"}',
                     [{"book": "Физика", "page": 2, "page_end": 2, "text": TEXT}])
    cases += mg.negatives(2)
    loaded = metrics.load_cases(mg.to_jsonl(cases).splitlines())
    assert len(loaded) == 3
    assert sum(1 for c in loaded if c.get("should_find") is False) == 2
    for line in mg.to_jsonl(cases).strip().splitlines():
        json.loads(line)


def test_main_refuses_to_overwrite(tmp_path, capsys):
    out = tmp_path / "golden.jsonl"
    out.write_text("проверенный набор", encoding="utf-8")
    assert mg.main(["--out", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == "проверенный набор"
