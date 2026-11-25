"""Навыки автопилота: сохранение удачных прогонов и подсказка для похожих задач."""

from modules import skills


STEP = {"action": "click", "text": "Войти"}


def trace(*targets):
    return [{"action": "click", "text": t} for t in targets]


def test_domain_of_strips_scheme_www_and_path():
    assert skills.domain_of("https://www.ozon.ru/cart?x=1") == "ozon.ru"
    assert skills.domain_of("http://mail.yandex.ru") == "mail.yandex.ru"
    assert skills.domain_of("ozon.ru/foo") == "ozon.ru"
    assert skills.domain_of("") == ""
    assert skills.domain_of("https://user:pw@site.ru:8080/x") == "site.ru"


def test_goal_similarity_same_words():
    assert skills.goal_similarity("открой почту", "открой почту") == 1.0
    assert skills.goal_similarity("проверь почту", "открой почту") > 0.2
    assert skills.goal_similarity("купи билет", "проверь почту") == 0.0


def test_save_requires_success(tmp_path):
    path = tmp_path / "s.jsonl"
    assert skills.save_trace("цель", trace("a", "b"), ok=False, url="a.ru", path=path) is None
    assert not path.exists()


def test_save_requires_two_steps(tmp_path):
    path = tmp_path / "s.jsonl"
    assert skills.save_trace("цель", trace("a"), ok=True, url="a.ru", path=path) is None


def test_save_skips_done_and_ask(tmp_path):
    path = tmp_path / "s.jsonl"
    steps = [{"action": "click", "text": "a"},
             {"action": "done", "summary": "готово"},
             {"action": "type", "field": "поиск"},
             {"action": "ask", "summary": "нужен код"}]
    skill = skills.save_trace("найди товар", steps, ok=True, url="a.ru", path=path)
    assert skill is not None
    assert [s["action"] for s in skill.steps] == ["click", "type"]


def test_save_replaces_same_goal_on_same_site(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace("a", "b"), ok=True, url="mail.ru", path=path)
    skills.save_trace("открой почту", trace("c", "d"), ok=True, url="mail.ru", path=path)
    items = skills.find("открой почту", "https://mail.ru", path=path, limit=5)
    assert len(items) == 1
    assert [s["target"] for s in items[0].steps] == ["c", "d"]
    assert items[0].uses == 2


def test_same_goal_on_other_site_is_separate(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace("a", "b"), ok=True, url="mail.ru", path=path)
    skills.save_trace("открой почту", trace("c", "d"), ok=True, url="gmail.com", path=path)
    assert len(skills.find("открой почту", path=path, limit=5)) == 2


def test_find_requires_goal_overlap(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("купи билет на поезд", trace("a", "b"), ok=True, url="rzd.ru", path=path)
    assert skills.find("свари борщ", "rzd.ru", path=path) == []


def test_find_prefers_same_site(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace("a", "b"), ok=True, url="mail.ru", path=path)
    skills.save_trace("открой почту", trace("c", "d"), ok=True, url="gmail.com", path=path)
    best = skills.find("открой почту", "https://mail.ru/inbox", path=path, limit=1)[0]
    assert best.domain == "mail.ru"


def test_hint_mentions_site_and_steps(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace("Войти", "Почта"), ok=True,
                      url="mail.ru", path=path)
    text = skills.hint("открой почту", "https://mail.ru", path=path)
    assert "mail.ru" in text
    assert "Войти" in text and "Почта" in text


def test_hint_empty_without_match(tmp_path):
    assert skills.hint("свари борщ", path=tmp_path / "нет.jsonl") == ""


def test_hint_limited_length(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace(*[f"шаг{i}" for i in range(20)]),
                      ok=True, url="mail.ru", path=path)
    text = skills.hint("открой почту", "mail.ru", path=path)
    assert len(text) < 400
    assert "шаг5" in text and "шаг6" not in text


def test_summarize_is_compact():
    assert skills.summarize_steps([{"action": "click", "text": "ОК"}]) == "click «ОК»"
    assert skills.summarize_steps([{"action": "scroll"}]) == "scroll"


def test_broken_lines_are_skipped(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("не json\n" + '{"goal": "открой почту", "domain": "m.ru", '
                    '"steps": [{"action": "click", "target": "a"}]}\n', encoding="utf-8")
    assert skills.stats(path)["total"] == 1


def test_forget_removes_matching(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace("a", "b"), ok=True, url="mail.ru", path=path)
    skills.save_trace("купи билет", trace("c", "d"), ok=True, url="rzd.ru", path=path)
    assert skills.forget("открой почту", "https://mail.ru", path=path) == 1
    assert skills.stats(path)["total"] == 1


def test_stats_by_domain(tmp_path):
    path = tmp_path / "s.jsonl"
    skills.save_trace("открой почту", trace("a", "b"), ok=True, url="mail.ru", path=path)
    skills.save_trace("купи билет", trace("c", "d"), ok=True, url="rzd.ru", path=path)
    assert skills.stats(path) == {"total": 2, "by_domain": {"mail.ru": 1, "rzd.ru": 1}}


def test_file_is_capped(tmp_path, monkeypatch):
    path = tmp_path / "s.jsonl"
    monkeypatch.setattr(skills, "MAX_SKILLS", 3)
    for i in range(6):
        skills.save_trace(f"задача номер {i}", trace("a", "b"), ok=True,
                          url=f"site{i}.ru", path=path)
    items = skills._read(path)
    assert len(items) == 3
    assert items[-1].goal.endswith("5")
