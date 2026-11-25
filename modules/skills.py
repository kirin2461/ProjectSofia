"""Навыки автопилота: удачный прогон превращается в макрос для похожих задач.

Зачем: типовые дела («залогинься», «найди товар», «скачай счёт») каждый раз
начинаются с нуля, хотя прошлый раз нужные шаги уже были найдены. Здесь
сохраняется не весь прогон, а последовательность действий, которая привела к
успеху, — и подставляется в промпт, когда цель и сайт похожи.

Макрос — подсказка, а не сценарий: модель всё равно смотрит на страницу и
может сделать иначе. Поэтому потеряться на изменившемся сайте он не мешает.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/skills.jsonl")
MAX_SKILLS = 300            # старые вытесняются: файл не должен расти вечно
MIN_GOAL_SIM = 0.45         # ниже — это разные задачи, подсказка только помешает
MAX_STEPS_IN_HINT = 6

_WORD = re.compile(r"[0-9a-zа-яё]+")
_STOP = {
    "и", "в", "на", "с", "по", "для", "мне", "нужно", "надо", "пожалуйста",
    "the", "a", "to", "of", "for", "please", "open", "go",
}


def _path(path: Optional[Path | str]) -> Path:
    return Path(path) if path else DEFAULT_PATH


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower().replace("ё", "е"))
            if len(w) > 1 and w not in _STOP}


def goal_similarity(a: str, b: str) -> float:
    """Пересечение значимых слов (0..1): «открой почту» ≈ «проверь почту»."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def domain_of(url: str) -> str:
    """«https://www.ozon.ru/cart» → «ozon.ru»."""
    host = re.sub(r"^[a-z]+://", "", (url or "").strip().lower())
    host = host.split("/")[0].split("?")[0].split("#")[0]
    host = host.split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _clean_steps(steps: Iterable[dict]) -> List[dict]:
    """Только действия с целью: без done/ask и без дублей подряд."""
    out: List[dict] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        kind = str(step.get("action") or "").strip().lower()
        if kind in ("", "done", "ask"):
            continue
        target = step.get("text") or step.get("url") or step.get("field") or ""
        item = {"action": kind, "target": str(target)[:80]}
        if out and out[-1] == item:
            continue
        out.append(item)
    return out


@dataclass
class Skill:
    """Сохранённый удачный прогон."""

    goal: str
    domain: str
    steps: List[dict] = field(default_factory=list)
    at: float = 0.0
    uses: int = 0

    def to_dict(self) -> dict:
        return {"goal": self.goal, "domain": self.domain,
                "steps": self.steps, "at": self.at, "uses": self.uses}

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(
            goal=str(data.get("goal") or "")[:300],
            domain=str(data.get("domain") or "")[:120],
            steps=[s for s in (data.get("steps") or []) if isinstance(s, dict)],
            at=float(data.get("at") or 0.0),
            uses=int(data.get("uses") or 0),
        )


def _read(path: Optional[Path | str] = None) -> List[Skill]:
    target = _path(path)
    if not target.exists():
        return []
    out: List[Skill] = []
    try:
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("goal"):
                    out.append(Skill.from_dict(data))
    except OSError as e:
        logger.warning("навыки не прочитаны: %s", e)
        return []
    return out


def _write(path: Optional[Path | str], skills: List[Skill]) -> None:
    target = _path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for skill in skills[-MAX_SKILLS:]:
                f.write(json.dumps(skill.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(target)          # перезапись целиком: файл не портится при сбое
    except OSError as e:
        logger.warning("навыки не сохранены: %s", e)


def save_trace(
    goal: str,
    steps: Iterable[dict],
    ok: bool,
    url: str = "",
    path: Optional[Path | str] = None,
) -> Optional[Skill]:
    """Сохраняет прогон, если он удался. Неудачи не копим — учиться нечему."""
    goal = (goal or "").strip()
    clean = _clean_steps(steps)
    if not ok or not goal or len(clean) < 2:
        return None
    skill = Skill(goal=goal[:300], domain=domain_of(url), steps=clean,
                  at=time.time(), uses=1)

    existing = _read(path)
    merged: List[Skill] = []
    for item in existing:
        # та же задача на том же сайте — заменяем свежим путём к цели
        if item.domain == skill.domain and goal_similarity(item.goal, goal) >= 0.8:
            skill.uses = item.uses + 1
            continue
        merged.append(item)
    merged.append(skill)
    _write(path, merged)
    logger.info("навык сохранён: %s (%s, шагов %d)",
                goal[:60], skill.domain or "без сайта", len(clean))
    return skill


def find(goal: str, url: str = "", path: Optional[Path | str] = None,
         limit: int = 1) -> List[Skill]:
    """Похожие навыки: тот же сайт весомее, совпадение цели — обязательно."""
    domain = domain_of(url)
    scored = []
    for skill in _read(path):
        sim = goal_similarity(goal, skill.goal)
        if sim < MIN_GOAL_SIM:
            continue
        score = sim + (0.25 if domain and skill.domain == domain else 0.0)
        scored.append((score, skill))
    scored.sort(key=lambda pair: (pair[0], pair[1].at), reverse=True)
    return [s for _, s in scored[:max(0, limit)]]


def summarize_steps(steps: Iterable[dict], limit: int = MAX_STEPS_IN_HINT) -> str:
    """«click «Войти» → type «почта»» — компактно и без JSON-шума."""
    parts = []
    for step in list(steps or [])[:limit]:
        kind = str(step.get("action") or "")
        target = str(step.get("target") or step.get("text") or step.get("url")
                     or step.get("field") or "")
        parts.append(f"{kind} «{target}»" if target else kind)
    return " → ".join(parts)


def hint(goal: str, url: str = "", path: Optional[Path | str] = None) -> str:
    """Подсказка для промпта автопилота; пустая, если похожего нет."""
    found = find(goal, url, path, limit=1)
    if not found:
        return ""
    skill = found[0]
    steps = summarize_steps(skill.steps)
    if not steps:
        return ""
    where = skill.domain or "этот сайт"
    return (f"Похожая задача уже решалась на {where}: {steps}. "
            "Можешь повторить этот путь, но сверяйся со страницей.")


def forget(goal: str, url: str = "", path: Optional[Path | str] = None) -> int:
    """Удаляет похожие навыки; возвращает, сколько убрано."""
    domain = domain_of(url)
    kept, removed = [], 0
    for skill in _read(path):
        same_site = not domain or skill.domain == domain
        if same_site and goal_similarity(goal, skill.goal) >= 0.8:
            removed += 1
            continue
        kept.append(skill)
    if removed:
        _write(path, kept)
    return removed


def stats(path: Optional[Path | str] = None) -> Dict[str, object]:
    """Сколько навыков и по каким сайтам — для /state."""
    skills = _read(path)
    by_domain: Dict[str, int] = {}
    for skill in skills:
        key = skill.domain or "—"
        by_domain[key] = by_domain.get(key, 0) + 1
    return {"total": len(skills), "by_domain": by_domain}
