"""Извлечение JSON из ответа модели — одно место на весь проект.

Раньше логика была скопирована в автопилоте и в сборщике датасета; копии
разъехались бы на первом же изменении формата. Здесь один разбор: терпит
```-обёртки, текст вокруг, вложенные объекты и экранированные кавычки.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

_FENCE_RE = re.compile(r"```(?:json|javascript|js)?", re.IGNORECASE)


def _objects(text: str):
    """Перебирает все законченные JSON-объекты в тексте."""
    cleaned = _FENCE_RE.sub("", text or "").replace("```", "")
    start = cleaned.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        yield json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    break
        start = cleaned.find("{", start + 1)


def first_json(text: str) -> Optional[dict]:
    """Первый объект в ответе модели или None."""
    for obj in _objects(text):
        if isinstance(obj, dict):
            return obj
    return None


def json_list(value: Any, limit: int = 3) -> List[str]:
    """Список строк из поля объекта: строки, списки, мусор — отбрасываем."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = " ".join(item.split()).strip()
        if len(item) >= 3 and item not in out:
            out.append(item)
    return out[:limit]
