"""Чтение и правка .env с сохранением комментариев и порядка строк."""

import re
from pathlib import Path
from typing import Dict

_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_env(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in (text or "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        # инлайн-комментарий вне кавычек: KEY=value  # comment
        if value and value[0] not in "\"'":
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos]
        result[key] = _unquote(value)
    return result


def read_env(path: Path) -> Dict[str, str]:
    try:
        return parse_env(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}


def upsert_text(text: str, updates: Dict[str, str]) -> str:
    """Заменить существующие ключи на месте, новые — дописать в конец."""
    lines = (text or "").splitlines()
    pending = dict(updates)
    out = []
    for line in lines:
        m = _KEY_RE.match(line)
        if m and m.group(1) in pending:
            key = m.group(1)
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    if pending:
        if out and out[-1].strip():
            out.append("")
        for key, value in pending.items():
            out.append(f"{key}={value}")
    return "\n".join(out).rstrip("\n") + "\n"


def upsert_env(path: Path, updates: Dict[str, str]) -> Dict[str, str]:
    path = Path(path)
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        current = ""
    new_text = upsert_text(current, updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return parse_env(new_text)
