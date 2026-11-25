"""Регрессия упаковки: модули, без которых frozen-сборка падает."""

import ast
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "sofia.spec"

# torch.utils._config_module → import unittest (torch.distributed, XTTS).
REQUIRED = {"unittest"}


def _excludes():
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "excludes" for t in node.targets)):
            return {ast.literal_eval(e) for e in node.value.elts}
    raise AssertionError("в sofia.spec нет списка excludes")


def test_spec_does_not_exclude_required_stdlib():
    bad = REQUIRED & _excludes()
    assert not bad, f"sofia.spec исключает нужные модули: {sorted(bad)}"


def test_spec_collects_tts_data():
    # Без TTS/.models.json XTTS не загружается в exe.
    text = SPEC.read_text(encoding="utf-8")
    assert 'collect_data_files(\n    "TTS"' in text or 'collect_data_files("TTS"' in text
    assert 'collect_data_files("ko_speech_tools")' in text


def test_spec_keeps_soar_and_clips():
    text = SPEC.read_text(encoding="utf-8")
    assert 'collect_submodules("clips")' in text
    assert "soar_sml" in text
