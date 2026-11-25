"""Потокобезопасность TTSEngine.

XTTS не умеет два синтеза одновременно: `tts_to_file` пишет в общий граф
модели, а ленивая `_load()` при гонке грузит модель дважды. В exe озвучка
запускается фоновой задачей на каждую реплику, поэтому при медленном CPU
два вызова легко накладываются — и процесс умирает в нативном коде.
"""

import threading
import time

from modules.tts_engine import TTSEngine


class _FakeTTS:
    """Заглушка XTTS: считает, сколько синтезов идёт одновременно."""

    def __init__(self, delay: float = 0.15):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._lock = threading.Lock()

    def tts_to_file(self, text=None, speaker_wav=None, language=None,
                    file_path=None, speaker=None):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            with open(file_path, "wb") as f:
                f.write(b"RIFF" + b"\0" * 32)
        finally:
            with self._lock:
                self.active -= 1


def _engine(delay: float = 0.15) -> TTSEngine:
    engine = TTSEngine(device="cpu")
    engine._tts = _FakeTTS(delay)
    engine._load_error = None
    return engine


def _run_parallel(engine: TTSEngine, texts=("раз", "два", "три")):
    results = []
    errors = []

    def worker(text):
        try:
            results.append(engine.synthesize(text))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in texts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results, errors


def test_synthesize_never_runs_twice_at_once():
    engine = _engine()
    results, errors = _run_parallel(engine)

    assert not errors
    assert engine._tts.max_active == 1, "два синтеза шли параллельно"
    assert engine._tts.calls == 3
    assert all(r for r in results)


def test_model_loads_once_under_concurrency(monkeypatch):
    engine = TTSEngine(device="cpu")
    engine._tts = None
    engine._load_error = None
    loads = []
    fake = _FakeTTS(delay=0.05)

    def slow_load():
        if engine._tts is not None:      # как в настоящем _load()
            return
        loads.append(time.time())
        time.sleep(0.1)
        engine._tts = fake

    monkeypatch.setattr(engine, "_load", slow_load)
    _run_parallel(engine, ("раз", "два"))

    assert len(loads) == 1, f"модель грузилась {len(loads)} раз"


def test_unload_waits_for_running_synthesis():
    engine = _engine(delay=0.2)
    started = threading.Event()

    def synth():
        started.set()
        engine.synthesize("привет")

    t = threading.Thread(target=synth)
    t.start()
    started.wait(timeout=5)
    time.sleep(0.05)
    engine.unload()          # не должен оборвать синтез на середине
    t.join(timeout=10)

    assert engine._tts is None
