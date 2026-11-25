"""Phase-1-01: XTTS v2 — клонирование голоса, полностью локально.

- Ленивая загрузка модели (~2 GB) при первом synthesize
- device: cuda (если доступна) → cpu
- Клонирование по всем .wav из speaker_wavs_dir
- Fallback на стандартного диктора, если образцов нет
- Временные файлы удаляются, ничего не пишется в рабочую директорию
"""

import io
import logging
import os
import tempfile
import threading
import wave
from pathlib import Path
from typing import List, Optional

# XTTS v2 распространяется под некоммерческой лицензией CPML и при первой
# загрузке спрашивает согласие в консоли. София работает сервером (без TTY),
# поэтому подтверждаем согласие заранее, иначе загрузка падает с EOF.
# Коммерческое использование требует лицензии Coqui (licensing@coqui.ai).
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from config.settings import (
    XTTS_MODEL,
    VOICE_SAMPLES_DIR,
    XTTS_LANGUAGE,
    TTS_DEVICE,
    TTS_MIN_FREE_VRAM,
)

logger = logging.getLogger(__name__)


def _soundfile_load_audio(audiopath, sampling_rate):
    """Замена TTS.tts.models.xtts.load_audio без torchaudio.load.

    torchaudio ≥ 2.9 читает файлы только через torchcodec, а тому нужен
    установленный FFmpeg. Без FFmpeg клонирование голоса падало с
    «Could not load libtorchcodec». WAV-образцы спокойно читает soundfile.
    """
    import soundfile as sf
    import torch
    import torchaudio

    data, lsr = sf.read(str(audiopath), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())  # (каналы, сэмплы)
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    audio.clip_(-1, 1)
    return audio


def _patch_xtts_audio_loader() -> bool:
    """Подменить загрузчик образцов в XTTS. True — если подмена удалась."""
    try:
        from TTS.tts.models import xtts
    except Exception as e:  # noqa: BLE001
        logger.debug("XTTS недоступна для патча загрузчика: %s", e)
        return False
    xtts.load_audio = _soundfile_load_audio
    return True


def _free_vram_gb() -> Optional[float]:
    """Свободная VRAM в GB. На Windows mem_get_info врёт (WDDM overcommit),
    поэтому сначала спрашиваем сам драйвер через nvidia-smi."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:  # noqa: BLE001
        return None


def _detect_device(min_free_gb: float = TTS_MIN_FREE_VRAM) -> str:
    """GPU только если реально хватает свободной VRAM, иначе CPU.

    llama-server с 9B-моделью занимает почти всю память 8-гиговой карты,
    поэтому «cuda есть» ≠ «cuda подойдёт».
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"

    free_gb = _free_vram_gb()
    if free_gb is not None and free_gb < min_free_gb:
        logger.warning(
            "Свободно %.1f GB VRAM (< %.1f GB) — XTTS будет на CPU",
            free_gb, min_free_gb,
        )
        return "cpu"
    return "cuda"


class TTSEngine:
    """Ленивая обёртка над coqui-tts (XTTS v2)."""

    def __init__(
        self,
        speaker_wavs_dir: str = VOICE_SAMPLES_DIR,
        device: Optional[str] = None,
        model_name: str = XTTS_MODEL,
        language: str = XTTS_LANGUAGE,
    ):
        self.speaker_wavs_dir = Path(speaker_wavs_dir)
        if device:
            self.device = device
        elif TTS_DEVICE != "auto":
            self.device = TTS_DEVICE
        else:
            self.device = _detect_device()
        self.model_name = model_name
        self.language = language
        self._tts = None
        self._load_error: Optional[str] = None
        # XTTS не потокобезопасен: tts_to_file пишет в общий граф модели,
        # а ленивая загрузка при гонке грузит модель дважды (в exe озвучка
        # идёт фоновой задачей на каждую реплику). Один синтез за раз.
        self._lock = threading.RLock()

    # ── Инициализация ───────────────────────────────────────
    @property
    def available(self) -> bool:
        """True, если модель уже загружена или может быть загружена."""
        return self._tts is not None or self._load_error is None

    def _load(self):
        if self._tts is not None or self._load_error is not None:
            return
        try:
            from TTS.api import TTS

            _patch_xtts_audio_loader()
            logger.info("Загрузка XTTS v2 на %s (это займёт время)...", self.device)
            try:
                self._tts = TTS(model_name=self.model_name).to(self.device)
            except RuntimeError as e:
                if self.device != "cuda" or "out of memory" not in str(e).lower():
                    raise
                self._tts = None
                logger.warning("CUDA OOM при загрузке XTTS — переключаюсь на CPU")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                self.device = "cpu"
                self._tts = TTS(model_name=self.model_name).to(self.device)
            logger.info("XTTS v2 загружена")
        except ImportError:
            self._load_error = "coqui-tts не установлен (pip install coqui-tts)"
            logger.error(self._load_error)
        except Exception as e:  # noqa: BLE001
            self._load_error = f"Не удалось загрузить XTTS: {e}"
            logger.error(self._load_error)

    def speaker_samples(self) -> List[str]:
        if not self.speaker_wavs_dir.exists():
            return []
        return [str(p) for p in sorted(self.speaker_wavs_dir.glob("*.wav"))]

    # ── Синтез ──────────────────────────────────────────────
    def synthesize(self, text: str, language: Optional[str] = None) -> Optional[bytes]:
        """Синтезировать речь. Возвращает WAV-bytes или None при ошибке."""
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            return self._synthesize_locked(text, language)

    def _synthesize_locked(self, text: str, language: Optional[str]) -> Optional[bytes]:
        self._load()
        if self._tts is None:
            return None

        language = language or self.language
        samples = self.speaker_samples()
        if not samples:
            logger.warning(
                "Нет WAV-образцов в %s — используется стандартный диктор. "
                "Добавьте 5–8 файлов для клонирования.",
                self.speaker_wavs_dir,
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.wav"
            try:
                if samples:
                    self._tts.tts_to_file(
                        text=text,
                        speaker_wav=samples,
                        language=language,
                        file_path=str(out_path),
                    )
                else:
                    # Fallback: стандартный диктор без клонирования
                    try:
                        self._tts.tts_to_file(
                            text=text,
                            language=language,
                            file_path=str(out_path),
                        )
                    except Exception:
                        # Для multi-speaker моделей нужен speaker
                        speakers = getattr(self._tts, "speakers", None)
                        speaker = speakers[0] if speakers else None
                        if speaker is None:
                            raise
                        self._tts.tts_to_file(
                            text=text,
                            speaker=speaker,
                            language=language,
                            file_path=str(out_path),
                        )
                return out_path.read_bytes()
            except Exception as e:  # noqa: BLE001
                logger.error("XTTS synth failed: %s", e)
                return None

    # ── Утилиты ─────────────────────────────────────────────
    @staticmethod
    def make_beep(duration_ms: int = 120, freq: int = 440) -> bytes:
        """Короткий сигнал (используется в игровом режиме вместо речи)."""
        import math
        import struct

        sample_rate = 22050
        n = int(sample_rate * duration_ms / 1000)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sample_rate)))
            for i in range(n)
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(frames)
        return buf.getvalue()

    def unload(self):
        """Выгрузить модель (для игрового режима — освободить VRAM)."""
        with self._lock:
            self._tts = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("XTTS выгружена из памяти")
