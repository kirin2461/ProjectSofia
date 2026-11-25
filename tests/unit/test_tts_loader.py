"""Загрузчик образцов голоса для XTTS.

torchaudio ≥ 2.9 читает файлы через torchcodec, а тот требует установленный
FFmpeg. На чистой Windows FFmpeg нет → клонирование голоса падало с
`Could not load libtorchcodec`. Загрузчик на soundfile убирает эту зависимость.
"""

import wave

import numpy as np
import pytest

from modules.tts_engine import _patch_xtts_audio_loader, _soundfile_load_audio


def _write_wav(path, seconds=1.0, rate=24000, channels=1, freq=220.0):
    n = int(rate * seconds)
    t = np.arange(n) / rate
    mono = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([mono] * channels, axis=1)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def test_loads_mono(tmp_path):
    import torch

    path = _write_wav(tmp_path / "a.wav")
    audio = _soundfile_load_audio(path, 24000)
    assert isinstance(audio, torch.Tensor)
    assert audio.shape[0] == 1
    assert abs(audio.shape[1] - 24000) <= 2
    assert float(audio.abs().max()) <= 1.0


def test_mixes_stereo_to_mono(tmp_path):
    path = _write_wav(tmp_path / "st.wav", channels=2)
    audio = _soundfile_load_audio(path, 24000)
    assert audio.shape[0] == 1


def test_resamples(tmp_path):
    path = _write_wav(tmp_path / "44k.wav", rate=44100)
    audio = _soundfile_load_audio(path, 22050)
    assert abs(audio.shape[1] - 22050) <= 8


def test_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        _soundfile_load_audio(tmp_path / "none.wav", 22050)


def test_patch_installs_loader_into_xtts():
    pytest.importorskip("TTS")
    assert _patch_xtts_audio_loader() is True
    from TTS.tts.models import xtts

    assert xtts.load_audio is _soundfile_load_audio
