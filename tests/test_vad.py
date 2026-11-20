import numpy as np

from assistant.audio.vad import EnergyVAD


def _chunks(data: np.ndarray, size: int = 1280) -> np.ndarray:
    count = len(data) // size
    return data[: count * size].reshape(count, size)


def _tone(seconds: float, amplitude: int = 12000, sample_rate: int = 16000) -> np.ndarray:
    count = int(seconds * sample_rate)
    ramp = np.linspace(0, np.pi / 2, count)
    freq = 2 * np.pi * 440 * np.arange(count) / sample_rate
    return (amplitude * np.sin(freq) * ramp).astype(np.int16)


def test_vad_detects_speech_then_silence():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30, silence_seconds=0.4)
    speech = _tone(0.8)
    silence = np.zeros(16000, dtype=np.int16)
    ended = False
    saw_speech = False
    for chunk in _chunks(np.concatenate([speech, silence])):
        active, just_ended = vad.process(chunk)
        saw_speech = saw_speech or active
        ended = ended or just_ended
    assert saw_speech
    assert ended


def test_vad_ignores_quiet_audio():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30, silence_seconds=0.4)
    quiet = np.zeros(16000, dtype=np.int16)
    saw_speech = False
    for chunk in _chunks(quiet):
        active, _ = vad.process(chunk)
        saw_speech = saw_speech or active
    assert not saw_speech
