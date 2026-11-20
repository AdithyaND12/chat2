"""Speech-to-text with Deepgram (primary) and OpenAI Whisper (fallback)."""

from __future__ import annotations

import io
import wave
from typing import Protocol

import httpx
import numpy as np

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)


def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
    return buffer.getvalue()


class SpeechToText(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class DeepgramSTT:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.deepgram_api_key
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        wav = audio_to_wav(audio, sample_rate)
        params = {
            "model": "nova-2",
            "smart_format": "true",
            "language": "en",
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": "1",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        response = httpx.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            content=wav,
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        alternatives = payload.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
        if not alternatives:
            return ""
        transcript = alternatives[0].get("transcript", "").strip()
        log.debug(
            "Deepgram transcript (confidence %.2f): %s",
            alternatives[0].get("confidence", 0.0),
            transcript,
        )
        return transcript


class WhisperSTT:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for Whisper fallback")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        wav = audio_to_wav(audio, sample_rate)
        files = {"file": ("utterance.wav", wav, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "json"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()


class CompositeSTT:
    """Try the configured provider first, then fall back to the other one."""

    def __init__(self, primary: SpeechToText | None = None) -> None:
        self.primary = primary or self._build_primary()
        self.fallback: SpeechToText | None = None
        if isinstance(self.primary, DeepgramSTT):
            try:
                self.fallback = WhisperSTT()
            except RuntimeError:
                self.fallback = None
        elif isinstance(self.primary, WhisperSTT):
            try:
                self.fallback = DeepgramSTT()
            except RuntimeError:
                self.fallback = None

    @staticmethod
    def _build_primary() -> SpeechToText:
        provider = settings.stt_provider.lower()
        if provider == "whisper":
            return WhisperSTT()
        try:
            return DeepgramSTT()
        except RuntimeError:
            log.warning("Deepgram key missing; falling back to Whisper")
            return WhisperSTT()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            text = self.primary.transcribe(audio, sample_rate)
            if text:
                return text
        except Exception as exc:
            log.warning("Primary STT failed: %s", exc)
        if self.fallback is not None:
            try:
                return self.fallback.transcribe(audio, sample_rate)
            except Exception as exc:
                log.error("Fallback STT failed: %s", exc)
        return ""


def transcribe(audio: np.ndarray, sample_rate: int | None = None) -> str:
    engine: SpeechToText = CompositeSTT()
    return engine.transcribe(audio, sample_rate or settings.sample_rate)
