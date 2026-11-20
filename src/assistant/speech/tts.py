"""Text-to-speech via edge-tts (free, neural voices)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import edge_tts

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)


def _convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert edge-tts MP3 output to PCM WAV using ffmpeg when available."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ar",
        "24000",
        "-ac",
        "1",
        str(dst),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("ffmpeg conversion failed: %s", exc)
        return False
    return result.returncode == 0 and dst.exists() and dst.stat().st_size > 0


class TextToSpeech:
    def __init__(self, voice: str | None = None, rate: str = "+0%") -> None:
        self.voice = voice or settings.edge_voice
        self.rate = rate

    async def synth_to_file(self, text: str, path: Path) -> Path:
        """Synthesize ``text`` to ``path``.

        Returns a WAV path when ffmpeg is available, otherwise an MP3 path.
        """
        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        tmp_mp3 = path.with_suffix(".mp3")
        await communicate.save(str(tmp_mp3))
        if _convert_to_wav(tmp_mp3, path):
            try:
                tmp_mp3.unlink()
            except OSError:
                pass
            return path
        mp3_path = path.with_suffix(".mp3")
        if mp3_path != tmp_mp3:
            os.replace(tmp_mp3, mp3_path)
        return mp3_path

    async def synth(self, text: str) -> bytes:
        import io

        buffer = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()


async def list_voices() -> list[dict]:
    return await edge_tts.list_voices()
