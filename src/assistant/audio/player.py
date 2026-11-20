"""Sequential audio playback with interrupt support."""

from __future__ import annotations

import asyncio
import queue
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from assistant.logging import get_logger

log = get_logger(__name__)


class AudioPlayer:
    def __init__(self, tts_engine=None, tmp_dir: str = "data/tts") -> None:
        self.tts_engine = tts_engine
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="audio-player", daemon=True)
        self._stop_event = threading.Event()
        self._playing = False
        self._proc: subprocess.Popen | None = None
        self._counter = 0

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, text: str) -> None:
        if not text.strip():
            return
        self._queue.put(text)

    def stop(self) -> None:
        """Drop queued utterances and stop the current playback."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._playing:
            try:
                sd.stop()
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def shutdown(self) -> None:
        self.stop()
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            text = self._queue.get()
            if text is None:
                break
            try:
                self._speak(text)
            except Exception as exc:
                log.exception("TTS playback failed: %s", exc)

    def _speak(self, text: str) -> None:
        if self.tts_engine is None:
            log.warning("No TTS engine configured; skipping speech for %r", text)
            return
        self._counter += 1
        path = self.tmp_dir / f"utterance_{self._counter}.wav"
        produced = asyncio.run(self.tts_engine.synth_to_file(text, path))
        try:
            if produced.suffix == ".mp3":
                self._play_mp3(produced)
            else:
                self._play(produced)
        finally:
            for candidate in (produced, produced.with_suffix(".wav"), produced.with_suffix(".mp3")):
                try:
                    candidate.unlink()
                except OSError:
                    pass

    def _play(self, path: Path) -> None:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
            raw = handle.readframes(frames)
        data = np.frombuffer(raw, dtype=np.int16)
        self._playing = True
        try:
            sd.play(data, samplerate=sample_rate)
            sd.wait()
        finally:
            self._playing = False

    def _play_mp3(self, path: Path) -> None:
        """Fallback playback when ffmpeg is unavailable to transcode to WAV."""
        if shutil.which("afplay"):
            command = ["afplay", str(path)]
        elif shutil.which("ffplay"):
            command = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        else:
            log.error("No mp3 player available (need ffmpeg, afplay, or ffplay)")
            return
        self._playing = True
        try:
            self._proc = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._proc.wait()
        finally:
            self._playing = False
            self._proc = None
