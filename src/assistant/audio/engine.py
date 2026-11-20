"""Audio pipeline orchestrator: microphone -> wake word -> VAD -> utterance."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from assistant.audio.capture import MicrophoneStream
from assistant.audio.vad import EnergyVAD
from assistant.audio.wakeword import WakeWordDetector
from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)

WakeCallback = Callable[[str], None]
UtteranceCallback = Callable[[np.ndarray], None]
StatusCallback = Callable[[str], None]


class AudioEngine:
    def __init__(
        self,
        on_wake: WakeCallback | None = None,
        on_utterance: UtteranceCallback | None = None,
        on_status: StatusCallback | None = None,
        wake_word: str | None = None,
        sensitivity: float | None = None,
    ) -> None:
        self.on_wake = on_wake or (lambda _label: None)
        self.on_utterance = on_utterance or (lambda _audio: None)
        self.on_status = on_status or (lambda _state: None)

        configured = (wake_word if wake_word is not None else settings.wake_word).strip().lower()
        self.wake_word_enabled = configured not in {"", "none", "off", "disabled"}
        self.detector: WakeWordDetector | None = None
        if self.wake_word_enabled:
            self.detector = WakeWordDetector(configured, sensitivity)

        self.mic = MicrophoneStream(
            sample_rate=settings.sample_rate,
            block_ms=80,
            on_status=lambda s: self.on_status(s),
        )
        self.vad = EnergyVAD(
            sample_rate=settings.sample_rate,
            frame_ms=settings.vad_frame_ms,
            silence_seconds=settings.silence_seconds,
        )

        self.listening = True
        self.state = "idle"
        self._utterance: list[np.ndarray] = []
        self._preroll: deque[np.ndarray] = deque(maxlen=8)
        self._started_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.mic.start()
        self._thread = threading.Thread(target=self._run, name="audio-engine", daemon=True)
        self._thread.start()
        self.on_status("idle")

    def stop(self) -> None:
        self._stop.set()
        self.mic.stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self.state = "idle"
        self._utterance.clear()

    def set_listening(self, enabled: bool) -> None:
        self.listening = enabled
        if not enabled:
            self.state = "idle"
            self._utterance.clear()
            self._preroll.clear()
            self.vad.reset()
            if self.detector is not None:
                self.detector.reset()
        self.on_status("idle" if enabled else "muted")

    def _run(self) -> None:
        while not self._stop.is_set():
            chunk = self.mic.read(timeout=0.2)
            if chunk is None:
                continue
            if not self.listening:
                continue
            try:
                self._process(chunk)
            except Exception as exc:
                log.exception("Audio processing error: %s", exc)

    def _process(self, chunk: np.ndarray) -> None:
        if self.state == "idle":
            self._preroll.append(chunk)
            if self.detector is not None:
                label = self.detector.process(chunk)
                if label is not None:
                    self._begin(label)
            else:
                speech, _ = self.vad.process(chunk)
                if speech:
                    self._begin(None)
            return

        self._utterance.append(chunk)
        _, ended = self.vad.process(chunk)
        duration = sum(len(c) for c in self._utterance) / self.mic.sample_rate
        if ended or duration >= settings.max_utterance_seconds:
            audio = np.concatenate(self._utterance) if self._utterance else chunk
            self._finish(audio)

    def _begin(self, label: str | None) -> None:
        self.state = "recording"
        self._utterance = list(self._preroll)
        self._started_at = time.time()
        self.vad.reset()
        self.on_status("recording")
        if label is not None:
            log.info("Wake word detected: %s", label)
            self.on_wake(label)

    def _finish(self, audio: np.ndarray) -> None:
        self.state = "idle"
        self._utterance.clear()
        self._preroll.clear()
        self.vad.reset()
        if self.detector is not None:
            self.detector.reset()
        self.on_status("idle")
        if len(audio) >= self.mic.sample_rate // 4:
            self.on_utterance(audio)
