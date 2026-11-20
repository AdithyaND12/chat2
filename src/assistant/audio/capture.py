"""Microphone capture using sounddevice."""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from assistant.logging import get_logger

log = get_logger(__name__)


class MicrophoneStream:
    """Captures mono int16 audio and hands chunks to a consumer thread."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 80,
        on_status=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.on_status = on_status
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=256)
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def chunk_duration_ms(self) -> int:
        return int(self.block_size / self.sample_rate * 1000)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("Audio input status: %s", status)
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stop.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        if self.on_status:
            self.on_status("listening")
        log.info("Microphone started at %d Hz", self.sample_rate)

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("Error closing microphone: %s", exc)
            self._stream = None
        if self.on_status:
            self.on_status("stopped")
        log.info("Microphone stopped")

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active
