"""Lightweight energy-based voice activity detection.

A small adaptive noise floor keeps the threshold usable in both quiet and
noisy rooms without pulling in a neural VAD dependency.
"""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_seconds: float = 1.2,
        min_speech_seconds: float = 0.25,
        noise_margin_db: float = 10.0,
        min_threshold_db: float = -50.0,
        max_threshold_db: float = -18.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = max(1, int(sample_rate * frame_ms / 1000))
        self.silence_frames = max(1, int(silence_seconds * 1000 / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_seconds * 1000 / frame_ms))
        self.noise_margin_db = noise_margin_db
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db

        self.noise_floor_db = -60.0
        self.threshold_db = self.noise_floor_db + noise_margin_db
        self.speech_frames = 0
        self.silence_count = 0
        self.in_speech = False
        self._pending: list[np.ndarray] = []

    def reset(self) -> None:
        self.speech_frames = 0
        self.silence_count = 0
        self.in_speech = False
        self._pending.clear()

    def _db(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
        if rms <= 0:
            return -120.0
        return 20.0 * float(np.log10(rms / 32768.0))

    def _split_frames(self, chunk: np.ndarray) -> list[np.ndarray]:
        frames = []
        start = 0
        while start + self.frame_size <= len(chunk):
            frames.append(chunk[start : start + self.frame_size])
            start += self.frame_size
        if start < len(chunk):
            self._pending.append(chunk[start:])
        return frames

    def process(self, chunk: np.ndarray) -> tuple[bool, bool]:
        """Return ``(speech_active, utterance_ended)`` for this chunk."""
        data = chunk
        if self._pending:
            data = np.concatenate([self._pending.pop(), chunk])
        frames = self._split_frames(data)

        speech_active = False
        for frame in frames:
            level = self._db(frame)
            self.noise_floor_db = min(self.noise_floor_db, level)
            self.threshold_db = float(
                np.clip(
                    self.noise_floor_db + self.noise_margin_db,
                    self.min_threshold_db,
                    self.max_threshold_db,
                )
            )
            is_speech = level >= self.threshold_db
            speech_active = speech_active or is_speech

            if is_speech:
                self.speech_frames += 1
                self.silence_count = 0
                self.in_speech = self.speech_frames >= self.min_speech_frames
            elif self.in_speech:
                self.silence_count += 1
                if self.silence_count >= self.silence_frames:
                    self.reset()
                    return True, True
        return speech_active, False
