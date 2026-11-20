"""Streaming wake-word detection backed by openWakeWord ONNX models."""

from __future__ import annotations

import numpy as np

from assistant.audio.models import ensure_core_models, ensure_wake_word
from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)


class WakeWordDetector:
    """Feed 80 ms audio frames (1280 int16 samples at 16 kHz) to openWakeWord."""

    def __init__(
        self,
        wake_word: str | None = None,
        sensitivity: float | None = None,
        models_dir: str | None = None,
    ) -> None:
        import openwakeword

        self.wake_word = (wake_word or settings.wake_word).strip().lower()
        self.threshold = settings.wake_word_sensitivity if sensitivity is None else sensitivity
        self.models_dir = models_dir or settings.models_dir

        wake_path = ensure_wake_word(self.wake_word, self.models_dir)
        core = ensure_core_models(self.models_dir)

        self.model = openwakeword.Model(
            wakeword_models=[str(wake_path)],
            inference_framework="onnx",
            melspec_model_path=str(core["melspectrogram.onnx"]),
            embedding_model_path=str(core["embedding_model.onnx"]),
        )
        log.info(
            "Wake word ready: %s (threshold %.2f, model %s)",
            self.wake_word,
            self.threshold,
            wake_path,
        )

    def process(self, chunk: np.ndarray) -> str | None:
        """Return the detected label if the score exceeds the threshold."""
        predictions = self.model.predict(chunk)
        for label, score in predictions.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value >= self.threshold:
                log.debug("Wake word score %.3f for %s", value, label)
                return str(label)
        return None

    def reset(self) -> None:
        self.model.reset()


def available_wake_words() -> list[str]:
    from assistant.audio.models import WAKE_WORDS

    return sorted(WAKE_WORDS)
