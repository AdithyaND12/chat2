"""Download and cache the ONNX models used by openWakeWord.

The openWakeWord wheel does not ship the model files on macOS, so they are
fetched once from the project's GitHub release and cached under ``models_dir``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)

RELEASE = "v0.5.1"
BASE_URL = f"https://github.com/dscripka/openWakeWord/releases/download/{RELEASE}"

WAKE_WORDS: dict[str, str] = {
    "alexa": "alexa_v0.1.onnx",
    "hey_mycroft": "hey_mycroft_v0.1.onnx",
    "hey_jarvis": "hey_jarvis_v0.1.onnx",
    "hey_rhasspy": "hey_rhasspy_v0.1.onnx",
    "timer": "timer_v0.1.onnx",
    "weather": "weather_v0.1.onnx",
}

CORE_MODELS: dict[str, str] = {
    "melspectrogram.onnx": "melspectrogram.onnx",
    "embedding_model.onnx": "embedding_model.onnx",
}


def _download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 16):
                handle.write(chunk)
    os.replace(tmp, dest)
    return dest


def ensure_model(filename: str, models_dir: str | Path | None = None) -> Path:
    root = Path(models_dir or settings.models_dir)
    dest = root / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{BASE_URL}/{filename}"
    return _download(url, dest)


def ensure_wake_word(name: str, models_dir: str | Path | None = None) -> Path:
    if name not in WAKE_WORDS:
        raise ValueError(f"Unknown wake word {name!r}. Available: {', '.join(sorted(WAKE_WORDS))}")
    return ensure_model(WAKE_WORDS[name], models_dir)


def ensure_core_models(models_dir: str | Path | None = None) -> dict[str, Path]:
    return {name: ensure_model(filename, models_dir) for name, filename in CORE_MODELS.items()}


def model_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()[:12]
