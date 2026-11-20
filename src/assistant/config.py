"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM (OpenRouter, OpenAI-compatible)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-sonnet-4-20250514"

    # Speech-to-text
    deepgram_api_key: str = ""
    stt_provider: str = "deepgram"
    openai_api_key: str = ""

    # Text-to-speech
    edge_voice: str = "en-US-AriaNeural"

    # Tool data sources
    gnews_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_api_key: str = ""

    # Voice pipeline
    wake_word: str = "hey_jarvis"
    wake_word_sensitivity: float = 0.5
    silence_seconds: float = 1.2
    sample_rate: int = 16000
    vad_frame_ms: int = 30
    max_utterance_seconds: float = 30.0

    # Assistant
    assistant_name: str = "Nova"
    timezone: str = "UTC"
    notifications_enabled: bool = True

    # Storage
    db_path: str = "data/assistant.db"
    checkpoint_path: str = "data/checkpoint.sqlite"
    models_dir: str = "data/models"
    log_level: str = "INFO"

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_stt(self) -> bool:
        if self.stt_provider == "deepgram":
            return bool(self.deepgram_api_key)
        if self.stt_provider == "whisper":
            return bool(self.openai_api_key)
        return bool(self.deepgram_api_key or self.openai_api_key)

    def ensure_paths(self) -> None:
        for raw in (self.db_path, self.checkpoint_path):
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def local_timezone_name() -> str:
    """Best-effort local timezone name, falling back to the configured value."""
    tz_env = os.environ.get("TZ")
    if tz_env:
        return tz_env
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("localtime").key  # type: ignore[attr-defined]
    except Exception:
        return settings.timezone
