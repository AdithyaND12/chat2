"""OpenRouter LLM factory (OpenAI-compatible endpoint)."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)


def build_llm(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
) -> ChatOpenAI:
    key = api_key if api_key is not None else settings.openrouter_api_key
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env to enable the conversational brain."
        )
    resolved_model = model or settings.openrouter_model
    resolved_base = base_url or settings.openrouter_base_url
    log.info("Using LLM %s via %s", resolved_model, resolved_base)
    return ChatOpenAI(
        model=resolved_model,
        api_key=key,
        base_url=resolved_base,
        temperature=temperature,
        max_tokens=1500,
        timeout=90.0,
    )
