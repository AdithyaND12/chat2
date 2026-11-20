"""News tool: GNews when a key is configured, Hacker News as a keyless fallback."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)


def _gnews(topic: str, max_results: int) -> list[dict]:
    key = settings.gnews_api_key
    if not key:
        raise RuntimeError("GNEWS_API_KEY not set")
    if topic.strip().lower() in {"", "top", "headlines", "top headlines", "latest", "news"}:
        url = "https://gnews.io/api/v4/top-headlines"
        params = {"max": max_results, "lang": "en", "apikey": key}
    else:
        url = "https://gnews.io/api/v4/search"
        params = {"q": topic, "max": max_results, "lang": "en", "apikey": key}
    response = httpx.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    return [
        {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": (article.get("source") or {}).get("name", ""),
            "description": article.get("description", ""),
        }
        for article in articles
    ]


def _hackernews(topic: str, max_results: int) -> list[dict]:
    if topic.strip().lower() in {"", "top", "headlines", "top headlines", "latest", "news"}:
        url = "https://hn.algolia.com/api/v1/search"
        params = {"tags": "front_page"}
    else:
        url = "https://hn.algolia.com/api/v1/search"
        params = {"query": topic, "tags": "story"}
    response = httpx.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    hits = response.json().get("hits", [])
    return [
        {
            "title": hit.get("title", ""),
            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "source": "Hacker News",
            "description": f"{hit.get('points', 0)} points",
        }
        for hit in hits[:max_results]
    ]


@tool
def get_news(topic: str = "top headlines", max_results: int = 5) -> str:
    """Fetch recent news headlines. Pass a topic like 'technology' or 'sports'."""
    try:
        try:
            articles = _gnews(topic, max_results)
            provider = "GNews"
        except RuntimeError:
            articles = _hackernews(topic, max_results)
            provider = "Hacker News"
        if not articles:
            return f"I couldn't find any news about {topic}."
        lines = [f"Here are the top stories from {provider}:"]
        for index, article in enumerate(articles, start=1):
            line = f"{index}. {article['title']}"
            if article.get("source"):
                line += f" ({article['source']})"
            lines.append(line)
            if article.get("description"):
                lines.append(f"   {article['description']}")
        return "\n".join(lines)
    except httpx.HTTPError as exc:
        log.warning("News request failed: %s", exc)
        return "I couldn't reach the news service right now."
    except Exception as exc:
        log.exception("News tool error: %s", exc)
        return "Something went wrong while fetching the news."
